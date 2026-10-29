import os
import argparse
import random
import json
import numpy as np
import torch

from opacus.validators import ModuleValidator

from src.dataset import get_client_dataloaders
from src.model import MedicalCNN, get_weights, set_weights
from src.client_dp import PrivateMedicalFLClient
from src.blockchain_client import BlockchainLogger
from src.nacs import baseline_contribution_score, NACSTracker
from src.utils import save_metrics_json, plot_multiseed_comparison
from run_phase1 import fed_avg, evaluate_global_model
from run_phase4_nacs import record_nacs_reward_on_chain


def set_random_seed(seed: int):
    """Set random seed across Python, NumPy, and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_live_status(status_dict: dict, path: str = "results/live_status.json"):
    """Write live status object to JSON file for UI polling."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(status_dict, f, indent=2)


def run_single_seed_nacs(
    seed: int,
    seed_index: int,
    total_seeds: int,
    num_rounds: int = 10,
    target_epsilons=[1.0, 3.0, 8.0, 15.0, 20.0],
    enable_blockchain: bool = True,
    alpha: float = 0.5,
):
    set_random_seed(seed)
    num_clients = len(target_epsilons)

    print("-" * 85, flush=True)
    print(
        f"RUNNING SEED {seed_index + 1}/{total_seeds} (Random Seed: {seed}) | "
        f"Blockchain Logging: {'ENABLED (Seed 0)' if enable_blockchain else 'DISABLED (Off-Chain to save gas)'}",
        flush=True,
    )
    print("-" * 85, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_epochs = 2
    lr = 0.005

    logger = None
    if enable_blockchain:
        try:
            logger = BlockchainLogger()
        except Exception as e:
            print(f"  [Warning] Could not connect to Blockchain node: {e}", flush=True)
            print("  Falling back to off-chain logging for this seed.", flush=True)
            enable_blockchain = False

    client_train_loaders, client_val_loaders, test_loader, _, client_sizes = get_client_dataloaders(
        num_clients=num_clients, alpha=alpha, batch_size=32, return_client_sizes=True
    )

    clients = [
        PrivateMedicalFLClient(
            client_id=i,
            train_loader=client_train_loaders[i],
            val_loader=client_val_loaders[i],
            device=device,
            target_epsilon=target_epsilons[i],
            target_delta=1e-5,
            total_rounds=num_rounds,
            local_epochs=local_epochs,
            lr=lr,
        )
        for i in range(num_clients)
    ]

    nacs_tracker = NACSTracker()

    base = MedicalCNN(in_channels=1, num_classes=2)
    global_model = (
        ModuleValidator.fix(base).to(device)
        if not ModuleValidator.is_valid(base)
        else base.to(device)
    )
    criterion = torch.nn.CrossEntropyLoss()

    initial_metrics = evaluate_global_model(global_model, test_loader, device, criterion)

    current_weights = get_weights(global_model)
    config = {"local_epochs": local_epochs, "lr": lr}

    seed_history = {
        str(i): {
            "client_id": i,
            "target_epsilon": target_epsilons[i],
            "client_size": client_sizes[i],
            "history": [],
        }
        for i in range(num_clients)
    }

    for rnd in range(1, num_rounds + 1):
        client_updates = []
        client_sizes_round = []

        for client in clients:
            cid = client.client_id
            target_eps = target_epsilons[cid]
            c_size = client_sizes[cid]

            loss_before, _, _ = client.evaluate(current_weights, config)
            updated_weights, num_samples, fit_metrics = client.fit(current_weights, config)
            loss_after = fit_metrics["train_loss"]
            noise_mult = fit_metrics.get("noise_multiplier", 0.0)

            client_updates.append(updated_weights)
            client_sizes_round.append(num_samples)

            b_score = baseline_contribution_score(loss_before, loss_after)
            n_score = nacs_tracker.update(
                client_id=cid,
                round_num=rnd,
                loss_before=loss_before,
                loss_after=loss_after,
                noise_multiplier=noise_mult,
                client_size=c_size,
                all_client_sizes=client_sizes,
            )
            latest_scores = nacs_tracker.get_latest_scores(cid)

            if enable_blockchain and logger is not None:
                try:
                    logger.log_update(round_num=rnd, client_id=cid, weights=updated_weights)
                    record_nacs_reward_on_chain(
                        logger=logger, round_num=rnd, client_id=cid, nacs_score=n_score
                    )
                except Exception as ex:
                    pass

            seed_history[str(cid)]["history"].append({
                "round": rnd,
                "target_epsilon": target_eps,
                "noise_multiplier": noise_mult,
                "loss_before": float(loss_before),
                "loss_after": float(loss_after),
                "client_size": c_size,
                "all_client_sizes": client_sizes,
                "baseline_score": b_score,
                "nacs_score": n_score,
                "score_size_only": latest_scores["size_only"],
                "score_size_quality": latest_scores["size_quality"],
                "score_size_quality_noise": latest_scores["size_quality_noise"],
                "score_full_nacs": latest_scores["full_nacs"],
            })

        current_weights = fed_avg(client_updates, client_sizes)
        set_weights(global_model, current_weights)
        metrics = evaluate_global_model(global_model, test_loader, device, criterion)

        # Update live status for UI
        write_live_status({
            "status": "running",
            "phase": "4-multiseed",
            "current_seed": seed_index + 1,
            "total_seeds": total_seeds,
            "current_round": rnd,
            "total_rounds": num_rounds,
            "last_accuracy": float(metrics["accuracy"]),
            "last_f1": float(metrics["f1"]),
        })

        print(
            f"  Seed {seed_index + 1}/{total_seeds} | Round {rnd}/{num_rounds} complete -> "
            f"Global Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}",
            flush=True,
        )

    # Compute variance reduction for this seed
    seed_reductions = {}
    for i in range(num_clients):
        cid_str = str(i)
        c_hist = seed_history[cid_str]["history"]
        baseline_scores = [h["baseline_score"] for h in c_hist]
        nacs_scores = [h["nacs_score"] for h in c_hist]

        b_var = float(np.var(baseline_scores))
        n_var = float(np.var(nacs_scores))
        reduction_pct = ((b_var - n_var) / b_var * 100.0) if b_var > 0 else 0.0
        seed_reductions[i] = float(reduction_pct)

    return seed_reductions


def run_phase4_multiseed_experiment(
    num_seeds: int = 5,
    num_rounds: int = 10,
    target_epsilons=[1.0, 3.0, 8.0, 15.0, 20.0],
):
    print("=" * 85, flush=True)
    print("PHASE 4: Multi-Seed NACS Robustness Benchmark", flush=True)
    print(f"  Total Seeds : {num_seeds} (Seeds 42..{42 + num_seeds - 1})", flush=True)
    print(f"  Rounds/Seed : {num_rounds}", flush=True)
    print(f"  Clients     : {len(target_epsilons)} (epsilons: {target_epsilons})", flush=True)
    print(
        "  Note        : On-chain blockchain logging active for SEED 0 only to conserve gas.",
        flush=True,
    )
    print("=" * 85, flush=True)

    base_seed = 42
    all_seed_results = {i: [] for i in range(len(target_epsilons))}

    write_live_status({
        "status": "running",
        "phase": "4-multiseed",
        "current_seed": 1,
        "total_seeds": num_seeds,
        "current_round": 0,
        "total_rounds": num_rounds,
    })

    for s_idx in range(num_seeds):
        current_seed = base_seed + s_idx
        # Enable blockchain logging ONLY for seed index 0
        enable_bc = (s_idx == 0)

        seed_reductions = run_single_seed_nacs(
            seed=current_seed,
            seed_index=s_idx,
            total_seeds=num_seeds,
            num_rounds=num_rounds,
            target_epsilons=target_epsilons,
            enable_blockchain=enable_bc,
        )

        for client_id, red_pct in seed_reductions.items():
            all_seed_results[client_id].append(red_pct)

    # Compute final statistics across seeds
    final_summary = {}
    print("\n" + "=" * 95, flush=True)
    print("PHASE 4 MULTI-SEED ROBUSTNESS RESULTS (MEAN +/- STD OVER SEEDS)", flush=True)
    print("=" * 95, flush=True)
    print(
        f"{'Client':<10} | {'Target eps':<12} | {'Per-Seed Reductions (%)':<35} | {'Mean ± Std Var Reduction':<25}",
        flush=True,
    )
    print("-" * 95, flush=True)

    for i in range(len(target_epsilons)):
        eps = target_epsilons[i]
        vals = all_seed_results[i]
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))

        final_summary[str(i)] = {
            "client_id": i,
            "target_epsilon": eps,
            "mean_variance_reduction": mean_val,
            "std_variance_reduction": std_val,
            "per_seed_values": vals,
        }

        vals_str = ", ".join([f"{v:.1f}%" for v in vals])
        print(
            f"Client {i + 1:<3d} | {eps:<12.1f} | {vals_str:<35} | {mean_val:>6.2f}% ± {std_val:<5.2f}%",
            flush=True,
        )

    print("=" * 95 + "\n", flush=True)

    # Save final JSON summary
    json_path = "results/phase4_nacs_multiseed.json"
    save_metrics_json(final_summary, json_path)

    # Update live_status.json to complete
    write_live_status({
        "status": "complete",
        "phase": "4-multiseed",
        "current_seed": num_seeds,
        "total_seeds": num_seeds,
        "current_round": num_rounds,
        "total_rounds": num_rounds,
        "results": final_summary,
    })

    # Generate comparison plot
    plot_multiseed_comparison(json_path=json_path, save_path="results/phase4_multiseed_comparison.png")

    return final_summary


def main():
    parser = argparse.ArgumentParser(description="Phase 4 Multi-Seed NACS Robustness Benchmark")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds (default: 5)")
    parser.add_argument("--rounds", type=int, default=10, help="FL rounds per seed (default: 10)")
    args = parser.parse_args()

    run_phase4_multiseed_experiment(num_seeds=args.seeds, num_rounds=args.rounds)


if __name__ == "__main__":
    main()
