import os
import argparse
import random
import json
import numpy as np
import torch

from opacus.validators import ModuleValidator

from src.dataset_har import get_har_client_dataloaders
from src.model_har import HARClassifier, get_weights, set_weights
from src.client_dp import PrivateMedicalFLClient
from src.blockchain_client import BlockchainLogger
from src.nacs import baseline_contribution_score, NACSTracker
from src.utils import (
    save_metrics_json,
    plot_phase5_har_comparison,
    plot_multiseed_comparison,
    print_domain_generalizability_summary,
)
from src.training_health import check_training_health
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


def run_phase5_har_experiment(
    target_epsilons=[1.0, 3.0, 8.0, 15.0, 20.0],
    num_rounds=10,
    seed: int = 42,
    enable_blockchain: bool = True,
):
    set_random_seed(seed)
    num_clients = len(target_epsilons)

    print("=" * 95, flush=True)
    print("PHASE 5: Cross-Domain Generalizability Benchmark — UCI HAR Dataset", flush=True)
    print("  Note: Validation of framework generalizability on wearable/phone sensor data.", flush=True)
    print(f"  Clients: {num_clients} (Partitioned by REAL Subject ID) | Epsilons: {target_epsilons}", flush=True)
    print(f"  Rounds : {num_rounds}  | Smart Contract: ContributionLedger", flush=True)
    print("=" * 95, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_epochs = 2
    lr = 0.005

    # Connect to Blockchain logger
    logger = None
    if enable_blockchain:
        try:
            logger = BlockchainLogger()
        except Exception as e:
            print(f"  [Warning] Could not connect to Blockchain node: {e}", flush=True)
            print("  Falling back to off-chain logging.", flush=True)
            enable_blockchain = False

    # Load UCI HAR dataset partitions (partitioned by real subject ID)
    client_train_loaders, client_val_loaders, test_loader, _, client_sizes = get_har_client_dataloaders(
        num_clients=num_clients, batch_size=32, seed=seed, return_client_sizes=True
    )

    # Instantiate DP clients using HARClassifier MLP
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
            model_cls=HARClassifier,
        )
        for i in range(num_clients)
    ]

    nacs_tracker = NACSTracker()

    # Global HARClassifier model for server evaluation
    base = HARClassifier(input_dim=561, num_classes=6)
    global_model = (
        ModuleValidator.fix(base).to(device)
        if not ModuleValidator.is_valid(base)
        else base.to(device)
    )
    criterion = torch.nn.CrossEntropyLoss()

    initial_metrics = evaluate_global_model(global_model, test_loader, device, criterion)
    print(
        f"Round 0 (Initial) -> Test Accuracy: {initial_metrics['accuracy']:.4f}, "
        f"F1: {initial_metrics['f1']:.4f}, Loss: {initial_metrics['loss']:.4f}\n",
        flush=True,
    )

    history = {
        "domain": "UCI HAR (Human Activity Recognition)",
        "target_epsilons": target_epsilons,
        "rounds": [0],
        "accuracy": [initial_metrics["accuracy"]],
        "f1": [initial_metrics["f1"]],
        "precision": [initial_metrics["precision"]],
        "recall": [initial_metrics["recall"]],
        "loss": [initial_metrics["loss"]],
        "per_client": {
            str(i): {
                "client_id": i,
                "target_epsilon": target_epsilons[i],
                "noise_multiplier": 0.0,
                "client_size": client_sizes[i],
                "history": [],
            }
            for i in range(num_clients)
        },
        "blockchain_log": [],
    }

    current_weights = get_weights(global_model)
    config = {"local_epochs": local_epochs, "lr": lr}

    for rnd in range(1, num_rounds + 1):
        print(f"--- Phase 5 (HAR) FL Round {rnd}/{num_rounds} ---", flush=True)
        client_updates = []
        client_sizes_round = []

        for client in clients:
            cid = client.client_id
            target_eps = target_epsilons[cid]
            c_size = client_sizes[cid]

            # 1. Pre-training loss evaluation
            loss_before, _, _ = client.evaluate(current_weights, config)

            # 2. Local DP fit
            updated_weights, num_samples, fit_metrics = client.fit(current_weights, config)
            loss_after = fit_metrics["train_loss"]
            noise_mult = fit_metrics.get("noise_multiplier", 0.0)

            client_updates.append(updated_weights)
            client_sizes_round.append(num_samples)

            # 3. Compute baseline score and 4-config NACS scores
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

            # 4. Log update hash and NACS score on-chain if connected
            tx_hash = "off-chain"
            gas_used = 0
            reward_tx_hash = "off-chain"
            reward_gas = 0

            if enable_blockchain and logger is not None:
                try:
                    tx_hash, gas_used = logger.log_update(
                        round_num=rnd, client_id=cid, weights=updated_weights
                    )
                    reward_tx_hash, reward_gas = record_nacs_reward_on_chain(
                        logger=logger, round_num=rnd, client_id=cid, nacs_score=n_score
                    )
                except Exception as ex:
                    print(f"    [Blockchain Log Error] Client {cid + 1}: {ex}")

            history["per_client"][str(cid)]["noise_multiplier"] = noise_mult
            history["per_client"][str(cid)]["client_size"] = c_size
            history["per_client"][str(cid)]["history"].append({
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

            history["blockchain_log"].append({
                "round": rnd,
                "client_id": cid,
                "update_tx_hash": tx_hash,
                "update_gas_used": gas_used,
                "reward_tx_hash": reward_tx_hash,
                "reward_gas_used": reward_gas,
                "nacs_score": n_score,
            })

            print(
                f"  Client {cid + 1} (Subject {cid + 1}, eps={target_eps:>4.1f}, sigma={noise_mult:.3f}): "
                f"loss {loss_before:.3f}->{loss_after:.3f} | "
                f"Baseline: {b_score:>3d} | NACS: {n_score:>3d} | "
                f"Gas: {gas_used + reward_gas}",
                flush=True,
            )

        # Server Aggregation (FedAvg)
        current_weights = fed_avg(client_updates, client_sizes)
        set_weights(global_model, current_weights)

        # Global Evaluation
        metrics = evaluate_global_model(global_model, test_loader, device, criterion)
        print(
            f"  >> Round {rnd} HAR Global Test Accuracy: {metrics['accuracy']:.4f} | "
            f"F1: {metrics['f1']:.4f} | Loss: {metrics['loss']:.4f}\n",
            flush=True,
        )

        history["rounds"].append(rnd)
        history["accuracy"].append(metrics["accuracy"])
        history["f1"].append(metrics["f1"])
        history["precision"].append(metrics["precision"])
        history["recall"].append(metrics["recall"])
        history["loss"].append(metrics["loss"])

        # Automatic training-health check
        # (NOTE: run_phase1.py / run_phase2_dp.py / run_phase4_nacs.py could reuse this check, but remain untouched per instructions)
        is_unhealthy, diag_msg = check_training_health(history, num_classes=6, min_check_round=2)
        if is_unhealthy:
            print("\n" + "!" * 95, flush=True)
            print(f"  [AUTOMATIC HEALTH CHECK ABORT] {diag_msg}", flush=True)
            print("!" * 95 + "\n", flush=True)

            history["status"] = "aborted_not_learning"
            history["diagnostic_message"] = diag_msg
            save_metrics_json(history, "results/phase5_har.json")

            write_live_status({
                "status": "aborted_not_learning",
                "phase": "5-har",
                "current_round": rnd,
                "total_rounds": num_rounds,
                "diagnostic_message": diag_msg,
            })
            return history

    # Score Variance Comparison Analysis
    comparison = {}
    print("=" * 95, flush=True)
    print("PHASE 5 (HAR DOMAIN): CONTRIBUTION SCORE VARIANCE COMPARISON", flush=True)
    print("=" * 95, flush=True)
    print(
        f"{'Client':<10} | {'Target eps':<10} | {'Noise Mult (sigma)':<18} | "
        f"{'Baseline Var':<14} | {'NACS Var':<12} | {'Var Reduction (%)':<18}",
        flush=True,
    )
    print("-" * 95, flush=True)

    for i in range(num_clients):
        cid_str = str(i)
        c_hist = history["per_client"][cid_str]["history"]
        target_eps = history["per_client"][cid_str]["target_epsilon"]
        noise_mult = history["per_client"][cid_str]["noise_multiplier"]

        baseline_scores = [h["baseline_score"] for h in c_hist]
        nacs_scores = [h["nacs_score"] for h in c_hist]

        b_var = float(np.var(baseline_scores))
        n_var = float(np.var(nacs_scores))
        reduction_pct = ((b_var - n_var) / b_var * 100.0) if b_var > 0 else 0.0

        comparison[cid_str] = {
            "client_id": i,
            "target_epsilon": target_eps,
            "noise_multiplier": noise_mult,
            "baseline_variance": b_var,
            "nacs_variance": n_var,
            "variance_reduction_percent": float(reduction_pct),
        }

        print(
            f"Client {i + 1:<3d} | {target_eps:<10.1f} | {noise_mult:<14.4f} | "
            f"{b_var:<14.2f} | {n_var:<12.2f} | {reduction_pct:>+17.2f}%",
            flush=True,
        )

    print("=" * 95 + "\n", flush=True)

    history["comparison"] = comparison

    # Save to results/phase5_har.json
    json_path = "results/phase5_har.json"
    save_metrics_json(history, json_path)

    # Save final global model state dict (Part B Step 1)
    os.makedirs("results", exist_ok=True)
    torch.save(global_model.state_dict(), "results/global_model_har.pt")
    print(f"[Model Weights Saved] Final global HAR model state dict saved to results/global_model_har.pt", flush=True)

    # Generate comparison plot
    plot_phase5_har_comparison(phase5_json=json_path, save_path="results/phase5_har_comparison.png")

    return history


def run_phase5_multiseed_experiment(
    num_seeds: int = 5,
    num_rounds: int = 10,
    target_epsilons=[1.0, 3.0, 8.0, 15.0, 20.0],
):
    print("=" * 95, flush=True)
    print("PHASE 5: Multi-Seed Robustness Benchmark — UCI HAR Domain", flush=True)
    print(f"  Total Seeds : {num_seeds} (Seeds 42..{42 + num_seeds - 1})", flush=True)
    print(f"  Rounds/Seed : {num_rounds}", flush=True)
    print(f"  Clients     : {len(target_epsilons)} (epsilons: {target_epsilons})", flush=True)
    print("=" * 95, flush=True)

    base_seed = 42
    all_seed_results = {i: [] for i in range(len(target_epsilons))}

    for s_idx in range(num_seeds):
        current_seed = base_seed + s_idx
        enable_bc = (s_idx == 0)

        print(f"\n--- Running Phase 5 HAR Seed {s_idx + 1}/{num_seeds} (Seed: {current_seed}) ---", flush=True)
        h = run_phase5_har_experiment(
            target_epsilons=target_epsilons,
            num_rounds=num_rounds,
            seed=current_seed,
            enable_blockchain=enable_bc,
        )

        for cid_str, comp in h["comparison"].items():
            cid = int(cid_str)
            all_seed_results[cid].append(comp["variance_reduction_percent"])

    final_summary = {}
    print("\n" + "=" * 95, flush=True)
    print("PHASE 5 HAR MULTI-SEED ROBUSTNESS RESULTS (MEAN +/- STD OVER SEEDS)", flush=True)
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

    json_path = "results/phase5_har_multiseed.json"
    save_metrics_json(final_summary, json_path)
    plot_multiseed_comparison(json_path=json_path, save_path="results/phase5_multiseed_comparison.png")

    return final_summary


def main():
    parser = argparse.ArgumentParser(description="Phase 5: Cross-Domain HAR Experiment")
    parser.add_argument("--rounds", type=int, default=10, help="Number of FL rounds (default: 10)")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds for benchmark (default: 5)")
    args = parser.parse_args()

    # 1. Run single experiment & on-chain logging
    run_phase5_har_experiment(num_rounds=args.rounds, seed=42, enable_blockchain=True)

    # 2. Run multi-seed benchmark
    if args.seeds > 1:
        run_phase5_multiseed_experiment(num_seeds=args.seeds, num_rounds=args.rounds)

    # 3. Print side-by-side generalizability comparison summary
    print_domain_generalizability_summary()


if __name__ == "__main__":
    main()
