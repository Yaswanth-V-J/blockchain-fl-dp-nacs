import os
import argparse
import random
import json
import numpy as np
import torch

from opacus.validators import ModuleValidator

from src.dataset_nih import get_nih_client_dataloaders
from src.model_nih import NIHChestCNN, get_weights, set_weights
from src.client_dp import PrivateMedicalFLClient
from src.blockchain_client import BlockchainLogger
from src.nacs import baseline_contribution_score, NACSTracker
from src.utils import (
    save_metrics_json,
    evaluate_global_multilabel_model,
    plot_multiseed_comparison,
)
from run_phase1 import fed_avg
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


def run_single_seed_nih(
    seed: int,
    seed_index: int,
    total_seeds: int,
    num_rounds: int = 10,
    target_epsilons=[1.0, 3.0, 8.0, 15.0, 20.0],
    enable_blockchain: bool = True,
    subset_size: int = 4000,
):
    set_random_seed(seed)
    num_clients = len(target_epsilons)

    print("-" * 85, flush=True)
    print(
        f"RUNNING SEED {seed_index + 1}/{total_seeds} (Random Seed: {seed}) | "
        f"Blockchain Logging: {'ENABLED (Seed 1)' if enable_blockchain else 'DISABLED (Off-Chain)'}",
        flush=True,
    )
    print("-" * 85, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_epochs = 2
    lr = 0.0003

    logger = None
    if enable_blockchain:
        try:
            logger = BlockchainLogger()
        except Exception as e:
            print(f"  [Warning] Could not connect to Blockchain node: {e}", flush=True)
            print("  Falling back to off-chain logging for this seed.", flush=True)
            enable_blockchain = False

    client_train_loaders, client_val_loaders, test_loader, _, client_sizes = get_nih_client_dataloaders(
        num_clients=num_clients,
        batch_size=32,
        seed=seed,
        subset_size=subset_size,
        target_size=(64, 64),
        return_client_sizes=True,
    )

    # Calculate per-label pos_weight to address extreme class imbalance (~94.4% negative)
    all_train_targets = []
    for loader in client_train_loaders:
        for _, labels in loader:
            all_train_targets.append(labels)
    all_train_targets = torch.cat(all_train_targets, dim=0).numpy()
    pos_counts = all_train_targets.sum(axis=0)
    neg_counts = len(all_train_targets) - pos_counts
    pos_weight = np.clip(neg_counts / np.maximum(pos_counts, 1.0), 1.0, 10.0)
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

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
            model_cls=NIHChestCNN,
            criterion=criterion,
        )
        for i in range(num_clients)
    ]

    nacs_tracker = NACSTracker()

    base = NIHChestCNN(in_channels=1, num_classes=14)
    global_model = (
        ModuleValidator.fix(base).to(device)
        if not ModuleValidator.is_valid(base)
        else base.to(device)
    )

    history = {
        "seed": seed,
        "target_epsilons": target_epsilons,
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
        "micro_f1": [],
        "loss": [],
    }

    current_weights = get_weights(global_model)
    config = {"local_epochs": local_epochs, "lr": lr}

    for rnd in range(1, num_rounds + 1):
        client_updates = []

        for client in clients:
            cid = client.client_id
            target_eps = target_epsilons[cid]
            c_size = client_sizes[cid]

            loss_before, _, _ = client.evaluate(current_weights, config)
            updated_weights, num_samples, fit_metrics = client.fit(current_weights, config)
            loss_after = fit_metrics["train_loss"]
            noise_mult = fit_metrics.get("noise_multiplier", 0.0)

            client_updates.append(updated_weights)

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

            if enable_blockchain and logger is not None:
                try:
                    logger.log_update(round_num=rnd, client_id=cid, weights=updated_weights)
                    record_nacs_reward_on_chain(logger=logger, round_num=rnd, client_id=cid, nacs_score=n_score)
                except Exception:
                    pass

            history["per_client"][str(cid)]["noise_multiplier"] = noise_mult
            history["per_client"][str(cid)]["history"].append({
                "round": rnd,
                "baseline_score": b_score,
                "nacs_score": n_score,
            })

        current_weights = fed_avg(client_updates, client_sizes)
        set_weights(global_model, current_weights)

        metrics = evaluate_global_multilabel_model(global_model, test_loader, device, criterion)
        history["micro_f1"].append(metrics["micro_f1"])
        history["loss"].append(metrics["loss"])

        print(
            f"  Seed {seed} | Round {rnd}/{num_rounds} -> Global Test Micro-F1: {metrics['micro_f1']:.4f} | Loss: {metrics['loss']:.4f}",
            flush=True,
        )

    client_comparison = {}
    for cid in range(num_clients):
        c_hist = history["per_client"][str(cid)]["history"]
        b_scores = [item["baseline_score"] for item in c_hist]
        n_scores = [item["nacs_score"] for item in c_hist]

        b_var = float(np.var(b_scores)) if len(b_scores) > 1 else 0.0
        n_var = float(np.var(n_scores)) if len(n_scores) > 1 else 0.0
        var_red = ((b_var - n_var) / b_var * 100) if b_var > 0 else 0.0

        client_comparison[str(cid)] = {
            "target_epsilon": target_epsilons[cid],
            "baseline_variance": b_var,
            "nacs_variance": n_var,
            "variance_reduction_percent": var_red,
        }

    history["comparison"] = client_comparison
    return history


def run_phase6_multiseed_experiment(
    seeds: list[int] = list(range(42, 52)),
    num_rounds: int = 10,
    target_epsilons=[1.0, 3.0, 8.0, 15.0, 20.0],
    enable_blockchain_seed0: bool = True,
    subset_size: int = 4000,
):
    print("=" * 95, flush=True)
    print("PHASE 6 MULTI-SEED STUDY: NIH Chest X-ray Multi-Label FL (10 Seeds)", flush=True)
    print(f"  Seeds ({len(seeds)} total): {seeds}", flush=True)
    print(f"  Target Epsilons: {target_epsilons} | Rounds: {num_rounds} | Subset: {subset_size}", flush=True)
    print("=" * 95, flush=True)

    all_seed_histories = []
    num_clients = len(target_epsilons)

    for idx, seed in enumerate(seeds):
        use_bc = enable_blockchain_seed0 if idx == 0 else False
        seed_res = run_single_seed_nih(
            seed=seed,
            seed_index=idx,
            total_seeds=len(seeds),
            num_rounds=num_rounds,
            target_epsilons=target_epsilons,
            enable_blockchain=use_bc,
            subset_size=subset_size,
        )
        all_seed_histories.append(seed_res)

        write_live_status({
            "status": "RUNNING_MULTISEED",
            "domain": "NIH Chest X-ray",
            "seed_index": idx + 1,
            "total_seeds": len(seeds),
            "latest_micro_f1": seed_res["micro_f1"][-1],
        })

    # Aggregate metrics across seeds
    client_aggregate = {str(cid): {"baseline_vars": [], "nacs_vars": [], "reductions": []} for cid in range(num_clients)}
    final_f1s = []

    for s_hist in all_seed_histories:
        final_f1s.append(s_hist["micro_f1"][-1])
        comp = s_hist["comparison"]
        for cid in range(num_clients):
            client_aggregate[str(cid)]["baseline_vars"].append(comp[str(cid)]["baseline_variance"])
            client_aggregate[str(cid)]["nacs_vars"].append(comp[str(cid)]["nacs_variance"])
            client_aggregate[str(cid)]["reductions"].append(comp[str(cid)]["variance_reduction_percent"])

    summary_comparison = {}
    print("\n" + "=" * 95, flush=True)
    print("PHASE 6 NIH MULTI-SEED SUMMARY RESULTS (10 SEEDS MEAN ± STD)", flush=True)
    print("=" * 95, flush=True)

    for cid in range(num_clients):
        eps = target_epsilons[cid]
        c_agg = client_aggregate[str(cid)]

        b_mean = float(np.mean(c_agg["baseline_vars"]))
        b_std = float(np.std(c_agg["baseline_vars"]))
        n_mean = float(np.mean(c_agg["nacs_vars"]))
        n_std = float(np.std(c_agg["nacs_vars"]))
        red_mean = float(np.mean(c_agg["reductions"]))
        red_std = float(np.std(c_agg["reductions"]))

        summary_comparison[str(cid)] = {
            "client_id": cid,
            "target_epsilon": eps,
            "baseline_var_mean": b_mean,
            "baseline_var_std": b_std,
            "nacs_var_mean": n_mean,
            "nacs_var_std": n_std,
            "reduction_mean_percent": red_mean,
            "reduction_std_percent": red_std,
        }

        print(
            f"Client {cid + 1} (eps={eps:>4.1f}): "
            f"Baseline Var = {b_mean:>6.2f} ± {b_std:<5.2f} -> NACS Var = {n_mean:>6.2f} ± {n_std:<5.2f} "
            f"(Reduction = {red_mean:>+6.1f}% ± {red_std:<4.1f}%)",
            flush=True,
        )

    mean_f1 = float(np.mean(final_f1s))
    std_f1 = float(np.std(final_f1s))
    print(f"\nGlobal Test Micro-F1 (10 Seeds): {mean_f1 * 100:.2f}% ± {std_f1 * 100:.2f}%", flush=True)

    multiseed_output = {
        "domain": "NIH Chest X-ray Multi-Label FL",
        "seeds": seeds,
        "target_epsilons": target_epsilons,
        "mean_final_micro_f1": mean_f1,
        "std_final_micro_f1": std_f1,
        "client_summary": summary_comparison,
        "all_seeds_data": all_seed_histories,
    }

    output_json = "results/phase6_nih_multiseed.json"
    save_metrics_json(multiseed_output, output_json)

    # Plot multi-seed variance reduction bar chart
    plot_multiseed_comparison(output_json, "results/phase6_nih_multiseed_comparison.png")

    write_live_status({
        "status": "COMPLETED_MULTISEED",
        "domain": "NIH Chest X-ray",
        "mean_micro_f1": mean_f1,
        "std_micro_f1": std_f1,
    })

    return multiseed_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 6: Multi-Seed NIH Chest X-ray Benchmark")
    parser.add_argument("--rounds", type=int, default=10, help="Number of FL rounds")
    parser.add_argument("--subset_size", type=int, default=4000, help="Subset image count")
    parser.add_argument("--num_seeds", type=int, default=10, help="Number of random seeds")
    args = parser.parse_args()

    seeds = list(range(42, 42 + args.num_seeds))
    run_phase6_multiseed_experiment(seeds=seeds, num_rounds=args.rounds, subset_size=args.subset_size)
