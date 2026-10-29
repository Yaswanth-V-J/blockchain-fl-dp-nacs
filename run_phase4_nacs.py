import os
import argparse
import numpy as np
import torch

from opacus.validators import ModuleValidator

from src.dataset import get_client_dataloaders
from src.model import MedicalCNN, get_weights, set_weights
from src.client_dp import PrivateMedicalFLClient
from src.blockchain_client import BlockchainLogger
from src.nacs import baseline_contribution_score, NACSTracker
from src.utils import save_metrics_json, plot_phase4_nacs_comparison
from run_phase1 import fed_avg, evaluate_global_model


def record_nacs_reward_on_chain(logger: BlockchainLogger, round_num: int, client_id: int, nacs_score: int):
    """
    Calls smart contract's recordReward() with the NACS score.
    Returns (tx_hash_hex, gas_used).
    """
    tx_hash = logger.contract.functions.recordReward(
        round_num, client_id, int(nacs_score)
    ).transact({"from": logger.account})

    receipt = logger.w3.eth.wait_for_transaction_receipt(tx_hash)
    tx_hash_hex = receipt["transactionHash"].hex()
    gas_used = receipt["gasUsed"]
    return tx_hash_hex, gas_used


def run_phase4_nacs_experiment(
    target_epsilons=[1.0, 3.0, 8.0, 15.0, 20.0],
    num_rounds=10,
    alpha=0.5,
):
    num_clients = len(target_epsilons)
    print("=" * 85, flush=True)
    print(f"PHASE 4: Noise-Aware Adaptive Contribution Scoring (NACS) Experiment", flush=True)
    print(f"  Clients: {num_clients} | Heterogeneous target_epsilons: {target_epsilons}", flush=True)
    print(f"  Rounds : {num_rounds}  | Smart Contract: ContributionLedger", flush=True)
    print("=" * 85, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_epochs = 2
    lr = 0.005

    # Connect to Blockchain logger
    logger = BlockchainLogger()

    # Load dataset partitions
    client_train_loaders, client_val_loaders, test_loader, _, client_sizes = get_client_dataloaders(
        num_clients=num_clients, alpha=alpha, batch_size=32, return_client_sizes=True
    )

    # Instantiate DP clients with heterogeneous privacy budgets
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

    # Global model for server evaluation
    base = MedicalCNN(in_channels=1, num_classes=2)
    global_model = ModuleValidator.fix(base).to(device) if not ModuleValidator.is_valid(base) else base.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    initial_metrics = evaluate_global_model(global_model, test_loader, device, criterion)
    print(f"Round 0 (Initial) -> Test Accuracy: {initial_metrics['accuracy']:.4f}, "
          f"F1: {initial_metrics['f1']:.4f}, Loss: {initial_metrics['loss']:.4f}\n", flush=True)

    history = {
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
        print(f"--- NACS FL Round {rnd}/{num_rounds} ---", flush=True)
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

            # 4. Log update hash on-chain
            tx_hash, gas_used = logger.log_update(
                round_num=rnd, client_id=cid, weights=updated_weights
            )

            # 5. Log NACS score (score-of-record) to smart contract
            reward_tx_hash, reward_gas = record_nacs_reward_on_chain(
                logger=logger, round_num=rnd, client_id=cid, nacs_score=n_score
            )

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
                f"  Client {cid + 1} (eps={target_eps:>4.1f}, sigma={noise_mult:.3f}): "
                f"loss {loss_before:.3f}->{loss_after:.3f} | "
                f"Baseline: {b_score:>3d} | NACS: {n_score:>3d} | "
                f"Gas: {gas_used + reward_gas}",
                flush=True
            )

        # Server Aggregation (FedAvg)
        current_weights = fed_avg(client_updates, client_sizes)
        set_weights(global_model, current_weights)

        # Global Evaluation
        metrics = evaluate_global_model(global_model, test_loader, device, criterion)
        print(
            f"  >> Round {rnd} Global Test Accuracy: {metrics['accuracy']:.4f} | "
            f"F1: {metrics['f1']:.4f} | Loss: {metrics['loss']:.4f}\n",
            flush=True
        )

        history["rounds"].append(rnd)
        history["accuracy"].append(metrics["accuracy"])
        history["f1"].append(metrics["f1"])
        history["precision"].append(metrics["precision"])
        history["recall"].append(metrics["recall"])
        history["loss"].append(metrics["loss"])

    # Step 4 — Variance Comparison Analysis
    comparison = {}
    print("=" * 95, flush=True)
    print("PHASE 4: CONTRIBUTION SCORE VARIANCE COMPARISON (BASELINE VS. NACS)", flush=True)
    print("=" * 95, flush=True)
    print(
        f"{'Client':<10} | {'Target eps':<10} | {'Noise Mult (sigma)':<18} | "
        f"{'Baseline Var':<14} | {'NACS Var':<12} | {'Var Reduction (%)':<18}",
        flush=True
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
            flush=True
        )

    print("=" * 95 + "\n", flush=True)

    history["comparison"] = comparison

    # Save to results/phase4_nacs.json
    json_path = "results/phase4_nacs.json"
    save_metrics_json(history, json_path)

    # Step 5 — Generate comparison plot
    plot_phase4_nacs_comparison(phase4_json=json_path, save_path="results/phase4_nacs_comparison.png")

    return history


def main():
    parser = argparse.ArgumentParser(description="Phase 4: NACS FL Experiment")
    parser.add_argument(
        "--rounds", type=int, default=10, help="Number of FL rounds (default: 10)"
    )
    args = parser.parse_args()

    run_phase4_nacs_experiment(num_rounds=args.rounds)


if __name__ == "__main__":
    main()
