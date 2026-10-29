import os
import argparse
import random
import json
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import roc_auc_score

from opacus.validators import ModuleValidator

from src.dataset_nih import (
    get_nih_client_dataloaders,
    NIH_LABELS,
    NUM_CLASSES,
    DEFAULT_SUBSET_SIZE,
)
from src.model_nih import (
    NIHChestCNN,
    get_weights,
    set_weights,
    build_nih_model,
)
from src.client_dp import PrivateMedicalFLClient
from src.blockchain_client import BlockchainLogger
from src.nacs import baseline_contribution_score, NACSTracker
from src.utils import (
    save_metrics_json,
    compute_multilabel_metrics,
    plot_phase6_nih_comparison,
    plot_multiseed_comparison,
    print_domain_generalizability_summary,
)
from src.training_health import check_multilabel_training_health
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


def evaluate_auroc_metrics(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.BCEWithLogitsLoss,
    labels_list: List[str] = NIH_LABELS,
) -> Dict[str, Any]:
    """
    Evaluates multi-label model on test set:
    - Computes per-label AUROC and mean AUROC using sklearn.metrics.roc_auc_score.
    - If a test label contains no positive or no negative examples, skips that label with a clear explanation.
    - Computes mean AUROC strictly over valid evaluated labels.
    - Also calculates multi-label accuracy, micro/macro F1, precision, recall, and BCE loss.
    """
    model.eval()
    all_logits = []
    all_targets = []
    test_losses = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_losses.append(loss.item())

            all_logits.append(outputs.cpu())
            all_targets.append(labels.cpu())

    cat_logits = torch.cat(all_logits, dim=0).numpy()
    cat_targets = torch.cat(all_targets, dim=0).numpy()

    per_label_auroc: Dict[str, float] = {}
    skipped_labels: List[Dict[str, str]] = []
    evaluated_labels: List[str] = []

    for idx, label_name in enumerate(labels_list):
        y_true = cat_targets[:, idx]
        y_score = cat_logits[:, idx]
        unique_classes = np.unique(y_true)

        if len(unique_classes) < 2:
            reason = "no positive examples" if 1 not in unique_classes else "no negative examples"
            print(f"  [AUROC Note] Skipping AUROC for label '{label_name}': {reason} in test partition.")
            skipped_labels.append({"label": label_name, "reason": reason})
        else:
            try:
                score = float(roc_auc_score(y_true, y_score))
                per_label_auroc[label_name] = score
                evaluated_labels.append(label_name)
            except Exception as ex:
                skipped_labels.append({"label": label_name, "reason": str(ex)})

    if len(evaluated_labels) > 0:
        mean_auroc = float(np.mean([per_label_auroc[l] for l in evaluated_labels]))
    else:
        mean_auroc = 0.0

    # Compute standard classification metrics for backwards compatibility (threshold calibrated to ~5% positive prevalence)
    metrics = compute_multilabel_metrics(cat_targets, cat_logits, threshold=0.05)
    metrics["loss"] = float(np.mean(test_losses))
    metrics["mean_auroc"] = mean_auroc
    metrics["per_label_auroc"] = per_label_auroc
    metrics["evaluated_labels"] = evaluated_labels
    metrics["skipped_labels"] = skipped_labels

    return metrics


def calculate_training_pos_weights(train_loaders, device: torch.device) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Computes per-label positive weight:
      pos_weight = (1 - p) / p
    where p is label prevalence calculated exclusively from training labels.
    Weights are clipped to a maximum of 20.0. Zero-positive labels are safely handled.
    """
    all_targets = []
    if isinstance(train_loaders, list):
        for loader in train_loaders:
            for _, labels in loader:
                all_targets.append(labels)
    else:
        for _, labels in train_loaders:
            all_targets.append(labels)

    all_targets_np = torch.cat(all_targets, dim=0).numpy()
    total_samples = len(all_targets_np)
    pos_counts = all_targets_np.sum(axis=0)

    pos_weights = []
    for count in pos_counts:
        p = count / total_samples
        if p > 0:
            w = float(min((1.0 - p) / p, 20.0))
        else:
            w = 1.0  # Safe fallback for zero positive instances
        pos_weights.append(w)

    pos_weights_np = np.array(pos_weights, dtype=np.float32)
    pos_weight_tensor = torch.tensor(pos_weights_np, dtype=torch.float32).to(device)
    return pos_weight_tensor, pos_weights_np


def run_single_client_sanity_check(
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.BCEWithLogitsLoss,
    lr: float = 1e-3,
    epochs: int = 3,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Mandatory Step 4 Single-Client Sanity Check:
    1. Trains a single model on Client 0's patient-disjoint NIH data without FL, DP, or blockchain.
    2. Evaluates on the held-out global test set.
    3. Prints initial and post-training per-label AUROC and mean AUROC.
    4. Aborts if mean AUROC is below 0.55.
    """
    print("\n" + "=" * 95)
    print("MANDATORY STEP 4 SANITY CHECK: Single-Client Non-DP Training on NIH Chest X-ray")
    print(f"  Configuration: Local Epochs={epochs}, Optimizer=Adam, LR={lr}, Loss=BCEWithLogitsLoss")
    print("=" * 95)

    model = build_nih_model(in_channels=1, num_classes=NUM_CLASSES, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 1. Initial Evaluation on held-out global test set
    initial_metrics = evaluate_auroc_metrics(model, test_loader, device, criterion)
    print(f"\n[Sanity Check] Initial (Untrained) Global Test AUROC:")
    for lbl in initial_metrics["evaluated_labels"]:
        print(f"  - {lbl:<15}: {initial_metrics['per_label_auroc'][lbl]:.4f}")
    print(f"  >> Initial Mean AUROC: {initial_metrics['mean_auroc']:.4f}\n")

    # 2. Local Training for 3 epochs
    for ep in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        print(f"  Epoch {ep}/{epochs} - Train Loss: {np.mean(epoch_losses):.4f}")

    # 3. Post-training Evaluation on held-out global test set
    post_metrics = evaluate_auroc_metrics(model, test_loader, device, criterion)
    post_mean_auroc = post_metrics["mean_auroc"]

    print(f"\n[Sanity Check] Post-Training Global Test AUROC (after {epochs} epochs):")
    for lbl in post_metrics["evaluated_labels"]:
        print(f"  - {lbl:<15}: {post_metrics['per_label_auroc'][lbl]:.4f}")
    print(f"  >> Post-Training Mean AUROC: {post_mean_auroc:.4f}")

    sanity_results = {
        "epochs": epochs,
        "lr": lr,
        "initial_per_label_auroc": initial_metrics["per_label_auroc"],
        "initial_mean_auroc": initial_metrics["mean_auroc"],
        "post_per_label_auroc": post_metrics["per_label_auroc"],
        "post_mean_auroc": post_mean_auroc,
        "evaluated_labels": post_metrics["evaluated_labels"],
        "skipped_labels": post_metrics["skipped_labels"],
        "passed": False,
    }

    # 4. Check AUROC against 0.55 threshold
    if post_mean_auroc < 0.55:
        print("\n" + "!" * 95)
        print("  [SANITY CHECK FAILED] NIH training is not learning!")
        print(f"  Post-training Mean AUROC is {post_mean_auroc:.4f}, which is below required threshold 0.55.")
        print("  Aborting experiment. Federated learning, DP, NACS, and blockchain will NOT proceed.")
        print("!" * 95 + "\n")
        sanity_results["passed"] = False
        return False, sanity_results

    print("\n" + "=" * 95)
    print(f"  [SANITY CHECK PASSED] Post-training Mean AUROC ({post_mean_auroc:.4f}) reaches >= 0.55 threshold.")
    print("  Proceeding to Federated Learning + Differential Privacy + NACS + Blockchain...")
    print("=" * 95 + "\n")
    sanity_results["passed"] = True
    return True, sanity_results


def run_phase6_nih_experiment(
    target_epsilons: List[float] = [1.0, 3.0, 8.0, 15.0, 20.0],
    num_rounds: int = 10,
    seed: int = 42,
    enable_blockchain: bool = True,
    subset_size: int = DEFAULT_SUBSET_SIZE,
):
    set_random_seed(seed)
    num_clients = len(target_epsilons)

    print("=" * 95, flush=True)
    print("PHASE 6: Cross-Domain Benchmark 3 — NIH Chest X-ray Dataset (Multi-Label 6-Class)", flush=True)
    print("  Framework: Blockchain Federated-Learning Medical Marketplace with DP & NACS", flush=True)
    print(f"  Clients  : {num_clients} (Partitioned by REAL Patient ID) | Epsilons: {target_epsilons}", flush=True)
    print(f"  Rounds   : {num_rounds}  | Smart Contract: ContributionLedger", flush=True)
    print(f"  Scope    : {subset_size} sampled images | Res: 64x64 grayscale [0, 1] | 6 Clinical Targets", flush=True)
    print("=" * 95, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_epochs = 3
    lr = 1e-3

    logger = None
    if enable_blockchain:
        try:
            logger = BlockchainLogger()
        except Exception as e:
            print(f"  [Warning] Could not connect to Blockchain node: {e}", flush=True)
            print("  Falling back to off-chain logging.", flush=True)
            enable_blockchain = False

    # 1. Load patient-disjoint NIH Chest X-ray dataset partitions
    client_train_loaders, client_val_loaders, test_loader, _, client_sizes = get_nih_client_dataloaders(
        num_clients=num_clients,
        batch_size=32,
        seed=seed,
        subset_size=subset_size,
        target_size=(64, 64),
        return_client_sizes=True,
    )

    # 2. Calculate pos_weight exclusively from training labels: pos_weight = (1 - p) / p, clipped to max 20
    pos_weight_tensor, pos_weights_np = calculate_training_pos_weights(client_train_loaders, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    print(f"[Training Loss] BCEWithLogitsLoss pos_weight per label: {dict(zip(NIH_LABELS, np.round(pos_weights_np, 2)))}")

    # 3. Compute Trivial All-Negative Baseline on Test Set
    all_test_targets = []
    for _, labels in test_loader:
        all_test_targets.append(labels)
    all_test_targets = torch.cat(all_test_targets, dim=0).numpy()
    all_neg_preds = np.zeros_like(all_test_targets)
    trivial_metrics = compute_multilabel_metrics(all_test_targets, all_neg_preds)
    trivial_baseline_f1 = float(trivial_metrics["micro_f1"])
    print(f"[Baseline Floor] All-Negative Test Micro-F1: {trivial_baseline_f1:.4f} ({trivial_baseline_f1 * 100:.2f}%)\n")

    # 4. Mandatory Step 4 Single-Client Sanity Check
    sanity_passed, sanity_results = run_single_client_sanity_check(
        train_loader=client_train_loaders[0],
        test_loader=test_loader,
        device=device,
        criterion=criterion,
        lr=lr,
        epochs=local_epochs,
    )

    if not sanity_passed:
        # Abort loud and clear as required
        history = {
            "status": "aborted_sanity_check_failed",
            "domain": "NIH Chest X-ray (Multi-Label Clinical Imaging)",
            "num_classes": NUM_CLASSES,
            "labels": NIH_LABELS,
            "subset_size": subset_size,
            "sanity_check": sanity_results,
            "diagnostic_message": f"Sanity check failed: Mean AUROC {sanity_results['post_mean_auroc']:.4f} < 0.55 threshold.",
        }
        save_metrics_json(history, "results/phase6_nih.json")
        write_live_status({
            "status": "aborted_sanity_check_failed",
            "phase": "PHASE 6 (NIH Multi-Label)",
            "diagnostic_message": history["diagnostic_message"],
        })
        return history

    # 5. Instantiate FL Clients with Differential Privacy (Opacus)
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

    # Central Global Model for Server Evaluation
    global_model = build_nih_model(in_channels=1, num_classes=NUM_CLASSES, device=device)

    # Initial Global Evaluation before FL rounds
    initial_metrics = evaluate_auroc_metrics(global_model, test_loader, device, criterion)
    print(
        f"Round 0 (Initial Global) -> Mean AUROC: {initial_metrics['mean_auroc']:.4f}, "
        f"Micro-F1: {initial_metrics['micro_f1']:.4f}, Loss: {initial_metrics['loss']:.4f}\n",
        flush=True,
    )

    history = {
        "status": "RUNNING",
        "domain": "NIH Chest X-ray (Multi-Label Clinical Imaging)",
        "num_classes": NUM_CLASSES,
        "labels": NIH_LABELS,
        "subset_size": subset_size,
        "image_resolution": "64x64",
        "partition_basis": "Patient ID",
        "target_epsilons": target_epsilons,
        "optimizer": "Adam",
        "learning_rate": lr,
        "local_epochs": local_epochs,
        "pos_weights": {label: float(pos_weights_np[idx]) for idx, label in enumerate(NIH_LABELS)},
        "trivial_baseline_f1": trivial_baseline_f1,
        "sanity_check": sanity_results,
        "evaluated_labels": initial_metrics["evaluated_labels"],
        "skipped_labels": initial_metrics["skipped_labels"],
        "rounds": [0],
        "mean_auroc": [initial_metrics["mean_auroc"]],
        "per_label_auroc": [initial_metrics["per_label_auroc"]],
        "accuracy": [initial_metrics["accuracy"]],
        "micro_f1": [initial_metrics["micro_f1"]],
        "macro_f1": [initial_metrics["macro_f1"]],
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
        print(f"--- Phase 6 (NIH Multi-Label) FL Round {rnd}/{num_rounds} ---", flush=True)
        client_updates = []
        client_sizes_round = []

        for client in clients:
            cid = client.client_id
            target_eps = target_epsilons[cid]
            c_size = client_sizes[cid]

            # 1. Pre-training evaluation
            loss_before, _, _ = client.evaluate(current_weights, config)

            # 2. Local DP-SGD fit
            updated_weights, num_samples, fit_metrics = client.fit(current_weights, config)
            loss_after = fit_metrics["train_loss"]
            noise_mult = fit_metrics.get("noise_multiplier", 0.0)

            client_updates.append(updated_weights)
            client_sizes_round.append(num_samples)

            # 3. NACS contribution scoring
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

            # 4. Smart contract logging
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
                f"  Client {cid + 1} (Patient set {cid + 1}, eps={target_eps:>4.1f}, sigma={noise_mult:.3f}): "
                f"loss {loss_before:.3f}->{loss_after:.3f} | "
                f"Baseline: {b_score:>3d} | NACS: {n_score:>3d} | "
                f"Gas: {gas_used + reward_gas}",
                flush=True,
            )

        # 5. Server Aggregation (FedAvg)
        current_weights = fed_avg(client_updates, client_sizes)
        set_weights(global_model, current_weights)
        os.makedirs("results", exist_ok=True)
        torch.save(global_model.state_dict(), "results/global_model_nih.pt")

        # 6. Global Model Evaluation on Central Patient-Disjoint Test Set
        metrics = evaluate_auroc_metrics(global_model, test_loader, device, criterion)
        current_mean_auroc = metrics["mean_auroc"]

        print(
            f"  >> Round {rnd} Global Test Mean AUROC: {current_mean_auroc:.4f} | "
            f"Micro-F1: {metrics['micro_f1']:.4f} | Loss: {metrics['loss']:.4f}",
            flush=True,
        )
        for lbl in metrics["evaluated_labels"]:
            print(f"      - {lbl:<15}: AUROC = {metrics['per_label_auroc'][lbl]:.4f}", flush=True)
        print("", flush=True)

        history["rounds"].append(rnd)
        history["mean_auroc"].append(current_mean_auroc)
        history["per_label_auroc"].append(metrics["per_label_auroc"])
        history["accuracy"].append(metrics["accuracy"])
        history["micro_f1"].append(metrics["micro_f1"])
        history["macro_f1"].append(metrics["macro_f1"])
        history["f1"].append(metrics["micro_f1"])
        history["precision"].append(metrics["precision"])
        history["recall"].append(metrics["recall"])
        history["loss"].append(metrics["loss"])

        # Mandatory Health Check: After round 3, if global mean AUROC is below 0.55, abort loudly
        if rnd >= 3 and current_mean_auroc < 0.55:
            diag_msg = (
                f"[TRAINING HEALTH ALERT] Global model Mean AUROC at Round {rnd} is {current_mean_auroc:.4f}, "
                f"which is below required threshold 0.55. Aborting experiment to prevent non-learning run."
            )
            print("\n" + "!" * 95, flush=True)
            print(f"  {diag_msg}", flush=True)
            print("!" * 95 + "\n", flush=True)

            history["status"] = "aborted_not_learning"
            history["diagnostic_message"] = diag_msg
            save_metrics_json(history, "results/phase6_nih.json")

            write_live_status({
                "status": "aborted_not_learning",
                "phase": "PHASE 6 (NIH Multi-Label)",
                "round": rnd,
                "total_rounds": num_rounds,
                "mean_auroc": current_mean_auroc,
                "diagnostic_message": diag_msg,
            })
            return history

        write_live_status({
            "status": "RUNNING",
            "phase": "PHASE 6 (NIH Multi-Label)",
            "round": rnd,
            "total_rounds": num_rounds,
            "mean_auroc": current_mean_auroc,
            "micro_f1": metrics["micro_f1"],
            "loss": metrics["loss"],
        })

    # Variance reduction comparison
    print("\n" + "=" * 95, flush=True)
    print("PHASE 6 NIH RESULTS: NOISE-BLIND BASELINE VS NACS REWARD ALLOCATION", flush=True)
    print("=" * 95, flush=True)

    client_comparison = {}
    for cid in range(num_clients):
        c_history = history["per_client"][str(cid)]["history"]
        b_scores = [item["baseline_score"] for item in c_history]
        n_scores = [item["nacs_score"] for item in c_history]

        b_var = float(np.var(b_scores)) if len(b_scores) > 1 else 0.0
        n_var = float(np.var(n_scores)) if len(n_scores) > 1 else 0.0
        var_reduction = ((b_var - n_var) / b_var * 100) if b_var > 0 else 0.0

        client_comparison[str(cid)] = {
            "client_id": cid,
            "target_epsilon": target_epsilons[cid],
            "noise_multiplier": history["per_client"][str(cid)]["noise_multiplier"],
            "client_size": client_sizes[cid],
            "baseline_mean_score": float(np.mean(b_scores)),
            "baseline_variance": b_var,
            "nacs_mean_score": float(np.mean(n_scores)),
            "nacs_variance": n_var,
            "variance_reduction_percent": var_reduction,
        }

        print(
            f"Client {cid + 1} (Patient set {cid + 1}, eps={target_epsilons[cid]:>4.1f}): "
            f"Baseline Var = {b_var:>6.2f} -> NACS Var = {n_var:>6.2f} "
            f"({var_reduction:>+6.1f}% reduction)",
            flush=True,
        )

    history["status"] = "COMPLETED"
    history["final_mean_auroc"] = history["mean_auroc"][-1]
    history["final_per_label_auroc"] = history["per_label_auroc"][-1]
    history["comparison"] = client_comparison

    output_path = "results/phase6_nih.json"
    save_metrics_json(history, output_path)

    # Save final global model
    os.makedirs("results", exist_ok=True)
    torch.save(global_model.state_dict(), "results/global_model_nih.pt")
    print(f"[Model Weights Saved] Final global NIH model state dict saved to results/global_model_nih.pt", flush=True)

    # Plot results
    plot_phase6_nih_comparison(output_path, "results/phase6_nih_comparison.png")

    # Domain generalizability summary
    print_domain_generalizability_summary(
        phase4_path="results/phase4_nacs.json",
        phase5_path="results/phase5_har.json",
        phase6_path=output_path,
    )

    write_live_status({
        "status": "COMPLETED",
        "phase": "PHASE 6 (NIH Multi-Label)",
        "final_mean_auroc": history["final_mean_auroc"],
        "final_micro_f1": history["micro_f1"][-1],
        "final_loss": history["loss"][-1],
    })

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 6: NIH Chest X-ray Multi-Label FL Validation")
    parser.add_argument("--rounds", type=int, default=10, help="Number of FL rounds (default: 10)")
    parser.add_argument("--subset_size", type=int, default=DEFAULT_SUBSET_SIZE, help="Stratified subset image count (default: 15000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_blockchain", action="store_true", help="Disable blockchain logging")
    args = parser.parse_args()

    run_phase6_nih_experiment(
        num_rounds=args.rounds,
        subset_size=args.subset_size,
        seed=args.seed,
        enable_blockchain=not args.no_blockchain,
    )
