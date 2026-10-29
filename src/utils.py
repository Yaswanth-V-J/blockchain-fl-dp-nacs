import os
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(y_true, y_pred):
    """Compute accuracy, precision, recall, and macro F1 score."""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }


def compute_multilabel_metrics(y_true, y_pred_logits_or_probs, threshold: float = 0.15):
    """
    Compute multi-label metrics for NIH Chest X-ray (14 disease findings):
    - Subset accuracy (exact multi-label match ratio across all 14 targets)
    - Micro F1 score (with threshold=0.15 calibrated for ~5.6% positive prior)
    - Macro F1 score
    - Micro / Macro Precision & Recall
    """
    import torch
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred_logits_or_probs, torch.Tensor):
        y_pred_logits_or_probs = y_pred_logits_or_probs.cpu().numpy()

    # Apply sigmoid if inputs are raw logits
    if y_pred_logits_or_probs.min() < 0.0 or y_pred_logits_or_probs.max() > 1.0:
        probs = 1.0 / (1.0 + np.exp(-np.clip(y_pred_logits_or_probs, -50, 50)))
    else:
        probs = y_pred_logits_or_probs

    preds = (probs >= threshold).astype(np.float32)

    # Exact Match Ratio (Subset Accuracy)
    exact_match = float(np.mean((preds == y_true).all(axis=1)))

    micro_prec, micro_rec, micro_f1, _ = precision_recall_fscore_support(
        y_true, preds, average='micro', zero_division=0
    )
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_true, preds, average='macro', zero_division=0
    )

    return {
        "accuracy": exact_match,
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "f1": float(micro_f1),  # Standard F1 key used for unified round tracking
        "precision": float(micro_prec),
        "recall": float(micro_rec),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
    }


def evaluate_global_multilabel_model(model, test_loader, device, criterion):
    """Evaluates multi-label global model on centralized test set using BCEWithLogitsLoss."""
    import torch
    model.eval()
    all_preds = []
    all_targets = []
    test_losses = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_losses.append(loss.item())

            all_preds.append(outputs.cpu())
            all_targets.append(labels.cpu())

    cat_preds = torch.cat(all_preds, dim=0)
    cat_targets = torch.cat(all_targets, dim=0)

    metrics = compute_multilabel_metrics(cat_targets, cat_preds)
    metrics["loss"] = float(np.mean(test_losses))
    return metrics


def save_metrics_json(metrics_dict: dict, file_path: str):
    """Save metrics dictionary to JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"[Results Saved] Metrics JSON saved to {file_path}")


def plot_phase1_baseline(history: dict, save_path: str = "results/phase1_baseline.png"):
    """
    Plot round-by-round accuracy and F1 score for Phase 1 Plain FL baseline.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    rounds = history.get("rounds", [])
    accuracy = history.get("accuracy", [])
    f1_scores = history.get("f1", [])
    losses = history.get("loss", [])

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = 'tab:blue'
    ax1.set_xlabel('FL Round', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy / F1 Score', color=color, fontsize=12, fontweight='bold')
    line1 = ax1.plot(rounds, accuracy, 'o-', color='tab:blue', linewidth=2.5, label='Global Accuracy')
    line2 = ax1.plot(rounds, f1_scores, 's--', color='tab:green', linewidth=2.5, label='Global Macro F1')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Loss', color=color, fontsize=12, fontweight='bold')
    line3 = ax2.plot(rounds, losses, '^:', color='tab:red', linewidth=2, label='Global Loss')
    ax2.tick_params(axis='y', labelcolor=color)

    # Added legend
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', frameon=True, facecolor='white', framealpha=0.9)

    plt.title('Phase 1 Baseline: Federated Learning (FedAvg) Performance', fontsize=14, fontweight='bold', pad=15)
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plot Saved] Baseline FL performance plot saved to {save_path}")


def plot_phase2_comparison(
    phase1_json: str = "results/phase1_baseline.json",
    results_dir: str = "results",
    save_path: str = "results/phase2_epsilon_comparison.png"
):
    """
    Plot Phase 2 Differential Privacy comparison:
    Final Accuracy vs Epsilon (with Phase 1 baseline reference line),
    and Round-by-Round Accuracy curves across epsilon values.
    """
    import glob

    # Load Phase 1 Baseline
    baseline_acc = None
    if os.path.exists(phase1_json):
        with open(phase1_json, 'r') as f:
            p1_data = json.load(f)
            baseline_acc = p1_data["accuracy"][-1]

    # Load all Phase 2 DP files
    dp_files = sorted(glob.glob(os.path.join(results_dir, "phase2_dp_eps*.json")))
    
    eps_values = []
    final_accuracies = []
    final_f1s = []
    run_histories = {}

    for filepath in dp_files:
        with open(filepath, 'r') as f:
            data = json.load(f)
            eps_label = data.get("target_epsilon", "No-DP")
            if eps_label is None:
                eps_label = "No-DP (P2 Sanity)"
            
            # Record metrics
            run_histories[str(eps_label)] = data
            
            # For Accuracy vs Epsilon plot (numerical x-axis for DP runs)
            target_eps = data.get("target_epsilon")
            if target_eps is not None:
                eps_values.append(float(target_eps))
                final_accuracies.append(data["accuracy"][-1])
                final_f1s.append(data["f1"][-1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Subplot 1: Final Accuracy vs Target Epsilon ---
    if eps_values:
        sorted_pairs = sorted(zip(eps_values, final_accuracies, final_f1s))
        s_eps, s_acc, s_f1 = zip(*sorted_pairs)
        
        ax1.plot(s_eps, s_acc, 'o-', color='tab:blue', linewidth=2.5, markersize=8, label='DP-SGD Accuracy')
        ax1.plot(s_eps, s_f1, 's--', color='tab:green', linewidth=2.5, markersize=8, label='DP-SGD Macro F1')
        
        for e, a in zip(s_eps, s_acc):
            ax1.annotate(f"{a:.3f}", (e, a), textcoords="offset points", xytext=(0, 10), ha='center', fontweight='bold')

    if baseline_acc is not None:
        ax1.axhline(y=baseline_acc, color='tab:red', linestyle='--', linewidth=2, label=f'Phase 1 Baseline (No-DP = {baseline_acc:.3f})')

    ax1.set_xlabel('Privacy Budget Epsilon (ε)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Final Accuracy / F1 (Round 10)', fontsize=12, fontweight='bold')
    ax1.set_title('Final Accuracy vs. Differential Privacy Epsilon (ε)', fontsize=13, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='lower right', frameon=True, facecolor='white')

    # --- Subplot 2: Round-by-Round Accuracy Curves across Epsilon ---
    color_map = {'1.0': 'tab:purple', '3.0': 'tab:orange', '8.0': 'tab:cyan', '20.0': 'tab:green', 'No-DP (P2 Sanity)': 'tab:pink'}
    
    if os.path.exists(phase1_json):
        with open(phase1_json, 'r') as f:
            p1_data = json.load(f)
            ax2.plot(p1_data["rounds"], p1_data["accuracy"], 'k-^', linewidth=2.5, label='Phase 1 Baseline (No-DP)')

    for label, data in run_histories.items():
        c = color_map.get(str(label), 'tab:blue')
        ax2.plot(data["rounds"], data["accuracy"], 'o--', color=c, linewidth=2, label=f'ε = {label}')

    ax2.set_xlabel('FL Round', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Global Test Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy Progression Across FL Rounds', fontsize=13, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='lower right', frameon=True, facecolor='white')

    plt.suptitle('Phase 2: Differential Privacy (DP-SGD) Trade-off Evaluation', fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot Saved] Phase 2 Epsilon Comparison Plot saved to {save_path}")


def plot_phase4_nacs_comparison(
    phase4_json: str = "results/phase4_nacs.json",
    save_path: str = "results/phase4_nacs_comparison.png",
):
    """
    Plot Phase 4 NACS comparison:
    One figure with 5 subplots (one per client), showing baseline_score vs nacs_score
    across rounds as two lines to highlight the adaptive score smoothing effect.
    """
    if not os.path.exists(phase4_json):
        print(f"[Plot Error] JSON file not found at {phase4_json}")
        return

    with open(phase4_json, "r") as f:
        data = json.load(f)

    per_client = data.get("per_client", {})
    client_ids = sorted([int(cid) for cid in per_client.keys()])

    num_clients = len(client_ids)
    if num_clients == 0:
        print("[Plot Error] No client data found in JSON.")
        return

    fig, axes = plt.subplots(1, num_clients, figsize=(4 * num_clients, 4.5), sharey=True)
    if num_clients == 1:
        axes = [axes]

    for idx, cid in enumerate(client_ids):
        ax = axes[idx]
        c_data = per_client[str(cid)]
        history = c_data.get("history", [])

        rounds = [item["round"] for item in history]
        baseline_scores = [item["baseline_score"] for item in history]
        nacs_scores = [item["nacs_score"] for item in history]
        target_eps = c_data.get("target_epsilon", "N/A")
        noise_mult = c_data.get("noise_multiplier", 0.0)

        ax.plot(
            rounds,
            baseline_scores,
            "o--",
            color="tab:red",
            linewidth=2,
            label="Baseline (Noise-Blind)",
        )
        ax.plot(
            rounds,
            nacs_scores,
            "s-",
            color="tab:blue",
            linewidth=2.5,
            label="NACS (Noise-Aware)",
        )

        ax.set_title(
            f"Client {cid + 1}\n(ε={target_eps}, σ={noise_mult:.2f})",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("FL Round", fontsize=10, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("Contribution Score (0-100)", fontsize=11, fontweight="bold")
        ax.set_ylim(-5, 105)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper right", fontsize=8, frameon=True, facecolor="white")

    plt.suptitle(
        "Phase 4: Noise-Blind Baseline vs. NACS Contribution Scores Across Heterogeneous DP Budgets",
        fontsize=14,
        fontweight="bold",
        y=1.03,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plot Saved] Phase 4 NACS Comparison Plot saved to {save_path}")


def plot_multiseed_comparison(
    json_path: str = "results/phase4_nacs_multiseed.json",
    save_path: str = "results/phase4_multiseed_comparison.png",
):
    """
    Plot Phase 4 NACS Multi-Seed Comparison:
    Bar chart per client of mean variance reduction (%) with error bars (std).
    """
    if not os.path.exists(json_path):
        print(f"[Plot Error] JSON file not found at {json_path}")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    client_dict = data.get("client_summary", data.get("per_client", data))
    client_ids = sorted([int(cid) for cid in client_dict.keys() if str(cid).isdigit()])
    if not client_ids:
        print("[Plot Error] No client data found in JSON.")
        return

    labels = []
    means = []
    stds = []

    for cid in client_ids:
        c_info = client_dict[str(cid)]
        eps = c_info.get("target_epsilon", "N/A")
        labels.append(f"Client {cid + 1}\n(eps={eps})")
        m = c_info.get("reduction_mean_percent", c_info.get("mean_variance_reduction", 0.0))
        s = c_info.get("reduction_std_percent", c_info.get("std_variance_reduction", 0.0))
        means.append(m)
        stds.append(s)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=6,
        color="tab:blue",
        alpha=0.85,
        edgecolor="navy",
        linewidth=1.2,
    )

    for bar, m, s in zip(bars, means, stds):
        yval = bar.get_height()
        ax.annotate(
            f"{m:.1f}% \u00b1 {s:.1f}%",
            (bar.get_x() + bar.get_width() / 2.0, yval + (s if s > 0 else 0) + 1.5),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_ylabel("Mean Variance Reduction (%)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_title(
        "NACS Robustness Benchmark: Mean Variance Reduction Across Random Seeds",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    max_y = max([m + s for m, s in zip(means, stds)]) if means else 100
    ax.set_ylim(0, max(100, max_y + 15))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plot Saved] Multi-seed comparison plot saved to {save_path}")


def plot_phase5_har_comparison(
    phase5_json: str = "results/phase5_har.json",
    save_path: str = "results/phase5_har_comparison.png",
):
    """
    Plot Phase 5 HAR comparison:
    Subplots for 5 HAR clients showing Baseline (Noise-Blind) vs NACS (Noise-Aware) contribution scores.
    """
    if not os.path.exists(phase5_json):
        print(f"[Plot Error] JSON file not found at {phase5_json}")
        return

    with open(phase5_json, "r") as f:
        data = json.load(f)

    per_client = data.get("per_client", {})
    client_ids = sorted([int(cid) for cid in per_client.keys()])

    num_clients = len(client_ids)
    if num_clients == 0:
        print("[Plot Error] No client data found in JSON.")
        return

    fig, axes = plt.subplots(1, num_clients, figsize=(4 * num_clients, 4.5), sharey=True)
    if num_clients == 1:
        axes = [axes]

    for idx, cid in enumerate(client_ids):
        ax = axes[idx]
        c_data = per_client[str(cid)]
        history = c_data.get("history", [])

        rounds = [item["round"] for item in history]
        baseline_scores = [item["baseline_score"] for item in history]
        nacs_scores = [item["nacs_score"] for item in history]
        target_eps = c_data.get("target_epsilon", "N/A")
        noise_mult = c_data.get("noise_multiplier", 0.0)

        ax.plot(
            rounds,
            baseline_scores,
            "o--",
            color="tab:red",
            linewidth=2,
            label="Baseline (Noise-Blind)",
        )
        ax.plot(
            rounds,
            nacs_scores,
            "s-",
            color="tab:blue",
            linewidth=2.5,
            label="NACS (Noise-Aware)",
        )

        ax.set_title(
            f"Client {cid + 1} (Subj {cid + 1})\n(ε={target_eps}, σ={noise_mult:.2f})",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("FL Round", fontsize=10, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("Contribution Score (0-100)", fontsize=11, fontweight="bold")
        ax.set_ylim(-5, 105)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper right", fontsize=8, frameon=True, facecolor="white")

    plt.suptitle(
        "Phase 5 (HAR Domain): Noise-Blind Baseline vs. NACS Contribution Scores",
        fontsize=14,
        fontweight="bold",
        y=1.03,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plot Saved] Phase 5 HAR Comparison Plot saved to {save_path}")


def print_domain_generalizability_summary(
    phase4_path: str = "results/phase4_nacs.json",
    phase5_path: str = "results/phase5_har.json",
    phase6_path: str = "results/phase6_nih.json",
    phase4_multiseed_path: str = "results/phase4_nacs_multiseed.json",
    phase5_multiseed_path: str = "results/phase5_har_multiseed.json",
    phase6_multiseed_path: str = "results/phase6_nih_multiseed.json",
):
    """
    Prints a side-by-side summary table comparing all THREE domains:
    1. PneumoniaMNIST (Medical 2D Binary Image)
    2. UCI HAR (Everyday Wearable Sensor 6-class Tabular)
    3. NIH Chest X-ray (Clinical 2D Multi-label 14-class Image)
    """
    print("\n" + "=" * 125)
    print("THREE-DOMAIN FRAMEWORK GENERALIZABILITY BENCHMARK SUMMARY")
    print("=" * 125)

    p4_data, p5_data, p6_data = {}, {}, {}
    if os.path.exists(phase4_path):
        with open(phase4_path, "r") as f:
            p4_data = json.load(f)
    if os.path.exists(phase5_path):
        with open(phase5_path, "r") as f:
            p5_data = json.load(f)
    if os.path.exists(phase6_path):
        with open(phase6_path, "r") as f:
            p6_data = json.load(f)

    p4_acc = p4_data.get("accuracy", [0.0])[-1] if "accuracy" in p4_data else 0.0
    p5_acc = p5_data.get("accuracy", [0.0])[-1] if "accuracy" in p5_data else 0.0
    p6_micro_f1 = (
        p6_data.get("micro_f1", p6_data.get("f1", [0.0]))[-1]
        if ("micro_f1" in p6_data or "f1" in p6_data)
        else 0.0
    )

    print(f"  * Domain 1: PneumoniaMNIST Medical Images (2D, 28x28, Binary)     | Final Acc : {p4_acc * 100:.2f}% | Partition: Simulated Hospitals")
    print(f"  * Domain 2: UCI HAR Wearable Sensors      (1D, 561-tab, 6-class)  | Final Acc : {p5_acc * 100:.2f}% | Partition: Real Subject ID")
    print(f"  * Domain 3: NIH Chest X-ray Clinical      (2D, 64x64, 14-label)   | Micro-F1  : {p6_micro_f1 * 100:.2f}% | Partition: Real Patient ID")
    print("-" * 125)

    p4_comp = p4_data.get("comparison", {})
    p5_comp = p5_data.get("comparison", {})
    p6_comp = p6_data.get("comparison", {})

    print(
        f"{'Client':<8} | {'Target eps':<10} | "
        f"{'Med NACS Var':<13} | {'Med Red (%)':<12} | "
        f"{'HAR NACS Var':<13} | {'HAR Red (%)':<12} | "
        f"{'NIH NACS Var':<13} | {'NIH Red (%)':<12}"
    )
    print("-" * 125)

    all_cids = sorted(list(set(list(p4_comp.keys()) + list(p5_comp.keys()) + list(p6_comp.keys()))))
    for cid in all_cids:
        c4 = p4_comp.get(str(cid), {})
        c5 = p5_comp.get(str(cid), {})
        c6 = p6_comp.get(str(cid), {})

        eps = c4.get("target_epsilon", c5.get("target_epsilon", c6.get("target_epsilon", "N/A")))

        m_nvar = c4.get("nacs_variance", 0.0)
        m_red = c4.get("variance_reduction_percent", 0.0)
        h_nvar = c5.get("nacs_variance", 0.0)
        h_red = c5.get("variance_reduction_percent", 0.0)
        n_nvar = c6.get("nacs_variance", 0.0)
        n_red = c6.get("variance_reduction_percent", 0.0)

        print(
            f"Client {int(cid)+1:<2} | {eps:<10} | "
            f"{m_nvar:<13.2f} | {m_red:>+11.2f}% | "
            f"{h_nvar:<13.2f} | {h_red:>+11.2f}% | "
            f"{n_nvar:<13.2f} | {n_red:>+11.2f}%"
        )

    print("=" * 125 + "\n")


def plot_phase6_nih_comparison(
    phase6_json: str = "results/phase6_nih.json",
    save_path: str = "results/phase6_nih_comparison.png",
):
    """
    Plot Phase 6 NIH Chest X-ray comparison:
    Subplots for 5 NIH clients showing Baseline (Noise-Blind) vs NACS (Noise-Aware) contribution scores.
    """
    if not os.path.exists(phase6_json):
        print(f"[Plot Error] JSON file not found at {phase6_json}")
        return

    with open(phase6_json, "r") as f:
        data = json.load(f)

    per_client = data.get("per_client", {})
    client_ids = sorted([int(cid) for cid in per_client.keys()])

    num_clients = len(client_ids)
    if num_clients == 0:
        print("[Plot Error] No client data found in JSON.")
        return

    fig, axes = plt.subplots(1, num_clients, figsize=(4 * num_clients, 4.5), sharey=True)
    if num_clients == 1:
        axes = [axes]

    for idx, cid in enumerate(client_ids):
        ax = axes[idx]
        c_data = per_client[str(cid)]
        history = c_data.get("history", [])

        rounds = [item["round"] for item in history]
        baseline_scores = [item["baseline_score"] for item in history]
        nacs_scores = [item["nacs_score"] for item in history]
        target_eps = c_data.get("target_epsilon", "N/A")
        noise_mult = c_data.get("noise_multiplier", 0.0)

        ax.plot(
            rounds,
            baseline_scores,
            "o--",
            color="tab:red",
            linewidth=2,
            label="Baseline (Noise-Blind)",
        )
        ax.plot(
            rounds,
            nacs_scores,
            "s-",
            color="tab:blue",
            linewidth=2.5,
            label="NACS (Noise-Aware)",
        )

        ax.set_title(
            f"Client {cid + 1} (Patient Partition)\n(ε={target_eps}, σ={noise_mult:.2f})",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("FL Round", fontsize=10, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("Contribution Score (0-100)", fontsize=11, fontweight="bold")
        ax.set_ylim(-5, 105)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper right", fontsize=8, frameon=True, facecolor="white")

    plt.suptitle(
        "Phase 6 (NIH Chest X-ray Multi-Label Domain): Baseline vs. NACS Contribution Scores",
        fontsize=14,
        fontweight="bold",
        y=1.03,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Plot Saved] Phase 6 NIH Comparison Plot saved to {save_path}")





