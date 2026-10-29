import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def analyze_ablation_study(
    json_path: str = "results/phase4_nacs.json",
    save_plot_path: str = "results/ablation_study_comparison.png",
):
    if not os.path.exists(json_path):
        print(
            f"[Error] Results JSON file not found at '{json_path}'. Please run phase 4 NACS experiment first."
        )
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    per_client = data.get("per_client", {})
    if not per_client:
        print(f"[Error] No per_client data found in '{json_path}'.")
        return None

    client_summary = []

    for cid_str, c_info in sorted(per_client.items(), key=lambda x: int(x[0])):
        cid = int(cid_str)
        target_eps = c_info.get("target_epsilon", 0.0)
        noise_mult = c_info.get("noise_multiplier", 0.0)
        c_size = c_info.get("client_size", 0)

        history = c_info.get("history", [])
        if not history:
            continue

        # Get latest round
        last_entry = history[-1]

        # If client_size wasn't at root, check last entry
        if c_size == 0 and "client_size" in last_entry:
            c_size = last_entry["client_size"]

        # Extract 4 configuration scores for final round
        s_size = last_entry.get("score_size_only", 0)
        s_sq = last_entry.get("score_size_quality", 0)
        s_sqn = last_entry.get("score_size_quality_noise", 0)
        s_full = last_entry.get("score_full_nacs", last_entry.get("nacs_score", 0))

        # Also compute average scores across all rounds
        avg_size = float(np.mean([h.get("score_size_only", 0) for h in history]))
        avg_sq = float(np.mean([h.get("score_size_quality", 0) for h in history]))
        avg_sqn = float(np.mean([h.get("score_size_quality_noise", 0) for h in history]))
        avg_full = float(np.mean([h.get("score_full_nacs", h.get("nacs_score", 0)) for h in history]))

        client_summary.append({
            "client_id": cid,
            "target_epsilon": target_eps,
            "noise_multiplier": noise_mult,
            "client_size": c_size,
            "score_size_only": s_size,
            "score_size_quality": s_sq,
            "score_size_quality_noise": s_sqn,
            "score_full_nacs": s_full,
            "avg_size_only": avg_size,
            "avg_size_quality": avg_sq,
            "avg_size_quality_noise": avg_sqn,
            "avg_full_nacs": avg_full,
        })

    df = pd.DataFrame(client_summary)

    # Compute rankings under Config A (size_only) vs Config D (full_nacs)
    # Higher score = better rank (rank 1 is top score)
    df["rank_size_only"] = df["score_size_only"].rank(ascending=False, method="min").astype(int)
    df["rank_full_nacs"] = df["score_full_nacs"].rank(ascending=False, method="min").astype(int)
    df["rank_changed"] = df["rank_size_only"] != df["rank_full_nacs"]

    print("=" * 115)
    print(f"4-CONFIGURATION NACS ABLATION STUDY ANALYSIS ({json_path})")
    print("=" * 115)
    print(
        f"{'Client':<10} | {'Epsilon (eps)':<13} | {'Noise (sigma)':<13} | {'Samples (N)':<12} | "
        f"{'Config A (Size)':<15} | {'Config B (S+Q)':<14} | {'Config C (S+Q+N)':<16} | {'Config D (Full NACS)':<20} | "
        f"{'Rank A -> D':<12}"
    )
    print("-" * 115)

    for _, row in df.iterrows():
        cid_name = f"Client {int(row['client_id']) + 1}"
        rank_str = f"R{int(row['rank_size_only'])} -> R{int(row['rank_full_nacs'])}"
        if row["rank_changed"]:
            rank_str += " *"
        print(
            f"{cid_name:<10} | {row['target_epsilon']:<12.1f} | {row['noise_multiplier']:<10.3f} | {row['client_size']:<12d} | "
            f"{row['score_size_only']:<15d} | {row['score_size_quality']:<14d} | {row['score_size_quality_noise']:<16d} | {row['score_full_nacs']:<20d} | "
            f"{rank_str:<12}"
        )

    print("=" * 115)

    # Ranking change conclusion
    rankings_changed = df["rank_changed"].any()
    if rankings_changed:
        changed_clients = df[df["rank_changed"]]
        print("\n[FINDING] RANKING CHANGED between Config A (size_only) and Config D (full_nacs)!")
        for _, r in changed_clients.iterrows():
            print(
                f"  - Client {int(r['client_id'])+1} (N={r['client_size']}, eps={r['target_epsilon']:.1f}): "
                f"Ranked #{int(r['rank_size_only'])} under size_only -> Ranked #{int(r['rank_full_nacs'])} under full_nacs "
                f"(Size score: {r['score_size_only']} vs Full NACS: {r['score_full_nacs']})"
            )
        print(
            "  => This directly supports the claim that NACS prevents clients from winning purely due to large dataset size."
        )
    else:
        print("\n[FINDING] RANKINGS REMAINED IDENTICAL between Config A (size_only) and Config D (full_nacs).")
        print(
            "  => Client rankings were preserved across dataset-size baseline and full NACS in this run."
        )

    print("=" * 115 + "\n")

    # Generate multi-config bar plot
    os.makedirs(os.path.dirname(save_plot_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(df))
    width = 0.2

    ax.bar(x - 1.5 * width, df["score_size_only"], width, label="Config A (Size Only)", color="#4C72B0")
    ax.bar(x - 0.5 * width, df["score_size_quality"], width, label="Config B (Size + Quality)", color="#55A868")
    ax.bar(x + 0.5 * width, df["score_size_quality_noise"], width, label="Config C (Size + Qual + Noise)", color="#DD8452")
    ax.bar(x + 1.5 * width, df["score_full_nacs"], width, label="Config D (Full NACS EMA)", color="#8172B3")

    ax.set_xlabel("Participating Clients", fontsize=12, fontweight="bold")
    ax.set_ylabel("Contribution Score (0-100)", fontsize=12, fontweight="bold")
    ax.set_title("NACS 4-Configuration Ablation Study Comparison", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"Client {int(i)+1}\n(N={int(s)}, eps={e})"
            for i, s, e in zip(df["client_id"], df["client_size"], df["target_epsilon"])
        ]
    )
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend(fontsize=10, loc="upper right")

    plt.tight_layout()
    plt.savefig(save_plot_path, dpi=300)
    plt.close()
    print(f"[Plot Saved] Ablation comparison plot saved to '{save_plot_path}'")

    return df


def main():
    parser = argparse.ArgumentParser(description="Analyze 4-Configuration NACS Ablation Study and Domain Generalizability")
    parser.add_argument(
        "--json_path",
        type=str,
        default="results/phase4_nacs.json",
        help="Path to results JSON",
    )
    parser.add_argument(
        "--save_plot",
        type=str,
        default="results/ablation_study_comparison.png",
        help="Path to save ablation plot",
    )
    args = parser.parse_args()

    analyze_ablation_study(json_path=args.json_path, save_plot_path=args.save_plot)
    
    from src.utils import print_domain_generalizability_summary
    print_domain_generalizability_summary()


if __name__ == "__main__":
    main()
