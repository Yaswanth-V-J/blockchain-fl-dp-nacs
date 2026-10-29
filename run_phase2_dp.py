import os
import torch
import numpy as np

from opacus.validators import ModuleValidator

from src.dataset import get_client_dataloaders
from src.model import MedicalCNN, get_weights, set_weights
from src.client_dp import PrivateMedicalFLClient
from src.utils import compute_metrics, save_metrics_json, plot_phase2_comparison
from run_phase1 import fed_avg, evaluate_global_model


def run_dp_experiment_for_epsilon(target_epsilon, num_clients=5, num_rounds=10, alpha=0.5):
    eps_label = "None (No-DP)" if target_epsilon is None else str(target_epsilon)
    print("\n" + "=" * 75)
    print(f"RUNNING DP FL EXPERIMENT: target_epsilon = {eps_label}")
    print(f"  Privacy budget is TOTAL across all {num_rounds} rounds (not per-round)")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_epochs = 2
    lr = 0.005

    # 1. Load exact same dataset partition (seed=42 for fair comparison with Phase 1)
    client_train_loaders, client_val_loaders, test_loader, client_indices = get_client_dataloaders(
        num_clients=num_clients, alpha=alpha, batch_size=32
    )

    # 2. Instantiate DP clients — PrivacyEngine created ONCE here per client,
    #    calibrated for total_rounds * local_epochs total training epochs.
    clients = [
        PrivateMedicalFLClient(
            client_id=i,
            train_loader=client_train_loaders[i],
            val_loader=client_val_loaders[i],
            device=device,
            target_epsilon=target_epsilon,
            target_delta=1e-5,
            total_rounds=num_rounds,
            local_epochs=local_epochs,
            lr=lr,
        )
        for i in range(num_clients)
    ]

    # 3. Separate unwrapped global model for server-side evaluation.
    #    Uses GroupNorm (same architecture as DP clients' unwrapped module)
    #    so set_weights / get_weights keys are compatible.
    base = MedicalCNN(in_channels=1, num_classes=2)
    if not ModuleValidator.is_valid(base):
        global_model = ModuleValidator.fix(base).to(device)
    else:
        global_model = base.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    # Evaluation on initial (random) weights
    initial_metrics = evaluate_global_model(global_model, test_loader, device, criterion)
    print(f"Round 0 (Initial) -> Test Accuracy: {initial_metrics['accuracy']:.4f}, "
          f"F1: {initial_metrics['f1']:.4f}, Loss: {initial_metrics['loss']:.4f}")

    history = {
        "target_epsilon": target_epsilon,
        "rounds": [0],
        "accuracy": [initial_metrics["accuracy"]],
        "f1": [initial_metrics["f1"]],
        "precision": [initial_metrics["precision"]],
        "recall": [initial_metrics["recall"]],
        "loss": [initial_metrics["loss"]],
        "epsilon_spent": [0.0],
    }

    current_weights = get_weights(global_model)
    config = {"local_epochs": local_epochs, "lr": lr}

    for rnd in range(1, num_rounds + 1):
        print(f"\n--- DP FL Round {rnd}/{num_rounds} (Target epsilon={eps_label}) ---")
        client_updates = []
        client_sizes = []
        round_epsilons = []

        for client in clients:
            updated_weights, num_samples, fit_metrics = client.fit(current_weights, config)
            client_updates.append(updated_weights)
            client_sizes.append(num_samples)
            eps_spent = fit_metrics.get("epsilon_spent", 0.0)
            round_epsilons.append(eps_spent)

            eps_str = f"{eps_spent:.4f}" if eps_spent is not None else "N/A"
            print(f"  Hospital Client {client.client_id + 1}: "
                  f"train loss = {fit_metrics['train_loss']:.4f} | "
                  f"Cumulative epsilon spent: {eps_str}")

        # Server Aggregation (FedAvg)
        current_weights = fed_avg(client_updates, client_sizes)
        set_weights(global_model, current_weights)

        # Global Evaluation
        metrics = evaluate_global_model(global_model, test_loader, device, criterion)
        max_eps_this_round = (
            float(np.max([e for e in round_epsilons if e is not None]))
            if round_epsilons and round_epsilons[0] is not None
            else 0.0
        )

        print(f"  >> Global Test Accuracy: {metrics['accuracy']:.4f} | "
              f"F1: {metrics['f1']:.4f} | "
              f"Max cumulative epsilon: {max_eps_this_round:.4f}")

        history["rounds"].append(rnd)
        history["accuracy"].append(metrics["accuracy"])
        history["f1"].append(metrics["f1"])
        history["precision"].append(metrics["precision"])
        history["recall"].append(metrics["recall"])
        history["loss"].append(metrics["loss"])
        history["epsilon_spent"].append(max_eps_this_round)

    # Save metrics JSON
    file_tag = "None" if target_epsilon is None else str(target_epsilon)
    json_path = f"results/phase2_dp_eps{file_tag}.json"
    save_metrics_json(history, json_path)

    return history


def main():
    print("=" * 75)
    print("PHASE 2: Differential Privacy (DP-SGD via Opacus) Experiment Suite")
    print("  Privacy budget = TOTAL across all rounds (not per-round)")
    print("  Target Epsilon Values: [1.0, 3.0, 8.0, 20.0, None (No-DP)]")
    print("=" * 75)

    epsilon_targets = [1.0, 3.0, 8.0, 20.0, None]
    results_summary = []

    for eps in epsilon_targets:
        history = run_dp_experiment_for_epsilon(target_epsilon=eps, num_clients=5, num_rounds=10)
        results_summary.append({
            "target_epsilon": eps,
            "final_accuracy": history["accuracy"][-1],
            "final_f1": history["f1"][-1],
            "final_epsilon_spent": history["epsilon_spent"][-1],
        })

    # Generate multi-epsilon comparison plot
    print("\n[Plotting] Generating Phase 2 Differential Privacy Comparison Plot...")
    plot_phase2_comparison()

    print("\n" + "=" * 75)
    print("PHASE 2 EXPERIMENT SUITE COMPLETE!")
    print("Summary of Final Round Accuracy vs Privacy Budget Epsilon:")
    print(f"{'Target Epsilon':<20} | {'Final eps Spent':<15} | {'Final Accuracy':<15} | {'Final Macro F1':<15}")
    print("-" * 75)
    for res in results_summary:
        eps_str = "None (No-DP)" if res["target_epsilon"] is None else f"{res['target_epsilon']:.1f}"
        spent_str = f"{res['final_epsilon_spent']:.4f}"
        print(f"{eps_str:<20} | {spent_str:<15} | {res['final_accuracy']*100:.2f}%{'':<9} | {res['final_f1']*100:.2f}%")
    print("=" * 75)


if __name__ == "__main__":
    main()
