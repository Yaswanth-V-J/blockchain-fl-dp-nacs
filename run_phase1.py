import os
import torch
import numpy as np

from src.dataset import get_client_dataloaders
from src.model import MedicalCNN, get_weights, set_weights
from src.client import MedicalFLClient
from src.utils import compute_metrics, save_metrics_json, plot_phase1_baseline


def evaluate_global_model(model, test_loader, device, criterion):
    """Evaluates global model on centralized test set."""
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

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    metrics = compute_metrics(all_targets, all_preds)
    metrics["loss"] = float(np.mean(test_losses))
    return metrics


def fed_avg(client_weights: list, client_sizes: list) -> list:
    """Standard Federated Averaging (FedAvg) of model weights weighted by client sample sizes."""
    total_samples = sum(client_sizes)
    num_layers = len(client_weights[0])
    avg_weights = []

    for layer_idx in range(num_layers):
        orig_dtype = client_weights[0][layer_idx].dtype
        layer_sum = np.zeros_like(client_weights[0][layer_idx], dtype=np.float64)
        for w, n in zip(client_weights, client_sizes):
            layer_sum += w[layer_idx].astype(np.float64) * (n / total_samples)
        avg_weights.append(layer_sum.astype(orig_dtype))

    return avg_weights


def run_phase1_simulation(num_clients: int = 5, num_rounds: int = 10, alpha: float = 0.5):
    print("=" * 70)
    print(f"PHASE 1: Plain FL Baseline (FedAvg, N={num_clients} Clients, {num_rounds} Rounds)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load non-IID partitioned dataset across simulated hospital clients
    print("\n[Step 1] Loading and partitioning MedMNIST Pneumonia dataset (non-IID Dirichlet alpha={})...".format(alpha))
    client_train_loaders, client_val_loaders, test_loader, client_indices = get_client_dataloaders(
        num_clients=num_clients, alpha=alpha, batch_size=32
    )

    for i, loader in enumerate(client_train_loaders):
        print(f"  - Hospital Client {i+1}: {len(loader.dataset)} training samples")
    print(f"  - Central Server Test Set: {len(test_loader.dataset)} test samples")

    # 2. Instantiate clients and global server model
    print("\n[Step 2] Initializing Global Model & Flower Hospital Clients...")
    global_model = MedicalCNN(in_channels=1, num_classes=2).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    clients = [
        MedicalFLClient(
            client_id=i,
            train_loader=client_train_loaders[i],
            val_loader=client_val_loaders[i],
            device=device
        )
        for i in range(num_clients)
    ]

    # Initial global model evaluation
    initial_metrics = evaluate_global_model(global_model, test_loader, device, criterion)
    print(f"Round 0 (Initial) -> Test Accuracy: {initial_metrics['accuracy']:.4f}, F1: {initial_metrics['f1']:.4f}, Loss: {initial_metrics['loss']:.4f}")

    history = {
        "rounds": [0],
        "accuracy": [initial_metrics["accuracy"]],
        "f1": [initial_metrics["f1"]],
        "precision": [initial_metrics["precision"]],
        "recall": [initial_metrics["recall"]],
        "loss": [initial_metrics["loss"]]
    }

    # 3. Perform Federated Learning Rounds
    print("\n[Step 3] Executing Federated Learning Rounds...")
    current_weights = get_weights(global_model)

    for rnd in range(1, num_rounds + 1):
        print(f"\n--- FL Round {rnd}/{num_rounds} ---")
        client_updates = []
        client_sizes = []

        # Local training per client
        config = {"local_epochs": 2, "lr": 0.005}
        for client in clients:
            updated_weights, num_samples, fit_metrics = client.fit(current_weights, config)
            client_updates.append(updated_weights)
            client_sizes.append(num_samples)
            print(f"  Hospital Client {client.client_id + 1}: fit loss = {fit_metrics['train_loss']:.4f}")

        # Server Aggregation (FedAvg)
        current_weights = fed_avg(client_updates, client_sizes)
        set_weights(global_model, current_weights)

        # Global Evaluation
        metrics = evaluate_global_model(global_model, test_loader, device, criterion)
        print(f"  >> Global Test Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | Loss: {metrics['loss']:.4f}")

        history["rounds"].append(rnd)
        history["accuracy"].append(metrics["accuracy"])
        history["f1"].append(metrics["f1"])
        history["precision"].append(metrics["precision"])
        history["recall"].append(metrics["recall"])
        history["loss"].append(metrics["loss"])

    # 4. Save results & generate plots
    json_path = "results/phase1_baseline.json"
    plot_path = "results/phase1_baseline.png"
    
    save_metrics_json(history, json_path)
    plot_phase1_baseline(history, plot_path)

    # Save final global model state dict (Part B Step 1)
    os.makedirs("results", exist_ok=True)
    torch.save(global_model.state_dict(), "results/global_model_medical.pt")
    print(f"[Model Weights Saved] Final global Medical model state dict saved to results/global_model_medical.pt")

    print("\n" + "=" * 70)
    print(f"PHASE 1 COMPLETE!")
    print(f"Final Global Accuracy: {history['accuracy'][-1]:.4f}")
    print(f"Final Global Macro F1: {history['f1'][-1]:.4f}")
    print(f"Results JSON: {json_path}")
    print(f"Plot Image: {plot_path}")
    print("=" * 70)

    return history


if __name__ == "__main__":
    run_phase1_simulation()
