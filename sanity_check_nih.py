import torch
import torch.nn as nn
import numpy as np

from src.dataset_nih import get_nih_client_dataloaders
from src.model_nih import NIHChestCNN
from src.utils import compute_multilabel_metrics


def run_sanity_check(epochs: int = 15, lr: float = 0.001):
    print("=" * 80)
    print("STEP 2 SANITY CHECK: Single-Client Non-DP Training on NIH Chest X-ray")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataloaders for 5 clients, we will use Client 0 only
    client_train_loaders, client_val_loaders, test_loader, _, _ = get_nih_client_dataloaders(
        num_clients=5, batch_size=32, seed=42, subset_size=4000, target_size=(64, 64), return_client_sizes=True
    )

    train_loader = client_train_loaders[0]
    val_loader = client_val_loaders[0]

    # Calculate per-label pos_weight to address extreme class imbalance (~94.4% negative)
    all_train_targets = []
    for _, labels in train_loader:
        all_train_targets.append(labels)
    all_train_targets = torch.cat(all_train_targets, dim=0).numpy()
    pos_counts = all_train_targets.sum(axis=0)
    neg_counts = len(all_train_targets) - pos_counts
    pos_weight = np.clip(neg_counts / np.maximum(pos_counts, 1.0), 1.0, 10.0)
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32).to(device)

    print(f"Calculated pos_weight tensor: {pos_weight}")

    # Compute Trivial Baseline (All-Negative) on Validation Set
    all_val_targets = []
    for _, labels in val_loader:
        all_val_targets.append(labels)
    all_val_targets = torch.cat(all_val_targets, dim=0).numpy()
    all_neg_preds = np.zeros_like(all_val_targets)
    trivial_metrics = compute_multilabel_metrics(all_val_targets, all_neg_preds)
    trivial_floor = float(trivial_metrics["micro_f1"])

    print(f"[Step 1 Baseline Floor] All-Negative Micro-F1: {trivial_floor * 100:.2f}%\n")

    # Instantiate Model, Optimizer, Criterion
    model = NIHChestCNN(in_channels=1, num_classes=14).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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

        mean_train_loss = float(np.mean(epoch_losses))

        # Evaluate on Validation Set
        model.eval()
        val_losses = []
        val_outputs = []
        val_targets = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                v_loss = criterion(outputs, labels)
                val_losses.append(v_loss.item())
                val_outputs.append(outputs.cpu())
                val_targets.append(labels.cpu())

        val_outputs_cat = torch.cat(val_outputs, dim=0)
        val_targets_cat = torch.cat(val_targets, dim=0)

        metrics = compute_multilabel_metrics(val_targets_cat, val_outputs_cat)
        micro_f1 = metrics["micro_f1"]
        mean_val_loss = float(np.mean(val_losses))

        diff_above_floor = (micro_f1 - trivial_floor) * 100.0
        print(
            f"Epoch {ep:2d}/{epochs} | Train Loss: {mean_train_loss:.4f} | "
            f"Val Loss: {mean_val_loss:.4f} | Val Micro-F1: {micro_f1 * 100:.2f}% "
            f"({diff_above_floor:+.2f}% vs trivial floor)"
        )

    print("\n" + "=" * 80)
    if micro_f1 > (trivial_floor + 0.05):
        print(f"[PASSED] SANITY CHECK PASSED: Micro-F1 ({micro_f1 * 100:.2f}%) is clearly above floor ({trivial_floor * 100:.2f}%).")
    else:
        print(f"[FAILED] SANITY CHECK FAILED: Micro-F1 ({micro_f1 * 100:.2f}%) did not climb sufficiently above floor ({trivial_floor * 100:.2f}%).")
    print("=" * 80)


if __name__ == "__main__":
    run_sanity_check()
