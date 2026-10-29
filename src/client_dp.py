import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import flwr as fl
from typing import Dict, List, Tuple, Optional
import numpy as np
import warnings

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

from src.model import MedicalCNN, get_weights, set_weights
from src.utils import compute_metrics


class PrivateMedicalFLClient(fl.client.NumPyClient):
    """
    Flower Client with Differential Privacy (DP-SGD via Opacus).

    Privacy Accounting:
        target_epsilon is the TOTAL privacy budget for the entire federated
        training run (all rounds combined), NOT a per-round budget.  The
        PrivacyEngine is created once in __init__() — calibrated for
        total_rounds * local_epochs total training epochs — and persists
        across all fit() calls.  The internal RDP accountant therefore
        tracks cumulative epsilon correctly: it rises monotonically each
        round and should approach target_epsilon only after the final round.

    BatchNorm Handling:
        MedicalCNN uses BatchNorm2d, which Opacus does not support.
        ModuleValidator.fix() automatically replaces BatchNorm with
        GroupNorm at construction time, without modifying the original
        MedicalCNN class used by Phase 1.
    """

    def __init__(
        self,
        client_id: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        target_epsilon: Optional[float] = None,
        target_delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        total_rounds: int = 10,
        local_epochs: int = 2,
        lr: float = 0.005,
        model_cls: Optional[type] = None,
        criterion: Optional[nn.Module] = None,
    ):
        self.client_id = client_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.max_grad_norm = max_grad_norm
        self.local_epochs = local_epochs
        self.lr = lr
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.privacy_engine = None
        self.dp_train_loader = None

        # Flag small datasets that may cause DP noise issues
        num_samples = len(self.train_loader.dataset)
        if num_samples < 50:
            print(
                f"  [DP Warning] Hospital Client {client_id + 1} has only "
                f"{num_samples} samples — Opacus noise calibration may be "
                f"unstable (risk of NaN losses)."
            )

        # Build model with BatchNorm -> GroupNorm conversion for Opacus if needed
        if model_cls is None:
            base_model = MedicalCNN(in_channels=1, num_classes=2)
        else:
            base_model = model_cls()

        if not ModuleValidator.is_valid(base_model):
            model = ModuleValidator.fix(base_model).to(self.device)
        else:
            model = base_model.to(self.device)

        if self.target_epsilon is not None and self.target_epsilon > 0:
            # ---- DP path: create engine ONCE for the full training run ----
            # total_epochs covers every local epoch across every FL round,
            # so the noise multiplier is calibrated for the complete run.
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            self.privacy_engine = PrivacyEngine()

            total_epochs = total_rounds * local_epochs

            (
                self.model,            # GradSampleModule wrapper
                self.optimizer,        # DP-aware optimizer (persisted)
                self.dp_train_loader,  # Poisson-sampled DataLoader
            ) = self.privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=self.train_loader,
                target_epsilon=self.target_epsilon,
                target_delta=self.target_delta,
                epochs=total_epochs,
                max_grad_norm=self.max_grad_norm,
            )
        else:
            # ---- Non-DP path: plain model, optimizer created per round ----
            self.model = model
            self.optimizer = None  # Created fresh each fit() to match Phase 1

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _unwrapped_model(self) -> nn.Module:
        """Return the underlying nn.Module, unwrapped from GradSampleModule
        if DP is active, or the plain model otherwise."""
        if self.privacy_engine is not None:
            return self.model._module
        return self.model

    # ------------------------------------------------------------------ #
    # Flower NumPyClient interface
    # ------------------------------------------------------------------ #

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        return get_weights(self._unwrapped_model())

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        epochs = config.get("local_epochs", self.local_epochs)

        if self.privacy_engine is not None:
            # --- DP training (persisted engine / optimizer) ---
            # Load federated-averaged weights into the unwrapped module
            # so keys match (no _module. prefix).
            set_weights(self._unwrapped_model(), parameters)
            if self.optimizer is not None:
                self.optimizer.state.clear()
            self.model.train()

            epoch_losses = []
            for epoch in range(epochs):
                batch_losses = []
                for images, labels in self.dp_train_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    batch_losses.append(loss.item())
                epoch_losses.append(np.mean(batch_losses))

            avg_loss = float(np.mean(epoch_losses))
            # Cumulative epsilon across ALL rounds this client has trained —
            # this value increases each round as the accountant accumulates.
            epsilon_spent = float(
                self.privacy_engine.get_epsilon(self.target_delta)
            )
            noise_multiplier = float(
                getattr(
                    self.privacy_engine,
                    "noise_multiplier",
                    getattr(self.optimizer, "noise_multiplier", 0.0),
                )
            )
            # Return weights from the unwrapped module for FedAvg compatibility
            updated_weights = get_weights(self._unwrapped_model())

        else:
            # --- Non-DP training (fresh optimizer each round, like Phase 1) ---
            set_weights(self.model, parameters)
            self.model.train()

            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=config.get("lr", self.lr)
            )

            epoch_losses = []
            for epoch in range(epochs):
                batch_losses = []
                for images, labels in self.train_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    batch_losses.append(loss.item())
                epoch_losses.append(np.mean(batch_losses))

            avg_loss = float(np.mean(epoch_losses))
            epsilon_spent = 0.0
            noise_multiplier = 0.0
            updated_weights = get_weights(self.model)

        num_samples = len(self.train_loader.dataset)
        fit_metrics = {
            "train_loss": avg_loss,
            "client_id": self.client_id,
            "epsilon_spent": epsilon_spent,
            "noise_multiplier": noise_multiplier,
        }
        return updated_weights, num_samples, fit_metrics

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        set_weights(self._unwrapped_model(), parameters)
        self.model.eval()

        all_preds = []
        all_targets = []
        val_losses = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_losses.append(loss.item())

                all_preds.append(outputs.cpu())
                all_targets.append(labels.cpu())

        cat_preds = torch.cat(all_preds, dim=0)
        cat_targets = torch.cat(all_targets, dim=0)

        if isinstance(self.criterion, nn.BCEWithLogitsLoss) or (cat_targets.dim() > 1 and cat_targets.size(1) > 1):
            from src.utils import compute_multilabel_metrics
            metrics = compute_multilabel_metrics(cat_targets, cat_preds)
        else:
            argmax_preds = torch.argmax(cat_preds, dim=1)
            metrics = compute_metrics(cat_targets.numpy(), argmax_preds.numpy())

        val_loss = float(np.mean(val_losses))
        num_samples = len(self.val_loader.dataset)
        metrics["val_loss"] = val_loss

        return val_loss, num_samples, metrics
