import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import flwr as fl
from typing import Dict, List, Tuple
import numpy as np

from src.model import MedicalCNN, get_weights, set_weights
from src.utils import compute_metrics


class MedicalFLClient(fl.client.NumPyClient):
    """
    Flower Client representing a simulated hospital.
    Executes local training and validation of MedicalCNN model.
    """
    def __init__(self, client_id: int, train_loader: DataLoader, val_loader: DataLoader, device: torch.device):
        self.client_id = client_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model = MedicalCNN(in_channels=1, num_classes=2).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        return get_weights(self.model)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
        set_weights(self.model, parameters)
        self.model.train()
        
        epochs = config.get("local_epochs", 2)
        lr = config.get("lr", 0.01)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

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
        updated_weights = get_weights(self.model)
        num_samples = len(self.train_loader.dataset)
        
        return updated_weights, num_samples, {"train_loss": avg_loss, "client_id": self.client_id}

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict]:
        set_weights(self.model, parameters)
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
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        metrics = compute_metrics(all_targets, all_preds)
        val_loss = float(np.mean(val_losses))
        num_samples = len(self.val_loader.dataset)
        metrics["val_loss"] = val_loss

        return val_loss, num_samples, metrics
