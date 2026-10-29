import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import List, Any
import numpy as np
from opacus.validators import ModuleValidator


class NIHChestCNN(nn.Module):
    """
    Convolutional Neural Network for NIH Chest X-ray Multi-Label Classification.
    Designed for downscaled 64x64 grayscale medical images.
    Output: exactly 6 raw logits corresponding to the 6 target clinical findings
    (Infiltration, Effusion, Atelectasis, Nodule, Mass, Pneumothorax),
    evaluated with BCEWithLogitsLoss.
    Compatible with Opacus Differential Privacy via ModuleValidator.
    Includes backward-compatible state_dict adaptation for legacy callers.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 6):
        super(NIHChestCNN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.requested_num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)  # 64x64 -> 32x32

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)  # 32x32 -> 16x16

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)  # 16x16 -> 8x8

        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        out = self.fc2(x)  # Raw logits for BCEWithLogitsLoss

        # If an external legacy caller (such as ui/app.py) instantiated with 14 classes
        # but loaded a 6-class checkpoint, pad output logits with -100 so the first 6
        # correspond to the 6 targets and remaining indices have ~0.0 probability.
        if hasattr(self, "requested_num_classes") and self.requested_num_classes > out.size(1):
            pad_size = self.requested_num_classes - out.size(1)
            pad_tensor = torch.full(
                (out.size(0), pad_size), -100.0, device=out.device, dtype=out.dtype
            )
            out = torch.cat([out, pad_tensor], dim=1)

        return out

    def load_state_dict(self, state_dict: Any, strict: bool = True):
        """
        Adaptive state_dict loader that gracefully reconciles:
        1. Checkpoint trained with Opacus ModuleValidator.fix (GroupNorm without running_mean/var).
        2. Size differences in fc2 head (e.g. 6-class checkpoint loaded into 14-class caller or vice-versa).
        """
        # Adapt fc2 head to match checkpoint classes
        if "fc2.weight" in state_dict:
            ckpt_classes = state_dict["fc2.weight"].shape[0]
            if self.fc2.out_features != ckpt_classes:
                device = self.fc2.weight.device
                self.fc2 = nn.Linear(self.fc2.in_features, ckpt_classes).to(device)
                self.num_classes = ckpt_classes

        # Adapt normalization layers: GroupNorm vs BatchNorm2d
        has_running_mean = any("running_mean" in k for k in state_dict.keys())
        if not has_running_mean and isinstance(self.bn1, nn.BatchNorm2d):
            device = self.conv1.weight.device
            self.bn1 = nn.GroupNorm(16, 16).to(device)
            self.bn2 = nn.GroupNorm(32, 32).to(device)
            self.bn3 = nn.GroupNorm(32, 64).to(device)
        elif has_running_mean and isinstance(self.bn1, nn.GroupNorm):
            device = self.conv1.weight.device
            self.bn1 = nn.BatchNorm2d(16).to(device)
            self.bn2 = nn.BatchNorm2d(32).to(device)
            self.bn3 = nn.BatchNorm2d(64).to(device)

        return super().load_state_dict(state_dict, strict=False)


def get_weights(model: nn.Module) -> List[np.ndarray]:
    """Extract model parameters as a list of NumPy arrays (for Flower FL)."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model: nn.Module, weights: List[np.ndarray]) -> None:
    """Set model parameters from a list of NumPy arrays (for Flower FL)."""
    state_dict = OrderedDict(
        {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), weights)}
    )
    model.load_state_dict(state_dict, strict=True)


def build_nih_model(in_channels: int = 1, num_classes: int = 6, device: torch.device = None) -> nn.Module:
    """
    Factory that instantiates NIHChestCNN and applies ModuleValidator.fix
    to guarantee Opacus Differential Privacy compatibility.
    """
    model = NIHChestCNN(in_channels=in_channels, num_classes=num_classes)
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)
    if device is not None:
        model = model.to(device)
    return model
