import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np


class HARClassifier(nn.Module):
    """
    Multi-Layer Perceptron (MLP) Classifier for UCI Human Activity Recognition (HAR).
    Input: 561 tabular sensor feature vector.
    Output: 6 activity classes (Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying).
    """
    def __init__(self, input_dim: int = 561, hidden_dim1: int = 128, hidden_dim2: int = 64, num_classes: int = 6):
        super(HARClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten if 3D tensor (e.g. batch_size, 1, 561)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_weights(model: nn.Module) -> list[np.ndarray]:
    """Extract model parameters as a list of NumPy arrays (for Flower FL)."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model: nn.Module, weights: list[np.ndarray]) -> None:
    """Set model parameters from a list of NumPy arrays (for Flower FL)."""
    state_dict = OrderedDict(
        {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), weights)}
    )
    model.load_state_dict(state_dict, strict=True)
