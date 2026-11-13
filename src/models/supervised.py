"""
Supervised Deep Learning Classifier for Credit Card Fraud.
Uses Tabular Residual Blocks, LayerNorm/BatchNorm, GeLU, and Focal Loss.
"""

from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Binary Focal Loss for severe class imbalance.
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_weight = alpha_t * torch.pow((1.0 - p_t), self.gamma)
        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class TabularResBlock(nn.Module):
    """
    Residual block designed for tabular data representations.
    """
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class DeepFraudClassifier(nn.Module):
    """
    Deep Tabular ResNet classifier for high-precision fraud detection.
    """
    def __init__(
        self,
        input_dim: int = 33,
        hidden_dims: List[int] = (128, 64, 32),
        dropout: float = 0.25,
    ):
        super().__init__()
        self.input_dim = input_dim

        # Input projection
        first_dim = hidden_dims[0] if len(hidden_dims) > 0 else 64
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, first_dim),
            nn.BatchNorm1d(first_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2.0),
        )

        # Residual blocks
        self.res1 = TabularResBlock(first_dim, dropout=dropout)

        # Transition to lower dimensions
        layers = []
        curr_dim = first_dim
        for next_dim in hidden_dims[1:]:
            layers.append(nn.Linear(curr_dim, next_dim))
            layers.append(nn.BatchNorm1d(next_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            curr_dim = next_dim

        self.dense_stack = nn.Sequential(*layers)
        self.head = nn.Linear(curr_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Outputs raw uncalibrated logits."""
        h = self.in_proj(x)
        h = self.res1(h)
        h = self.dense_stack(h)
        logits = self.head(h)
        return logits

    def predict_proba(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Numpy-compatible inference method for Scikit-Learn / SHAP / LIME.
        Returns array of shape (N, 2): [P(Normal), P(Fraud)].
        """
        self.eval()
        device = next(self.parameters()).device
        if isinstance(x, np.ndarray):
            tensor_x = torch.as_tensor(x, dtype=torch.float32, device=device)
        else:
            tensor_x = x.to(device)

        with torch.no_grad():
            logits = self.forward(tensor_x)
            fraud_probs = torch.sigmoid(logits).cpu().numpy().flatten()
            normal_probs = 1.0 - fraud_probs
            return np.column_stack([normal_probs, fraud_probs])
