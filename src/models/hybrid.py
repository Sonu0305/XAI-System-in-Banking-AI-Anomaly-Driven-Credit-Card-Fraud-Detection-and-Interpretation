"""
Hybrid Pipeline: Unsupervised Anomaly Representation + Supervised Deep Classifier.
Augments raw feature representations with Autoencoder bottleneck latents and reconstruction errors.
"""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from src.models.autoencoder import DeepAutoencoder
from src.models.supervised import DeepFraudClassifier


class HybridFraudModel(nn.Module):
    """
    End-to-end Hybrid Architecture:
    Extracts manifold representations from DeepAutoencoder and fuses them with raw features
    into a Deep Fraud Classifier.
    """
    def __init__(
        self,
        autoencoder: DeepAutoencoder,
        input_dim: int = 33,
        latent_dim: int = 8,
        hidden_dims: Tuple[int, ...] = (160, 80, 40),
        dropout: float = 0.25,
        freeze_autoencoder: bool = True,
    ):
        super().__init__()
        self.autoencoder = autoencoder
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.augmented_dim = input_dim + latent_dim + 1  # [raw, latent, recon_error]

        if freeze_autoencoder:
            for p in self.autoencoder.parameters():
                p.requires_grad = False

        self.classifier = DeepFraudClassifier(
            input_dim=self.augmented_dim,
            hidden_dims=list(hidden_dims),
            dropout=dropout,
        )

    def extract_anomaly_features(self, x: torch.Tensor) -> torch.Tensor:
        """Computes latent representation and scalar reconstruction error."""
        self.autoencoder.eval()
        with torch.set_grad_enabled(not self.autoencoder.encoder[0].weight.requires_grad):
            x_recon, z = self.autoencoder(x)
            recon_error = torch.mean((x - x_recon) ** 2, dim=1, keepdim=True)
            augmented = torch.cat([x, z, recon_error], dim=1)
        return augmented

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagates raw features through anomaly fusion and deep classifier."""
        x_aug = self.extract_anomaly_features(x)
        logits = self.classifier(x_aug)
        return logits

    def predict_proba(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Numpy-compatible probability output for SHAP / LIME.
        Returns shape (N, 2): [P(Normal), P(Fraud)].
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
