"""
Deep Autoencoder for Unsupervised Anomaly Detection.
Trained solely on legitimate banking transactions to learn the normal transaction manifold.
High reconstruction errors signal anomalous / potentially fraudulent activity.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


class DeepAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 33,
        latent_dim: int = 8,
        encoder_dims: Tuple[int, ...] = (64, 32, 16),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder_dims = encoder_dims
        self.threshold: float = 0.0

        # Build Encoder
        enc_layers = []
        in_d = input_dim
        for h_dim in encoder_dims:
            enc_layers.append(nn.Linear(in_d, h_dim))
            enc_layers.append(nn.BatchNorm1d(h_dim))
            enc_layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout > 0:
                enc_layers.append(nn.Dropout(dropout))
            in_d = h_dim
        enc_layers.append(nn.Linear(in_d, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Build Decoder (mirror of encoder)
        dec_layers = []
        in_d = latent_dim
        for h_dim in reversed(encoder_dims):
            dec_layers.append(nn.Linear(in_d, h_dim))
            dec_layers.append(nn.BatchNorm1d(h_dim))
            dec_layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout > 0:
                dec_layers.append(nn.Dropout(dropout))
            in_d = h_dim
        dec_layers.append(nn.Linear(in_d, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Projects input into latent bottleneck representation."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstructs input from latent bottleneck."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, latent_embedding)."""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

    def compute_reconstruction_error(
        self, x: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        """
        Computes reconstruction error per sample or per feature.
        If reduction == 'mean': returns 1D tensor of MSE per sample.
        If reduction == 'none': returns 2D tensor of squared errors per feature (useful for XAI).
        """
        self.eval()
        with torch.no_grad():
            x_recon, _ = self.forward(x)
            diff_sq = (x - x_recon) ** 2
            if reduction == "mean":
                return torch.mean(diff_sq, dim=1)
            elif reduction == "none":
                return diff_sq
            else:
                raise ValueError(f"Unknown reduction: {reduction}")

    def calibrate_threshold(self, normal_val_tensor: torch.Tensor, percentile: float = 98.0) -> float:
        """
        Calibrates decision threshold based on reconstruction error of normal validation transactions.
        """
        errors = self.compute_reconstruction_error(normal_val_tensor, reduction="mean").cpu().numpy()
        self.threshold = float(np.percentile(errors, percentile))
        return self.threshold

    def predict_anomaly(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (is_anomaly_binary, continuous_anomaly_score).
        """
        scores = self.compute_reconstruction_error(x, reduction="mean").cpu().numpy()
        is_anomaly = (scores >= self.threshold).astype(np.int64)
        return is_anomaly, scores
