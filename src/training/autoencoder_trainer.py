"""
Trainer for Deep Autoencoder anomaly detection.
Supports AMP, multi-GPU, early stopping, and automatic threshold calibration.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.autoencoder import DeepAutoencoder
from src.utils.device import get_amp_context, prepare_model_for_device


class AutoencoderTrainer:
    def __init__(
        self,
        model: DeepAutoencoder,
        device: torch.device,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        use_amp: bool = True,
        save_path: str = "results/models/autoencoder.pt",
    ):
        self.device = device
        self.use_amp = use_amp
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        self.model = prepare_model_for_device(model, device)
        self.raw_model = model.module if isinstance(self.model, nn.DataParallel) else self.model
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2
        )
        self.criterion = nn.MSELoss()

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = len(dataloader)

        for batch in dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with get_amp_context(self.device, self.use_amp):
                recon, _ = self.model(x)
                loss = self.criterion(recon, x)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / max(1, n_batches)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = len(dataloader)

        for batch in dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(self.device, non_blocking=True)
            with get_amp_context(self.device, self.use_amp):
                recon, _ = self.model(x)
                loss = self.criterion(recon, x)
            total_loss += loss.item()

        return total_loss / max(1, n_batches)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        val_normal_tensor: torch.Tensor,
        epochs: int = 25,
        patience: int = 5,
        threshold_percentile: float = 98.0,
    ) -> Dict[str, list]:
        """
        Trains autoencoder with early stopping and threshold calibration.
        """
        best_val_loss = float("inf")
        patience_counter = 0
        history = {"train_loss": [], "val_loss": []}

        print(f"[INFO] Training Deep Autoencoder on {self.device}...")
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)
            self.scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            print(f"  Epoch {epoch:02d}/{epochs:02d} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.raw_model.state_dict(), self.save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[INFO] Early stopping triggered at epoch {epoch}")
                    break

        # Load best weights
        if self.save_path.exists():
            self.raw_model.load_state_dict(torch.load(self.save_path, map_location=self.device, weights_only=True))

        # Calibrate anomaly threshold
        val_norm_device = val_normal_tensor.to(self.device)
        threshold = self.raw_model.calibrate_threshold(val_norm_device, percentile=threshold_percentile)
        print(f"[INFO] Autoencoder anomaly threshold set at {threshold_percentile}th percentile: {threshold:.6f}")

        # Save again with updated threshold attribute
        torch.save({
            "model_state_dict": self.raw_model.state_dict(),
            "threshold": threshold,
            "threshold_percentile": threshold_percentile,
        }, self.save_path)

        return history
