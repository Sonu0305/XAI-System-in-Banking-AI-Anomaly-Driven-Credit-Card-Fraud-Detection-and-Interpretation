"""
Trainer for Supervised Deep Learning & Hybrid Fraud Detection Models.
Optimized for imbalanced classification: early stops on validation AUPRC.
Supports AMP, multi-GPU DataParallel, and Focal Loss.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from src.models.supervised import FocalLoss
from src.utils.device import get_amp_context, prepare_model_for_device


class SupervisedTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.0008,
        weight_decay: float = 1e-4,
        use_amp: bool = True,
        loss_type: str = "focal",
        pos_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        save_path: str = "results/models/supervised_model.pt",
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
            self.optimizer, mode="max", factor=0.5, patience=2
        )

        if loss_type == "focal":
            self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            weight_tensor = torch.tensor([pos_weight], device=device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = len(dataloader)

        for x, y in dataloader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with get_amp_context(self.device, self.use_amp):
                logits = self.model(x)
                loss = self.criterion(logits, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / max(1, n_batches)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float, float]:
        """
        Computes (val_loss, val_auprc, val_roc_auc).
        AUPRC is the primary metric for fraud detection on extreme imbalance.
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        n_batches = len(dataloader)

        for x, y in dataloader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            with get_amp_context(self.device, self.use_amp):
                logits = self.model(x)
                loss = self.criterion(logits, y)

            probs = torch.sigmoid(logits)
            total_loss += loss.item()
            all_preds.append(probs.cpu().numpy())
            all_targets.append(y.cpu().numpy())

        val_loss = total_loss / max(1, n_batches)
        y_pred = np.vstack(all_preds).flatten()
        y_true = np.vstack(all_targets).flatten()

        auprc = float(average_precision_score(y_true, y_pred))
        roc_auc = float(roc_auc_score(y_true, y_pred))

        return val_loss, auprc, roc_auc

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 30,
        patience: int = 6,
    ) -> Dict[str, list]:
        """
        Executes supervised model training with early stopping on AUPRC.
        """
        best_auprc = 0.0
        patience_counter = 0
        history = {"train_loss": [], "val_loss": [], "val_auprc": [], "val_roc_auc": []}

        print(f"[INFO] Training Supervised Model on {self.device}...")
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_auprc, val_roc_auc = self.evaluate(val_loader)
            self.scheduler.step(val_auprc)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_auprc"].append(val_auprc)
            history["val_roc_auc"].append(val_roc_auc)

            print(
                f"  Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | "
                f"Val AUPRC: {val_auprc:.4f} | Val ROC-AUC: {val_roc_auc:.4f}"
            )

            if val_auprc > best_auprc:
                best_auprc = val_auprc
                patience_counter = 0
                torch.save(self.raw_model.state_dict(), self.save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[INFO] Early stopping triggered at epoch {epoch} (Best AUPRC: {best_auprc:.4f})")
                    break

        if self.save_path.exists():
            self.raw_model.load_state_dict(torch.load(self.save_path, map_location=self.device, weights_only=True))
            print(f"[SUCCESS] Best model loaded with Validation AUPRC: {best_auprc:.4f}")

        return history
