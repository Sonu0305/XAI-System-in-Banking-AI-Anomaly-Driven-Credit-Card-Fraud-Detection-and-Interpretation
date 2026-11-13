"""
PyTorch Dataset and DataLoader wrappers for tabular fraud data.
"""

from typing import Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from src.data.preprocessing import ProcessedDataset


class FraudDataset(Dataset):
    """
    Standard PyTorch Dataset for credit card transactions.
    """
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        if y is not None:
            self.y = torch.as_tensor(y, dtype=torch.float32).unsqueeze(1)
        else:
            self.y = None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def build_dataloaders(
    data: ProcessedDataset,
    batch_size: int = 512,
    num_workers: int = 2,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Constructs DataLoaders for:
    1. Unsupervised Autoencoder (X_train_normal only)
    2. Supervised Training (X_train, y_train)
    3. Validation (X_val, y_val)
    4. Testing (X_test, y_test)
    """
    # Autoencoder loader uses only normal transactions
    ae_dataset = FraudDataset(data.X_train_normal)
    ae_loader = DataLoader(
        ae_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    train_dataset = FraudDataset(data.X_train, data.y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_dataset = FraudDataset(data.X_val, data.y_val)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_dataset = FraudDataset(data.X_test, data.y_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return ae_loader, train_loader, val_loader, test_loader
