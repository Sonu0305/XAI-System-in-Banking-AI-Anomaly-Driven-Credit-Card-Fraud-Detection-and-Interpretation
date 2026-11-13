"""
Data preprocessing, feature engineering, and stratified splitting pipeline.
Ensures zero data leakage by fitting all scalers strictly on training data.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler


@dataclass
class ProcessedDataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    X_train_normal: np.ndarray  # Only Class 0 for Autoencoder unsupervised training
    feature_names: List[str]
    scaler: Any = None
    pos_weight: float = 1.0


class DataPipeline:
    def __init__(
        self,
        raw_csv_path: str = "data/raw/creditcard.csv",
        processed_dir: str = "data/processed",
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42,
        engineer_time_features: bool = True,
        scaling_method: str = "robust",
    ):
        self.raw_csv_path = Path(raw_csv_path)
        self.processed_dir = Path(processed_dir)
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.engineer_time_features = engineer_time_features
        self.scaling_method = scaling_method
        self.scaler = None
        self.feature_names: List[str] = []

    def load_raw_data(self) -> pd.DataFrame:
        """Loads the raw credit card dataset."""
        if not self.raw_csv_path.exists():
            raise FileNotFoundError(f"Raw dataset file missing: {self.raw_csv_path}")
        return pd.read_csv(self.raw_csv_path)

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates cyclic time features and log-transformed amount features.
        """
        data = df.copy()
        if self.engineer_time_features and "Time" in data.columns:
            # 48 hours total in dataset: compute hour of day (0-23)
            hours = (data["Time"] / 3600.0) % 24.0
            data["Hour_Sin"] = np.sin(2.0 * np.pi * hours / 24.0)
            data["Hour_Cos"] = np.cos(2.0 * np.pi * hours / 24.0)

        # Log transform for Amount (handles high positive skewness)
        if "Amount" in data.columns:
            data["Amount_Log"] = np.log1p(np.maximum(0, data["Amount"]))

        return data

    def run(self, save_artifacts: bool = True) -> ProcessedDataset:
        """
        Executes end-to-end preprocessing:
        1. Loads raw CSV
        2. Applies feature engineering
        3. Performs stratified train/val/test split
        4. Scales Amount, Time, and PCA features using training statistics
        5. Saves processed arrays and scalers to disk
        """
        df = self.load_raw_data()
        df = self.engineer_features(df)

        target_col = "Class"
        y = df[target_col].to_numpy(dtype=np.int64)
        X_df = df.drop(columns=[target_col])
        self.feature_names = list(X_df.columns)

        # First split: train+val (85%) and test (15%)
        X_temp, X_test_df, y_temp, y_test = train_test_split(
            X_df,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        # Second split: train (70% total) and val (15% total)
        val_relative_size = self.val_size / (1.0 - self.test_size)
        X_train_df, X_val_df, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_relative_size,
            random_state=self.random_state,
            stratify=y_temp,
        )

        # Fit Scaler strictly on X_train_df
        if self.scaling_method == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()

        X_train = self.scaler.fit_transform(X_train_df).astype(np.float32)
        X_val = self.scaler.transform(X_val_df).astype(np.float32)
        X_test = self.scaler.transform(X_test_df).astype(np.float32)

        # Subset normal transactions for Autoencoder training
        normal_mask = (y_train == 0)
        X_train_normal = X_train[normal_mask].copy()

        # Compute positive class weight (N_negative / N_positive)
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        pos_weight = float(n_neg) / float(max(1, n_pos))

        if save_artifacts:
            self.processed_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                self.processed_dir / "creditcard_splits.npz",
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                X_train_normal=X_train_normal,
            )
            joblib.dump(self.scaler, self.processed_dir / "scaler.joblib")
            joblib.dump(self.feature_names, self.processed_dir / "feature_names.joblib")
            print(f"[INFO] Processed artifacts saved in {self.processed_dir}")

        return ProcessedDataset(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            X_train_normal=X_train_normal,
            feature_names=self.feature_names,
            scaler=self.scaler,
            pos_weight=pos_weight,
        )

    def load_processed(self) -> ProcessedDataset:
        """Loads pre-saved processed data from disk."""
        npz_path = self.processed_dir / "creditcard_splits.npz"
        scaler_path = self.processed_dir / "scaler.joblib"
        names_path = self.processed_dir / "feature_names.joblib"

        if not npz_path.exists():
            return self.run(save_artifacts=True)

        data = np.load(npz_path)
        scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        feature_names = joblib.load(names_path) if names_path.exists() else []

        y_train = data["y_train"]
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        pos_weight = float(n_neg) / float(max(1, n_pos))

        return ProcessedDataset(
            X_train=data["X_train"],
            y_train=data["y_train"],
            X_val=data["X_val"],
            y_val=data["y_val"],
            X_test=data["X_test"],
            y_test=data["y_test"],
            X_train_normal=data["X_train_normal"],
            feature_names=feature_names,
            scaler=scaler,
            pos_weight=pos_weight,
        )
