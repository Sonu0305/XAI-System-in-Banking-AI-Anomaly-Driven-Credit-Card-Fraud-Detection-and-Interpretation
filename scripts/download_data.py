"""
Dataset download and extraction script for Kaggle Credit Card Fraud Detection.
Uses Kaggle API credentials loaded via .env / ~/.kaggle/access_token.
"""

import argparse
import os
import sys
from pathlib import Path
import zipfile

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.utils.credentials import load_kaggle_credentials
from src.utils.config import load_config


def download_dataset(dataset_name: str, target_dir: str = "data/raw", force: bool = False) -> Path:
    """
    Downloads and extracts a Kaggle dataset into target_dir.
    """
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    expected_csv = target_path / "creditcard.csv"

    if expected_csv.exists() and not force:
        print(f"[INFO] Dataset already present at: {expected_csv}")
        return expected_csv

    print(f"[INFO] Authenticating Kaggle API...")
    if not load_kaggle_credentials():
        raise RuntimeError(
            "Could not load Kaggle credentials! Please ensure .env points to ~/.kaggle/access_token "
            "or set KAGGLE_USERNAME and KAGGLE_KEY."
        )

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    print(f"[INFO] Downloading dataset '{dataset_name}' to '{target_dir}'...")
    api.dataset_download_files(dataset_name, path=str(target_path), unzip=True, quiet=False)

    # In case unzip didn't extract directly or was zipped
    for zip_file in target_path.glob("*.zip"):
        print(f"[INFO] Extracting {zip_file}...")
        with zipfile.ZipFile(zip_file, "r") as zf:
            zf.extractall(target_path)
        os.remove(zip_file)

    if not expected_csv.exists():
        # Check if CSV has a different case or name
        csv_files = list(target_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV file found in {target_dir} after download!")
        print(f"[INFO] Found CSV: {csv_files[0]}")
        return csv_files[0]

    # Verify integrity
    df = pd.read_csv(expected_csv)
    print(f"[SUCCESS] Downloaded and verified dataset: {expected_csv}")
    print(f"          Total rows: {len(df):,}, Columns: {len(df.columns)}")
    if "Class" in df.columns:
        counts = df["Class"].value_counts().to_dict()
        fraud_ratio = (counts.get(1, 0) / len(df)) * 100
        print(f"          Class 0 (Normal): {counts.get(0, 0):,}")
        print(f"          Class 1 (Fraud):  {counts.get(1, 0):,} ({fraud_ratio:.4f}%)")

    return expected_csv


def main():
    parser = argparse.ArgumentParser(description="Download Credit Card Fraud Dataset from Kaggle")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--force", action="store_true", help="Force re-download even if dataset exists")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_name = cfg["data"].get("primary_dataset", "mlg-ulb/creditcardfraud")
    target_dir = os.path.dirname(cfg["data"].get("raw_file", "data/raw/creditcard.csv"))

    download_dataset(dataset_name=dataset_name, target_dir=target_dir, force=args.force)


if __name__ == "__main__":
    main()
