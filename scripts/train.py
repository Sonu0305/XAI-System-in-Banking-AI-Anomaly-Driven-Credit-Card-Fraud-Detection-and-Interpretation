"""
End-to-End Training Orchestration Script.
Trains:
1. Unsupervised Deep Autoencoder (on normal transactions)
2. Supervised Deep Tabular ResNet Classifier (with Focal Loss)
3. Hybrid Anomaly-Augmented Deep Model
4. Tree Baselines (XGBoost & LightGBM)
Saves all model checkpoints and training logs in results/models/.
"""

import argparse
import sys
from pathlib import Path
import joblib
import numpy as np
import torch

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import build_dataloaders
from src.data.preprocessing import DataPipeline
from src.models.autoencoder import DeepAutoencoder
from src.models.baselines import LightGBMBaseline, XGBoostBaseline
from src.models.hybrid import HybridFraudModel
from src.models.supervised import DeepFraudClassifier
from src.training.autoencoder_trainer import AutoencoderTrainer
from src.training.supervised_trainer import SupervisedTrainer
from src.utils.config import load_config
from src.utils.device import get_dataloader_kwargs, get_device, get_device_info, set_seed


def main():
    parser = argparse.ArgumentParser(description="Train Fraud Detection Models")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--epochs-ae", type=int, default=None, help="Override Autoencoder epochs")
    parser.add_argument("--epochs-sup", type=int, default=None, help="Override Supervised epochs")
    parser.add_argument("--quick", action="store_true", help="Quick mode for rapid testing/verification")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg["project"].get("seed", 42)
    set_seed(seed)

    device_pref = cfg["project"].get("device", "auto")
    device = get_device(device_pref)
    dev_info = get_device_info()

    print("=" * 70)
    print("AI ANOMALY-DRIVEN CREDIT CARD FRAUD DETECTION PIPELINE")
    print(f"Device: {device} | CUDA Available: {dev_info['cuda_available']} | Dual GPU: {dev_info['is_dual_gpu']}")
    if dev_info["device_names"]:
        print(f"GPU Hardware: {', '.join(dev_info['device_names'])}")
    print("=" * 70)

    # 1. Data Pipeline
    print("\n[STEP 1/5] Loading & Preprocessing Dataset...")
    pipeline = DataPipeline(
        raw_csv_path=cfg["data"].get("raw_file", "data/raw/creditcard.csv"),
        processed_dir=cfg["data"].get("processed_dir", "data/processed"),
        test_size=cfg["data"].get("test_size", 0.15),
        val_size=cfg["data"].get("val_size", 0.15),
        random_state=seed,
        scaling_method=cfg["data"].get("scaling", "robust"),
    )
    data = pipeline.load_processed()
    input_dim = data.X_train.shape[1]
    print(f"[INFO] Dataset ready. Training samples: {len(data.X_train):,}, Features: {input_dim}")
    print(f"[INFO] Normal training samples for Autoencoder: {len(data.X_train_normal):,}")

    dl_kwargs = get_dataloader_kwargs(device, num_workers=cfg["project"].get("num_workers", 2))
    ae_batch_size = cfg["anomaly_model"].get("batch_size", 512)
    sup_batch_size = cfg["supervised_model"].get("batch_size", 512)

    ae_loader, train_loader, val_loader, test_loader = build_dataloaders(
        data,
        batch_size=sup_batch_size,
        num_workers=dl_kwargs["num_workers"],
        pin_memory=dl_kwargs["pin_memory"],
    )

    # 2. Train Deep Autoencoder
    print("\n[STEP 2/5] Training Unsupervised Deep Autoencoder...")
    ae_cfg = cfg["anomaly_model"]
    ae_epochs = 2 if args.quick else (args.epochs_ae or ae_cfg.get("epochs", 20))
    autoencoder = DeepAutoencoder(
        input_dim=input_dim,
        latent_dim=ae_cfg.get("latent_dim", 8),
        encoder_dims=tuple(ae_cfg.get("encoder_dims", [64, 32, 16])),
        dropout=ae_cfg.get("dropout", 0.1),
    )

    ae_trainer = AutoencoderTrainer(
        model=autoencoder,
        device=device,
        learning_rate=ae_cfg.get("learning_rate", 0.001),
        weight_decay=ae_cfg.get("weight_decay", 1e-5),
        use_amp=cfg["project"].get("use_amp", True),
        save_path=ae_cfg.get("save_path", "results/models/autoencoder.pt"),
    )

    # Convert validation normal subset for threshold calibration
    val_normal_tensor = torch.as_tensor(data.X_val[data.y_val == 0], dtype=torch.float32)
    ae_trainer.fit(
        train_loader=ae_loader,
        val_loader=val_loader,
        val_normal_tensor=val_normal_tensor,
        epochs=ae_epochs,
        patience=ae_cfg.get("early_stopping_patience", 5),
        threshold_percentile=ae_cfg.get("threshold_percentile", 98.0),
    )

    # 3. Train Supervised Deep Learning Model
    print("\n[STEP 3/5] Training Supervised Tabular Deep Classifier (Focal Loss)...")
    sup_cfg = cfg["supervised_model"]
    sup_epochs = 2 if args.quick else (args.epochs_sup or sup_cfg.get("epochs", 25))
    deep_classifier = DeepFraudClassifier(
        input_dim=input_dim,
        hidden_dims=sup_cfg.get("hidden_dims", [128, 64, 32]),
        dropout=sup_cfg.get("dropout", 0.25),
    )

    sup_trainer = SupervisedTrainer(
        model=deep_classifier,
        device=device,
        learning_rate=sup_cfg.get("learning_rate", 0.0008),
        weight_decay=sup_cfg.get("weight_decay", 1e-4),
        use_amp=cfg["project"].get("use_amp", True),
        loss_type="focal",
        pos_weight=data.pos_weight,
        focal_alpha=sup_cfg["focal_loss"].get("alpha", 0.25),
        focal_gamma=sup_cfg["focal_loss"].get("gamma", 2.0),
        save_path=sup_cfg.get("save_path", "results/models/supervised_deep.pt"),
    )

    sup_trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=sup_epochs,
        patience=sup_cfg.get("early_stopping_patience", 6),
    )

    # 4. Train Hybrid Model (Anomaly Fusion + Deep Classifier)
    print("\n[STEP 4/5] Training Hybrid Anomaly-Augmented Deep Model...")
    hyb_cfg = cfg["hybrid_model"]
    hybrid_model = HybridFraudModel(
        autoencoder=ae_trainer.raw_model,
        input_dim=input_dim,
        latent_dim=ae_cfg.get("latent_dim", 8),
        hidden_dims=tuple(hyb_cfg.get("hidden_dims", [160, 80, 40])),
        dropout=hyb_cfg.get("dropout", 0.25),
        freeze_autoencoder=True,
    )

    hybrid_trainer = SupervisedTrainer(
        model=hybrid_model,
        device=device,
        learning_rate=hyb_cfg.get("learning_rate", 0.0008),
        use_amp=cfg["project"].get("use_amp", True),
        loss_type="focal",
        pos_weight=data.pos_weight,
        save_path=hyb_cfg.get("save_path", "results/models/hybrid_deep.pt"),
    )

    hybrid_trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=sup_epochs,
        patience=hyb_cfg.get("early_stopping_patience", 6),
    )

    # 5. Train Baseline Tree Models (XGBoost & LightGBM)
    print("\n[STEP 5/5] Training Gradient Boosted Baselines...")
    xgb_cfg = cfg["baselines"]["xgboost"]
    xgb_n = 30 if args.quick else xgb_cfg.get("n_estimators", 250)
    xgb_model = XGBoostBaseline(
        n_estimators=xgb_n,
        max_depth=xgb_cfg.get("max_depth", 5),
        learning_rate=xgb_cfg.get("learning_rate", 0.05),
        scale_pos_weight=min(100.0, data.pos_weight / 5.0),  # Moderated class weight for stability
        random_state=seed,
    )
    print("  Fitting XGBoost...")
    xgb_model.fit(data.X_train, data.y_train, eval_set=[(data.X_val, data.y_val)])
    joblib.dump(xgb_model, "results/models/xgboost.joblib")

    lgb_cfg = cfg["baselines"]["lightgbm"]
    lgb_n = 30 if args.quick else lgb_cfg.get("n_estimators", 250)
    lgb_model = LightGBMBaseline(
        n_estimators=lgb_n,
        max_depth=lgb_cfg.get("max_depth", 6),
        learning_rate=lgb_cfg.get("learning_rate", 0.05),
        scale_pos_weight=min(100.0, data.pos_weight / 5.0),
        random_state=seed,
    )
    print("  Fitting LightGBM...")
    lgb_model.fit(data.X_train, data.y_train)
    joblib.dump(lgb_model, "results/models/lightgbm.joblib")

    print("\n" + "=" * 70)
    print("[SUCCESS] All models trained and saved to results/models/!")
    print("=" * 70)


if __name__ == "__main__":
    main()
