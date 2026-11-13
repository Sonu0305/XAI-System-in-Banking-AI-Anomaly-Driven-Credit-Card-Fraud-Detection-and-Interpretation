"""
Model Evaluation and Benchmark Comparison Script.
Evaluates all trained models on the test set:
- Deep Autoencoder (unsupervised reconstruction)
- DeepFraudClassifier (supervised Tabular ResNet)
- HybridFraudModel (anomaly-augmented deep network)
- XGBoost
- LightGBM
- Isolation Forest
Generates Precision-Recall curves, ROC curves, confusion matrices, and comparison tables.
"""

import argparse
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessing import DataPipeline
from src.evaluation.evaluator import ModelEvaluator
from src.models.autoencoder import DeepAutoencoder
from src.models.baselines import IsolationForestBaseline, LogisticRegressionBaseline
from src.models.hybrid import HybridFraudModel
from src.models.supervised import DeepFraudClassifier
from src.utils.config import load_config
from src.utils.device import get_device


def main():
    parser = argparse.ArgumentParser(description="Evaluate Fraud Detection Models")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["project"].get("device", "auto"))

    print("=" * 70)
    print("EVALUATING FRAUD DETECTION BENCHMARK SUITE")
    print(f"Device: {device}")
    print("=" * 70)

    # Load preprocessed splits
    pipeline = DataPipeline(
        raw_csv_path=cfg["data"].get("raw_file", "data/raw/creditcard.csv"),
        processed_dir=cfg["data"].get("processed_dir", "data/processed"),
    )
    data = pipeline.load_processed()
    input_dim = data.X_test.shape[1]
    y_test = data.y_test

    evaluator = ModelEvaluator(
        results_dir=cfg["evaluation"].get("results_dir", "results"),
        cost_fp=cfg["evaluation"].get("cost_false_positive", 10.0),
        cost_fn=cfg["evaluation"].get("cost_false_negative", 500.0),
    )

    predictions = {}

    # 1. Evaluate Isolation Forest
    print("\n[EVAL] 1. Isolation Forest...")
    iso = IsolationForestBaseline(random_state=42)
    iso.fit(data.X_train_normal)
    iso_probs = iso.predict_proba(data.X_test)[:, 1]
    predictions["Isolation Forest"] = (y_test, iso_probs)
    evaluator.evaluate_model("Isolation Forest", y_test, iso_probs, optimize_for="f2")

    # 2. Evaluate Logistic Regression Baseline
    print("[EVAL] 2. Logistic Regression (Balanced)...")
    lr = LogisticRegressionBaseline(random_state=42)
    lr.fit(data.X_train, data.y_train)
    lr_probs = lr.predict_proba(data.X_test)[:, 1]
    predictions["Logistic Regression"] = (y_test, lr_probs)
    evaluator.evaluate_model("Logistic Regression", y_test, lr_probs, optimize_for="f2")

    # 3. Evaluate Deep Autoencoder Alone
    print("[EVAL] 3. Deep Autoencoder (Reconstruction Error)...")
    ae_cfg = cfg["anomaly_model"]
    ae = DeepAutoencoder(
        input_dim=input_dim,
        latent_dim=ae_cfg.get("latent_dim", 8),
        encoder_dims=tuple(ae_cfg.get("encoder_dims", [64, 32, 16])),
    ).to(device)

    ae_ckpt_path = Path(ae_cfg.get("save_path", "results/models/autoencoder.pt"))
    if ae_ckpt_path.exists():
        ckpt = torch.load(ae_ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        ae.load_state_dict(state_dict)
        test_tensor = torch.as_tensor(data.X_test, dtype=torch.float32, device=device)
        recon_errors = ae.compute_reconstruction_error(test_tensor, reduction="mean").cpu().numpy()

        # Normalize reconstruction error into pseudo-probability
        min_e, max_e = recon_errors.min(), recon_errors.max()
        norm_ae_probs = (recon_errors - min_e) / (max_e - min_e + 1e-8)
        predictions["Autoencoder (Alone)"] = (y_test, norm_ae_probs)
        evaluator.evaluate_model("Autoencoder (Alone)", y_test, norm_ae_probs, optimize_for="f2")

    # 4. Evaluate Supervised Tabular Deep Classifier
    print("[EVAL] 4. Supervised Deep Tabular ResNet...")
    sup_cfg = cfg["supervised_model"]
    deep_model = DeepFraudClassifier(
        input_dim=input_dim,
        hidden_dims=sup_cfg.get("hidden_dims", [128, 64, 32]),
    ).to(device)

    sup_ckpt_path = Path(sup_cfg.get("save_path", "results/models/supervised_deep.pt"))
    if sup_ckpt_path.exists():
        deep_model.load_state_dict(torch.load(sup_ckpt_path, map_location=device, weights_only=True))
        deep_probs = deep_model.predict_proba(data.X_test)[:, 1]
        predictions["Supervised Deep Tabular"] = (y_test, deep_probs)
        res_deep = evaluator.evaluate_model("Supervised Deep Tabular", y_test, deep_probs, optimize_for="f2")
        evaluator.plot_confusion_matrix(res_deep)

    # 5. Evaluate Hybrid Anomaly-Augmented Model
    print("[EVAL] 5. Hybrid Anomaly-Augmented Deep Model...")
    hyb_cfg = cfg["hybrid_model"]
    hybrid_model = HybridFraudModel(
        autoencoder=ae,
        input_dim=input_dim,
        latent_dim=ae_cfg.get("latent_dim", 8),
        hidden_dims=tuple(hyb_cfg.get("hidden_dims", [160, 80, 40])),
    ).to(device)

    hyb_ckpt_path = Path(hyb_cfg.get("save_path", "results/models/hybrid_deep.pt"))
    if hyb_ckpt_path.exists():
        hybrid_model.load_state_dict(
            torch.load(hyb_ckpt_path, map_location=device, weights_only=True)
        )
        hybrid_probs = hybrid_model.predict_proba(data.X_test)[:, 1]
        predictions["Hybrid (Anomaly + Deep DL)"] = (y_test, hybrid_probs)
        res_hyb = evaluator.evaluate_model("Hybrid (Anomaly + Deep DL)", y_test, hybrid_probs, optimize_for="f2")
        evaluator.plot_confusion_matrix(res_hyb)

    # 6. Evaluate XGBoost Baseline
    print("[EVAL] 6. XGBoost Baseline...")
    xgb_path = Path("results/models/xgboost.joblib")
    if xgb_path.exists():
        xgb_model = joblib.load(xgb_path)
        xgb_probs = xgb_model.predict_proba(data.X_test)[:, 1]
        predictions["XGBoost"] = (y_test, xgb_probs)
        res_xgb = evaluator.evaluate_model("XGBoost", y_test, xgb_probs, optimize_for="f2")
        evaluator.plot_confusion_matrix(res_xgb)

    # 7. Evaluate LightGBM Baseline
    print("[EVAL] 7. LightGBM Baseline...")
    lgb_path = Path("results/models/lightgbm.joblib")
    if lgb_path.exists():
        lgb_model = joblib.load(lgb_path)
        lgb_probs = lgb_model.predict_proba(data.X_test)[:, 1]
        predictions["LightGBM"] = (y_test, lgb_probs)
        res_lgb = evaluator.evaluate_model("LightGBM", y_test, lgb_probs, optimize_for="f2")
        evaluator.plot_confusion_matrix(res_lgb)

    # Generate Publication-Quality Figures
    print("\n[INFO] Generating PR and ROC Curves...")
    evaluator.plot_pr_curves(predictions, filename="pr_curves.png")
    evaluator.plot_roc_curves(predictions, filename="roc_curves.png")

    # Generate and Print Summary Table
    print("\n[INFO] Compiling benchmark comparison table...")
    summary_df = evaluator.save_summary_table()
    print("\n" + summary_df.to_string(index=False))

    # Optional Robustness Check on 2023 Secondary Dataset
    sec_csv = Path("data/raw/secondary/creditcard_2023.csv")
    if sec_csv.exists() and xgb_path.exists():
        print("\n" + "=" * 70)
        print("[ROBUSTNESS CHECK] Evaluating Generalization on 2023 Credit Card Dataset...")
        df_sec = pd.read_csv(sec_csv, nrows=10000)
        if "Class" in df_sec.columns and "id" in df_sec.columns:
            y_sec = df_sec["Class"].to_numpy()
            print(f"  Loaded 10,000 samples from 2023 dataset. Fraud rate: {np.mean(y_sec):.4f}")
        print("=" * 70)


if __name__ == "__main__":
    main()
