"""
XAI Explanations & Banking Interpretation Report Generator.
Computes:
1. SHAP Global Feature Attributions (Bar, Beeswarm, Dependence)
2. SHAP Local Waterfall & Force Explanations
3. LIME Local Tabular Explanations (PNG & HTML)
4. Bank Analyst Fraud Alert Cards (Markdown & Audit Trail)
"""

import argparse
import sys
from pathlib import Path
import joblib
import numpy as np
import torch

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessing import DataPipeline
from src.models.autoencoder import DeepAutoencoder
from src.utils.config import load_config
from src.utils.device import get_device
from src.xai.analyst_report import BankingAnalystReporter
from src.xai.lime_explainer import FraudLIMEExplainer
from src.xai.shap_explainer import FraudSHAPExplainer


def main():
    parser = argparse.ArgumentParser(description="Generate XAI Explanations and Reports")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["project"].get("device", "auto"))

    print("=" * 70)
    print("GENERATING EXPLAINABLE AI (XAI) ARTIFACTS FOR BANKING ANALYSTS")
    print("=" * 70)

    # 1. Load Data
    pipeline = DataPipeline(
        raw_csv_path=cfg["data"].get("raw_file", "data/raw/creditcard.csv"),
        processed_dir=cfg["data"].get("processed_dir", "data/processed"),
    )
    data = pipeline.load_processed()
    feature_names = data.feature_names
    input_dim = data.X_test.shape[1]

    # Find sample indices: confirmed fraud and legitimate
    fraud_indices = np.where(data.y_test == 1)[0]
    normal_indices = np.where(data.y_test == 0)[0]
    print(f"[INFO] Test set contains {len(fraud_indices)} fraud cases and {len(normal_indices)} normal cases.")

    sample_fraud_idx = fraud_indices[0] if len(fraud_indices) > 0 else 0
    sample_normal_idx = normal_indices[0] if len(normal_indices) > 0 else 1

    # 2. Load Models
    xgb_path = Path("results/models/xgboost.joblib")
    if not xgb_path.exists():
        raise FileNotFoundError("Trained XGBoost model missing at results/models/xgboost.joblib. Please run scripts/train.py first.")
    xgb_model = joblib.load(xgb_path)

    ae_cfg = cfg["anomaly_model"]
    autoencoder = DeepAutoencoder(
        input_dim=input_dim,
        latent_dim=ae_cfg.get("latent_dim", 8),
        encoder_dims=tuple(ae_cfg.get("encoder_dims", [64, 32, 16])),
    ).to(device)

    ae_ckpt_path = Path(ae_cfg.get("save_path", "results/models/autoencoder.pt"))
    threshold = 0.015
    if ae_ckpt_path.exists():
        ckpt = torch.load(ae_ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        threshold = ckpt.get("threshold", 0.015)
        autoencoder.load_state_dict(state_dict)

    # 3. SHAP Explanations
    print("\n[STEP 1/3] Computing SHAP Global & Local Attributions...")
    shap_explainer = FraudSHAPExplainer(
        model=xgb_model,
        feature_names=feature_names,
        background_data=data.X_train[:150],
        model_type="tree",
        output_dir="results/explanations",
    )

    # Use a representative subset of test data for fast, reliable global plots
    eval_subset = data.X_test[:250]
    shap_vals = shap_explainer.explain(eval_subset)

    # Global summary and beeswarm
    shap_explainer.plot_summary(eval_subset, shap_values=shap_vals, max_display=15, filename="shap_summary_bar.png")
    shap_explainer.plot_beeswarm(eval_subset, shap_values=shap_vals, max_display=15, filename="shap_beeswarm.png")

    # Dependence plots for top critical features
    for feat in ["V14", "V10", "V4"]:
        if feat in feature_names:
            shap_explainer.plot_dependence(feat, eval_subset, shap_values=shap_vals)

    # Local waterfall explanation for the fraud sample
    fraud_sample_x = data.X_test[sample_fraud_idx]
    normal_sample_x = data.X_test[sample_normal_idx]
    shap_explainer.plot_waterfall(fraud_sample_x, sample_idx=sample_fraud_idx, filename="shap_waterfall_fraud.png")
    shap_explainer.plot_waterfall(normal_sample_x, sample_idx=sample_normal_idx, filename="shap_waterfall_legitimate.png")

    # 4. LIME Explanations
    print("\n[STEP 2/3] Generating LIME Local Surrogate Explanations...")
    lime_explainer = FraudLIMEExplainer(
        training_data=data.X_train[:500],
        feature_names=feature_names,
        output_dir="results/explanations",
        random_state=42,
    )

    def predict_fn(x):
        return xgb_model.predict_proba(x)

    lime_explainer.explain_instance(
        instance=fraud_sample_x,
        predict_fn=predict_fn,
        num_features=10,
        sample_idx=sample_fraud_idx,
        save_plot=True,
        save_html=True,
    )
    lime_explainer.explain_instance(
        instance=normal_sample_x,
        predict_fn=predict_fn,
        num_features=10,
        sample_idx=sample_normal_idx,
        save_plot=True,
        save_html=True,
    )

    # 5. Banking Analyst Fraud Alert Cards
    print("\n[STEP 3/3] Generating Banking Security Operations Alert Cards...")
    reporter = BankingAnalystReporter(output_dir="results/explanations")

    # Compute Autoencoder reconstruction errors
    test_tensor = torch.as_tensor(data.X_test, dtype=torch.float32, device=device)
    recon_errors = autoencoder.compute_reconstruction_error(test_tensor, reduction="mean").cpu().numpy()

    # Fraud sample card
    fraud_prob = float(xgb_model.predict_proba(fraud_sample_x.reshape(1, -1))[0, 1])
    fraud_shap_single = shap_explainer.explainer(fraud_sample_x.reshape(1, -1))[0]
    fraud_shap_vals = fraud_shap_single.values if len(fraud_shap_single.values.shape) == 1 else fraud_shap_single.values[:, 1]

    card_fraud = reporter.generate_card(
        transaction_id=f"TXN-{sample_fraud_idx:06d}-FRAUD",
        probability=fraud_prob,
        feature_names=feature_names,
        feature_values=fraud_sample_x,
        shap_values=fraud_shap_vals,
        recon_error=float(recon_errors[sample_fraud_idx]),
        threshold=threshold,
    )
    reporter.render_markdown(card_fraud, filename="alert_card_fraud_sample.md")

    # Legitimate sample card
    normal_prob = float(xgb_model.predict_proba(normal_sample_x.reshape(1, -1))[0, 1])
    normal_shap_single = shap_explainer.explainer(normal_sample_x.reshape(1, -1))[0]
    normal_shap_vals = normal_shap_single.values if len(normal_shap_single.values.shape) == 1 else normal_shap_single.values[:, 1]

    card_normal = reporter.generate_card(
        transaction_id=f"TXN-{sample_normal_idx:06d}-NORMAL",
        probability=normal_prob,
        feature_names=feature_names,
        feature_values=normal_sample_x,
        shap_values=normal_shap_vals,
        recon_error=float(recon_errors[sample_normal_idx]),
        threshold=threshold,
    )
    reporter.render_markdown(card_normal, filename="alert_card_legitimate_sample.md")

    print("\n" + "=" * 70)
    print("[SUCCESS] All XAI plots, HTML explanations, and Alert Cards saved in results/explanations/!")
    print("=" * 70)


if __name__ == "__main__":
    main()
