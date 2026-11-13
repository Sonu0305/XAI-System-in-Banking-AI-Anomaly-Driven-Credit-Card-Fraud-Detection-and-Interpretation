"""
Unit and integration test suite for the XAI Credit Card Fraud Detection system.
"""

import sys
from pathlib import Path
import numpy as np
import pytest
import torch

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessing import DataPipeline
from src.evaluation.metrics import compute_metrics, find_optimal_threshold
from src.models.autoencoder import DeepAutoencoder
from src.models.hybrid import HybridFraudModel
from src.models.supervised import DeepFraudClassifier, FocalLoss
from src.utils.device import get_amp_context, get_device, get_device_info, set_seed
from src.xai.analyst_report import BankingAnalystReporter


def test_device_utilities():
    set_seed(42)
    device = get_device("auto")
    assert isinstance(device, torch.device)
    info = get_device_info()
    assert "cuda_available" in info
    assert "is_dual_gpu" in info


def test_data_pipeline():
    pipeline = DataPipeline(
        raw_csv_path="data/raw/creditcard.csv",
        processed_dir="data/processed",
        test_size=0.15,
        val_size=0.15,
        random_state=42,
    )
    data = pipeline.load_processed()

    assert data.X_train.shape[0] > 0
    assert data.X_val.shape[0] > 0
    assert data.X_test.shape[0] > 0
    assert data.X_train_normal.shape[0] > 0
    assert data.X_train.shape[1] == len(data.feature_names)
    assert data.pos_weight > 100.0  # Demonstrates severe class imbalance


def test_autoencoder_architecture():
    batch_size = 16
    in_dim = 33
    latent_dim = 8
    model = DeepAutoencoder(input_dim=in_dim, latent_dim=latent_dim)
    dummy_x = torch.randn(batch_size, in_dim)

    recon, z = model(dummy_x)
    assert recon.shape == (batch_size, in_dim)
    assert z.shape == (batch_size, latent_dim)

    errors = model.compute_reconstruction_error(dummy_x, reduction="mean")
    assert errors.shape == (batch_size,)
    assert (errors >= 0).all()

    thresh = model.calibrate_threshold(dummy_x, percentile=95.0)
    assert thresh > 0


def test_supervised_classifier_and_focal_loss():
    batch_size = 16
    in_dim = 33
    model = DeepFraudClassifier(input_dim=in_dim, hidden_dims=[64, 32])
    dummy_x = torch.randn(batch_size, in_dim)
    dummy_y = torch.randint(0, 2, (batch_size, 1)).float()

    logits = model(dummy_x)
    assert logits.shape == (batch_size, 1)

    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    loss = criterion(logits, dummy_y)
    assert loss.item() >= 0
    assert not torch.isnan(loss)

    probs = model.predict_proba(dummy_x.numpy())
    assert probs.shape == (batch_size, 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_hybrid_model():
    batch_size = 16
    in_dim = 33
    latent_dim = 8
    ae = DeepAutoencoder(input_dim=in_dim, latent_dim=latent_dim)
    hybrid = HybridFraudModel(autoencoder=ae, input_dim=in_dim, latent_dim=latent_dim)
    dummy_x = torch.randn(batch_size, in_dim)

    logits = hybrid(dummy_x)
    assert logits.shape == (batch_size, 1)

    probs = hybrid.predict_proba(dummy_x.numpy())
    assert probs.shape == (batch_size, 2)


def test_evaluation_metrics():
    y_true = np.array([0, 0, 0, 0, 1, 1, 0, 1, 0, 0])
    y_scores = np.array([0.1, 0.2, 0.05, 0.15, 0.85, 0.9, 0.3, 0.75, 0.2, 0.1])

    best_thresh, score = find_optimal_threshold(y_true, y_scores, metric="f2")
    assert 0 < best_thresh < 1

    res = compute_metrics(y_true, y_scores, threshold=best_thresh, model_name="TestModel")
    assert 0 <= res.auprc <= 1
    assert 0 <= res.roc_auc <= 1
    assert 0 <= res.f1 <= 1
    assert 0 <= res.f2 <= 1
    assert res.total_cost >= 0


def test_banking_analyst_report():
    reporter = BankingAnalystReporter(output_dir="results/explanations")
    feature_names = ["Amount", "Hour_Sin", "V14", "V10"]
    feat_vals = np.array([125.0, 0.5, -4.5, -3.2])
    shap_vals = np.array([0.12, -0.05, 1.45, 0.88])

    card = reporter.generate_card(
        transaction_id="TXN-TEST-001",
        probability=0.88,
        feature_names=feature_names,
        feature_values=feat_vals,
        shap_values=shap_vals,
        recon_error=0.085,
        threshold=0.020,
    )

    assert "CRITICAL" in card.risk_tier
    assert card.is_anomaly is True
    assert len(card.top_red_flags) > 0

    md = reporter.render_markdown(card, filename="test_alert_card.md")
    assert "Bank Security Operations Alert Card" in md

