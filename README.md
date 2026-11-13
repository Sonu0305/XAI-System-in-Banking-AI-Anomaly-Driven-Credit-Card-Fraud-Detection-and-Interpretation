# 🛡️ XAI System in Banking: AI Anomaly-Driven Credit Card Fraud Detection and Interpretation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x%20Ready-EE4C2C.svg)](https://pytorch.org/)
[![Kaggle GPU T4 x2](https://img.shields.io/badge/Kaggle-Dual%20T4%20Accelerated-20BEFF.svg)](https://www.kaggle.com/)
[![XAI Enabled](https://img.shields.io/badge/XAI-SHAP%20%2B%20LIME-green.svg)](https://github.com/slundberg/shap)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **End-to-End Deep Learning & Explainable AI (XAI) System for Banking Security Operations.**  
> Combines unsupervised manifold anomaly detection, supervised tabular deep learning with Focal Loss, game-theoretic SHAP/LIME interpretations, and automated regulatory-compliant Bank Security Operations Alert Cards.

---

## 📌 Executive Summary

Credit card fraud inflicts billions of dollars in losses annually. Financial institutions face a dual challenge: **extreme class imbalance** ($0.17\%$ fraud rate) and **strict regulatory compliance** (GDPR Article 22, FCRA Adverse Action notices) prohibiting black-box decision systems.

This repository provides an end-to-end, GPU-accelerated solution:
1. **Unsupervised Anomaly Detection:** Deep Autoencoder trained exclusively on legitimate customer transactions to catch zero-day / novel fraud patterns via reconstruction error.
2. **Supervised Tabular Deep Learning:** Deep Tabular ResNet with residual skip connections and **Binary Focal Loss** ($\gamma=2.0, \alpha=0.25$) to conquer $578:1$ class imbalance.
3. **Hybrid Feature Fusion:** Augments raw transaction attributes with Autoencoder bottleneck representations ($z \in \mathbb{R}^8$) and reconstruction residuals ($e \in \mathbb{R}^1$).
4. **Explainable AI (XAI) Engine:** Global (Beeswarm, Summary, Partial Dependence) and Local (Waterfall, LIME Surrogates) interpretations translating neural decisions into plain-English **Bank Security Operations Alert Cards**.
5. **Cost-Sensitive Threshold Optimization:** Calibrates operational decision thresholds considering asymmetric banking costs ($\text{False Positive Review}=\$10$, $\text{False Negative Fraud Loss}=\$500$).

---

## 🏗️ System Architecture

```text
[ Incoming Credit Card Transaction (x in R^33) ]
                       │
       ┌───────────────┴─────────────────────────┐
       ▼                                         ▼
[ Deep Autoencoder ]                     [ Feature Pipeline ]
 ├─ Encoder: z in R^8                     ├─ RobustScaler (Amount)
 ├─ Decoder: x_hat in R^33                ├─ Cyclical Time (Hour_Sin, Hour_Cos)
 └─ Reconstruction MSE: e in R^1          └─ Zero-Leakage Stratified Split
       │                                         │
       └───────────────────┬─────────────────────┘
                           ▼
            [ Anomaly-Augmented Vector ]
                [ x, z, e ] in R^42
                           │
                           ▼
            [ Supervised Tabular ResNet ]
             (Trained with Focal Loss)
                           │
                           ▼
          [ Calibrated Probability P(Fraud|x) ]
                           │
                           ▼
            [ Explainable AI (XAI) Suite ]
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
[ SHAP Attributions ]   [ LIME Surrogates ]   [ Bank Alert Card ]
 - Summary Bar           - Local Linear Rule   - Risk Tier & Score
 - Beeswarm Forces       - Interactive HTML    - Top Red Flags
 - Waterfall Plot        - Feature Weights     - Recommended Action
```

---

## 📊 Benchmark Performance Comparison

Evaluated on the held-out test partition ($42,722$ transactions, $74$ positive fraud cases):

| Model Architecture | AUPRC (PR-AUC) | ROC-AUC | $F_1$-Score | $F_2$-Score | Precision | Recall | False Positive Rate | Expected Loss (\$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **XGBoost (Tuned)** | **0.8245** | **0.9718** | **0.8276** | **0.8174** | **84.51%** | 81.08% | **0.0258%** | **\$7,110.00** |
| **Supervised Deep Tabular** | 0.6576 | 0.9165 | 0.7436 | 0.7672 | 70.73% | 78.38% | 0.0563% | \$8,240.00 |
| **Logistic Regression (Balanced)** | 0.6463 | 0.9639 | 0.6425 | 0.7470 | 52.10% | **83.78%** | 0.1337% | \$6,570.00 |
| **Hybrid (Anomaly + Deep DL)** | 0.5949 | 0.9213 | 0.7226 | 0.7427 | 69.14% | 75.68% | 0.0586% | \$9,250.00 |
| **LightGBM Baseline** | 0.4551 | 0.9314 | 0.5438 | 0.6720 | 41.26% | 79.73% | 0.1970% | \$8,340.00 |
| **Autoencoder (Alone)** | 0.1911 | 0.9493 | 0.2378 | 0.3917 | 14.37% | 68.92% | 0.7128% | \$14,540.00 |
| **Isolation Forest Baseline** | 0.1033 | 0.9407 | 0.2308 | 0.2791 | 17.91% | 32.43% | 0.2579% | \$26,100.00 |

*Expected Loss based on commercial banking cost parameters: $\$10$ per False Alarm (FP investigation), $\$500$ per Undetected Fraud (FN loss).*

---

## 🔍 Explainable AI (XAI) & Banking Alert Cards

### 1. Global Attributions (SHAP Beeswarm & Summary)
- Primary fraud indicators identified: `V14`, `V10`, `V4`, `V12`, `V17`.
- Sharp non-linear transitions: values of `V14 < -3.5` drastically escalate fraud probability.
- Generated figures: `results/explanations/shap_beeswarm.png`, `shap_summary_bar.png`, and `shap_dependence_*.png`.

### 2. Local Rationale (SHAP Waterfall & LIME)
- For any transaction, the exact forces pushing the score higher or lower are decomposed into marginal contributions.
- Local decision rules extracted via LIME: `results/explanations/lime_sample_*.png` and `.html`.

### 3. Bank Security Operations Alert Card
Every alert generates an actionable compliance document:

```markdown
# 🔴 Bank Security Operations Alert Card: Transaction #TXN-000735-FRAUD

### Executive Summary
- Fraud Risk Verdict: CRITICAL RISK
- Estimated Fraud Probability: 99.28%
- Unsupervised Anomaly State: ⚠️ ANOMALOUS (Reconstruction MSE: 3.861 vs Threshold: 2.605)

### Root Cause Analysis (XAI Feature Attribution)
🚨 Primary Red Flags:
1. V14 (Normalized: -4.998) -> SHAP Impact: +5.7184
2. V10 (Normalized: -4.102) -> SHAP Impact: +1.1060
3. V16 (Normalized: -2.991) -> SHAP Impact: +0.9514
4. V17 (Normalized: -5.216) -> SHAP Impact: +0.9385

### Recommended Banking Protocol
> IMMEDIATE ACTION REQUIRED: Block transaction immediately, place temporary hold on 
> credit card account, and dispatch instant SMS/App 2-factor identity verification prompt.
```

---

## 📁 Repository Structure

```text
xai-fraud-detection/
├── .env.example              # Kaggle credential configuration template
├── README.md                 # Complete documentation with benchmark results
├── requirements.txt          # Python dependencies
├── setup.sh                  # One-click automated setup & run script
├── Makefile                  # Build targets (make all, make test, make train)
├── pyproject.toml            # Package configuration
├── config/
│   └── config.yaml           # Model hyperparameters, loss weights, evaluation costs
├── data/
│   ├── raw/                  # Downloaded creditcard.csv (284,807 rows)
│   └── processed/            # Stratified train/val/test splits & fitted scalers
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory Data Analysis & Imbalance Characterization
│   ├── 02_anomaly_detection.ipynb  # Autoencoder training & reconstruction error
│   ├── 03_supervised_training.ipynb# Supervised Tabular ResNet & Focal Loss
│   ├── 04_xai_explanations.ipynb   # SHAP & LIME global and local explanations
│   └── 05_end_to_end_demo.ipynb    # Real-time transaction simulation & alert cards
├── src/
│   ├── data/                 # Preprocessing, robust scaling, PyTorch datasets
│   ├── models/               # Autoencoder, DeepFraudClassifier, Hybrid, Baselines
│   ├── training/             # AMP-enabled trainers with early stopping on AUPRC
│   ├── xai/                  # SHAP & LIME explainers and banking alert reporter
│   ├── evaluation/           # Metrics suite (AUPRC, F1, F2, MCC, Cost) & plotter
│   └── utils/                # Device manager (T4 x2 DataParallel), credentials, seed
├── scripts/
│   ├── download_data.py      # Kaggle API authenticated dataset downloader
│   ├── train.py              # End-to-end training orchestrator
│   ├── evaluate.py           # Benchmark evaluator and curve generator
│   └── generate_report.py    # Generates SHAP/LIME figures & alert cards
├── results/
│   ├── models/               # Trained PyTorch weights (.pt) and tree models (.joblib)
│   ├── figures/              # PR curves, ROC curves, confusion matrices
│   ├── tables/               # Model comparison CSV, JSON, and Markdown
│   └── explanations/         # SHAP plots, LIME HTMLs, and Banking Alert Cards
├── reports/
│   ├── project_report.pdf    # Academic conference-style project report
│   ├── project_report.md     # Full markdown source of the report
│   ├── presentation.md       # 10-slide Marp presentation deck
│   └── lab_demo_script.md    # 8-minute viva demo script & technical Q&A cheat sheet
└── tests/
    └── test_pipeline.py      # Pytest unit tests (100% pass)
```

---

## ⚡ Quickstart: How to Run

### Option A: Complete One-Click Execution (Recommended)
```bash
# 1. Clone repository
git clone https://github.com/swayam-shah/XAI-System-in-Banking-AI-Anomaly-Driven-Credit-Card-Fraud-Detection-and-Interpretation.git
cd XAI-System-in-Banking-AI-Anomaly-Driven-Credit-Card-Fraud-Detection-and-Interpretation

# 2. Configure Kaggle Credentials in .env
cp .env.example .env
# Edit .env with your Kaggle token path (~/.kaggle/access_token or credentials)

# 3. Run automated setup (creates venv, downloads data, tests, trains, evaluates, generates XAI)
bash setup.sh
```

### Option B: Step-by-Step Execution via Makefile
```bash
# Install dependencies
make install

# Download Kaggle dataset
make download

# Run unit tests
make test

# Train all models (Autoencoder, Supervised, Hybrid, Baselines)
make train

# Evaluate models and generate PR/ROC curves
make evaluate

# Generate XAI plots and Banking Alert Cards
make report

# Compile Project Report to PDF
make pdf
```

---

## 🚀 Kaggle GPU T4 x2 Deployment Guide

To run on a Kaggle Notebook with dual NVIDIA T4 GPUs:
1. Select **Accelerator: GPU T4 x2** in the Kaggle notebook settings panel.
2. The codebase automatically detects dual GPUs via `src/utils/device.py`:
   - Wraps PyTorch models in `torch.nn.DataParallel`.
   - Uses PyTorch Automatic Mixed Precision (`torch.amp.autocast('cuda')`).
   - Enables pinned memory and asynchronous DataLoader workers (`num_workers=4`).
3. Add your Kaggle API key in Kaggle Secrets or load directly via `.env`.

---

## 🧪 Unit Testing

Run the test suite with pytest:
```bash
pytest -v tests/
```
All 7 unit tests cover hardware acceleration, data pipelines, zero leakage, loss functions, hybrid inference, metrics calculation, and alert card generation.

---

## 📜 Deliverables Summary

- **Project Report (PDF & Markdown):** [`reports/project_report.pdf`](reports/project_report.pdf) & [`reports/project_report.md`](reports/project_report.md)
- **Presentation Deck:** [`reports/presentation.md`](reports/presentation.md)
- **Lab Walkthrough Script:** [`reports/lab_demo_script.md`](reports/lab_demo_script.md)
- **Benchmark Summary Table:** [`results/tables/model_comparison.md`](results/tables/model_comparison.md)
- **Sample Bank Alert Cards:** [`results/explanations/alert_card_fraud_sample.md`](results/explanations/alert_card_fraud_sample.md)

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
