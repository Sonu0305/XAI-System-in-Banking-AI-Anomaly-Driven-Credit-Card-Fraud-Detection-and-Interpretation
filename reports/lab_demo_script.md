# 🎙️ Lab Examination Demonstration Script & Viva Walkthrough
## Project: XAI System in Banking: AI Anomaly-Driven Credit Card Fraud Detection and Interpretation

**Target Duration:** 8–10 Minutes  
**Audience:** Course Professors, External Evaluators, Lab Examiners  
**Materials Required:** Terminal, Browser/Jupyter Notebooks, PDF Project Report

---

## ⏱️ Timeline Overview

| Section | Topic | Duration | Key Visual / Artifact |
|:---|:---|:---:|:---|
| **Part 1** | Problem Statement & Architecture | 1.5 min | Architecture Diagram |
| **Part 2** | Data Pipeline & Zero-Leakage Splitting | 1.5 min | `data/processed/` & Class Ratio |
| **Part 3** | Dual-Stage Model Training & Loss Mechanics | 2.0 min | `scripts/train.py` & Focal Loss |
| **Part 4** | Benchmark Results & Metric Justification | 2.0 min | `results/tables/model_comparison.md` |
| **Part 5** | Live XAI & Bank Alert Card Demo | 2.0 min | `alert_card_fraud_sample.md` & SHAP |
| **Part 6** | Examiner Viva Q&A | 2.0 min | Technical Defense Guide |

---

## 🎬 Step-by-Step Presentation Script

### Part 1: Introduction & Architecture (0:00 - 1:30)

> **Speaker:**  
> *"Good morning, respected examiners. Today I am presenting our Deep Learning project: **XAI System in Banking: AI Anomaly-Driven Credit Card Fraud Detection and Interpretation**.*
>
> *Credit card fraud poses a dual dilemma in banking: on one hand, class imbalance is extreme ($0.17\%$ fraud prevalence), which causes standard neural networks to collapse towards predicting the majority class. On the other hand, strict financial regulations—such as GDPR Article 22 and FCRA Adverse Action notices—strictly forbid black-box decision making.*
>
> *Our solution is a four-tier architecture: First, an unsupervised Deep Autoencoder trained solely on normal transactions to learn the legitimate spending manifold; Second, a Supervised Deep Tabular ResNet trained with Focal Loss to prioritize hard fraud cases; Third, a Hybrid representation fusing manifold latents with raw features; and Fourth, a comprehensive Explainable AI layer that translates raw neural outputs into human-readable Bank Security Operations Alert Cards."*

**Action:** Open terminal and run:
```bash
python scripts/download_data.py
```

---

### Part 2: Data Pipeline & Zero-Leakage Stratification (1:30 - 3:00)

> **Speaker:**  
> *"We utilize the European Cardholder benchmark containing $284,807$ real transactions. Notice that there are only $492$ fraudulent cases—an imbalance ratio of approximately $578:1$.*
>
> *To maintain complete mathematical validity and zero data leakage:
> 1. All scalers (`RobustScaler` for transaction amounts and cyclical sine/cosine encoders for transaction time) are fitted strictly on the $70\%$ training partition.
> 2. The Autoencoder training set ($X_{\text{train\_normal}}$) contains strictly legitimate transactions ($199,020$ records).
> 3. Stratified splitting preserves the exact $0.1727\%$ fraud prevalence across train, validation, and test sets."*

**Action:** Show test execution:
```bash
pytest -v tests/
```
> *"Our test suite verifies that device acceleration, data shapes, focal loss, and reporting utilities pass with $100\%$ success."*

---

### Part 3: Dual-Stage Modeling & Focal Loss (3:00 - 5:00)

> **Speaker:**  
> *"In Stage 1, the Deep Autoencoder compresses 33 features down to an 8-dimensional bottleneck. When normal transactions pass through, reconstruction error is low. When an anomalous fraud transaction occurs, the error spikes. We calibrate our threshold at the $98^{\text{th}}$ percentile of normal validation errors.
>
> In Stage 2, for supervised classification, standard Cross-Entropy fails because easy negatives overwhelm the gradient. We implement Binary Focal Loss:
> $$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
> Setting $\gamma = 2.0$ dynamically suppresses loss from well-classified legitimate transactions, allowing the network to learn subtle fraud patterns."*

**Action:** Display model checkpoints:
```bash
ls -lh results/models/
```

---

### Part 4: Benchmark Comparison & Financial Evaluation (5:00 - 7:00)

> **Speaker:**  
> *"Let us examine the empirical benchmark across seven architectures on the held-out test set ($42,722$ transactions):"*

**Action:** Display table:
```bash
cat results/tables/model_comparison.md
```

> **Speaker:**  
> *"Notice key insights:
> 1. **Why AUPRC over ROC-AUC:** While ROC-AUC appears high across all models ($>0.91$) due to the massive true negative volume, **AUPRC** (Area Under the Precision-Recall Curve) correctly differentiates model quality. XGBoost achieves $0.8245$ AUPRC, while the Deep Tabular ResNet achieves $0.6576$.
> 2. **Financial Cost Metric:** In commercial banking, a False Positive costs $\sim\$10$ (analyst verification), but a False Negative costs $\sim\$500$ (chargeback fraud loss). Our threshold calibration minimizes expected banking loss down to $\$7,110$, compared to $\$26,100$ for Isolation Forest."*

**Action:** Open Precision-Recall and ROC curves in `results/figures/`:
```bash
ls -lh results/figures/pr_curves.png results/figures/roc_curves.png
```

---

### Part 5: Live XAI Demonstration & Banking Alert Cards (7:00 - 9:00)

> **Speaker:**  
> *"Now let us inspect the core highlight of our project: the Explainable AI (XAI) Suite."*

**Action:** Show generated XAI figures and open `results/explanations/alert_card_fraud_sample.md`:
```bash
cat results/explanations/alert_card_fraud_sample.md
```

> **Speaker:**  
> *"Here is a real transaction evaluated by our pipeline: Transaction `#TXN-000735`:
> - **Supervised Probability:** $99.28\%$
> - **Autoencoder Anomaly Score:** $3.861$ MSE (exceeds normal threshold of $2.605$)
> - **SHAP Root Cause Analysis:** Feature `V14 = -4.998` contributed $+5.72$ log-odds towards fraud, supported by `V10` and `V16`.
> - **Actionable Banking Protocol:** The system outputs an immediate instruction: *'Block transaction, freeze card, and prompt 2FA verification.'*
> - **Regulatory Compliance:** It logs an immutable reason-code audit trail compliant with GDPR Article 22 and FCRA."*

---

## 🧠 Examiner Viva Q&A Cheat Sheet

### Q1: Why use an Autoencoder if XGBoost has a higher AUPRC?
**Answer:**  
*"Supervised models like XGBoost only recognize historical, previously labeled fraud patterns. If a fraud syndicate introduces a novel zero-day exploit, supervised trees often fail. The Autoencoder learns the boundary of legitimate transactions; any deviation—known or unknown—triggers an elevated reconstruction error. In production, the Autoencoder acts as a frontline anomaly filter and enriches the supervised model with manifold coordinates."*

### Q2: Why did you use Focal Loss instead of SMOTE?
**Answer:**  
*"SMOTE synthesizes artificial minority examples through k-nearest neighbors in input space. In tabular financial data, linear interpolation between distant fraud vectors often creates synthetic points in legitimate transaction space, degrading precision. Focal Loss operates directly on the loss landscape by dynamically scaling loss based on $(1-p_t)^\gamma$, penalizing hard-to-classify real instances without distorting the underlying data distribution."*

### Q3: How do you guarantee zero data leakage?
**Answer:**  
*"All transformations (RobustScaler, log1p, cyclical encoding) are fitted strictly on the $70\%$ training partition. The validation and test sets are purely transformed using the training statistics. Furthermore, the Autoencoder is trained strictly on class 0 transactions from the training set, ensuring no test or validation data ever influences parameter estimation."*

### Q4: How is this system optimized for Kaggle GPU T4 x2?
**Answer:**  
*"Our `device.py` detects multiple GPUs via `torch.cuda.device_count() >= 2` and wraps models in `nn.DataParallel`. We utilize PyTorch Automatic Mixed Precision (`torch.amp.autocast('cuda')`), DataLoader pinned memory (`pin_memory=True`), and asynchronous worker prefetching (`num_workers=4`), doubling throughput and keeping VRAM utilization optimized."*

