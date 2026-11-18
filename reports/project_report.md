---
title: "XAI System in Banking: AI Anomaly-Driven Credit Card Fraud Detection and Interpretation"
subtitle: "Deep Learning Lab Examination Project & Comprehensive Technical Report"
author: "Lead Deep Learning / XAI Engineer"
date: "November 2026"
geometry: "margin=1in"
fontsize: 11pt
documentclass: article
header-includes:
  - \input{reports/preamble.tex}
---

\begin{abstractbox}
\textbf{\large Executive Abstract}\vspace{5pt}

Credit card fraud represents an escalating multi-billion-dollar global challenge characterized by extreme class imbalance, shifting transaction patterns, and strict regulatory demands for algorithmic transparency. In this work, we propose and validate an end-to-end Explainable AI (XAI) banking fraud detection system. Our architecture combines: \textbf{(1)} an unsupervised Deep Autoencoder trained exclusively on legitimate banking transactions to model normal manifold dynamics and compute continuous reconstruction error anomaly scores; \textbf{(2)} a supervised Tabular Deep ResNet classifier optimized with Binary Focal Loss to counteract extreme class imbalance ($\sim 578:1$); \textbf{(3)} a hybrid representation model fusing raw features with latent bottleneck embeddings and reconstruction error residuals; and \textbf{(4)} an interpretability layer uniting game-theoretic SHAP (Shapley Additive exPlanations) and local surrogate LIME (Local Interpretable Model-agnostic Explanations). Evaluated on 284,807 European cardholder transactions, our system achieves competitive discriminative performance (Hybrid AUPRC \textbf{0.8550}, Supervised Deep Tabular ROC-AUC \textbf{0.9789}, XGBoost AUPRC 0.8245, False Positive Rate $<0.03\%$) and reduces expected financial investigation and fraud losses down to \$6,600. Furthermore, the system automatically compiles audit-compliant, human-readable Bank Security Operations Alert Cards satisfying GDPR Article 22 and FCRA Adverse Action requirements.

\vspace{6pt}
\noindent\textbf{Keywords:} Explainable AI (XAI), Credit Card Fraud Detection, Anomaly Detection, Deep Autoencoder, Tabular ResNet, Focal Loss, SHAP, LIME, Cost-Sensitive Learning, Banking Security Operations.
\end{abstractbox}

\newpage

# 1. Introduction & Banking Fraud Landscape

Modern financial transactions occur in milliseconds across global payment networks. While automated payment processing enables seamless electronic commerce, it introduces vulnerabilities exploited by organized fraud syndicates. Fraud typologies encompass card-not-present (CNP) exploits, credential stuffing, identity synthesis, and algorithmic micro-probing.

Designing machine learning systems for banking fraud faces three foundational obstacles:

1. **Extreme Class Imbalance:** In real-world payment networks, fraudulent transactions account for fewer than $0.2\%$ of total volume (in our benchmark, $492$ frauds out of $284,807$ total records, or $0.1727\%$). Standard empirical risk minimization collapses toward the majority class, predicting zero fraud with misleadingly high 99.8\% accuracy.
2. **Concept Drift & Zero-Day Fraud:** Fraud syndicates continually mutate behaviors and tactics. Purely supervised classifiers trained solely on historical labeled fraud fail to detect novel attack mechanisms and zero-day intrusion patterns.
3. **The "Black-Box" Interpretability Deficit:** Strict banking regulations (such as European GDPR Article 22 and the US Fair Credit Reporting Act) mandate that automated financial decisions affecting cardholders must be explainable. An alert that flags a transaction without actionable rationale cannot be acted upon effectively by fraud operations analysts and risks customer attrition.

To resolve these interconnected challenges, this project introduces a unified, GPU-accelerated pipeline integrating unsupervised anomaly detection, supervised deep representation learning, cost-sensitive threshold calibration, and multi-tier explainability.

# 2. Related Work

### 2.1 Anomaly Detection in Financial Streams
Unsupervised anomaly detection aims to identify instances that deviate significantly from majority patterns without requiring positive class labels. Methods including Isolation Forests, Local Outlier Factors (LOF), and One-Class SVMs have been applied to tabular fraud data. Deep Autoencoders advance this paradigm by learning non-linear manifold projections, capturing subtle multi-attribute correlations across legitimate spending profiles.

### 2.2 Deep Learning for Tabular Imbalance & Focal Loss
While Tree-based Ensembles (XGBoost, LightGBM, CatBoost) historically dominate tabular benchmarks, modern deep tabular architectures---such as TabNet and Tabular ResNets with residual skip connections---offer differentiable feature integration and seamless fusion of manifold embeddings. To counteract severe label skewness, Lin et al. (2017) introduced **Focal Loss**, dynamically down-weighting the loss assigned to easily classified negative instances and concentrating gradients on hard, borderline fraud examples.

### 2.3 Explainable AI (XAI) in Regulated Finance
XAI techniques bridge complex neural representations and regulatory auditability. Lundberg & Lee (2017) introduced **SHAP**, framing feature importance through cooperative game theory (Shapley values) to guarantee local accuracy, missingness, and consistency. Ribeiro et al. (2016) developed **LIME**, which fits locally faithful linear surrogate models around individual query instances to generate interpretable decision rules.

# 3. Methodology & System Architecture

The architecture comprises four coordinated modules depicted in Figure \ref{fig:architecture}:

\begin{figure}[htbp]
\centering
\resizebox{0.96\linewidth}{!}{
\begin{tikzpicture}[
    node distance=0.85cm and 1.2cm,
    box/.style={rectangle, draw=SteelBlue!80!black, fill=SteelBlue!8, rounded corners=3pt, thick, align=center, inner sep=6pt, font=\small},
    databox/.style={rectangle, draw=Navy!80!black, fill=Navy!6, rounded corners=3pt, thick, align=center, inner sep=6pt, font=\small},
    modelbox/.style={rectangle, draw=TealAccent!80!black, fill=TealAccent!8, rounded corners=3pt, thick, align=center, inner sep=6pt, font=\small},
    xaibox/.style={rectangle, draw=AlertRed!80!black, fill=AlertRed!8, rounded corners=3pt, thick, align=center, inner sep=6pt, font=\small},
    arrow/.style={-{Latex[length=2.5mm]}, thick, draw=Navy!80!black}
]
\node[databox] (input) {\textbf{Incoming Financial Transaction} \\ $x \in \mathbb{R}^{33}$ (Amount, Cyclical Time, PCA Features $V_1\dots V_{28}$)};
\node[modelbox, below left=0.85cm and 0.2cm of input] (ae) {\textbf{Stage 1: Deep Autoencoder} \\ (Trained strictly on Normal Data) \\ Bottleneck: $z \in \mathbb{R}^8$ \quad MSE Residual: $e \in \mathbb{R}^1$};
\node[box, below right=0.85cm and 0.2cm of input] (pipe) {\textbf{Feature Transformation Pipeline} \\ RobustScaler (Amounts, Log1p) \\ Cyclical Sine/Cosine Hour Encoding};
\node[databox, below=2.65cm of input] (fused) {\textbf{Anomaly-Augmented Vector} \\ $x_{\text{augmented}} = [x \,\|\, z \,\|\, e] \in \mathbb{R}^{42}$};
\node[modelbox, below=0.75cm of fused] (resnet) {\textbf{Stage 2: Supervised Tabular ResNet} \\ Residual Skip Connections, LayerNorm, Dropout ($0.25$) \\ Optimized via Binary Focal Loss ($\gamma=2.0, \alpha=0.25$)};
\node[databox, below=0.75cm of resnet] (prob) {\textbf{Calibrated Fraud Probability} \\ $P(\text{Fraud}|x) \in [0, 1]$ \& Cost-Optimal Threshold $\tau^*$ Calibration};
\node[xaibox, below=0.75cm of prob] (xai) {\textbf{Stage 3: Explainable AI (XAI) Suite} \\ Global/Local SHAP Shapley Attributions $\bullet$ LIME Decision Surrogates \\ Automated Bank Security Operations Alert Card};

\draw[arrow] (input) -| (ae);
\draw[arrow] (input) -| (pipe);
\draw[arrow] (ae) |- (fused);
\draw[arrow] (pipe) |- (fused);
\draw[arrow] (fused) -- (resnet);
\draw[arrow] (resnet) -- (prob);
\draw[arrow] (prob) -- (xai);
\end{tikzpicture}
}
\caption{End-to-End System Architecture for Anomaly-Driven Credit Card Fraud Detection and Explainability.}
\label{fig:architecture}
\end{figure}

### 3.1 Data Pipeline & Stratified Splitting

To eliminate data leakage, all feature scalers are fitted strictly on training observations:

- **Transaction Amount:** Normalized via `RobustScaler` (median-centered and interquartile range scaled) combined with $\log(1 + x)$ transformation to handle high positive skewness and extreme tail spenders.
- **Transaction Time:** Transformed into cyclical trigonometric coordinates to model 24-hour diurnal spending rhythmicity:
  $$\text{Hour} = \left(\frac{\text{Time}}{3600}\right) \pmod{24}$$
  $$x_{\sin} = \sin\left(\frac{2\pi \cdot \text{Hour}}{24}\right), \quad x_{\cos} = \cos\left(\frac{2\pi \cdot \text{Hour}}{24}\right)$$
- **Splitting Strategy:** Stratified $70\%$ train ($199,364$ records, $344$ frauds), $15\%$ validation ($42,721$ records, $74$ frauds), and $15\%$ test ($42,722$ records, $74$ frauds).

### 3.2 Unsupervised Anomaly Detection: Deep Autoencoder

The Autoencoder comprises a symmetrical bottleneck architecture with encoder $f_\theta: \mathbb{R}^{33} \to \mathbb{R}^8$ and decoder $g_\phi: \mathbb{R}^8 \to \mathbb{R}^{33}$:
$$z = f_\theta(x) = \text{Linear}_{8}(\text{LReLU}(\text{BN}(\text{Linear}_{16}(\dots)))), \quad \hat{x} = g_\phi(z)$$

The model is trained **strictly on normal transactions** ($X_{\text{train\_normal}}$) using Mean Squared Error:
$$\mathcal{L}_{\text{AE}}(\theta, \phi) = \frac{1}{|X_{\text{train\_normal}}|} \sum_{x \in X_{\text{train\_normal}}} \|x - g_\phi(f_\theta(x))\|_2^2$$

The anomaly score is defined as the continuous reconstruction error MSE:
$$\text{Score}_{\text{anomaly}}(x) = \frac{1}{D} \sum_{d=1}^D (x_d - \hat{x}_d)^2$$

The baseline decision boundary $\tau_{\text{AE}}$ is calibrated at the $98^{\text{th}}$ percentile of validation normal reconstruction errors.

### 3.3 Supervised Tabular ResNet & Focal Loss

The supervised network incorporates tabular residual blocks with batch normalization, GeLU non-linearities, and dropout ($0.25$). To combat the $578:1$ class imbalance, we employ Binary Focal Loss:
$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
where $p_t = p$ if $y=1$ else $1-p$. The focusing parameter $\gamma = 2.0$ dynamically suppresses loss from well-classified negatives, while $\alpha = 0.25$ balances precision and recall trade-offs.

### 3.4 Hybrid Fusion Architecture

The hybrid model concatenates the original feature vector $x \in \mathbb{R}^{33}$, the latent bottleneck representation $z \in \mathbb{R}^8$, and the scalar reconstruction error $e \in \mathbb{R}^1$:
$$x_{\text{augmented}} = [x \,\|\, z \,\|\, e] \in \mathbb{R}^{42}$$
This augmented representation is processed by a secondary deep classification network, providing structural manifold context alongside raw transaction attributes.

### 3.5 Cost-Sensitive Evaluation & Threshold Calibration

In commercial banking, classification errors carry asymmetric financial implications:
$$\text{Total Financial Cost} = \text{FP} \times C_{\text{FP}} + \text{FN} \times C_{\text{FN}}$$
Where $C_{\text{FP}} = \$10$ represents fraud operations analyst verification overhead, and $C_{\text{FN}} = \$500$ represents average direct chargeback fraud loss. The operational threshold $\tau^*$ is calibrated to minimize this monetary loss or maximize the $F_2$-score (weighting recall twice as heavily as precision).

\clearpage

# 4. Experimental Results & Comparative Evaluation

All models were evaluated on the held-out test partition ($42,722$ transactions, $74$ positive fraud cases).

### 4.1 Benchmark Performance Table

Table \ref{tab:benchmark_results} summarizes empirical performance across seven architectures evaluated on the held-out test partition.

\begin{table}[htbp]
\centering
\caption{Empirical Benchmark Performance Comparison on Held-Out Test Set (42,722 Transactions, 74 Fraud Cases). Evaluated at cost-optimal operational thresholds ($C_{\text{FP}} = \$10$, $C_{\text{FN}} = \$500$).}
\label{tab:benchmark_results}
\vspace{3pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l ccccccccc}
\toprule
\textbf{Model Architecture} & \textbf{AUPRC} & \textbf{ROC-AUC} & $\mathbf{F_1}$\textbf{-Score} & $\mathbf{F_2}$\textbf{-Score} & \textbf{Precision} & \textbf{Recall} & \textbf{MCC} & \textbf{FPR} & \textbf{Loss (\$)} \\
\midrule
\rowcolor{HighlightRow}
\textbf{Hybrid (Anomaly + Deep DL)} & \textbf{0.8550} & 0.9609 & \textbf{0.8414} & \textbf{0.8311} & \textbf{0.8592} & 0.8243 & \textbf{0.8413} & 0.000234 & \$6,600 \\
\textbf{Supervised Deep Tabular}   & 0.8321 & \textbf{0.9789} & 0.8299 & 0.8266 & 0.8356 & 0.8243 & 0.8297 & 0.000281 & \$6,620 \\
\textbf{XGBoost (Tuned)}           & 0.8245 & 0.9718 & 0.8276 & 0.8174 & 0.8451 & 0.8108 & 0.8275 & \textbf{0.000258} & \$7,110 \\
\textbf{Logistic Regression (Bal)} & 0.6463 & 0.9639 & 0.6425 & 0.7470 & 0.5210 & \textbf{0.8378} & 0.6600 & 0.001337 & \textbf{\$6,570} \\
\textbf{Autoencoder (Alone)}       & 0.4883 & 0.9525 & 0.4926 & 0.5882 & 0.3876 & 0.6757 & 0.5107 & 0.001852 & \$12,790 \\
\textbf{LightGBM Baseline}         & 0.1384 & 0.9076 & 0.2676 & 0.4449 & 0.1608 & 0.7973 & 0.3560 & 0.007222 & \$10,580 \\
\textbf{Isolation Forest Baseline} & 0.1033 & 0.9407 & 0.2308 & 0.2791 & 0.1791 & 0.3243 & 0.2393 & 0.002579 & \$26,100 \\
\bottomrule
\end{tabular}%
}
\end{table}

### 4.2 Precision-Recall & ROC Curve Analysis

In heavily imbalanced regimes, ROC curves present an overly optimistic assessment because the massive True Negative volume keeps False Positive Rates deceptively low ($<0.01$). **Precision-Recall curves (PR-AUC / AUPRC)** directly focus on the minority class trade-off.

\begin{figure}[htbp]
\centering
\begin{minipage}{0.49\textwidth}
  \centering
  \includegraphics[width=\linewidth]{results/figures/pr_curves.png}
  \caption*{(a) Precision-Recall (PR) Curves}
\end{minipage}\hfill
\begin{minipage}{0.49\textwidth}
  \centering
  \includegraphics[width=\linewidth]{results/figures/roc_curves.png}
  \caption*{(b) Receiver Operating Characteristic (ROC) Curves}
\end{minipage}
\caption{Discriminative Evaluation Curves across all benchmark architectures on the held-out test partition ($42,722$ transactions, $74$ fraud cases). AUPRC accurately captures model differentiation under severe imbalance ($578:1$).}
\label{fig:curves}
\end{figure}

As shown in Figure \ref{fig:curves}(a), the **Hybrid (Anomaly + Deep DL)** model dominates the precision-recall envelope across practical recall operating points ($0.75 - 0.85$), achieving an AUPRC of **$0.8550$**, followed by the **Supervised Deep Tabular ResNet** ($0.8321$) and **XGBoost** ($0.8245$).

### 4.3 Classification Trade-Offs & Confusion Matrices

Figure \ref{fig:confusion_matrices} illustrates confusion matrices for the three leading models on the test set:

\begin{figure}[htbp]
\centering
\begin{minipage}{0.32\textwidth}
  \centering
  \includegraphics[width=\linewidth]{results/figures/confusion_matrix_hybrid_(anomaly_+_deep_dl).png}
  \caption*{(a) Hybrid Deep DL}
\end{minipage}\hfill
\begin{minipage}{0.32\textwidth}
  \centering
  \includegraphics[width=\linewidth]{results/figures/confusion_matrix_supervised_deep_tabular.png}
  \caption*{(b) Supervised Deep Tabular}
\end{minipage}\hfill
\begin{minipage}{0.32\textwidth}
  \centering
  \includegraphics[width=\linewidth]{results/figures/confusion_matrix_xgboost.png}
  \caption*{(c) XGBoost Ensemble}
\end{minipage}
\caption{Confusion Matrices for top-3 models on the held-out test partition at cost-optimal operational thresholds.}
\label{fig:confusion_matrices}
\end{figure}

The Hybrid model successfully detects $61$ out of $74$ fraud events ($82.43\%$ recall) while producing only $10$ false positives out of $42,648$ legitimate transactions ($\text{FPR} = 0.0234\%$).

\begin{takeawaybox}
\textbf{Key Empirical Takeaway:} Fusing unsupervised manifold latents ($z \in \mathbb{R}^8$) and reconstruction residuals ($e \in \mathbb{R}^1$) with supervised tabular representations delivers a $+3.05\%$ AUPRC improvement over tuned XGBoost, while cutting false positives by $9.3\%$.
\end{takeawaybox}

# 5. Explainable AI (XAI) Analysis

Financial institutions operating in Europe (GDPR) and North America (FCRA) must provide cardholders with Adverse Action rationales and maintain internal audit trails. Our system provides both population-level (global) and transaction-level (local) explainability.

### 5.1 Global Feature Attributions (SHAP)

Computing Shapley values across the test population reveals that features **`V14`**, **`V10`**, **`V4`**, **`V12`**, and **`V17`** dominate decision boundaries:

- Negative values of `V14` and `V10` exert severe positive pressure on the fraud probability.
- Elevated values of `V4` indicate heightened anomaly associated with automated card testing.
- Directional impact is confirmed via SHAP beeswarm and partial dependence plots, demonstrating sharp non-linear risk transitions when $V14 < -3.5$.

\clearpage

\begin{figure}[htbp]
\centering
\begin{minipage}{0.49\textwidth}
  \centering
  \includegraphics[width=\linewidth]{results/explanations/shap_summary_bar.png}
  \caption*{(a) Mean Absolute SHAP Importance}
\end{minipage}\hfill
\begin{minipage}{0.49\textwidth}
  \centering
  \includegraphics[width=\linewidth]{results/explanations/shap_beeswarm.png}
  \caption*{(b) SHAP Feature Beeswarm Attribution}
\end{minipage}
\caption{Global SHAP Feature Attributions across test transactions. Features $V_{14}$, $V_{10}$, and $V_4$ dominate decision boundaries.}
\label{fig:shap_global}
\end{figure}

Figure \ref{fig:shap_dependence} illustrates the non-linear risk transitions uncovered by partial dependence analysis:

\begin{figure}[htbp]
\centering
\begin{minipage}{0.32\textwidth}
  \centering
  \includegraphics[width=\linewidth,height=4.4cm,keepaspectratio]{results/explanations/shap_dependence_v14.png}
  \caption*{(a) $V_{14}$ Partial Dependence}
\end{minipage}\hfill
\begin{minipage}{0.32\textwidth}
  \centering
  \includegraphics[width=\linewidth,height=4.4cm,keepaspectratio]{results/explanations/shap_dependence_v10.png}
  \caption*{(b) $V_{10}$ Partial Dependence}
\end{minipage}\hfill
\begin{minipage}{0.32\textwidth}
  \centering
  \includegraphics[width=\linewidth,height=4.4cm,keepaspectratio]{results/explanations/shap_dependence_v4.png}
  \caption*{(c) $V_4$ Partial Dependence}
\end{minipage}
\caption{SHAP Partial Dependence plots highlighting non-linear risk transitions for the top three critical risk features.}
\label{fig:shap_dependence}
\end{figure}

\clearpage

### 5.2 Local Case-Level Explanations (SHAP Waterfall & LIME)

For every alerted transaction, the system computes local additive attributions and linear surrogate rules:

- **Confirmed Fraud Case (TXN-000735):** The model assigns a **$99.28\%$** fraud probability. As shown in Figure \ref{fig:local_xai}(a), $V_{14} = -4.998$ contributes $+5.72$ log-odds, $V_{10} = -4.102$ contributes $+1.11$, and $V_{16} = -2.991$ contributes $+0.95$.
- **LIME Local Decision Rules:** As shown in Figure \ref{fig:local_xai}(b), LIME identifies that conditions $V_{14} \le -1.41$ and $V_{10} \le -1.02$ are the local bounding conditions triggering the alert.

\begin{figure}[htbp]
\centering
\begin{minipage}{0.49\textwidth}
  \centering
  \includegraphics[width=\linewidth]{results/explanations/shap_waterfall_fraud.png}
  \caption*{(a) SHAP Waterfall Attribution}
\end{minipage}\hfill
\begin{minipage}{0.49\textwidth}
  \centering
  \includegraphics[width=\linewidth]{results/explanations/lime_sample_735.png}
  \caption*{(b) LIME Local Surrogate Explanation}
\end{minipage}
\caption{Local Decision Explanations for Confirmed Fraud Case (Transaction \#TXN-000735). Both SHAP and LIME identify anomalous negative shifts in $V_{14}$ and $V_{10}$ as the predominant drivers of the $99.28\%$ fraud prediction.}
\label{fig:local_xai}
\end{figure}

\clearpage

### 5.3 Automated Bank Security Operations Alert Cards

The system automatically translates complex Shapley math into human-actionable Security Operations Alert Cards:

\begin{alertcardbox}[Bank Security Operations Alert: Transaction \#TXN-000735-FRAUD]{AlertRed}
\textbf{Audit Status:} Complete \quad\textbar\quad \textbf{Model Engine:} AI Anomaly-Driven Credit Fraud Pipeline

\vspace{4pt}
\noindent\textbf{\large Executive Risk Assessment}
\begin{itemize}
  \item \textbf{Fraud Risk Verdict:} \colorbox{BadgeRedBg}{\textcolor{AlertRed}{\textbf{CRITICAL RISK (TIER 1)}}}
  \item \textbf{Estimated Fraud Probability:} \textbf{99.28\%}
  \item \textbf{Unsupervised Manifold State:} \textbf{ANOMALOUS} (Reconstruction MSE: \texttt{3.5071} vs Threshold: \texttt{1.5049}, $2.33\times$ boundary)
\end{itemize}

\vspace{3pt}
\noindent\textbf{\large Root Cause Analysis (XAI Feature Attribution)}
\begin{itemize}
  \item \textbf{Primary Red Flags (Driving Risk UP):}
    \begin{enumerate}
      \item \texttt{V14} (Normalized Value: $-4.998$) $\to$ SHAP Impact: \textbf{+5.7184}
      \item \texttt{V10} (Normalized Value: $-4.102$) $\to$ SHAP Impact: \textbf{+1.1060}
      \item \texttt{V16} (Normalized Value: $-2.991$) $\to$ SHAP Impact: \textbf{+0.9514}
      \item \texttt{V17} (Normalized Value: $-5.216$) $\to$ SHAP Impact: \textbf{+0.9385}
    \end{enumerate}
  \item \textbf{Mitigating Factors (Driving Risk DOWN):}
    \begin{enumerate}
      \item \texttt{V7} (Normalized Value: $+1.318$) $\to$ SHAP Impact: \textbf{-0.8917}
      \item \texttt{Hour\_Cos} (Normalized Value: $-0.475$) $\to$ SHAP Impact: \textbf{-0.2988}
    \end{enumerate}
\end{itemize}

\vspace{3pt}
\noindent\textbf{\large Recommended Banking Protocol}
\begin{quote}
\textbf{IMMEDIATE ACTION REQUIRED:} Block transaction immediately, place temporary freeze on credit card account, and dispatch instant SMS/App 2-factor identity verification prompt to cardholder.
\end{quote}

\vspace{3pt}
\noindent\textbf{\large Regulatory \& Compliance Audit Trail}
\begin{verbatim}
Transaction evaluated by XAI Hybrid Banking Pipeline. Supervised fraud probability:
99.28%. Autoencoder reconstruction residual is 2.33x of normal boundary. Key risk
drivers identified via SHAP Shapley values. Meets GDPR Art. 22 Adverse Action standards.
\end{verbatim}
\end{alertcardbox}

# 6. Hardware Acceleration & Dual NVIDIA T4 Deployment

The codebase is engineered for dual NVIDIA T4 GPU environments on Kaggle:

- **Multi-GPU Parallelization:** Automatically detects dual T4 GPUs via `torch.cuda.device_count() >= 2` and wraps neural architectures in `torch.nn.DataParallel`.
- **Automatic Mixed Precision (AMP):** Employs `torch.amp.autocast('cuda')` during forward passes, halving memory bandwidth overhead and doubling tensor core throughput.
- **DataLoader Efficiency:** Uses pinned memory (`pin_memory=True`) and multi-process asynchronous workers (`num_workers=4`) to eliminate CPU-to-GPU memory transfer bottlenecks.
- **Graceful Fallback:** If executed locally without CUDA, the pipeline dynamically adapts to multi-threaded CPU execution.

\clearpage

# 7. Ethical Considerations, Limitations & Future Work

### 7.1 Fair Lending & Financial Ethics
Machine learning models deployed in credit card underwriting or fraud monitoring must avoid disparate impact against marginalized demographics. Because the benchmark features are PCA-transformed, direct demographic leakage is mitigated; however, commercial deployments must audit proxy variables to ensure compliance with the Equal Credit Opportunity Act (ECOA).

### 7.2 System Limitations
1. Benchmark PCA features mask original transaction metadata (such as merchant category codes and geolocation).
2. The dataset captures a 48-hour snapshot; longitudinal drift over months or years requires streaming retraining pipelines.

### 7.3 Future Research Directions
1. Incorporate Graph Neural Networks (GNNs) to model cardholder-merchant bipartite transaction graphs.
2. Implement Self-Supervised Masked Autoencoders for tabular sequence modeling.
3. Deploy real-time Kafka / Ray streaming inference pipelines.

# 8. Conclusion

We have designed, implemented, and benchmarked an end-to-end Explainable AI Credit Card Fraud Detection system tailored for modern banking operations. By uniting unsupervised Deep Autoencoders for anomaly detection, supervised Tabular ResNets with Focal Loss for imbalanced classification, and game-theoretic SHAP/LIME interpretability, the system delivers both superior predictive performance (Hybrid AUPRC **0.8550**) and actionable, compliant transparency. The entire repository is modular, tested, GPU-accelerated, and submission-ready.

# References

1. Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). Calibrating probability with undersampling for unbalanced classification. *IEEE Symposium Series on Computational Intelligence*.
2. Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. *IEEE International Conference on Computer Vision (ICCV)*.
3. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems (NeurIPS)*.
4. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. *ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*.
5. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *ACM SIGKDD International Conference*.
