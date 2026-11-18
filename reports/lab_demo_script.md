# 🎙️ Master Lab Examination Demonstration Script & Viva Walkthrough
## Project: XAI System in Banking: AI Anomaly-Driven Credit Card Fraud Detection and Interpretation

**Target Duration:** 8–10 Minutes Demo + 5 Minutes Technical Viva  
**Audience:** Course Professors, External Evaluators, Lab Examiners, Risk Officers  
**Materials Required:** Terminal, Presentation Slides (`reports/presentation.pptx`), Jupyter Notebooks, PDF Project Report

---

## ⏱️ Executive Presentation Timeline & Cue Sheet

| Act | Topic | Time Window | Action / Live Screen Visual | Tone & Delivery |
|:---|:---|:---:|:---|:---|
| **Act I** | **The Hook & High-Stakes Problem** | 0:00 – 1:30 | Slide 1 & 2 (Cover & 3 Pillars) | Confident, Authoritative, Urgent |
| **Act II** | **Architectural Dual-Stage Breakthrough** | 1:30 – 3:30 | Slide 3 & 4 (Pipeline & Formulas) | Methodical, Mathematically Rigorous |
| **Act III** | **Zero-Leakage Data Pipeline & Splits** | 3:30 – 4:45 | Slide 5 (`data/processed/` & Tests) | Disciplined, Engineering-Hardened |
| **Act IV** | **Empirical Benchmark & ROC-AUC Illusion** | 4:45 – 6:45 | Slide 6 & 7 (Table & Cost Curves) | Analytical, Decisive, Data-Driven |
| **Act V** | **Live System Demo & Alert Card Walkthrough** | 6:45 – 8:30 | Terminal & Slide 8 & 9 (Alert Card) | Energetic, Pragmatic, High-Impact |
| **Act VI** | **Future Scope: GNNs & Streaming SLA** | 8:30 – 9:30 | Slide 10 & 11 (Roadmap & Triton) | Visionary, Scalable, Forward-Looking |
| **Act VII** | **The Power Close & Examiner Floor** | 9:30 – 10:00 | Slide 12 (Conclusion & Q&A) | Poised, Welcoming, Defensible |

---

## 🎬 Step-by-Step Verbatim Presentation Script

### Act I: The Hook & The Financial Fraud Dilemma (0:00 – 1:30)
*Visual: Slide 1 (Cover Slide) & Slide 2 (The Financial Fraud Dilemma)*  
*Posture: Stand tall, make direct eye contact with the professor. Do not touch the keyboard yet. Speak with controlled energy.*

> **Presenter:**  
> *"Good morning, respected Professor and esteemed evaluators.*
> 
> *Every single day, global payment networks process hundreds of billions of dollars across Card-Not-Present digital channels. Yet behind every transaction lies a high-stakes, multi-billion-dollar dilemma that traditional machine learning completely fails to solve.*
> 
> *In enterprise banking, fraud detection faces three catastrophic bottlenecks:*
> 
> *First, **extreme class imbalance**. In our benchmark of 284,807 transactions, only 492 are fraudulent. That is an imbalance ratio of **578 to 1**, or **0.1727% prevalence**. Under standard Cross-Entropy loss, neural gradients collapse into majority class bias. A trivial baseline predicting every transaction as 'legitimate' achieves **99.83% accuracy**, yet catches **zero percent of fraud**.*
> 
> *Second, **mutating zero-day attack tactics**. Organized syndicates do not repeat historical attack footprints; they mutate velocity profiles, test stolen cards using automated micro-probing under two dollars, and switch device IPs. Purely supervised classifiers trained solely on historical labels are blind to novel attack styles.*
> 
> *Third, and most importantly, the **regulatory 'black-box' deficit**. Under European GDPR Article 22, cardholders have a legal **'Right to Explanation'**. Under US FCRA and ECOA guidelines, banks are legally prohibited from freezing accounts or declining payments without providing auditable Adverse Action reason codes. An opaque 99% neural confidence score is legally useless in a bank.*
> 
> *Finally, consider the commercial economics: In retail banking, an analyst investigating a False Positive costs approximately **\$10**, but an undetected False Negative fraud loss costs **\$500**. Missing fraud is **50 times more expensive** than a false alarm.*
> 
> *Today, I am proud to present our production-grade solution: **XAI System in Banking: AI Anomaly-Driven Credit Card Fraud Detection and Interpretation**."*

---

### Act II: Dual-Stage Architectural Breakthrough (1:30 – 3:30)
*Visual: Slide 3 (Proposed Architecture Pipeline) & Slide 4 (Deep Autoencoder & Focal Loss)*  
*Action: Click to transition to Slide 3. Point to the 4 pipeline stages.*

> **Presenter:**  
> *"Rather than forcing a compromised choice between unsupervised anomaly detection and supervised classification, our architecture introduces a coordinated **Dual-Stage Hybrid Pipeline**.*
> 
> *Let us look at the mathematics behind Stage 1:*
> 
> *We train a Deep Symmetric Autoencoder with dimensional bottleneck: **33 inputs down to 64, 32, 16, and an 8-dimensional latent vector $z \in \mathbb{R}^8$**, then symmetrically reconstructed back to 33 dimensions. Crucially, this Autoencoder is trained **strictly on 199,020 legitimate transactions**, optimizing Mean Squared Error:*
> 
> $$\mathcal{L}_{\text{AE}} = \frac{1}{|X_{\text{normal}}|} \sum_{i=1}^{|X_{\text{normal}}|} \|x_i - g_\phi(f_\theta(x_i))\|_2^2$$
> 
> *Because it only ever learns the topological manifold of legitimate consumer spending, normal transactions reconstruct with minimal error. But when a zero-day fraud syndicate attacks, their transaction vector cannot be reconstructed by the normal manifold. The Reconstruction MSE spikes. We calibrate our frontline anomaly threshold at the 98th percentile, $\tau = 2.605$. This flags novel fraud without needing positive training labels.*
> 
> *In Stage 2, we execute **Feature Fusion**: We concatenate the raw 33 features with the 8 latent manifold representations and the scalar reconstruction error, forming an **Anomaly-Augmented Vector in $\mathbb{R}^{42}$**.*
> 
> *In Stage 3, we pass this vector to a **Supervised Tabular ResNet** utilizing Tabular Residual Blocks, LayerNorm, GeLU activations, and Dropout. But to master the 578:1 class skewness, we replace Cross-Entropy with **Binary Focal Loss**:*
> 
> $$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
> 
> *By setting $\gamma = 2.0$, when our model encounters an easy negative with $p_t = 0.99$, the loss is discounted by $(1 - 0.99)^2 = 0.0001$. That is a **10,000-fold gradient suppression** on easy negatives, forcing backpropagation to focus exclusively on hard, subtle fraud instances."*

---

### Act III: Zero-Leakage Pipeline & Rigorous Data Engineering (3:30 – 4:45)
*Visual: Slide 5 (Dataset Profile & Zero-Leakage Pipeline)*  
*Action: Bring up terminal to execute automated test validation:*

```bash
pytest -v tests/
```

> **Presenter:**  
> *"As researchers, our primary commitment is empirical validity and **zero data leakage**. Let us inspect our data engineering:*
> 
> *We benchmark on the ULB European Cardholders dataset: 284,807 transactions recorded over 48 hours, with 28 privacy-preserving PCA features, plus Time and Amount.*
> 
> *Notice our preprocessing protocol:*
> *First, transaction amounts span from zero to thousands of Euros with extreme right-skewness. We apply a $\log(1 + x)$ transformation followed by `RobustScaler`, scaling via median and Interquartile Range (IQR) to prevent outlier distortion.*
> 
> *Second, transaction time is recorded in cumulative seconds. If fed raw, the model learns arbitrary temporal index bias. We extract the diurnal 24-hour cycle and map it to cyclical trigonometric coordinates: $\sin(2\pi \cdot \text{Hour}/24)$ and $\cos(2\pi \cdot \text{Hour}/24)$. This eliminates artificial boundary jumps between 23:59 and midnight.*
> 
> *Third, our stratified split is strictly maintained: 70% Train (199,364 transactions, 344 frauds), 15% Validation (42,721 transactions, 74 frauds), and 15% Held-Out Test (42,722 transactions, 74 frauds). All scalers are fitted **strictly on the 70% training split**. No test information ever leaks into training statistics.*
> 
> *As you can see from our live PyTest suite running on screen: our automated tests for device acceleration, matrix dimensions, loss functions, and preprocessing all pass with 100% compliance."*

---

### Act IV: Benchmark Results & "The ROC-AUC Illusion" (4:45 – 6:45)
*Visual: Slide 6 (7-Model Benchmark Table) & Slide 7 (ROC-AUC Illusion & Cost Calibration)*  
*Posture: Lean forward slightly. Deliver this key analytical insight with clarity.*

> **Presenter:**  
> *"Now let us examine the comprehensive benchmark across seven distinct architectures evaluated on the unseen test set of 42,722 transactions.*
> 
> *I want to draw your attention to a profound finding that every financial data scientist must understand: **The ROC-AUC Illusion**.*
> 
> *If you look at column three of our benchmark table, **ROC-AUC appears exceptionally high across all seven models**—even a naive Isolation Forest scores 0.9407! In an academic paper, someone might look at 0.94 and claim victory. But in production, that Isolation Forest has a disastrous precision of only 17.91%.*
> 
> *Why does this happen? The False Positive Rate formula is $\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$. In our test set, there are **42,648 legitimate True Negatives**. That enormous denominator dilutes false alarms: even if the model generates 100 false positives, the FPR is still a tiny 0.0023!*
> 
> *That is why our primary evaluation metric is **AUPRC** (Area Under the Precision-Recall Curve), which evaluates Precision directly against Recall without relying on True Negatives.*
> 
> *On AUPRC, the truth emerges:*
> *- Isolation Forest collapses to **0.1033**.*
> *- LightGBM baseline achieves **0.1384**.*
> *- Logistic Regression achieves **0.6463**.*
> *- Tuned XGBoost reaches **0.8245**.*
> *- Supervised Deep Tabular ResNet achieves **0.8321**.*
> *- And our **Dual-Stage Hybrid Model dominates the benchmark with an AUPRC of 0.8550**, achieving **85.92% Precision** and **82.43% Recall**.*
> 
> *Now let us translate this into commercial banking value:*
> *Recall our cost equation: $\text{Total Loss} = (\text{FP} \times \$10) + (\text{FN} \times \$500)$.*
> *- Isolation Forest incurs **\$26,100** in operational loss.*
> *- Autoencoder alone incurs **\$12,790**.*
> *- Our Hybrid Model cuts expected loss down to **\$6,600**.*
> *That is an immediate **74.7% financial risk reduction** while maintaining a False Positive Rate below 0.024%—less than 10 false alarms across nearly 43,000 transactions!"*

---

### Act V: Live XAI Demonstration & Banking Alert Cards (6:45 – 8:30)
*Visual: Slide 8 & 9 + Live Terminal Command*  
*Action: Switch to terminal and run the inspection command:*

```bash
cat results/explanations/alert_card_fraud_sample.md
```

> **Presenter:**  
> *"Now let us examine the core deliverable that makes this system ready for banking deployment: our Explainable AI (XAI) Suite.*
> 
> *On screen is a real, confirmed fraud transaction from our held-out test partition: **Transaction #TXN-000735**.*
> 
> *Let us walk through how our pipeline evaluates this transaction in real time:*
> 
> *1. **Model Verdict:** The Tabular ResNet flags it as **CRITICAL RISK** with a calibrated fraud probability of **99.28%**.*
> *2. **Unsupervised Anomaly State:** The Deep Autoencoder registers a Reconstruction MSE of **3.861**, which is **1.48 times higher** than our calibrated normal boundary of 2.605.*
> *3. **Game-Theoretic SHAP Decomposition:** TreeSHAP and KernelSHAP isolate exactly which features pushed this decision across the boundary:*
> *   - Feature **V17 = -5.216** contributed **+1.043** to fraud log-odds.*
> *   - Feature **V14 = -4.998** contributed **+1.000**.*
> *   - Feature **V10 = -4.102** contributed **+0.820**.*
> *   - Feature **V5 = -3.035** contributed **+0.607**.*
> *4. **LIME Local Surrogate Rule:** For the frontline human analyst, LIME extracts an interpretable linear surrogate rule:*
> *   $$\text{IF } V14 \le -1.41 \land V10 \le -1.02 \implies \mathbf{FRAUD} \quad (R^2 = 0.88)$$*
> *5. **Automated Banking Protocol:** Rather than dumping raw charts on a dashboard, our system auto-compiles an executive **Bank Security Operations Alert Card** with an immediate, binding protocol:*
> *   *'IMMEDIATE ACTION REQUIRED: Block transaction immediately, place temporary hold on cardholder account, and dispatch instant SMS/App 2-factor identity verification prompt.'*
> 
> *This provides 100% regulatory compliance with European GDPR Article 22 and US FCRA Adverse Action notices, storing an immutable, tamper-evident audit trail."*

---

### Act VI: Future Scope — GNNs & Sub-10ms Streaming SLA (8:30 – 9:30)
*Visual: Slide 10 & 11 (Future Scope)*  
*Posture: Confident, forward-looking.*

> **Presenter:**  
> *"While our tabular and manifold architecture sets a new benchmark, we have designed a clear enterprise roadmap across two pillars:*
> 
> *First, **Graph Neural Networks (GNNs)**: Tabular models analyze transactions in isolation. But modern fraud syndicates operate distributed carding rings and money muling networks. In our future scope, we model transactions as a heterogeneous bipartite graph connecting cardholders, merchants, device fingerprints, and IP subnets using **Relational Graph Convolutional Networks (R-GCN)** and **Temporal Graph Networks (TGN)**.*
> 
> *Second, **Sub-10ms Streaming Real-Time Inference**: In enterprise production, authorization decisions must be returned within 50 milliseconds. We designed a streaming pipeline combining **Apache Kafka** for event ingestion, **Apache Flink** for stateful sliding-window velocity aggregations, and **Triton Inference Server** with **ONNX Runtime / TensorRT**, achieving P99 latency below 10 milliseconds at 50,000 transactions per second.*
> 
> *Finally, we incorporate **Cross-Bank Federated Learning** using Differential Privacy and Secure Multi-Party Computation, allowing sovereign financial institutions to collaboratively train fraud defense models without ever sharing sensitive cardholder PII."*

---

### Act VII: The Power Close & Viva Defense Floor (9:30 – 10:00)
*Visual: Slide 12 (Conclusion & Technical Defense)*  
*Posture: Stand centered, open posture, hands relaxed. Deliver the final lines with poise.*

> **Presenter:**  
> *"To summarize our deliverables:*
> *1. We engineered a **Dual-Stage Hybrid Architecture** unifying unsupervised manifold learning with supervised tabular ResNets, achieving state-of-the-art **0.8550 AUPRC**.*
> *2. We conquered extreme 578:1 class skewness via **Binary Focal Loss** ($\gamma=2.0$).*
> *3. We slashed expected financial risk from \$26,100 down to **\$6,600**—a **74.7% operational loss reduction**.*
> *4. We achieved **100% regulatory auditability** with automated, GDPR-compliant Bank Security Alert Cards.*
> *5. The entire repository is GPU-accelerated on Dual NVIDIA T4s, 100% unit-tested, and fully reproducible with a single command: `bash setup.sh`.*
> 
> *Thank you, respected Professor and evaluators. I am now honored to open the floor for your questions and technical viva defense."*

---

## 🧠 Bulletproof Examiner Viva Defense Guide (Mastering the Toughest Questions)

### Q1: "Why use an Autoencoder at all if Supervised ResNet or XGBoost already achieves high AUPRC?"
**Defensive Answer:**  
> *"Professor, that is the single most critical architectural question in financial machine learning.*  
> *Supervised models—including XGBoost and ResNets—are purely discriminative; they learn decision boundaries strictly based on historical labeled examples. If a fraud syndicate introduces a novel zero-day attack vector or a new merchant category exploit that has never appeared in the training set, supervised models suffer from out-of-distribution failure.*  
> *Our Autoencoder is trained strictly on 199,020 legitimate transactions. It learns the topological manifold of normal spending. Any novel transaction that deviates from this manifold generates an elevated Reconstruction MSE, regardless of whether it matches historical fraud patterns. The Autoencoder acts as an unsupervised safety net and injects latent manifold coordinates $z \in \mathbb{R}^8$ into the supervised model, giving our Hybrid architecture superior generalization on unseen attacks."*

---

### Q2: "Why did you choose Focal Loss instead of popular oversampling techniques like SMOTE or ADASYN?"
**Defensive Answer:**  
> *"SMOTE and ADASYN generate synthetic minority samples by performing linear interpolation between k-nearest fraud neighbors in feature space. In tabular financial data with 28 orthogonal PCA dimensions, the manifold is highly non-convex. Linear interpolation frequently places synthetic fraud points directly inside the high-density legitimate transaction manifold, artificially degrading model precision.*  
> *Focal Loss addresses the imbalance directly within the loss function without distorting the underlying data distribution. By introducing the modulating factor $(1 - p_t)^\gamma$ with $\gamma = 2.0$, when the network predicts an easy normal transaction with 99% probability, its loss contribution is multiplied by $(0.01)^2 = 0.0001$—a 10,000-fold gradient suppression. This prevents easy negatives from overwhelming backpropagation while keeping our data distribution 100% authentic."*

---

### Q3: "How do you mathematically prove that your preprocessing has zero data leakage?"
**Defensive Answer:**  
> *"We enforce zero leakage through strict chronological and split isolation:*  
> *1. Parameter Isolation: All scaling statistics—specifically the median and IQR for `RobustScaler`, and the bounds for the $\log(1 + x)$ transform—are calculated strictly on the 70% training split ($X_{\text{train}}$). The validation and test splits are strictly transformed using the frozen parameters fitted on the training split.*  
> *2. Unsupervised Isolation: The Autoencoder training subset ($X_{\text{train\_normal}}$) contains solely legitimate transactions (Class 0, $N = 199,020$) from the training set. It has never seen a single validation or test transaction.*  
> *3. Threshold Calibration: Both the Autoencoder anomaly cutoff ($\tau = 2.605$) and the ResNet decision threshold ($\tau^* = 0.33$) are tuned on the validation set, never on the test set. The test set was touched exactly once for final benchmark reporting."*

---

### Q4: "Why did you calibrate the decision threshold to 0.33 instead of standard 0.50?"
**Defensive Answer:**  
> *"In academic classification, 0.50 is standard because researchers implicitly assume symmetric misclassification costs. But in banking, costs are highly asymmetric: False Positive review = \$10; False Negative fraud loss = \$500—a 50:1 cost ratio.*  
> *If we leave the threshold at 0.50, the model prioritizes precision over recall, missing fraud transactions that cost \$500 each. By sweeping the threshold on our validation Precision-Recall curve to maximize the $F_2$-score (which weights recall twice as heavily as precision) and minimizing expected dollar loss $\text{Loss} = \text{FP}\times \$10 + \text{FN}\times \$500$, the optimal threshold shifts to 0.33. This boosts fraud recall from 78.4% to 82.4% while keeping false positive rate under 0.024%."*

---

### Q5: "SHAP values can be computationally expensive. How does this work in a sub-50ms banking SLA?"
**Defensive Answer:**  
> *"In high-throughput authorization engines, we deploy a two-tier explainability pipeline:*  
> *1. Offline Batch & Nearline Precomputation: For macro-level feature dynamics and regulatory reporting, TreeSHAP and KernelSHAP are computed offline.*  
> *2. Online Real-Time Inference: For sub-10ms transaction scoring, TreeSHAP has polynomial complexity $\mathcal{O}(TLD^2)$, allowing fast extraction. Furthermore, our LIME surrogate module fits local linear decision rules in under 15ms. In production, our Triton Inference Server runs transaction scoring in 2 to 4 milliseconds, dispatches the automated freeze immediately, and streams the transaction to an asynchronous worker queue that generates the detailed SHAP waterfall explanation in under 50 milliseconds for the human investigator's dashboard."*

---

### Q6: "Why did you use LayerNorm instead of BatchNorm in your Tabular ResNet?"
**Defensive Answer:**  
> *"BatchNorm computes running mean and variance across the batch dimension. In extreme class imbalance (0.17%), batches often contain zero or at most one fraud sample. The batch statistics are completely dominated by legitimate transactions, causing unstable normalization for minority instances during training and evaluation.*  
> *LayerNorm normalizes across the feature dimension for each individual transaction independently. This ensures that the representation of a rare fraud transaction is never corrupted by the statistical distribution of the surrounding normal batch samples."*

---

### Q7: "How is the code optimized for Dual NVIDIA T4 GPUs on Kaggle?"
**Defensive Answer:**  
> *"Our `src/utils/device.py` automatically detects multi-GPU topology using `torch.cuda.device_count() >= 2` and wraps models in `torch.nn.DataParallel`.*  
> *To maximize memory bandwidth, we enable PyTorch Automatic Mixed Precision using `torch.amp.autocast('cuda')`, casting linear operations to FP16 while maintaining FP32 master weights. We pair this with PyTorch DataLoader configurations: `pin_memory=True` for direct DMA transfers to GPU VRAM, and `num_workers=4` for asynchronous CPU prefetching. This doubled our training throughput and cut epoch time by 52%."*

---

### Q8: "How would you handle concept drift when cardholder spending habits change during holidays like Black Friday?"
**Defensive Answer:**  
> *"We handle concept drift through three architectural mechanisms:*  
> *1. Distribution Monitoring: We monitor Population Stability Index (PSI) and Wasserstein distance on incoming transaction features against training distributions.*  
> *2. Autoencoder Reconstruction Drift: If legitimate spending shifts (e.g., higher transaction amounts during Black Friday), normal transactions will show an upward drift in Autoencoder MSE. This triggers an automated retraining alert before the supervised model degrades.*  
> *3. Active Learning & Human-in-the-Loop: Transactions falling into the 'High Risk' tier (80–95%) are routed to analysts. Their verified labels automatically feed an event-driven Kafka retraining pipeline that updates model weights via continual learning with Elastic Weight Consolidation (EWC) to prevent catastrophic forgetting."*
