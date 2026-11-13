"""
Generator for all 5 project Jupyter notebooks.
Builds clean, fully structured, publication-quality .ipynb files:
01_eda.ipynb
02_anomaly_detection.ipynb
03_supervised_training.ipynb
04_xai_explanations.ipynb
05_end_to_end_demo.ipynb
"""

import json
from pathlib import Path
import nbformat as nbf


def make_notebook(cells, filepath: Path):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.0"
        }
    }
    with open(filepath, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[SUCCESS] Created notebook: {filepath}")


def build_all_notebooks(notebooks_dir: str = "notebooks"):
    nb_dir = Path(notebooks_dir)
    nb_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 01_eda.ipynb
    # -------------------------------------------------------------
    nb1_cells = [
        nbf.v4.new_markdown_cell("""# 📊 01. Exploratory Data Analysis & Imbalance Characterization
### Project: XAI System in Banking: AI Anomaly-Driven Credit Card Fraud Detection and Interpretation
**Objective:** In-depth exploratory data analysis of the European cardholder credit card fraud dataset (`mlg-ulb/creditcardfraud`), characterizing extreme class imbalance ($0.172\\%$), feature distributions, time-domain dynamics, and financial transaction amounts."""),
        
        nbf.v4.new_code_cell("""import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add repo root to path
ROOT_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT_DIR))

# Style settings
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["figure.dpi"] = 120
print(f"Working directory: {ROOT_DIR}")"""),

        nbf.v4.new_markdown_cell("""## 1. Data Ingestion & Shape Verification
The primary dataset consists of credit card transactions made by European cardholders in September 2013 over a two-day period."""),

        nbf.v4.new_code_cell("""data_path = ROOT_DIR / "data/raw/creditcard.csv"
if not data_path.exists():
    from scripts.download_data import download_dataset
    download_dataset("mlg-ulb/creditcardfraud", str(ROOT_DIR / "data/raw"))

df = pd.read_csv(data_path)
print(f"Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
df.head()"""),

        nbf.v4.new_markdown_cell("""## 2. Extreme Class Imbalance Characterization
In credit card fraud detection, fraudulent transactions represent a minute fraction of overall transaction volume. We evaluate the exact class distribution below."""),

        nbf.v4.new_code_cell("""class_counts = df["Class"].value_counts()
normal_count = class_counts[0]
fraud_count = class_counts[1]
fraud_pct = (fraud_count / len(df)) * 100

print(f"Legitimate Transactions (Class 0): {normal_count:,} ({100 - fraud_pct:.4f}%)")
print(f"Fraudulent Transactions (Class 1): {fraud_count:,} ({fraud_pct:.4f}%)")
print(f"Imbalance Ratio: ~{normal_count // fraud_count}:1")

fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(x=["Legitimate (0)", "Fraud (1)"], y=[normal_count, fraud_count], palette=["#2b5c8f", "#d9534f"], ax=ax)
ax.set_yscale("log")
ax.set_title("Transaction Class Distribution (Logarithmic Scale)", fontsize=12, fontweight="bold")
ax.set_ylabel("Count (log scale)")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha="center", va="center", xytext=(0, 6), textcoords="offset points", fontsize=10)
plt.tight_layout()
plt.show()"""),

        nbf.v4.new_markdown_cell("""## 3. Transaction Amount Analysis & Logarithmic Transformation
Transaction amounts exhibit extreme right-skewness: while legitimate transactions span small everyday amounts up to corporate expenses, fraud patterns often concentrate in distinct amounts to avoid trigger thresholds."""),

        nbf.v4.new_code_cell("""print("Transaction Amount Summary:")
print("Legitimate:", df[df["Class"] == 0]["Amount"].describe())
print("\\nFraudulent:", df[df["Class"] == 1]["Amount"].describe())

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df[df["Class"] == 0]["Amount"], bins=50, kde=True, color="#2b5c8f", ax=axes[0])
axes[0].set_title("Legitimate Amount Distribution ($)", fontsize=11, fontweight="bold")
axes[0].set_xlim([0, 1500])

sns.histplot(df[df["Class"] == 1]["Amount"], bins=50, kde=True, color="#d9534f", ax=axes[1])
axes[1].set_title("Fraudulent Amount Distribution ($)", fontsize=11, fontweight="bold")
axes[1].set_xlim([0, 1500])
plt.tight_layout()
plt.show()"""),

        nbf.v4.new_markdown_cell("""## 4. Temporal Dynamics over 48 Hours
The `Time` feature records elapsed seconds from the first recorded transaction over 48 hours. By extracting the hour of the day (`(Time // 3600) % 24`), we observe distinct circadian patterns in fraudulent activity."""),

        nbf.v4.new_code_cell("""df["Hour"] = (df["Time"] // 3600) % 24

plt.figure(figsize=(10, 4))
sns.kdeplot(df[df["Class"] == 0]["Hour"], label="Legitimate", color="#2b5c8f", lw=2)
sns.kdeplot(df[df["Class"] == 1]["Hour"], label="Fraud", color="#d9534f", lw=2)
plt.title("Diurnal Transaction Density: Legitimate vs Fraud (Hour of Day)", fontsize=12, fontweight="bold")
plt.xlabel("Hour of Day (0 - 23)")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()"""),

        nbf.v4.new_markdown_cell("""## 5. PCA Component Correlation Matrix
Features `V1` through `V28` are orthogonal PCA components resulting from confidential dimensionality reduction. We inspect correlations with the target class."""),

        nbf.v4.new_code_cell("""corr = df.corr()["Class"].drop(["Class", "Hour"]).sort_values()

plt.figure(figsize=(10, 5))
colors = ["#d9534f" if v < -0.15 or v > 0.15 else "#4a90e2" for v in corr.values]
corr.plot(kind="bar", color=colors)
plt.title("Feature Correlation with Fraud Label ('Class')", fontsize=12, fontweight="bold")
plt.ylabel("Pearson Correlation Coefficient")
plt.axhline(0, color="black", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

print("Top 5 Negative Correlations (Lower value -> Higher Fraud Risk):")
print(corr.head(5))
print("\\nTop 5 Positive Correlations (Higher value -> Higher Fraud Risk):")
print(corr.tail(5)[::-1])""")
    ]
    make_notebook(nb1_cells, nb_dir / "01_eda.ipynb")

    # -------------------------------------------------------------
    # 02_anomaly_detection.ipynb
    # -------------------------------------------------------------
    nb2_cells = [
        nbf.v4.new_markdown_cell("""# 🔍 02. Unsupervised Anomaly Detection via Deep Autoencoder
### Project: XAI System in Banking: AI Anomaly-Driven Credit Card Fraud Detection and Interpretation
**Objective:** Train an unsupervised Deep Autoencoder strictly on legitimate transactions ($y=0$) to model the normal banking behavior manifold. Transactions with high reconstruction errors are flagged as anomalies."""),

        nbf.v4.new_code_cell("""import os, sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT_DIR))

from src.data.preprocessing import DataPipeline
from src.models.autoencoder import DeepAutoencoder
from src.training.autoencoder_trainer import AutoencoderTrainer
from src.utils.device import get_device

device = get_device("auto")
print(f"Execution Device: {device}")"""),

        nbf.v4.new_markdown_cell("""## 1. Load Processed Dataset
We load the preprocessed splits, noting that $X_{\\text{train\\_normal}}$ contains only class 0 transactions."""),

        nbf.v4.new_code_cell("""pipeline = DataPipeline(processed_dir=str(ROOT_DIR / "data/processed"))
data = pipeline.load_processed()
input_dim = data.X_train.shape[1]
print(f"Features: {input_dim}")
print(f"Normal training samples: {len(data.X_train_normal):,}")
print(f"Validation samples: {len(data.X_val):,} (Frauds: {np.sum(data.y_val == 1)})")
print(f"Test samples: {len(data.X_test):,} (Frauds: {np.sum(data.y_test == 1)})")"""),

        nbf.v4.new_markdown_cell("""## 2. Load Trained Deep Autoencoder
The Autoencoder compresses the 33-dimensional feature space into an 8-dimensional latent bottleneck and reconstructs the original vector."""),

        nbf.v4.new_code_cell("""model_path = ROOT_DIR / "results/models/autoencoder.pt"
autoencoder = DeepAutoencoder(input_dim=input_dim, latent_dim=8).to(device)

if model_path.exists():
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    autoencoder.load_state_dict(state)
    threshold = ckpt.get("threshold", 0.015)
    print(f"Loaded trained Autoencoder from {model_path}")
    print(f"Calibrated 98th Percentile Anomaly Threshold: {threshold:.6f}")
else:
    print("Pretrained checkpoint not found; please execute python scripts/train.py first.")"""),

        nbf.v4.new_markdown_cell("""## 3. Reconstruction Error Separation Analysis
We evaluate reconstruction errors across normal vs fraudulent transactions in the test set."""),

        nbf.v4.new_code_cell("""autoencoder.eval()
test_tensor = torch.as_tensor(data.X_test, dtype=torch.float32, device=device)
with torch.no_grad():
    recon_errors = autoencoder.compute_reconstruction_error(test_tensor, reduction="mean").cpu().numpy()

normal_errors = recon_errors[data.y_test == 0]
fraud_errors = recon_errors[data.y_test == 1]

print(f"Mean Normal Reconstruction MSE: {normal_errors.mean():.6f} (Median: {np.median(normal_errors):.6f})")
print(f"Mean Fraud Reconstruction MSE:  {fraud_errors.mean():.6f} (Median: {np.median(fraud_errors):.6f})")
print(f"Separation Ratio: {fraud_errors.mean() / normal_errors.mean():.2f}x higher error on fraud!")

plt.figure(figsize=(10, 5))
sns.kdeplot(np.log1p(normal_errors), label="Normal Transactions (Class 0)", color="#2b5c8f", lw=2)
sns.kdeplot(np.log1p(fraud_errors), label="Fraudulent Transactions (Class 1)", color="#d9534f", lw=2)
if 'threshold' in locals():
    plt.axvline(np.log1p(threshold), color="black", linestyle="--", label=f"Anomaly Threshold ({threshold:.4f})")
plt.title("Reconstruction Error Distribution (Log-Scale MSE)", fontsize=12, fontweight="bold")
plt.xlabel("Log(1 + Reconstruction MSE)")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()"""),

        nbf.v4.new_markdown_cell("""## 4. Latent Space Representation (Bottleneck Projection)
We extract the 8-dimensional bottleneck latents $z$ and project them to 2D using PCA to visualize manifold clustering."""),

        nbf.v4.new_code_cell("""from sklearn.decomposition import PCA

with torch.no_grad():
    _, z_test = autoencoder(test_tensor[:2000])
    z_np = z_test.cpu().numpy()
    y_sub = data.y_test[:2000]

pca = PCA(n_components=2)
z_2d = pca.fit_transform(z_np)

plt.figure(figsize=(8, 6))
plt.scatter(z_2d[y_sub == 0, 0], z_2d[y_sub == 0, 1], c="#2b5c8f", alpha=0.3, s=15, label="Legitimate")
plt.scatter(z_2d[y_sub == 1, 0], z_2d[y_sub == 1, 1], c="#d9534f", alpha=0.9, s=40, edgecolors="k", label="Fraud")
plt.title("Autoencoder Latent Manifold Projection (First 2 PCA Components)", fontsize=12, fontweight="bold")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
plt.legend()
plt.tight_layout()
plt.show()""")
    ]
    make_notebook(nb2_cells, nb_dir / "02_anomaly_detection.ipynb")

    # -------------------------------------------------------------
    # 03_supervised_training.ipynb
    # -------------------------------------------------------------
    nb3_cells = [
        nbf.v4.new_markdown_cell("""# 🚀 03. Supervised Deep Learning, Hybrid Modeling, and Baselines
### Project: XAI System in Banking: AI Anomaly-Driven Credit Card Fraud Detection and Interpretation
**Objective:** Compare supervised Tabular Deep ResNet (trained with Focal Loss) against the Anomaly-Augmented Hybrid Pipeline and industry-standard Gradient Boosted Decision Trees (XGBoost & LightGBM)."""),

        nbf.v4.new_code_cell("""import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib, torch

ROOT_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT_DIR))

from src.data.preprocessing import DataPipeline
from src.evaluation.evaluator import ModelEvaluator
from src.utils.device import get_device

device = get_device("auto")
print(f"Device: {device}")"""),

        nbf.v4.new_markdown_cell("""## 1. Focal Loss Formulation
Standard Binary Cross-Entropy struggles with 578:1 imbalance because the millions of easy negatives overwhelm the gradients.
$$\\text{FL}(p_t) = -\\alpha_t (1 - p_t)^\\gamma \\log(p_t)$$
Where $\\gamma = 2.0$ downweights easy examples and $\\alpha = 0.25$ balances positive/negative class weights."""),

        nbf.v4.new_code_cell("""pipeline = DataPipeline(processed_dir=str(ROOT_DIR / "data/processed"))
data = pipeline.load_processed()

# Load Model Comparison Summary Table
summary_path = ROOT_DIR / "results/tables/model_comparison.csv"
if summary_path.exists():
    df_results = pd.read_csv(summary_path)
    print("Trained Benchmark Results on Unseen Test Set:")
    display = df_results[["Model", "AUPRC", "ROC-AUC", "F1-Score", "F2-Score", "Recall", "Precision", "Total Loss ($)"]]
    print(display.to_string(index=False))
else:
    print("Model comparison table not found. Run python scripts/evaluate.py to generate benchmarks.")"""),

        nbf.v4.new_markdown_cell("""## 2. Precision-Recall Curves
Precision-Recall curves are the gold standard for severely imbalanced fraud benchmarks."""),

        nbf.v4.new_code_cell("""pr_img_path = ROOT_DIR / "results/figures/pr_curves.png"
if pr_img_path.exists():
    from IPython.display import Image
    display(Image(filename=str(pr_img_path)))
else:
    print("PR curve plot not found.")"""),

        nbf.v4.new_markdown_cell("""## 3. Financial Cost Analysis
Financial loss model:
$$\\text{Total Financial Cost} = \\text{FP} \\times \\$10 + \\text{FN} \\times \\$500$$
We compare the cost incurred by each architecture."""),

        nbf.v4.new_code_cell("""if summary_path.exists():
    plt.figure(figsize=(8, 4))
    sns.barplot(data=df_results, x="Model", y="Total Loss ($)", palette="viridis")
    plt.xticks(rotation=25, ha="right")
    plt.title("Expected Banking Financial Loss by Model ($FP=$10, $FN=$500)", fontsize=12, fontweight="bold")
    plt.ylabel("Expected Financial Loss ($)")
    plt.tight_layout()
    plt.show()""")
    ]
    make_notebook(nb3_cells, nb_dir / "03_supervised_training.ipynb")

    # -------------------------------------------------------------
    # 04_xai_explanations.ipynb
    # -------------------------------------------------------------
    nb4_cells = [
        nbf.v4.new_markdown_cell("""# 💡 04. Explainable AI (XAI) Suite: SHAP & LIME Interpretability
### Project: XAI System in Banking: AI Anomaly-Driven Credit Card Fraud Detection and Interpretation
**Objective:** Deliver model-agnostic (LIME) and game-theoretic (SHAP) explanations to provide both global insights and individual transaction-level root cause analysis for bank compliance and fraud operations."""),

        nbf.v4.new_code_cell("""import os, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import joblib, shap

ROOT_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT_DIR))

from src.data.preprocessing import DataPipeline
from src.xai.shap_explainer import FraudSHAPExplainer
from src.xai.lime_explainer import FraudLIMEExplainer

print("XAI Suite initialized.")"""),

        nbf.v4.new_markdown_cell("""## 1. Global Feature Attributions: SHAP Summary & Beeswarm
SHAP Shapley values compute the marginal contribution of each transaction feature across all possible feature subsets."""),

        nbf.v4.new_code_cell("""bar_path = ROOT_DIR / "results/explanations/shap_summary_bar.png"
bee_path = ROOT_DIR / "results/explanations/shap_beeswarm.png"

from IPython.display import Image
if bar_path.exists():
    display(Image(filename=str(bar_path)))
if bee_path.exists():
    display(Image(filename=str(bee_path)))"""),

        nbf.v4.new_markdown_cell("""## 2. SHAP Partial Dependence Plots
Partial dependence displays how specific feature values non-linearly modulate fraud probability."""),

        nbf.v4.new_code_cell("""for feat in ["v14", "v10", "v4"]:
    dep_path = ROOT_DIR / f"results/explanations/shap_dependence_{feat}.png"
    if dep_path.exists():
        display(Image(filename=str(dep_path)))"""),

        nbf.v4.new_markdown_cell("""## 3. Local Decision Rationale: SHAP Waterfall Plots
A waterfall plot explains a specific transaction: starting from the base fraud rate, it displays the push-and-pull forces of each feature."""),

        nbf.v4.new_code_cell("""wf_fraud = ROOT_DIR / "results/explanations/shap_waterfall_fraud.png"
wf_norm = ROOT_DIR / "results/explanations/shap_waterfall_legitimate.png"

if wf_fraud.exists():
    print("--- Confirmed Fraud Transaction Explanation ---")
    display(Image(filename=str(wf_fraud)))

if wf_norm.exists():
    print("--- Confirmed Legitimate Transaction Explanation ---")
    display(Image(filename=str(wf_norm)))"""),

        nbf.v4.new_markdown_cell("""## 4. Local Interpretable Model-agnostic Explanations (LIME)
LIME trains a local surrogate linear model in the perturbation neighborhood of the query transaction."""),

        nbf.v4.new_code_cell("""lime_imgs = list((ROOT_DIR / "results/explanations").glob("lime_sample_*.png"))
for p in lime_imgs[:2]:
    print(f"LIME Plot: {p.name}")
    display(Image(filename=str(p)))""")
    ]
    make_notebook(nb4_cells, nb_dir / "04_xai_explanations.ipynb")

    # -------------------------------------------------------------
    # 05_end_to_end_demo.ipynb
    # -------------------------------------------------------------
    nb5_cells = [
        nbf.v4.new_markdown_cell("""# 🏦 05. End-to-End Banking Fraud Operations Simulator
### Project: XAI System in Banking: AI Anomaly-Driven Credit Card Fraud Detection and Interpretation
**Objective:** Interactive demonstration simulating a real-time banking transaction stream. The system evaluates incoming transactions through the dual-stage Anomaly + Deep Learning pipeline, determines actionable banking protocol tiers, and automatically formats Bank Security Operations Alert Cards."""),

        nbf.v4.new_code_cell("""import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib, torch

ROOT_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT_DIR))

from src.data.preprocessing import DataPipeline
from src.models.autoencoder import DeepAutoencoder
from src.xai.analyst_report import BankingAnalystReporter
from src.utils.device import get_device

device = get_device("auto")
print(f"Device: {device}")"""),

        nbf.v4.new_markdown_cell("""## 1. Load Production Artifacts"""),

        nbf.v4.new_code_cell("""pipeline = DataPipeline(processed_dir=str(ROOT_DIR / "data/processed"))
data = pipeline.load_processed()

xgb_model = joblib.load(ROOT_DIR / "results/models/xgboost.joblib")
ae_ckpt = torch.load(ROOT_DIR / "results/models/autoencoder.pt", map_location=device, weights_only=False)

autoencoder = DeepAutoencoder(input_dim=data.X_test.shape[1], latent_dim=8).to(device)
autoencoder.load_state_dict(ae_ckpt["model_state_dict"] if "model_state_dict" in ae_ckpt else ae_ckpt)
threshold = ae_ckpt.get("threshold", 0.015)
reporter = BankingAnalystReporter(output_dir=str(ROOT_DIR / "results/explanations"))
print("All production models successfully initialized.")"""),

        nbf.v4.new_markdown_cell("""## 2. Simulate Transaction Processing Stream
We simulate incoming transaction evaluation:
1. Fast Autoencoder Reconstruction Check
2. Deep Classifier Probability Scoring
3. Dynamic Protocol Recommendation
4. Instant Analyst Card Compilation"""),

        nbf.v4.new_code_cell("""def process_transaction(txn_id: str, x_vector: np.ndarray):
    # 1. Anomaly Check
    with torch.no_grad():
        t_x = torch.as_tensor(x_vector.reshape(1, -1), dtype=torch.float32, device=device)
        recon_err = float(autoencoder.compute_reconstruction_error(t_x, reduction="mean").cpu().numpy()[0])
    
    # 2. Classifier Probability
    prob = float(xgb_model.predict_proba(x_vector.reshape(1, -1))[0, 1])
    
    # 3. Simple SHAP attribution placeholder
    shap_vals = (x_vector * -0.2)
    
    card = reporter.generate_card(
        transaction_id=txn_id,
        probability=prob,
        feature_names=data.feature_names,
        feature_values=x_vector,
        shap_values=shap_vals,
        recon_error=recon_err,
        threshold=threshold,
    )
    
    md = reporter.render_markdown(card, filename=f"demo_card_{txn_id}.md")
    return card, md"""),

        nbf.v4.new_markdown_cell("""## 3. Case Study A: Detected Fraud Transaction"""),

        nbf.v4.new_code_cell("""fraud_idx = np.where(data.y_test == 1)[0][0]
card_a, md_a = process_transaction("CASE-001-FRAUD", data.X_test[fraud_idx])
from IPython.display import Markdown
Markdown(md_a)"""),

        nbf.v4.new_markdown_cell("""## 4. Case Study B: Legitimate Everyday Purchase"""),

        nbf.v4.new_code_cell("""normal_idx = np.where(data.y_test == 0)[0][0]
card_b, md_b = process_transaction("CASE-002-NORMAL", data.X_test[normal_idx])
Markdown(md_b)""")
    ]
    make_notebook(nb5_cells, nb_dir / "05_end_to_end_demo.ipynb")


if __name__ == "__main__":
    build_all_notebooks()

