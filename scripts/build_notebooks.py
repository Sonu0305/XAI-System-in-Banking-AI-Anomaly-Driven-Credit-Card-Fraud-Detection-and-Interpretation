"""
Generator and Executor for all 5 project Jupyter notebooks.
Builds clean, fully structured, publication-quality .ipynb files with complete execution outputs:
01_eda.ipynb
02_anomaly_detection.ipynb
03_supervised_training.ipynb
04_xai_explanations.ipynb
05_end_to_end_demo.ipynb
"""

import sys
import time
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

ROOT_DIR = Path(__file__).resolve().parent.parent


def make_notebook(cells, filepath: Path, execute: bool = True):
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
    print(f"[CREATED] Notebook template: {filepath}")

    if execute:
        print(f"[EXECUTING] Running cells for {filepath.name}...")
        client = NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(filepath.parent)}})
        t0 = time.time()
        client.execute()
        print(f"[SUCCESS] Executed {filepath.name} in {time.time() - t0:.2f}s!")
        with open(filepath, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print(f"[SAVED] Executed notebook: {filepath} ({filepath.stat().st_size / 1024:.1f} KB)")


def build_all_notebooks(notebooks_dir: str = "notebooks", execute: bool = True):
    nb_dir = ROOT_DIR / notebooks_dir
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

ROOT_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT_DIR))

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
sns.histplot(df[df["Class"] == 0]["Amount"], bins=50, ax=axes[0], color="#2b5c8f", kde=True)
axes[0].set_title("Legitimate Amount Distribution (Capped at 99th Pct)")
axes[0].set_xlim(0, df[df["Class"] == 0]["Amount"].quantile(0.99))

sns.histplot(df[df["Class"] == 1]["Amount"], bins=50, ax=axes[1], color="#d9534f", kde=True)
axes[1].set_title("Fraudulent Amount Distribution")
plt.tight_layout()
plt.show()

# Log amount transform
df["Amount_Log"] = np.log1p(df["Amount"])
fig, ax = plt.subplots(figsize=(7, 4))
sns.kdeplot(data=df, x="Amount_Log", hue="Class", common_norm=False, palette=["#2b5c8f", "#d9534f"], fill=True, ax=ax)
ax.set_title("Log-Transformed Transaction Amount Density")
plt.tight_layout()
plt.show()"""),

        nbf.v4.new_markdown_cell("""## 4. Time-Domain Periodicity & Hourly Cyclic Encoding
Transactions span 48 consecutive hours. Legitimate activity follows diurnal cycles, whereas automated bot attacks frequently strike during off-peak night hours."""),

        nbf.v4.new_code_cell("""df["Hour"] = (df["Time"] // 3600) % 24
fig, ax = plt.subplots(figsize=(10, 4))
sns.histplot(data=df, x="Hour", hue="Class", multiple="dodge", shrink=0.8, bins=24, palette=["#2b5c8f", "#d9534f"], ax=ax)
ax.set_yscale("log")
ax.set_title("Transactions by Hour of Day (0–23) on Log Scale")
plt.tight_layout()
plt.show()"""),

        nbf.v4.new_markdown_cell("""## 5. PCA Feature Correlation Spectrum with Fraud Label
We evaluate which anonymized features exhibit strongest correlation with `Class`."""),

        nbf.v4.new_code_cell("""correlations = df.corr()["Class"].drop(["Class", "Amount_Log"]).sort_values()
top_neg = correlations.head(5)
top_pos = correlations.tail(5)
top_corr = pd.concat([top_neg, top_pos])

fig, ax = plt.subplots(figsize=(10, 4))
colors = ["#2b5c8f" if c < 0 else "#d9534f" for c in top_corr.values]
ax.bar(top_corr.index, top_corr.values, color=colors)
ax.set_title("Top 10 Positively & Negatively Correlated Features with Class")
ax.set_ylabel("Pearson Correlation")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()""")
    ]
    make_notebook(nb1_cells, nb_dir / "01_eda.ipynb", execute=execute)

    # -------------------------------------------------------------
    # 02_anomaly_detection.ipynb
    # -------------------------------------------------------------
    nb2_cells = [
        nbf.v4.new_markdown_cell("""# 🔍 02. Unsupervised Anomaly Detection: Deep Autoencoder
### Project: XAI System in Banking: AI Anomaly-Driven Credit Card Fraud Detection and Interpretation
**Objective:** Train an unsupervised Deep Autoencoder strictly on legitimate banking transactions to learn normal manifold dynamics, compute reconstruction error residuals, and establish dynamic anomaly thresholds."""),

        nbf.v4.new_code_cell("""import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

ROOT_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT_DIR))

from src.data.preprocessing import DataPipeline
from src.models.autoencoder import DeepAutoencoder
from src.utils.device import get_device

device = get_device("auto")
print(f"Device: {device}")"""),

        nbf.v4.new_markdown_cell("""## 1. Load Preprocessed Zero-Leakage Dataset splits"""),

        nbf.v4.new_code_cell("""pipeline = DataPipeline(processed_dir=str(ROOT_DIR / "data/processed"))
data = pipeline.load_processed()

input_dim = data.X_train.shape[1]
print(f"Features: {input_dim}")
print(f"Normal training samples: {len(data.X_train_normal):,}")
print(f"Test samples: {len(data.X_test):,} (Fraud: {data.y_test.sum():,})")"""),

        nbf.v4.new_markdown_cell("""## 2. Load Production Autoencoder Checkpoint & Calibrated Threshold"""),

        nbf.v4.new_code_cell("""model_path = ROOT_DIR / "results/models/autoencoder.pt"
autoencoder = DeepAutoencoder(input_dim=input_dim, latent_dim=8).to(device)

if model_path.exists():
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    autoencoder.load_state_dict(state)
    threshold = ckpt.get("threshold", 1.5048)
    print(f"Loaded trained Autoencoder from {model_path}")
    print(f"Calibrated 98th Percentile Anomaly Threshold: {threshold:.6f}")
else:
    print("Pretrained checkpoint not found; please execute python scripts/train.py first.")"""),

        nbf.v4.new_markdown_cell("""## 3. Reconstruction Error Evaluation on Held-out Test Set"""),

        nbf.v4.new_code_cell("""autoencoder.eval()
with torch.no_grad():
    test_tensor = torch.as_tensor(data.X_test, dtype=torch.float32, device=device)
    recon_errors = autoencoder.compute_reconstruction_error(test_tensor, reduction="mean").cpu().numpy()

normal_errors = recon_errors[data.y_test == 0]
fraud_errors = recon_errors[data.y_test == 1]

fig, ax = plt.subplots(figsize=(8, 4.5))
sns.kdeplot(np.log1p(normal_errors), label="Normal Transactions (0)", fill=True, color="#2b5c8f", ax=ax)
sns.kdeplot(np.log1p(fraud_errors), label="Fraudulent Transactions (1)", fill=True, color="#d9534f", ax=ax)
ax.axvline(np.log1p(threshold), color="red", linestyle="--", linewidth=2, label=f"Threshold (τ = {threshold:.4f})")
ax.set_title("Autoencoder Reconstruction Error Density (Log Scale)")
ax.set_xlabel("log(1 + MSE Reconstruction Error)")
ax.legend()
plt.tight_layout()
plt.show()"""),

        nbf.v4.new_markdown_cell("""## 4. Latent Space Representation Visualization (t-SNE / PCA Projection)"""),

        nbf.v4.new_code_cell("""from sklearn.decomposition import PCA

with torch.no_grad():
    latent_z = autoencoder.encode(test_tensor).cpu().numpy()

pca = PCA(n_components=2)
z_2d = pca.fit_transform(latent_z)

fig, ax = plt.subplots(figsize=(7, 5))
norm_mask = (data.y_test == 0)
ax.scatter(z_2d[norm_mask, 0][:3000], z_2d[norm_mask, 1][:3000], c="#2b5c8f", alpha=0.3, s=10, label="Normal")
ax.scatter(z_2d[~norm_mask, 0], z_2d[~norm_mask, 1], c="#d9534f", alpha=0.9, s=25, label="Fraud")
ax.set_title("2D PCA Projection of Autoencoder Latent Space (Z ∈ ℝ⁸)")
ax.legend()
plt.tight_layout()
plt.show()""")
    ]
    make_notebook(nb2_cells, nb_dir / "02_anomaly_detection.ipynb", execute=execute)

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
import seaborn as sns
import joblib, torch
from IPython.display import Image, display

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
    df_view = df_results[["Model", "AUPRC", "ROC-AUC", "F1-Score", "F2-Score", "Recall", "Precision", "Total Loss ($)"]]
    print(df_view.to_string(index=False))
else:
    print("Model comparison table not found. Run python scripts/evaluate.py to generate benchmarks.")"""),

        nbf.v4.new_markdown_cell("""## 2. Precision-Recall Curves
Precision-Recall curves are the gold standard for severely imbalanced fraud benchmarks."""),

        nbf.v4.new_code_cell("""pr_img_path = ROOT_DIR / "results/figures/pr_curves.png"
if pr_img_path.exists():
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
    make_notebook(nb3_cells, nb_dir / "03_supervised_training.ipynb", execute=execute)

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
from IPython.display import Image, display

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
    make_notebook(nb4_cells, nb_dir / "04_xai_explanations.ipynb", execute=execute)

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
import joblib, torch, shap
from IPython.display import Markdown, display

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
threshold = ae_ckpt.get("threshold", 1.5048)
reporter = BankingAnalystReporter(output_dir=str(ROOT_DIR / "results/explanations"))
explainer = shap.TreeExplainer(getattr(xgb_model, "model", xgb_model))
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
    
    # 3. SHAP attribution
    shap_vals = explainer(x_vector.reshape(1, -1)).values[0]
    
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
display(Markdown(md_a))"""),

        nbf.v4.new_markdown_cell("""## 4. Case Study B: Legitimate Everyday Purchase"""),

        nbf.v4.new_code_cell("""normal_idx = np.where(data.y_test == 0)[0][0]
card_b, md_b = process_transaction("CASE-002-NORMAL", data.X_test[normal_idx])
display(Markdown(md_b))""")
    ]
    make_notebook(nb5_cells, nb_dir / "05_end_to_end_demo.ipynb", execute=execute)


if __name__ == "__main__":
    build_all_notebooks(execute=True)
