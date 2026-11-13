"""
Publication-quality plotting suite for model evaluation and financial impact analysis.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

# Matplotlib styling for high-clarity academic reports
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plots formatted normalized and raw confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
    annot = np.empty_like(cm).astype(str)
    for i in range(2):
        for j in range(2):
            annot[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j]:.1%})"

    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["Normal", "Fraud"],
        yticklabels=["Normal", "Fraud"],
        ax=ax,
    )
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Label", fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_precision_recall_curves(
    models_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],  # name -> (y_true, y_prob)
    title: str = "Precision-Recall Curves (Primary Imbalance Metric)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plots PR curves for multiple models with AUPRC legend."""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    for name, (y_true, y_prob) in models_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        from sklearn.metrics import average_precision_score
        auprc = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, lw=2, label=f"{name} (AUPRC = {auprc:.4f})")

    # Baseline no-skill line (proportion of positives)
    first_y = next(iter(models_dict.values()))[0]
    pos_ratio = np.mean(first_y)
    ax.axhline(y=pos_ratio, color="grey", linestyle="--", label=f"No Skill ({pos_ratio:.4f})")

    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel("Recall", fontweight="bold")
    ax.set_ylabel("Precision", fontweight="bold")
    ax.set_xlim([0.0, 1.02])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower left", frameon=True)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_roc_curves(
    models_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str = "Receiver Operating Characteristic (ROC) Curves",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plots ROC curves for multiple models."""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    for name, (y_true, y_prob) in models_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        from sklearn.metrics import roc_auc_score
        roc_auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Random Chance")
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate", fontweight="bold")
    ax.set_ylabel("True Positive Rate (Recall)", fontweight="bold")
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_cost_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fp: float = 10.0,
    cost_fn: float = 500.0,
    title: str = "Expected Banking Loss vs Decision Threshold",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plots expected total financial cost across varying decision thresholds."""
    from src.evaluation.metrics import compute_metrics, find_optimal_threshold
    thresholds = np.linspace(0.01, 0.99, 100)
    costs = [compute_metrics(y_true, y_prob, t, cost_fp, cost_fn)["Total_Cost"] for t in thresholds]

    best_thresh, min_cost = find_optimal_threshold(y_true, y_prob, metric="cost", cost_fp=cost_fp, cost_fn=cost_fn)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax.plot(thresholds, costs, color="crimson", lw=2.5, label="Total Financial Cost ($)")
    ax.axvline(best_thresh, color="darkgreen", linestyle="--", lw=2, label=f"Optimal Threshold = {best_thresh:.2f} (${min_cost:,.0f})")
    ax.axvline(0.50, color="navy", linestyle=":", lw=1.5, label="Default Threshold (0.50)")

    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel("Decision Threshold", fontweight="bold")
    ax.set_ylabel("Estimated Financial Loss ($)", fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_anomaly_distribution(
    normal_errors: np.ndarray,
    fraud_errors: np.ndarray,
    threshold: float,
    title: str = "Autoencoder Reconstruction Error Separation",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plots logarithmic reconstruction error distribution separating legitimate and fraud."""
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

    # Subsample normal errors for visualization density
    if len(normal_errors) > 5000:
        normal_sub = np.random.choice(normal_errors, size=5000, replace=False)
    else:
        normal_sub = normal_errors

    log_norm = np.log10(np.maximum(normal_sub, 1e-6))
    log_fraud = np.log10(np.maximum(fraud_errors, 1e-6))
    log_thresh = np.log10(max(threshold, 1e-6))

    sns.kdeplot(log_norm, ax=ax, label="Legitimate (Normal)", color="dodgerblue", fill=True, alpha=0.3)
    sns.kdeplot(log_fraud, ax=ax, label="Fraudulent", color="crimson", fill=True, alpha=0.4)
    ax.axvline(log_thresh, color="black", linestyle="--", lw=2, label=f"Anomaly Threshold ({threshold:.3f})")

    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel("Log10(Reconstruction Error)", fontweight="bold")
    ax.set_ylabel("Density", fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig
