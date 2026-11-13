"""
Model Evaluation and Comparison Suite.
Generates publication-quality PR-curves, ROC-curves, confusion matrices, and comparison tables.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from tabulate import tabulate

from src.evaluation.metrics import EvaluationResult, compute_metrics, find_optimal_threshold


class ModelEvaluator:
    def __init__(
        self,
        results_dir: str = "results",
        cost_fp: float = 10.0,
        cost_fn: float = 500.0,
    ):
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / "figures"
        self.tables_dir = self.results_dir / "tables"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

        self.cost_fp = cost_fp
        self.cost_fn = cost_fn
        self.results: List[EvaluationResult] = []

    def evaluate_model(
        self,
        name: str,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        optimize_for: str = "f2",
    ) -> EvaluationResult:
        """
        Calibrates optimal threshold and records evaluation metrics.
        """
        best_thresh, _ = find_optimal_threshold(
            y_true, y_scores, metric=optimize_for, cost_fp=self.cost_fp, cost_fn=self.cost_fn
        )
        res = compute_metrics(
            y_true=y_true,
            y_scores=y_scores,
            threshold=best_thresh,
            cost_fp=self.cost_fp,
            cost_fn=self.cost_fn,
            model_name=name,
        )
        self.results.append(res)
        return res

    def plot_pr_curves(self, predictions: Dict[str, Tuple[np.ndarray, np.ndarray]], filename: str = "pr_curves.png"):
        """
        Plots Precision-Recall curves for all compared models.
        """
        plt.figure(figsize=(9, 6), dpi=300)
        plt.title("Precision-Recall Curves (Credit Card Fraud Detection)", fontsize=14, fontweight="bold")

        for name, (y_true, y_scores) in predictions.items():
            prec, rec, _ = precision_recall_curve(y_true, y_scores)
            auprc = float(auc(rec, prec))
            plt.plot(rec, prec, lw=2.2, label=f"{name} (AUPRC = {auprc:.4f})")

        # Baseline no-skill line (fraud prevalence)
        first_y_true = next(iter(predictions.values()))[0]
        no_skill = np.sum(first_y_true == 1) / len(first_y_true)
        plt.axhline(y=no_skill, color="gray", linestyle="--", label=f"No Skill ({no_skill:.4f})")

        plt.xlabel("Recall (Detection Rate)", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.ylim([-0.02, 1.05])
        plt.xlim([-0.02, 1.02])
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="lower left", fontsize=10)
        plt.tight_layout()

        out_path = self.figures_dir / filename
        plt.savefig(out_path)
        plt.close()
        print(f"[INFO] Saved PR curve to: {out_path}")

    def plot_roc_curves(self, predictions: Dict[str, Tuple[np.ndarray, np.ndarray]], filename: str = "roc_curves.png"):
        """
        Plots ROC curves for all compared models.
        """
        plt.figure(figsize=(9, 6), dpi=300)
        plt.title("Receiver Operating Characteristic (ROC) Curves", fontsize=14, fontweight="bold")

        for name, (y_true, y_scores) in predictions.items():
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            plt.plot(fpr, tpr, lw=2.2, label=f"{name}")

        plt.plot([0, 1], [0, 1], color="navy", linestyle="--", label="Random Chance")
        plt.xlabel("False Positive Rate (FPR)", fontsize=12)
        plt.ylabel("True Positive Rate (Recall)", fontsize=12)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="lower right", fontsize=10)
        plt.tight_layout()

        out_path = self.figures_dir / filename
        plt.savefig(out_path)
        plt.close()
        print(f"[INFO] Saved ROC curve to: {out_path}")

    def plot_confusion_matrix(self, result: EvaluationResult, filename: Optional[str] = None):
        """
        Visualizes confusion matrix with counts and percentages.
        """
        cm = np.array([[result.tn, result.fp], [result.fn, result.tp]])
        plt.figure(figsize=(6, 5), dpi=300)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Predicted Legitimate", "Predicted Fraud"],
            yticklabels=["Actual Legitimate", "Actual Fraud"],
        )
        plt.title(f"Confusion Matrix: {result.model_name}\n(Threshold={result.optimal_threshold:.2f})", fontsize=12, fontweight="bold")
        plt.ylabel("Ground Truth")
        plt.xlabel("Prediction")
        plt.tight_layout()

        fname = filename or f"confusion_matrix_{result.model_name.lower().replace(' ', '_')}.png"
        out_path = self.figures_dir / fname
        plt.savefig(out_path)
        plt.close()
        print(f"[INFO] Saved confusion matrix to: {out_path}")

    def save_summary_table(self) -> pd.DataFrame:
        """
        Compiles all evaluation results into CSV, JSON, and Markdown tables.
        """
        records = []
        for r in self.results:
            records.append({
                "Model": r.model_name,
                "AUPRC": round(r.auprc, 4),
                "ROC-AUC": round(r.roc_auc, 4),
                "F1-Score": round(r.f1, 4),
                "F2-Score": round(r.f2, 4),
                "Precision": round(r.precision, 4),
                "Recall": round(r.recall, 4),
                "MCC": round(r.mcc, 4),
                "FPR": round(r.fpr, 6),
                "Optimal Threshold": round(r.optimal_threshold, 3),
                "Total Loss ($)": round(r.total_cost, 2),
            })

        df = pd.DataFrame(records).sort_values(by="AUPRC", ascending=False).reset_index(drop=True)

        # Save CSV and JSON
        df.to_csv(self.tables_dir / "model_comparison.csv", index=False)
        with open(self.tables_dir / "model_comparison.json", "w", encoding="utf-8") as f:
            f.write(df.to_json(orient="records", indent=2))

        # Save Markdown table
        md_table = tabulate(df, headers="keys", tablefmt="pipe", showindex=False)
        with open(self.tables_dir / "model_comparison.md", "w", encoding="utf-8") as f:
            f.write("# Model Performance Comparison\n\n")
            f.write(md_table + "\n")

        print("[SUCCESS] Model comparison tables successfully generated in results/tables/")
        return df
