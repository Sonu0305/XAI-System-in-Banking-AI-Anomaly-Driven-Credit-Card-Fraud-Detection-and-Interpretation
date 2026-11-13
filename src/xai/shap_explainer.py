"""
SHAP (SHapley Additive exPlanations) for Global and Local Fraud Interpretability.
Supports TreeExplainer for boosted trees and Kernel/Deep Explainer for PyTorch networks.
Generates publication-quality summary, beeswarm, waterfall, and dependence plots.
"""

from pathlib import Path
from typing import Any, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import shap


class FraudSHAPExplainer:
    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        background_data: np.ndarray,
        model_type: str = "tree",  # 'tree' or 'pytorch'
        output_dir: str = "results/explanations",
    ):
        self.model = model
        self.feature_names = feature_names
        self.background_data = background_data
        self.model_type = model_type
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.model_type == "tree":
            # Underlying model instance for TreeExplainer
            underlying = getattr(model, "model", model)
            self.explainer = shap.TreeExplainer(underlying)
        else:
            # Kernel/Sampling explainer for PyTorch functions
            def predict_fn(x):
                return self.model.predict_proba(x)[:, 1]

            # Use compact background to keep runtime snappy
            bg_subset = shap.sample(background_data, min(100, len(background_data)))
            self.explainer = shap.KernelExplainer(predict_fn, bg_subset)

    def explain(self, X: np.ndarray) -> shap.Explanation:
        """Computes SHAP values for given dataset X."""
        shap_values = self.explainer(X)
        # Ensure feature names are aligned
        if hasattr(shap_values, "feature_names"):
            shap_values.feature_names = self.feature_names
        return shap_values

    def plot_summary(
        self,
        X: np.ndarray,
        shap_values: Optional[Any] = None,
        max_display: int = 15,
        filename: str = "shap_summary_bar.png",
    ):
        """Generates global feature importance bar plot."""
        if shap_values is None:
            shap_values = self.explain(X)

        plt.figure(figsize=(10, 6), dpi=300)
        # If binary classification returns 3D or list, extract fraud class
        vals = shap_values.values if hasattr(shap_values, "values") else shap_values
        if isinstance(vals, list):
            vals = vals[1]
        elif len(vals.shape) == 3:
            vals = vals[:, :, 1]

        shap.summary_plot(
            vals,
            X,
            feature_names=self.feature_names,
            plot_type="bar",
            max_display=max_display,
            show=False,
        )
        plt.title("Global Feature Importance (SHAP Mean |Value|)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        out_path = self.output_dir / filename
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Saved SHAP summary bar to: {out_path}")

    def plot_beeswarm(
        self,
        X: np.ndarray,
        shap_values: Optional[Any] = None,
        max_display: int = 15,
        filename: str = "shap_beeswarm.png",
    ):
        """Generates SHAP beeswarm plot displaying directional influence."""
        if shap_values is None:
            shap_values = self.explain(X)

        vals = shap_values.values if hasattr(shap_values, "values") else shap_values
        if isinstance(vals, list):
            vals = vals[1]
        elif len(vals.shape) == 3:
            vals = vals[:, :, 1]

        plt.figure(figsize=(10, 6), dpi=300)
        shap.summary_plot(
            vals,
            X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False,
        )
        plt.title("SHAP Beeswarm: Impact of Features on Fraud Probability", fontsize=13, fontweight="bold")
        plt.tight_layout()
        out_path = self.output_dir / filename
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Saved SHAP beeswarm to: {out_path}")

    def plot_waterfall(
        self,
        sample_x: np.ndarray,
        sample_idx: int = 0,
        filename: str = "shap_waterfall_fraud.png",
    ):
        """Generates local waterfall explanation for a single transaction."""
        explanation = self.explainer(sample_x.reshape(1, -1))
        # Handle binary classification shapes
        single_exp = explanation[0]
        if len(single_exp.values.shape) > 1 and single_exp.values.shape[-1] == 2:
            single_exp = single_exp[:, 1]

        plt.figure(figsize=(9, 6), dpi=300)
        shap.plots.waterfall(single_exp, max_display=12, show=False)
        plt.title(f"Local Decision Rationale: Transaction #{sample_idx}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        out_path = self.output_dir / filename
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Saved SHAP waterfall to: {out_path}")

    def plot_dependence(
        self,
        feature_name: str,
        X: np.ndarray,
        shap_values: Optional[Any] = None,
        filename: Optional[str] = None,
    ):
        """Plots SHAP dependence for a specific feature."""
        if shap_values is None:
            shap_values = self.explain(X)

        vals = shap_values.values if hasattr(shap_values, "values") else shap_values
        if isinstance(vals, list):
            vals = vals[1]
        elif len(vals.shape) == 3:
            vals = vals[:, :, 1]

        feat_idx = self.feature_names.index(feature_name) if feature_name in self.feature_names else 0
        plt.figure(figsize=(8, 5), dpi=300)
        shap.dependence_plot(
            feat_idx,
            vals,
            X,
            feature_names=self.feature_names,
            show=False,
        )
        plt.title(f"SHAP Dependence: {feature_name}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        fname = filename or f"shap_dependence_{feature_name.lower()}.png"
        out_path = self.output_dir / fname
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Saved SHAP dependence to: {out_path}")
