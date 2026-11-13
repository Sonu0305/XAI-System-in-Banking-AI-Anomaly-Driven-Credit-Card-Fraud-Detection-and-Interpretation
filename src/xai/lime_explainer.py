"""
LIME (Local Interpretable Model-agnostic Explanations) for Credit Card Fraud alerts.
Generates local tabular surrogate linear models and produces analyst-ready contribution charts.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np


class FraudLIMEExplainer:
    def __init__(
        self,
        training_data: np.ndarray,
        feature_names: List[str],
        class_names: List[str] = ("Legitimate", "Fraud"),
        output_dir: str = "results/explanations",
        random_state: int = 42,
    ):
        self.feature_names = feature_names
        self.class_names = list(class_names)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=training_data,
            feature_names=feature_names,
            class_names=self.class_names,
            mode="classification",
            random_state=random_state,
        )

    def explain_instance(
        self,
        instance: np.ndarray,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        num_features: int = 10,
        sample_idx: int = 0,
        save_plot: bool = True,
        save_html: bool = True,
    ) -> Dict[str, Any]:
        """
        Explains single transaction prediction using local surrogate model.
        Returns dictionary of top contributing rules and weights.
        """
        exp = self.explainer.explain_instance(
            data_row=instance,
            predict_fn=predict_fn,
            num_features=num_features,
            top_labels=1,
        )

        label_to_explain = exp.available_labels()[0]
        explanation_list = exp.as_list(label=label_to_explain)

        if save_html:
            html_path = self.output_dir / f"lime_sample_{sample_idx}.html"
            exp.save_to_file(str(html_path))
            print(f"[INFO] Saved LIME HTML explanation to: {html_path}")

        if save_plot:
            features = [x[0] for x in explanation_list]
            weights = [x[1] for x in explanation_list]
            colors = ["#d9534f" if w > 0 else "#5cb85c" for w in weights]  # Red for fraud, Green for safe

            plt.figure(figsize=(9, 5), dpi=300)
            y_pos = np.arange(len(features))
            plt.barh(y_pos, weights, color=colors, align="center")
            plt.yticks(y_pos, features, fontsize=9)
            plt.xlabel("LIME Feature Weight (Contribution to Fraud Score)", fontsize=11)
            plt.title(
                f"LIME Local Explanation: Transaction #{sample_idx} ({self.class_names[label_to_explain]})",
                fontsize=12,
                fontweight="bold",
            )
            plt.gca().invert_yaxis()
            plt.grid(True, linestyle=":", alpha=0.5)
            plt.tight_layout()

            plot_path = self.output_dir / f"lime_sample_{sample_idx}.png"
            plt.savefig(plot_path)
            plt.close()
            print(f"[INFO] Saved LIME plot to: {plot_path}")

        return {
            "sample_idx": sample_idx,
            "explained_label": self.class_names[label_to_explain],
            "rules": explanation_list,
        }
