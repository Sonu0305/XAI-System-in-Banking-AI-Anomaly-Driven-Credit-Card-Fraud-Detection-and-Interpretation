"""
Human-Readable Banking Analyst Fraud Alert Report Generator.
Transforms complex SHAP, LIME, and Autoencoder outputs into actionable, audit-compliant banking alert cards.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import numpy as np


@dataclass
class AnalystAlertCard:
    transaction_id: str
    risk_score: float
    risk_tier: str
    autoencoder_recon_error: float
    autoencoder_threshold: float
    is_anomaly: bool
    top_red_flags: List[Dict[str, Any]]
    top_mitigating_factors: List[Dict[str, Any]]
    recommended_action: str
    compliance_summary: str


class BankingAnalystReporter:
    def __init__(self, output_dir: str = "results/explanations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def determine_risk_tier(self, prob: float, is_anomaly: bool) -> str:
        if prob >= 0.80 or (prob >= 0.60 and is_anomaly):
            return "CRITICAL RISK"
        elif prob >= 0.50:
            return "HIGH RISK"
        elif prob >= 0.20:
            return "MODERATE SUSPICION"
        return "LOW RISK (LEGITIMATE)"

    def determine_recommendation(self, tier: str, is_anomaly: bool) -> str:
        if tier == "CRITICAL RISK":
            return (
                "IMMEDIATE ACTION REQUIRED: Block transaction immediately, place temporary hold "
                "on credit card account, and dispatch instant SMS/App 2-factor identity verification prompt."
            )
        elif tier == "HIGH RISK":
            return (
                "HOLD & REVIEW: Place transaction on pending hold. Route alert to Tier-2 Fraud "
                "Operations queue for manual callback verification within 15 minutes."
            )
        elif tier == "MODERATE SUSPICION":
            return (
                "PASS WITH MONITORING: Allow transaction but increment merchant-category velocity "
                "counter and log risk telemetry for subsequent transactions."
            )
        return "APPROVE: Transaction parameters align with standard legitimate customer behavior."

    def generate_card(
        self,
        transaction_id: str,
        probability: float,
        feature_names: List[str],
        feature_values: np.ndarray,
        shap_values: np.ndarray,
        recon_error: float,
        threshold: float,
    ) -> AnalystAlertCard:
        """
        Synthesizes model probabilities, Autoencoder reconstruction, and SHAP values into an Analyst Card.
        """
        is_anomaly = bool(recon_error >= threshold)
        tier = self.determine_risk_tier(probability, is_anomaly)
        rec = self.determine_recommendation(tier, is_anomaly)

        # Sort features by SHAP impact
        paired = []
        for name, val, s in zip(feature_names, feature_values, shap_values):
            paired.append({"feature": name, "value": float(val), "shap_impact": float(s)})

        red_flags = [p for p in sorted(paired, key=lambda x: x["shap_impact"], reverse=True) if p["shap_impact"] > 0][:4]
        mitigating = [p for p in sorted(paired, key=lambda x: x["shap_impact"]) if p["shap_impact"] < 0][:3]

        ratio = (recon_error / max(1e-6, threshold))
        compliance_text = (
            f"Transaction evaluated by XAI Hybrid Banking Pipeline. Supervised fraud probability: "
            f"{probability * 100:.2f}%. Autoencoder reconstruction residual is {ratio:.2f}x of the "
            f"normal transaction boundary. Key risk drivers identified via SHAP Shapley values."
        )

        return AnalystAlertCard(
            transaction_id=transaction_id,
            risk_score=float(probability),
            risk_tier=tier,
            autoencoder_recon_error=float(recon_error),
            autoencoder_threshold=float(threshold),
            is_anomaly=is_anomaly,
            top_red_flags=red_flags,
            top_mitigating_factors=mitigating,
            recommended_action=rec,
            compliance_summary=compliance_text,
        )

    def render_markdown(self, card: AnalystAlertCard, filename: Optional[str] = None) -> str:
        """
        Formats alert card into a Markdown briefing document.
        """
        anomaly_status = "⚠️ ANOMALOUS (High Reconstruction Error)" if card.is_anomaly else "✅ NORMAL PATTERN"
        badge = "🔴" if "CRITICAL" in card.risk_tier else ("🟠" if "HIGH" in card.risk_tier else "🟢")

        lines = [
            f"# {badge} Bank Security Operations Alert Card: Transaction #{card.transaction_id}",
            f"",
            f"**Audit Status:** Complete | **Model System:** AI Anomaly-Driven Credit Fraud Engine",
            f"",
            f"---",
            f"### Executive Summary",
            f"- **Fraud Risk Verdict:** `{card.risk_tier}`",
            f"- **Estimated Fraud Probability:** **`{card.risk_score * 100:.2f}%`**",
            f"- **Unsupervised Anomaly State:** {anomaly_status} (Reconstruction MSE: `{card.autoencoder_recon_error:.5f}` vs Threshold: `{card.autoencoder_threshold:.5f}`)",
            f"",
            f"---",
            f"### Root Cause Analysis (XAI Feature Attribution)",
            f"#### 🚨 Primary Red Flags (Driving Risk UP):",
        ]

        for i, rf in enumerate(card.top_red_flags, 1):
            lines.append(
                f"{i}. **`{rf['feature']}`** (Normalized Value: `{rf['value']:.3f}`) → SHAP Impact: `+{rf['shap_impact']:.4f}`"
            )

        if card.top_mitigating_factors:
            lines.append(f"\n#### 🛡️ Mitigating Factors (Driving Risk DOWN):")
            for i, mf in enumerate(card.top_mitigating_factors, 1):
                lines.append(
                    f"{i}. **`{mf['feature']}`** (Normalized Value: `{mf['value']:.3f}`) → SHAP Impact: `{mf['shap_impact']:.4f}`"
                )

        lines.extend([
            f"",
            f"---",
            f"### Recommended Banking Protocol",
            f"> {card.recommended_action}",
            f"",
            f"### Regulatory & Compliance Audit Trail",
            f"```text",
            f"{card.compliance_summary}",
            f"```",
        ])

        md_content = "\n".join(lines)
        fname = filename or f"alert_card_sample_{card.transaction_id}.md"
        out_path = self.output_dir / fname
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"[INFO] Saved analyst alert card to: {out_path}")
        return md_content
