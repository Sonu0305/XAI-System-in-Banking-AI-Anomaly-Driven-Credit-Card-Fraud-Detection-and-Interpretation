"""
Automated Bank Fraud Analyst Alert Dossier Generator.
Transforms mathematical XAI outputs into human-readable banking investigation reports.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class TransactionAlert:
    transaction_id: str
    fraud_probability: float
    is_anomaly: bool
    reconstruction_error: float
    anomaly_threshold: float
    top_risk_factors: List[Tuple[str, float, float]]  # (feature, value, contribution)
    mitigating_factors: List[Tuple[str, float, float]]
    original_amount: Optional[float] = None
    transaction_hour: Optional[float] = None


class BankAnalystReportGenerator:
    """
    Generates structured banking intelligence cards and compliance dossiers.
    """
    def __init__(self, output_dir: str = "results/explanations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def determine_risk_tier(self, prob: float) -> Tuple[str, str]:
        if prob >= 0.85:
            return "CRITICAL RISK", "IMMEDIATE CARD FREEZE & OUTBOUND CONTACT"
        elif prob >= 0.50:
            return "HIGH RISK", "CHALLENGE VIA 2-FACTOR AUTH / TEMPORARY HOLD"
        elif prob >= 0.20:
            return "MEDIUM RISK", "PASSIVE MONITORING & STEP-UP AT NEXT HIGH-VALUE TX"
        else:
            return "LOW RISK", "AUTO-APPROVE TRANSACTION"

    def format_alert_card(self, alert: TransactionAlert) -> str:
        tier, action = self.determine_risk_tier(alert.fraud_probability)
        anomaly_str = "ANOMALOUS (High Deviation)" if alert.is_anomaly else "NORMAL MANIFOLD"

        lines = [
            "=" * 78,
            f"          FINANCIAL INSTITUTION FRAUD ALERT DOSSIER",
            f"          Case ID: TX-{alert.transaction_id}  |  Status: UNDER REVIEW",
            "=" * 78,
            f"  RISK ASSESSMENT:",
            f"    * Fraud Probability Score: {alert.fraud_probability * 100:.2f}%",
            f"    * Assigned Threat Tier:   {tier}",
            f"    * Recommended Protocol:   {action}",
            "",
            f"  ANOMALY DETECTION STAGE (Autoencoder Engine):",
            f"    * Manifold Status:         {anomaly_str}",
            f"    * Reconstruction Error:    {alert.reconstruction_error:.4f} (Threshold: {alert.anomaly_threshold:.4f})",
            f"    * Deviation Multiplier:    {alert.reconstruction_error / max(alert.anomaly_threshold, 1e-6):.2f}x Baseline",
            "",
            f"  PRIMARY RISK DRIVERS (Features Pushing Score Toward FRAUD):",
        ]

        if alert.top_risk_factors:
            for feat, val, contrib in alert.top_risk_factors:
                lines.append(f"    - [{feat:<12}] Value: {val:>8.3f} | Risk Contribution: +{contrib:>6.3f} SHAP")
        else:
            lines.append("    - No severe anomalous drivers detected.")

        lines.extend([
            "",
            f"  MITIGATING FACTORS (Features Decreasing Risk Score):",
        ])

        if alert.mitigating_factors:
            for feat, val, contrib in alert.mitigating_factors:
                lines.append(f"    - [{feat:<12}] Value: {val:>8.3f} | Mitigating Effect:  {contrib:>6.3f} SHAP")
        else:
            lines.append("    - No significant mitigating factors.")

        lines.extend([
            "",
            f"  ANALYST EXPLANATION SUMMARY:",
            f"    The transaction exhibits statistical deviations in behavioral features",
            f"    consistently correlated with synthetic identity fraud and credential stuffing.",
            f"    The Autoencoder reconstruction error indicates high deviation from normal",
            f"    account holder spending habits.",
            "=" * 78,
            "",
        ])
        return "\n".join(lines)

    def save_alert_card(self, alert: TransactionAlert, filename: Optional[str] = None) -> Path:
        if filename is None:
            filename = f"analyst_alert_TX_{alert.transaction_id}.txt"
        out_path = self.output_dir / filename
        card_text = self.format_alert_card(alert)
        out_path.write_text(card_text, encoding="utf-8")
        return out_path
