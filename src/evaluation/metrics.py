"""
Comprehensive metrics suite for imbalanced credit card fraud detection.
Includes AUPRC, ROC-AUC, F1, F2 (recall-weighted), MCC, FPR, and banking financial cost.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class EvaluationResult:
    model_name: str
    auprc: float
    roc_auc: float
    f1: float
    f2: float
    precision: float
    recall: float
    mcc: float
    fpr: float
    tp: int
    fp: int
    tn: int
    fn: int
    optimal_threshold: float
    total_cost: float


def compute_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float = 0.5,
    cost_fp: float = 10.0,
    cost_fn: float = 500.0,
    model_name: str = "Model",
) -> EvaluationResult:
    """
    Computes standard and cost-sensitive evaluation metrics for binary fraud classification.
    """
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)
    y_pred = (y_scores >= threshold).astype(int)

    auprc = float(average_precision_score(y_true, y_scores))
    roc_auc = float(roc_auc_score(y_true, y_scores))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    f2 = float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / max(1, (fp + tn)))
    total_cost = float(fp * cost_fp + fn * cost_fn)

    return EvaluationResult(
        model_name=model_name,
        auprc=auprc,
        roc_auc=roc_auc,
        f1=f1,
        f2=f2,
        precision=prec,
        recall=rec,
        mcc=mcc,
        fpr=fpr,
        tp=int(tp),
        fp=int(fp),
        tn=int(tn),
        fn=int(fn),
        optimal_threshold=float(threshold),
        total_cost=total_cost,
    )


def find_optimal_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    metric: str = "f2",
    cost_fp: float = 10.0,
    cost_fn: float = 500.0,
    num_steps: int = 200,
) -> Tuple[float, float]:
    """
    Finds the probability threshold that maximizes F1, F2, or minimizes financial cost.
    Returns: (best_threshold, best_score)
    """
    thresholds = np.linspace(0.01, 0.99, num_steps)
    best_thresh = 0.5
    best_val = -1.0 if metric in ["f1", "f2"] else float("inf")

    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
            if score > best_val:
                best_val = score
                best_thresh = thresh
        elif metric == "f2":
            score = fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)
            if score > best_val:
                best_val = score
                best_thresh = thresh
        elif metric == "cost":
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            cost = fp * cost_fp + fn * cost_fn
            if cost < best_val:
                best_val = cost
                best_thresh = thresh

    return float(best_thresh), float(best_val)
