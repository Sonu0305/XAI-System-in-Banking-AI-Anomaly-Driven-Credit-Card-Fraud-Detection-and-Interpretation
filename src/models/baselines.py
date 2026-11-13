"""
Baseline models for credit card fraud benchmark comparison.
Includes Isolation Forest, Logistic Regression, XGBoost, LightGBM, and CatBoost.
"""

from typing import Any, Dict, Optional
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier


class BaselineModel:
    def __init__(self, name: str):
        self.name = name
        self.model = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "BaselineModel":
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class IsolationForestBaseline(BaselineModel):
    def __init__(self, contamination: float = 0.0017, random_state: int = 42, n_estimators: int = 150):
        super().__init__("IsolationForest")
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=n_estimators,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "IsolationForestBaseline":
        self.model.fit(X)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Invert score_samples: more negative score = more anomalous
        scores = -self.model.score_samples(X)
        min_s, max_s = scores.min(), scores.max()
        norm_scores = (scores - min_s) / (max_s - min_s + 1e-8)
        return np.column_stack([1.0 - norm_scores, norm_scores])


class LogisticRegressionBaseline(BaselineModel):
    def __init__(self, random_state: int = 42, max_iter: int = 1000):
        super().__init__("LogisticRegression")
        self.model = LogisticRegression(
            class_weight="balanced",
            max_iter=max_iter,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionBaseline":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class XGBoostBaseline(BaselineModel):
    def __init__(
        self,
        n_estimators: int = 250,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        scale_pos_weight: float = 1.0,
        random_state: int = 42,
    ):
        super().__init__("XGBoost")
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, eval_set: Optional[list] = None) -> "XGBoostBaseline":
        self.model.fit(X, y, eval_set=eval_set, verbose=False)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class LightGBMBaseline(BaselineModel):
    def __init__(
        self,
        n_estimators: int = 250,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        scale_pos_weight: float = 1.0,
        random_state: int = 42,
    ):
        super().__init__("LightGBM")
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LightGBMBaseline":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class CatBoostBaseline(BaselineModel):
    def __init__(
        self,
        iterations: int = 250,
        depth: int = 6,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ):
        super().__init__("CatBoost")
        self.model = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            random_seed=random_state,
            auto_class_weights="Balanced",
            verbose=False,
            thread_count=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CatBoostBaseline":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
