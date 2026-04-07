import numpy as np
import pandas as pd
from typing import Any, Dict


class RuleBasedFraudDetector:

    def __init__(self) -> None:
        self.rules: list = []
        self.thresholds: Dict[str, float] = {}

    def fit(self, X: Any, y: Any) -> "RuleBasedFraudDetector":
        return self

    def predict(self, X: Any) -> np.ndarray:
        n = len(X) if hasattr(X, "__len__") else 1
        return np.zeros(n, dtype=int)

    def predict_proba(self, X: Any) -> np.ndarray:
        n = len(X) if hasattr(X, "__len__") else 1
        proba = np.zeros((n, 2))
        proba[:, 0] = 1.0
        return proba

    def predict_with_rules(self, X: Any) -> Dict[str, Any]:
        return {
            "predictions": self.predict(X).tolist(),
            "triggered_rules": [],
            "rule_scores": [],
            "rule_explanations": [],
        }
