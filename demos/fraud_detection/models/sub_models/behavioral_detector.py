import numpy as np
from typing import Any
from sklearn.linear_model import LogisticRegression


class BehavioralFraudDetector:

    def __init__(self, min_transactions_for_profile: int = 5, anomaly_threshold: float = 0.7, **kwargs: Any) -> None:
        self.min_transactions_for_profile = min_transactions_for_profile
        self.anomaly_threshold = anomaly_threshold
        self._model = LogisticRegression(random_state=42, max_iter=500)
        self._is_fitted = False

    def fit(self, X: Any, y: Any) -> "BehavioralFraudDetector":
        import pandas as pd
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        self._model.fit(X_arr, y)
        self._is_fitted = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        import pandas as pd
        if not self._is_fitted:
            n = len(X) if hasattr(X, "__len__") else 1
            return np.zeros(n, dtype=int)
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        return self._model.predict(X_arr)

    def predict_proba(self, X: Any) -> np.ndarray:
        import pandas as pd
        if not self._is_fitted:
            n = len(X) if hasattr(X, "__len__") else 1
            proba = np.zeros((n, 2))
            proba[:, 0] = 1.0
            return proba
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        return self._model.predict_proba(X_arr)
