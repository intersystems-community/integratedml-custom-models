import numpy as np
from typing import Any
from sklearn.ensemble import IsolationForest


class AnomalyFraudDetector:

    def __init__(self, contamination: float = 0.02, enable_iris_vector_search: bool = False, **kwargs: Any) -> None:
        self.contamination = contamination
        self.enable_iris_vector_search = enable_iris_vector_search
        self._model = IsolationForest(contamination=contamination, random_state=42)
        self._is_fitted = False

    def fit(self, X: Any, y: Any = None) -> "AnomalyFraudDetector":
        import pandas as pd
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        self._model.fit(X_arr)
        self._is_fitted = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        import pandas as pd
        if not self._is_fitted:
            n = len(X) if hasattr(X, "__len__") else 1
            return np.zeros(n, dtype=int)
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        raw = self._model.predict(X_arr)
        return np.where(raw == -1, 1, 0).astype(int)

    def predict_proba(self, X: Any) -> np.ndarray:
        import pandas as pd
        n = len(X) if hasattr(X, "__len__") else 1
        if not self._is_fitted:
            proba = np.zeros((n, 2))
            proba[:, 0] = 1.0
            return proba
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        scores = -self._model.score_samples(X_arr)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        proba = np.stack([1.0 - scores, scores], axis=1)
        return proba
