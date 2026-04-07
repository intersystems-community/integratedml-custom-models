import numpy as np
from typing import Any, List
from sklearn.neural_network import MLPClassifier


class NeuralFraudDetector:

    def __init__(self, hidden_layers: List[int] = None, dropout_rate: float = 0.3, epochs: int = 50, **kwargs: Any) -> None:
        layers = tuple(hidden_layers) if hidden_layers else (128, 64, 32)
        self._model = MLPClassifier(
            hidden_layer_sizes=layers,
            max_iter=epochs,
            random_state=42,
            early_stopping=True,
        )
        self._is_fitted = False

    def fit(self, X: Any, y: Any) -> "NeuralFraudDetector":
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
