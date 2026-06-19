"""
IRISModel wrapper for Gaia Variable Star detection.

Deploy this file to the pathtoclassifiers directory referenced in:
    CREATE MODEL GaiaVariability PREDICTING (is_variable)
    FROM GaiaObservationStats
    USING {"pathtoclassifiers": "/path/to/iris_models", "iscmodelsdisabled": 1}

IRISModel contract (iris_automl/automl_model.py):
    - __init__(**kwargs) receives random_state, n_jobs, and USING-clause userparams
    - self.model must be a sklearn-compatible estimator (BaseEstimator)
    - self.name must be a unique string
    - IntegratedML pre-processes features (scaling, encoding, correlation reduction)
      before calling self.model.fit(X, y) — X arrives as a dense float array
      of already-transformed features; column indices are not stable across runs.
"""

from sklearn.ensemble import GradientBoostingClassifier


class IRISModel:
    def __init__(self, **kwargs):
        random_state = kwargs.get("random_state", 42)
        n_estimators = int(kwargs.get("n_estimators", 100))
        max_depth = int(kwargs.get("max_depth", 3))
        learning_rate = float(kwargs.get("learning_rate", 0.1))
        self.name = "gaia_variability_detector"
        self.model_type = "Gradient Boosting"
        self.package = "sklearn"
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )
