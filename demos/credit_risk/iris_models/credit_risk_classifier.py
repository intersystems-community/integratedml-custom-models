from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class IRISModel:
    name = "credit_risk_classifier"

    def __init__(self, **kwargs):
        self.model = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", GradientBoostingClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
            )),
        ])

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]

    def predict_proba(self, X):
        return self.model.predict_proba(X)
