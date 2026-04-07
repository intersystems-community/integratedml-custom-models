import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class IRISModel:
    name = "ensemble_fraud_detector"

    def __init__(self, **kwargs):
        rf = RandomForestClassifier(n_estimators=50, random_state=42, class_weight="balanced")
        lr = LogisticRegression(max_iter=500, random_state=42, class_weight="balanced")
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100, random_state=42)
        voting = VotingClassifier(
            estimators=[("rf", rf), ("lr", lr), ("mlp", mlp)],
            voting="soft",
        )
        self.model = Pipeline([("scaler", StandardScaler(with_mean=False)), ("clf", voting)])

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
