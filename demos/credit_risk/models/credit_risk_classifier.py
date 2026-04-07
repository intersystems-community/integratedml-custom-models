import json
import pickle
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from shared.models.classification import ClassificationModel


class _NumericPreprocessor:
    def __init__(self, pipeline, numeric_cols):
        self._pipeline = pipeline
        self._numeric_cols = numeric_cols

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            available = [c for c in self._numeric_cols if c in X.columns]
            X = X[available]
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        return self._pipeline.transform(X_arr)

    @property
    def shape(self):
        return None


def calculate_debt_to_income_ratio(debt: float, income: float) -> float:
    if debt <= 0:
        return 0.0
    if income <= 0:
        return float("inf")
    ratio = debt / income
    return min(ratio, 10.0)


def assess_credit_stability(employment_months: float, residence_months: float) -> float:
    cap = 50.0
    if employment_months < 0 and residence_months < 0:
        return 0.0
    if employment_months < 0:
        return min(max(residence_months, 0) / cap, 1.0)
    if residence_months < 0:
        return min(max(employment_months, 0) / cap, 1.0)
    emp_score = min(employment_months / cap, 1.0)
    res_score = min(residence_months / cap, 1.0)
    return 0.5 * emp_score + 0.5 * res_score


def calculate_financial_capacity_score(
    loan_amount: float, duration_months: float, income: float = 0.0
) -> float:
    if duration_months <= 0:
        return 0.0
    monthly_payment = loan_amount / duration_months
    if income <= 0:
        return min(1.0 - min(monthly_payment / (loan_amount + 1e-9), 1.0), 1.0)
    ratio = monthly_payment / income
    return max(0.0, min(1.0 - ratio, 1.0))


def encode_credit_purpose(purpose: str) -> int:
    mapping = {
        "car": 1,
        "furniture": 2,
        "education": 3,
        "vacation": 4,
        "business": 5,
        "home": 6,
        "repairs": 2,
        "retraining": 3,
        "electronics": 2,
    }
    return mapping.get(purpose.lower(), 3)


class CustomCreditRiskClassifier(ClassificationModel):

    def __init__(
        self,
        enable_debt_ratio: bool = True,
        enable_interaction_terms: bool = True,
        enable_risk_scoring: bool = True,
        decision_threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        if not isinstance(enable_debt_ratio, bool):
            raise ValueError("enable_debt_ratio must be a bool")
        if not isinstance(enable_interaction_terms, bool):
            raise ValueError("enable_interaction_terms must be a bool")
        if not isinstance(enable_risk_scoring, bool):
            raise ValueError("enable_risk_scoring must be a bool")
        if not (0.0 <= decision_threshold <= 1.0):
            raise ValueError("decision_threshold must be between 0.0 and 1.0")

        super().__init__(
            enable_debt_ratio=enable_debt_ratio,
            enable_interaction_terms=enable_interaction_terms,
            enable_risk_scoring=enable_risk_scoring,
            decision_threshold=decision_threshold,
            **kwargs,
        )
        self.enable_debt_ratio = enable_debt_ratio
        self.enable_interaction_terms = enable_interaction_terms
        self.enable_risk_scoring = enable_risk_scoring
        self.decision_threshold = decision_threshold

        self._pipeline: Optional[Pipeline] = None
        self._preprocessor: Optional[Any] = None
        self._model_metadata: Dict[str, Any] = {}
        self._numeric_feature_cols: Optional[list] = None

    @property
    def is_fitted(self) -> bool:
        return self._is_trained

    def _add_debt_ratio_features(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "credit_amount" in X.columns and "employment_duration" in X.columns:
            X["debt_to_income_ratio"] = X.apply(
                lambda r: calculate_debt_to_income_ratio(
                    r["credit_amount"], max(r["employment_duration"] * 1000, 1)
                ),
                axis=1,
            )
        if "credit_amount" in X.columns and "duration" in X.columns:
            X["monthly_payment_ratio"] = X.apply(
                lambda r: calculate_financial_capacity_score(
                    r["credit_amount"], r["duration"]
                ),
                axis=1,
            )
        if "age" in X.columns and "credit_amount" in X.columns:
            X["age_adjusted_credit"] = X["credit_amount"] / (X["age"].clip(lower=1))
        return X

    def _add_interaction_terms(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            for i, c1 in enumerate(numeric_cols[:3]):
                for c2 in numeric_cols[i + 1 : 4]:
                    col_name = f"{c1}_{c2}_interaction"
                    X[col_name] = X[c1] * X[c2]
        return X

    def _add_risk_scoring_features(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        emp_col = "employment_duration" if "employment_duration" in X.columns else None
        age_col = "age" if "age" in X.columns else None

        if emp_col and age_col:
            X["stability_score"] = X.apply(
                lambda r: assess_credit_stability(
                    r[emp_col] * 12, (r[age_col] - 18) * 12
                ),
                axis=1,
            )
            X["composite_risk_score"] = X["stability_score"].clip(0, 1)
        return X

    def _engineer_features(self, X: Any, is_training: bool = False) -> Any:
        if not isinstance(X, pd.DataFrame):
            return X
        if self.enable_debt_ratio:
            X = self._add_debt_ratio_features(X)
        if self.enable_interaction_terms:
            X = self._add_interaction_terms(X)
        if self.enable_risk_scoring:
            X = self._add_risk_scoring_features(X)
        return X

    def fit(self, X: Any, y: Any, **fit_params: Any) -> "CustomCreditRiskClassifier":
        if isinstance(X, pd.DataFrame) and len(X) == 0:
            raise ValueError("X must not be empty")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")

        original_features = list(X.columns) if isinstance(X, pd.DataFrame) else []
        X_eng = self._engineer_features(X, is_training=True)
        engineered_features = list(X_eng.columns) if isinstance(X_eng, pd.DataFrame) else []

        self._label_encoder_fit(y)

        if isinstance(X_eng, pd.DataFrame):
            numeric_cols = X_eng.select_dtypes(include=[np.number]).columns.tolist()
            self._numeric_feature_cols = numeric_cols
            X_eng = X_eng[numeric_cols]
        else:
            self._numeric_feature_cols = None

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        clf = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")

        raw_preprocessor = Pipeline([("imputer", imputer), ("scaler", scaler)])
        self._pipeline = Pipeline(
            [("imputer", imputer), ("scaler", scaler), ("clf", clf)]
        )

        X_arr = X_eng.values if isinstance(X_eng, pd.DataFrame) else np.asarray(X_eng)
        y_arr = np.asarray(y)

        self._pipeline.fit(X_arr, y_arr)
        raw_preprocessor.fit(X_arr)
        self._preprocessor = _NumericPreprocessor(raw_preprocessor, self._numeric_feature_cols or [])

        self._model_metadata = {
            "feature_engineering_enabled": {
                "debt_ratio": self.enable_debt_ratio,
                "interaction_terms": self.enable_interaction_terms,
                "risk_scoring": self.enable_risk_scoring,
            },
            "original_features": original_features,
            "engineered_features": engineered_features,
        }
        self._is_trained = True
        return self

    def _label_encoder_fit(self, y: Any) -> None:
        from sklearn.preprocessing import LabelEncoder
        self._label_encoder = LabelEncoder()
        self._label_encoder.fit(y)
        self._classes = self._label_encoder.classes_

    def _select_numeric(self, X: Any) -> Any:
        if isinstance(X, pd.DataFrame) and self._numeric_feature_cols is not None:
            available = [c for c in self._numeric_feature_cols if c in X.columns]
            return X[available]
        return X

    def _predict_impl(self, X: Any) -> np.ndarray:
        X = self._select_numeric(X)
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        proba = self._pipeline.predict_proba(X_arr)[:, 1]
        return (proba >= self.decision_threshold).astype(int)

    def predict_proba(self, X: Any) -> np.ndarray:
        self._require_trained()
        X_eng = self._engineer_features(X, is_training=False)
        X_eng = self._select_numeric(X_eng)
        X_arr = X_eng.values if isinstance(X_eng, pd.DataFrame) else np.asarray(X_eng)
        return self._pipeline.predict_proba(X_arr)

    def decision_function(self, X: Any) -> np.ndarray:
        self._require_trained()
        X_eng = self._engineer_features(X, is_training=False)
        X_eng = self._select_numeric(X_eng)
        X_arr = X_eng.values if isinstance(X_eng, pd.DataFrame) else np.asarray(X_eng)
        return self._pipeline.decision_function(X_arr)

    def get_feature_importance(self) -> Optional[np.ndarray]:
        if not self._is_trained:
            return None
        clf = self._pipeline.named_steps["clf"]
        if hasattr(clf, "coef_"):
            raw = np.abs(clf.coef_[0])
            total = raw.sum()
            return raw / total if total > 0 else raw
        return None

    def score(self, X: Any, y: Any) -> float:
        from sklearn.metrics import accuracy_score
        return float(accuracy_score(y, self.predict(X)))

    def get_risk_explanation(self, X: pd.DataFrame) -> Dict[str, Any]:
        self._require_trained()
        proba = self.predict_proba(X)[:, 1]
        recommendations = []
        for p in proba:
            if p >= 0.7:
                recommendations.append("High risk — consider additional verification")
            elif p >= 0.4:
                recommendations.append("Moderate risk — standard review recommended")
            else:
                recommendations.append("Low risk — standard approval process")
        return {
            "risk_probabilities": proba.tolist(),
            "risk_factors": [{}] * len(X),
            "recommendations": recommendations,
        }

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "enable_debt_ratio": self.enable_debt_ratio,
            "enable_interaction_terms": self.enable_interaction_terms,
            "enable_risk_scoring": self.enable_risk_scoring,
            "decision_threshold": self.decision_threshold,
        }

    def set_params(self, **params: Any) -> "CustomCreditRiskClassifier":
        for k, v in params.items():
            setattr(self, k, v)
            self.parameters[k] = v
        return self

    def save_model(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(cls, path: str) -> "CustomCreditRiskClassifier":
        with open(path, "rb") as f:
            return pickle.load(f)

    def to_json(self) -> str:
        return json.dumps(self.get_params())

    @classmethod
    def from_json(cls, json_str: str) -> "CustomCreditRiskClassifier":
        params = json.loads(json_str)
        return cls(**params)
