from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from demos.tabpfn_foundation.iris_models.tabpfn_classifier import IRISModel


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "patient_screening.csv"


@pytest.fixture(scope="module")
def patient_data():
    df = pd.read_csv(DATA_PATH)
    feature_cols = [c for c in df.columns
                    if c not in ("patient_id", "needs_followup")]
    return df, feature_cols


def _train_test_split(df, feature_cols, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    cut = int(0.75 * len(df))
    train_idx, test_idx = idx[:cut], idx[cut:]
    X_train = df.iloc[train_idx][feature_cols]
    y_train = df.iloc[train_idx]["needs_followup"].to_numpy()
    X_test = df.iloc[test_idx][feature_cols]
    y_test = df.iloc[test_idx]["needs_followup"].to_numpy()
    return X_train, y_train, X_test, y_test


def test_sklearn_fallback_reports_sklearn_backend(patient_data):
    df, feature_cols = patient_data
    X_train, y_train, _, _ = _train_test_split(df, feature_cols)
    model = IRISModel(force_fallback=True).fit(X_train, y_train)
    assert model.backend == "sklearn"


def test_predictions_are_in_known_class_set(patient_data):
    df, feature_cols = patient_data
    X_train, y_train, X_test, _ = _train_test_split(df, feature_cols)
    model = IRISModel(force_fallback=True).fit(X_train, y_train)
    preds = model.predict(X_test)
    assert preds.shape == (len(X_test),)
    assert set(np.unique(preds)).issubset({0, 1})


def test_fallback_auc_beats_random(patient_data):
    """The synthetic dataset is imbalanced (~22% positives), so accuracy can
    tie the majority baseline even when the model is learning real signal.
    AUC is the right metric: anything well above 0.5 means the probabilities
    rank positives above negatives, which is what a downstream triage
    workflow actually uses.
    """
    df, feature_cols = patient_data
    X_train, y_train, X_test, y_test = _train_test_split(df, feature_cols)
    model = IRISModel(force_fallback=True).fit(X_train, y_train)
    from sklearn.metrics import roc_auc_score

    proba = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, proba))
    assert auc >= 0.75, f"AUC {auc:.3f} below 0.75 — fallback not learning"


def test_predict_proba_returns_two_columns(patient_data):
    df, feature_cols = patient_data
    X_train, y_train, X_test, _ = _train_test_split(df, feature_cols)
    model = IRISModel(force_fallback=True).fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_predict_aligns_columns_when_test_order_differs(patient_data):
    df, feature_cols = patient_data
    X_train, y_train, X_test, _ = _train_test_split(df, feature_cols)
    model = IRISModel(force_fallback=True).fit(X_train, y_train)

    shuffled_cols = list(reversed(feature_cols))
    preds_shuffled = model.predict(X_test[shuffled_cols])
    preds_normal = model.predict(X_test)
    np.testing.assert_array_equal(preds_shuffled, preds_normal)


def test_handles_categorical_sex_column(patient_data):
    df, feature_cols = patient_data
    assert "sex" in feature_cols  # encoded internally
    X_train, y_train, X_test, _ = _train_test_split(df, feature_cols)
    model = IRISModel(force_fallback=True).fit(X_train, y_train)
    # No exception raised + finite predictions.
    preds = model.predict(X_test)
    assert np.isfinite(preds.astype(float)).all()


@pytest.mark.requires_tabpfn
def test_tabpfn_backend_when_installed(patient_data):
    df, feature_cols = patient_data
    X_train, y_train, X_test, y_test = _train_test_split(df, feature_cols)
    model = IRISModel(device="cpu", n_estimators=4).fit(X_train, y_train)
    assert model.backend == "tabpfn"
    from sklearn.metrics import roc_auc_score

    proba = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, proba))
    assert auc >= 0.80
