from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from demos.tabpfn_foundation.iris_models.tabpfn_regressor import IRISModel


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "building_energy.csv"


@pytest.fixture(scope="module")
def energy_data():
    df = pd.read_csv(DATA_PATH)
    feature_cols = [c for c in df.columns
                    if c not in ("building_id", "kwh_day")]
    return df, feature_cols


def _train_test_split(df, feature_cols, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    cut = int(0.75 * len(df))
    train_idx, test_idx = idx[:cut], idx[cut:]
    X_train = df.iloc[train_idx][feature_cols]
    y_train = df.iloc[train_idx]["kwh_day"].to_numpy()
    X_test = df.iloc[test_idx][feature_cols]
    y_test = df.iloc[test_idx]["kwh_day"].to_numpy()
    return X_train, y_train, X_test, y_test


def test_sklearn_fallback_reports_sklearn_backend(energy_data):
    df, feature_cols = energy_data
    X_train, y_train, _, _ = _train_test_split(df, feature_cols)
    model = IRISModel(force_fallback=True).fit(X_train, y_train)
    assert model.backend == "sklearn"


def test_predictions_are_finite_and_positive(energy_data):
    df, feature_cols = energy_data
    X_train, y_train, X_test, _ = _train_test_split(df, feature_cols)
    model = IRISModel(force_fallback=True).fit(X_train, y_train)
    preds = model.predict(X_test)
    assert preds.shape == (len(X_test),)
    assert np.isfinite(preds).all()
    assert (preds > 0).all()


def test_r2_beats_mean_baseline(energy_data):
    df, feature_cols = energy_data
    X_train, y_train, X_test, y_test = _train_test_split(df, feature_cols)
    model = IRISModel(force_fallback=True).fit(X_train, y_train)
    preds = model.predict(X_test)
    ss_res = float(((y_test - preds) ** 2).sum())
    ss_tot = float(((y_test - y_test.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot
    assert r2 >= 0.85, f"R^2 {r2:.3f} below 0.85"


def test_mape_within_demo_quality(energy_data):
    df, feature_cols = energy_data
    X_train, y_train, X_test, y_test = _train_test_split(df, feature_cols)
    model = IRISModel(force_fallback=True).fit(X_train, y_train)
    preds = model.predict(X_test)
    mape = float(np.mean(np.abs((y_test - preds) / y_test)))
    assert mape <= 0.20


def test_handles_categorical_hvac_type(energy_data):
    df, feature_cols = energy_data
    assert "hvac_type" in feature_cols
    X_train, y_train, X_test, _ = _train_test_split(df, feature_cols)
    model = IRISModel(force_fallback=True).fit(X_train, y_train)
    preds = model.predict(X_test)
    assert np.isfinite(preds).all()


@pytest.mark.requires_tabpfn
def test_tabpfn_backend_when_installed(energy_data):
    df, feature_cols = energy_data
    X_train, y_train, X_test, y_test = _train_test_split(df, feature_cols)
    model = IRISModel(device="cpu", n_estimators=4).fit(X_train, y_train)
    assert model.backend == "tabpfn"
    preds = model.predict(X_test)
    r2 = 1.0 - ((y_test - preds) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum()
    assert r2 >= 0.80
