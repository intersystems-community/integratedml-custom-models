import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from demos.kumo_tabular_foundation.iris_models.kumo_regressor import IRISModel
from demos.kumo_tabular_foundation.tests import stubs


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


def test_fallback_r2_beats_mean_baseline(energy_data):
    from sklearn.metrics import r2_score

    df, feature_cols = energy_data
    X_train, y_train, X_test, y_test = _train_test_split(df, feature_cols)
    model = IRISModel(force_fallback=True).fit(X_train, y_train)
    preds = model.predict(X_test)
    assert np.isfinite(preds).all()
    assert r2_score(y_test, preds) > 0.5


def test_categorical_codes_stable_across_batches(energy_data):
    df, feature_cols = energy_data
    assert "hvac_type" in feature_cols
    X_train, y_train, X_test, _ = _train_test_split(df, feature_cols)
    model = IRISModel(force_fallback=True).fit(X_train, y_train)
    batch = model.predict(X_test)
    rows = np.concatenate([model.predict(X_test.iloc[[i]])
                           for i in range(len(X_test))])
    np.testing.assert_allclose(rows, batch)


def test_accepts_scipy_sparse_input(energy_data):
    # IntegratedML's AutoML provider hands custom models scipy sparse matrices.
    import scipy.sparse
    from sklearn.base import clone, is_regressor

    df, feature_cols = energy_data
    num_cols = [c for c in feature_cols if c != "hvac_type"]
    X_train, y_train, X_test, _ = _train_test_split(df, num_cols)
    model = IRISModel(force_fallback=True)
    assert is_regressor(model)
    clone(model).fit(scipy.sparse.csr_matrix(X_train.to_numpy()), y_train)
    model.fit(scipy.sparse.csr_matrix(X_train.to_numpy()), y_train)
    preds = model.predict(scipy.sparse.csr_matrix(X_test.to_numpy()))
    assert preds.shape == (len(X_test),)
    assert np.isfinite(preds).all()


def test_kumo_wiring_with_stub_package(energy_data, monkeypatch):
    calls = stubs.install(monkeypatch)
    df, feature_cols = energy_data
    X_train, y_train, X_test, _ = _train_test_split(df, feature_cols)
    model = IRISModel(num_estimators=4).fit(X_train, y_train)
    assert model.backend == "kumo-tabular"

    preds = model.predict(X_test)
    assert preds.shape == (len(X_test),)  # (n, 1) output flattened
    np.testing.assert_allclose(preds, np.mean(y_train))

    assert calls["models"][0]["task"] == "regression"
    assert calls["stypes"][-1] == {"__target__": "numerical"}
    fwd = calls["forward"][-1]
    assert fwd["num_estimators"] == 4
    assert len(fwd["x_context"]) == len(X_train)
    assert len(fwd["x_query"]) == len(X_test)
    table = calls["tables"][-1]
    assert table["__target__"].iloc[len(X_train):].isna().all()


def test_pickled_model_drops_network_and_reloads(energy_data, monkeypatch):
    calls = stubs.install(monkeypatch)
    df, feature_cols = energy_data
    X_train, y_train, X_test, _ = _train_test_split(df, feature_cols)
    model = IRISModel().fit(X_train, y_train)
    model.predict(X_test)

    restored = pickle.loads(pickle.dumps(model))
    assert restored._impl._model is None and restored._impl.sdm is None
    assert restored.predict(X_test).shape == (len(X_test),)
    assert len(calls["models"]) == 2


def test_constructs_like_automl_and_clones(energy_data):
    # AutoML calls IRISModel(**user_params): userparams plus its own settings,
    # including random_state=None, then clones the model for cross-validation.
    from sklearn.base import clone

    df, feature_cols = energy_data
    X_train, y_train, X_test, _ = _train_test_split(df, feature_cols)
    model = IRISModel(
        num_estimators=8, max_context_rows=100, force_fallback=True,
        random_state=None, n_jobs=1, verbose=0, log=print,
        isc_models_disabled=1,
    )
    fitted = clone(model).fit(X_train, y_train)
    assert fitted.predict(X_test).shape == (len(X_test),)


@pytest.mark.requires_sdm
def test_kumo_backend_when_installed(energy_data):
    from sklearn.metrics import r2_score

    df, feature_cols = energy_data
    X_train, y_train, X_test, y_test = _train_test_split(df, feature_cols)
    model = IRISModel().fit(X_train, y_train)
    assert model.backend == "kumo-tabular"
    assert r2_score(y_test, model.predict(X_test)) > 0.8
