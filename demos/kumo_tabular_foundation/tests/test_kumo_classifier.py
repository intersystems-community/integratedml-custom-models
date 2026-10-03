import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from demos.kumo_tabular_foundation.iris_models.kumo_classifier import IRISModel
from demos.kumo_tabular_foundation.tests import stubs


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


def test_categorical_codes_stable_across_batches(patient_data):
    # PREDICT() can score one row at a time; a categorical value must get the
    # same code as at fit time no matter what else is in the batch.
    df, feature_cols = patient_data
    X_train, y_train, X_test, _ = _train_test_split(df, feature_cols)
    model = IRISModel(force_fallback=True).fit(X_train, y_train)
    batch = model.predict_proba(X_test)
    rows = np.vstack([model.predict_proba(X_test.iloc[[i]])
                      for i in range(len(X_test))])
    np.testing.assert_allclose(rows, batch)


def test_accepts_scipy_sparse_input(patient_data):
    # IntegratedML's AutoML provider hands custom models scipy sparse matrices.
    import scipy.sparse
    from sklearn.base import clone, is_classifier

    df, feature_cols = patient_data
    num_cols = [c for c in feature_cols if c != "sex"]
    X_train, y_train, X_test, _ = _train_test_split(df, num_cols)
    model = IRISModel(force_fallback=True)
    assert is_classifier(model)
    clone(model).fit(scipy.sparse.csr_matrix(X_train.to_numpy()), y_train)
    model.fit(scipy.sparse.csr_matrix(X_train.to_numpy()), y_train)
    proba = model.predict_proba(scipy.sparse.csr_matrix(X_test.to_numpy()))
    assert proba.shape == (len(X_test), 2)


def test_string_labels_round_trip(patient_data):
    df, feature_cols = patient_data
    X_train, y_train, X_test, _ = _train_test_split(df, feature_cols)
    labels = np.where(y_train == 1, "followup", "routine")
    model = IRISModel(force_fallback=True).fit(X_train, labels)
    assert list(model.classes_) == ["followup", "routine"]
    assert set(model.predict(X_test)).issubset({"followup", "routine"})


def test_kumo_wiring_with_stub_package(patient_data, monkeypatch):
    """The real-backend path follows the model card's sdm usage."""
    calls = stubs.install(monkeypatch)
    df, feature_cols = patient_data
    X_train, y_train, X_test, _ = _train_test_split(df, feature_cols)
    model = IRISModel(
        num_estimators=3, model_kwargs={"size": "small"}
    ).fit(X_train, y_train)
    assert model.backend == "kumo-tabular"

    proba = model.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    # Stub puts all mass on the last class -> classes_[-1].
    assert (model.predict(X_test) == 1).all()

    assert calls["models"] == [
        {"task": "classification", "device": calls["models"][0]["device"],
         "size": "small"}
    ]
    assert calls["models"][0]["device"].type == "cpu"
    assert calls["stypes"][-1] == {"__target__": "categorical"}
    fwd = calls["forward"][-1]
    assert fwd["num_estimators"] == 3
    assert len(fwd["x_context"]) == len(X_train)
    assert len(fwd["x_query"]) == len(X_test)
    assert "__target__" not in fwd["x_query"].columns
    np.testing.assert_array_equal(fwd["y_context"], y_train)
    table = calls["tables"][-1]
    assert table["__target__"].iloc[len(X_train):].isna().all()
    assert not pd.api.types.is_numeric_dtype(table["sex"])  # kept as strings


def test_pickled_model_drops_network_and_reloads(patient_data, monkeypatch):
    calls = stubs.install(monkeypatch)
    df, feature_cols = patient_data
    X_train, y_train, X_test, _ = _train_test_split(df, feature_cols)
    model = IRISModel().fit(X_train, y_train)
    model.predict_proba(X_test)
    assert model._impl._model is not None

    restored = pickle.loads(pickle.dumps(model))
    assert restored._impl._model is None and restored._impl.sdm is None
    assert restored.predict_proba(X_test).shape == (len(X_test), 2)
    assert len(calls["models"]) == 2  # reloaded after unpickling


def test_too_many_classes_raises_on_kumo_backend(patient_data, monkeypatch):
    stubs.install(monkeypatch)
    df, feature_cols = patient_data
    X_train, _, _, _ = _train_test_split(df, feature_cols)
    y = np.arange(len(X_train)) % 11
    with pytest.raises(ValueError, match="up to 10 classes"):
        IRISModel().fit(X_train, y)


def test_max_context_rows_samples_context(patient_data, monkeypatch):
    calls = stubs.install(monkeypatch)
    df, feature_cols = patient_data
    X_train, y_train, X_test, _ = _train_test_split(df, feature_cols)
    model = IRISModel(max_context_rows=50).fit(X_train, y_train)
    model.predict_proba(X_test)
    assert len(calls["forward"][-1]["x_context"]) == 50


def test_constructs_like_automl_and_clones(patient_data):
    # AutoML calls IRISModel(**user_params): userparams plus its own settings,
    # including random_state=None, then clones the model for cross-validation.
    from sklearn.base import clone

    df, feature_cols = patient_data
    X_train, y_train, X_test, _ = _train_test_split(df, feature_cols)
    model = IRISModel(
        num_estimators=8, max_context_rows=100, force_fallback=True,
        random_state=None, n_jobs=1, verbose=0, log=print,
        isc_models_disabled=1,
    )
    fitted = clone(model).fit(X_train, y_train)
    assert fitted.predict(X_test).shape == (len(X_test),)


@pytest.mark.requires_sdm
def test_kumo_backend_when_installed(patient_data):
    df, feature_cols = patient_data
    X_train, y_train, X_test, y_test = _train_test_split(df, feature_cols)
    model = IRISModel().fit(X_train, y_train)
    assert model.backend == "kumo-tabular"
    from sklearn.metrics import roc_auc_score

    proba = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, proba))
    assert auc >= 0.80
