import importlib
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse

PKG_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_DIR))

import install  # noqa: E402

FAKE_DATA_PREP = textwrap.dedent('''
    def _calc_feature_elimination_fit(df, y, log):
        def transform_fn(df):
            return df.iloc[:, :1]  # "eliminates" all but the first column
        return transform_fn

    def _build_transform_fn(df, y, log):
        # Like AutoML: looks the function up as a module global at call time.
        return _calc_feature_elimination_fit(df, y, log)
''')


@pytest.fixture
def fake_automl(tmp_path, monkeypatch):
    """A fake `iris_automl` package on sys.path, plus a clean hook state."""
    pkg = tmp_path / "iris_automl"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "automl_data_prep.py").write_text(FAKE_DATA_PREP)
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in ("iris_automl", "iris_automl.automl_data_prep",
                 "iris_automl_keep_features"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    meta_path = list(sys.meta_path)
    yield tmp_path
    sys.meta_path[:] = meta_path
    for name in ("iris_automl", "iris_automl.automl_data_prep",
                 "iris_automl_keep_features"):
        sys.modules.pop(name, None)


def _df():
    return pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})


def test_unpatched_fake_drops_columns(fake_automl):
    prep = importlib.import_module("iris_automl.automl_data_prep")
    fn = prep._build_transform_fn(_df(), None, lambda *a, **k: None)
    assert list(fn(_df()).columns) == ["a"]


def test_hook_patches_on_later_import(fake_automl):
    hook = importlib.import_module("iris_automl_keep_features")
    assert "iris_automl.automl_data_prep" not in sys.modules  # lazy

    prep = importlib.import_module("iris_automl.automl_data_prep")
    assert prep._keep_features_patched
    messages = []
    fn = prep._build_transform_fn(_df(), None, lambda *a, **k: messages.append(a))
    assert list(fn(_df()).columns) == ["a", "b"]
    assert any("keeping all" in str(m) for m in messages)
    assert prep._calc_feature_elimination_fit is hook._keep_all_features_fit


def test_patches_immediately_if_already_imported(fake_automl):
    prep = importlib.import_module("iris_automl.automl_data_prep")
    importlib.import_module("iris_automl_keep_features")
    assert prep._keep_features_patched


def test_sparse_input_returned_as_csc(fake_automl):
    hook = importlib.import_module("iris_automl_keep_features")
    fn = hook._keep_all_features_fit(_df(), None, lambda *a, **k: None)
    out = fn(scipy.sparse.coo_matrix(np.eye(3)))
    assert scipy.sparse.isspmatrix_csc(out) and out.shape == (3, 3)


def test_missing_function_leaves_module_alone(fake_automl, capsys):
    hook = importlib.import_module("iris_automl_keep_features")
    import types

    module = types.ModuleType("renamed")
    assert hook.patch(module) is False
    assert "not found" in capsys.readouterr().err


def test_install_preserves_existing_sitecustomize(tmp_path):
    site = tmp_path / "sitecustomize.py"
    site.write_text("import os  # existing\n")

    install.install(tmp_path)
    install.install(tmp_path)  # idempotent
    text = site.read_text()
    assert text.count(install.BEGIN) == 1
    assert "import os  # existing" in text
    assert (tmp_path / "iris_automl_keep_features.py").exists()
    assert install.status(tmp_path)

    install.uninstall(tmp_path)
    assert site.read_text() == "import os  # existing\n"
    assert not (tmp_path / "iris_automl_keep_features.py").exists()
    assert not install.status(tmp_path)


def test_uninstall_removes_sitecustomize_it_created(tmp_path):
    install.install(tmp_path)
    install.uninstall(tmp_path)
    assert not (tmp_path / "sitecustomize.py").exists()


def test_sitecustomize_block_imports_module(tmp_path, monkeypatch):
    install.install(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "iris_automl_keep_features", raising=False)
    meta_path = list(sys.meta_path)
    try:
        exec(compile((tmp_path / "sitecustomize.py").read_text(), "sc", "exec"), {})
        assert "iris_automl_keep_features" in sys.modules
    finally:
        sys.meta_path[:] = meta_path
        sys.modules.pop("iris_automl_keep_features", None)
