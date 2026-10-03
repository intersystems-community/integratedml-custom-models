"""Make the repo root importable so tests can use the full package path
(`demos.sdm_foundation.iris_models.<class>`). See the matching comment in
the AI Functions demo conftest for why we don't put each demo's directory
directly on sys.path — both demos legitimately ship an `iris_models/`
package and we don't want them shadowing each other.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_sdm: skip unless structured-data-models (`sdm`) is installed",
    )


def pytest_collection_modifyitems(config, items):
    """Mark tests that require NVIDIA's structured-data-models (`sdm`).

    Checks for `sdm.models.KumoTabular`, since an unrelated PyPI package is
    also named `sdm`.
    """
    try:
        import sdm

        have_sdm = hasattr(getattr(sdm, "models", None), "KumoTabular")
    except Exception:
        have_sdm = False
    skip_sdm = (
        None if have_sdm
        else __import__("pytest").mark.skip(
            reason="structured-data-models not installed"
        )
    )
    if skip_sdm is None:
        return
    for item in items:
        if "requires_sdm" in item.keywords:
            item.add_marker(skip_sdm)
