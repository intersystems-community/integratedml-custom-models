"""Make the repo root importable so tests can use the full package path
(`demos.tabpfn_foundation.iris_models.<class>`). See the matching comment in
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
        "requires_tabpfn: skip the test when the `tabpfn` package isn't installed",
    )


def pytest_collection_modifyitems(config, items):
    """Mark tests that require the real `tabpfn` package."""
    import importlib

    have_tabpfn = importlib.util.find_spec("tabpfn") is not None
    skip_tabpfn = (
        None if have_tabpfn
        else __import__("pytest").mark.skip(
            reason="tabpfn package not installed"
        )
    )
    if skip_tabpfn is None:
        return
    for item in items:
        if "requires_tabpfn" in item.keywords:
            item.add_marker(skip_tabpfn)
