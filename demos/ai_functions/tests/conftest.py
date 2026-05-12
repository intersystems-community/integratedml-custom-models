"""Make the repo root importable so tests can use the full package path
(`demos.ai_functions.iris_models.<function>`) — this is needed because
multiple demos legitimately ship an `iris_models/` package and we don't
want them shadowing each other when pytest collects more than one demo's
tests in a single session.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
