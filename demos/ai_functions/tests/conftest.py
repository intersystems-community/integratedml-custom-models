"""Make the demo's iris_models package importable from `tests/`."""

import sys
from pathlib import Path

DEMO_ROOT = Path(__file__).resolve().parent.parent
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))
