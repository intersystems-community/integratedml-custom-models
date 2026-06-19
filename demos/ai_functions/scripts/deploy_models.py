"""Stage each AI Function IRISModel into its own pathtoclassifiers directory.

IRIS's IntegratedML scans every `.py` file under `pathtoclassifiers` when
TRAIN MODEL runs and instantiates every IRISModel it finds. Putting all seven
AI Functions in a single directory would cause every CREATE MODEL statement
to load all seven, which is wasteful and ambiguous. To work around this we
mirror each AI Function file into its own per-function directory under
`iris_models/_staging/<function_name>/` so each `CREATE MODEL` resolves to
exactly one class.

Run this script once after editing any of the IRISModel files. It is also
called from the top-level `run_ai_functions_demo.py`.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


AI_FUNCTIONS = (
    "ai_sentiment",
    "ai_classify",
    "ai_summarize",
    "ai_translate",
    "embed_text",
    "ai_extract",
    "ai_complete",
)


def stage(iris_models_dir: Path) -> Path:
    staging_root = iris_models_dir / "_staging"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)

    for fn in AI_FUNCTIONS:
        src = iris_models_dir / f"{fn}.py"
        if not src.exists():
            raise FileNotFoundError(f"Missing AI Function source: {src}")
        target_dir = staging_root / fn
        target_dir.mkdir()
        shutil.copy2(src, target_dir / f"{fn}.py")
    return staging_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iris-models-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "iris_models",
        help="Directory containing the AI Function .py files.",
    )
    args = parser.parse_args()
    staging_root = stage(args.iris_models_dir)
    print(f"Staged AI Function models under {staging_root}")
    for sub in sorted(staging_root.iterdir()):
        print(f"  - {sub.name}")


if __name__ == "__main__":
    main()
