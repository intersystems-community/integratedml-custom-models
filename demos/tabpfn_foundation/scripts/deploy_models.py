"""Stage TabPFN classifier and regressor into per-model directories.

Same rationale as the AI Functions demo's deploy script: IRIS's IntegratedML
scans every `.py` under `pathtoclassifiers` (or `pathtoregressors`) when
TRAIN MODEL runs. We give the classifier and regressor each their own
directory so CREATE MODEL resolves to exactly one IRISModel class.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


MODELS = (
    "tabpfn_classifier",
    "tabpfn_regressor",
)


def stage(iris_models_dir: Path) -> Path:
    staging_root = iris_models_dir / "_staging"
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)

    for fn in MODELS:
        src = iris_models_dir / f"{fn}.py"
        if not src.exists():
            raise FileNotFoundError(f"Missing IRISModel source: {src}")
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
    )
    args = parser.parse_args()
    staging_root = stage(args.iris_models_dir)
    print(f"Staged TabPFN models under {staging_root}")
    for sub in sorted(staging_root.iterdir()):
        print(f"  - {sub.name}")


if __name__ == "__main__":
    main()
