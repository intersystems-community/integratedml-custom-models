"""Install or remove the AutoML keep-all-features patch in an IRIS instance.

Run with IRIS's Python, inside the IRIS container:

    docker exec iris /usr/irissys/bin/irispython \
        /opt/irisapp/scripts/automl_keep_features/install.py [--uninstall|--status]

It copies iris_automl_keep_features.py into IRIS's Python path and adds a
marked block to sitecustomize.py there that imports it, leaving any other
sitecustomize content alone. The patch applies to IRIS processes started
afterwards; processes that already imported AutoML keep the old behaviour.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

MODULE = "iris_automl_keep_features"
BEGIN = f"# >>> {MODULE} >>>"
END = f"# <<< {MODULE} <<<"
BLOCK = f"""{BEGIN}
try:
    import {MODULE}  # noqa: F401
except Exception as _exc:  # never break IRIS Python startup
    import sys as _sys

    _sys.stderr.write("{MODULE} not loaded: %r\\n" % (_exc,))
{END}
"""
DEFAULT_TARGET = Path("/usr/irissys/mgr/python")


def _without_block(text: str) -> str:
    if BEGIN not in text:
        return text
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    return (head.rstrip("\n") + "\n" + tail.lstrip("\n")).strip("\n") + "\n"


def install(target: Path) -> None:
    src = Path(__file__).resolve().parent / f"{MODULE}.py"
    shutil.copy2(src, target / f"{MODULE}.py")
    site = target / "sitecustomize.py"
    text = site.read_text() if site.exists() else ""
    text = _without_block(text).strip("\n")
    site.write_text((text + "\n\n" if text else "") + BLOCK)
    print(f"installed {MODULE} into {target}")


def uninstall(target: Path) -> None:
    (target / f"{MODULE}.py").unlink(missing_ok=True)
    site = target / "sitecustomize.py"
    if site.exists():
        text = _without_block(site.read_text())
        if text.strip():
            site.write_text(text)
        else:
            site.unlink()
    print(f"removed {MODULE} from {target}")


def status(target: Path) -> bool:
    site = target / "sitecustomize.py"
    active = (target / f"{MODULE}.py").exists() and site.exists() and (
        BEGIN in site.read_text()
    )
    print(f"{MODULE}: {'installed' if active else 'not installed'} in {target}")
    return active


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--uninstall", action="store_true")
    group.add_argument("--status", action="store_true")
    args = parser.parse_args(argv)
    if not args.target.is_dir():
        sys.exit(f"target directory not found: {args.target}")
    if args.status:
        status(args.target)
    elif args.uninstall:
        uninstall(args.target)
    else:
        install(args.target)


if __name__ == "__main__":
    main()
