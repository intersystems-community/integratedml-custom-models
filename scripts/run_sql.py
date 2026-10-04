"""Run the statements of an IntegratedML .sql file against IRIS.

From the host, over DB-API (`pip install intersystems-irispython`):

    python scripts/run_sql.py demos/kumo_tabular_foundation/sql/02_create_models.sql

Inside the IRIS container, with embedded Python:

    docker exec -e IRISNAMESPACE=USER iris /usr/irissys/bin/irispython \
        /opt/irisapp/scripts/run_sql.py \
        /opt/irisapp/demos/kumo_tabular_foundation/sql/02_create_models.sql

Prints OK/FAIL and the time for each statement, plus the first rows of each
result. Exits 1 if any statement failed. See iris_sql.py for connection
settings (IRIS_HOST, IRIS_PORT, IRIS_NAMESPACE, IRIS_USERNAME,
IRIS_PASSWORD).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from iris_sql import SQLError, connect, split_statements  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sql_files", nargs="+", type=Path)
    parser.add_argument("--mode", choices=("auto", "dbapi", "embedded"),
                        default="auto")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--namespace")
    parser.add_argument("--user")
    parser.add_argument("--password")
    parser.add_argument("--max-rows", type=int, default=10)
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args(argv)

    if args.mode == "embedded":
        db = connect("embedded")
    else:
        db = connect(args.mode, host=args.host, port=args.port,
                     namespace=args.namespace, user=args.user,
                     password=args.password)
    print(f"connected: {db.mode} {db.target}")

    failures = 0
    try:
        for path in args.sql_files:
            print(f"\n== {path}")
            for stmt in split_statements(path.read_text()):
                head = " ".join(stmt.split())[:90]
                t0 = time.time()
                try:
                    result = db.execute(stmt)
                except SQLError as exc:
                    failures += 1
                    print(f"FAIL ({time.time() - t0:5.1f}s) {head}\n     {exc}")
                    if args.stop_on_error:
                        return 1
                    continue
                print(f"OK   ({time.time() - t0:5.1f}s) {head}")
                if result is not None:
                    columns, rows = result
                    print(f"     {len(rows)} rows: {', '.join(columns)}")
                    for row in rows[: args.max_rows]:
                        print(f"     {row}")
    finally:
        db.close()
    print(f"\n{failures} statement(s) failed" if failures else "\nall OK")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
