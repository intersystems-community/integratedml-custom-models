"""Create the Kumo Tabular demo tables and load the CSVs into IRIS.

From the host, over DB-API (`pip install intersystems-irispython`; connection
settings as in scripts/iris_sql.py, default localhost:1972/USER as demo):

    python demos/kumo_tabular_foundation/scripts/load_data.py

or inside the IRIS container, with embedded Python:

    docker exec -e IRISNAMESPACE=USER iris /usr/irissys/bin/irispython \
        /opt/irisapp/demos/kumo_tabular_foundation/scripts/load_data.py

The `split` column uses the same seeded 75/25 shuffle as
run_kumo_tabular_demo.py, so SQL results line up with the local demo.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DEMO_DIR.parent.parent / "scripts"))

from iris_sql import connect, split_statements  # noqa: E402

DATA_DIR = DEMO_DIR / "data"
SETUP_SQL = DEMO_DIR / "sql" / "01_setup_tables.sql"

TABLES = (
    ("KumoTab.PatientScreening", "patient_screening.csv"),
    ("KumoTab.BuildingEnergy", "building_energy.csv"),
)


def split_labels(n: int, seed: int = 0, train_frac: float = 0.75):
    # Mirrors run_kumo_tabular_demo._split (numpy default_rng shuffle). Without
    # numpy it falls back to the stdlib RNG, which gives a different split.
    try:
        import numpy as np

        idx = np.arange(n)
        np.random.default_rng(seed).shuffle(idx)
        idx = idx.tolist()
    except ImportError:
        idx = list(range(n))
        random.Random(seed).shuffle(idx)
    cut = int(train_frac * n)
    labels = ["test"] * n
    for i in idx[:cut]:
        labels[i] = "train"
    return labels


def load(db, table: str, csv_path: Path) -> int:
    with csv_path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    cols = header + ["split"]
    placeholders = ", ".join("?" * len(cols))
    stmt = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
    for row, label in zip(rows, split_labels(len(rows))):
        db.execute(stmt, [*row, label])
    return len(rows)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("auto", "dbapi", "embedded"),
                        default="auto")
    args = parser.parse_args(argv)

    db = connect(args.mode)
    try:
        for stmt in split_statements(SETUP_SQL.read_text()):
            db.execute(stmt)
        for table, name in TABLES:
            n = load(db, table, DATA_DIR / name)
            _, rows = db.execute(
                f"SELECT split, COUNT(*) FROM {table} GROUP BY split"
            )
            print(f"{table}: loaded {n} rows {dict(map(tuple, rows))}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
