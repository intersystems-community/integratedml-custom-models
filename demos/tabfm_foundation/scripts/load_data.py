"""Create the TabFM demo tables and load the CSVs into IRIS.

Runs under IRIS embedded Python, inside the IRIS container:

    docker exec -e IRISNAMESPACE=USER iris \
        /usr/irissys/bin/irispython \
        /opt/irisapp/demos/tabfm_foundation/scripts/load_data.py

The `split` column uses the same seeded 75/25 shuffle as
run_tabfm_demo.py, so SQL results line up with the local demo.
"""

from __future__ import annotations

import csv
import random
import re
from pathlib import Path

import iris

DEMO_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = DEMO_DIR / "data"
SETUP_SQL = DEMO_DIR / "sql" / "01_setup_tables.sql"

TABLES = (
    ("TabFM.PatientScreening", "patient_screening.csv"),
    ("TabFM.BuildingEnergy", "building_energy.csv"),
)


def sql_statements(path: Path):
    text = re.sub(r"--[^\n]*", "", path.read_text())
    return [s.strip() for s in text.split(";") if s.strip()]


def split_labels(n: int, seed: int = 0, train_frac: float = 0.75):
    # Mirrors run_tabfm_demo._split (numpy default_rng shuffle) when numpy is
    # available; falls back to the stdlib RNG otherwise.
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


def load(table: str, csv_path: Path) -> int:
    with csv_path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    cols = header + ["split"]
    placeholders = ", ".join("?" * len(cols))
    stmt = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
    for row, label in zip(rows, split_labels(len(rows))):
        iris.sql.exec(stmt, *row, label)
    return len(rows)


def main() -> None:
    for stmt in sql_statements(SETUP_SQL):
        iris.sql.exec(stmt)
    for table, name in TABLES:
        n = load(table, DATA_DIR / name)
        counts = dict(
            iris.sql.exec(f"SELECT split, COUNT(*) FROM {table} GROUP BY split")
        )
        print(f"{table}: loaded {n} rows {counts}")


if __name__ == "__main__":
    main()
