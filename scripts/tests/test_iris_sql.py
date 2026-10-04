import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iris_sql import connect, split_statements  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def test_splits_and_drops_comments():
    sql = """
    -- header comment; with a semicolon
    SELECT 1;  -- trailing comment
    SELECT 'a;b' AS x;
    SELECT "odd;name" FROM t
    """
    assert split_statements(sql) == [
        "SELECT 1",
        "SELECT 'a;b' AS x",
        'SELECT "odd;name" FROM t',
    ]


def test_dashes_inside_quotes_are_kept():
    assert split_statements("SELECT '--not a comment' AS x;") == [
        "SELECT '--not a comment' AS x"
    ]


@pytest.mark.parametrize("path", sorted(REPO.glob("demos/*_foundation/sql/*.sql")))
def test_demo_sql_files_split_cleanly(path):
    statements = split_statements(path.read_text())
    assert statements
    for stmt in statements:
        assert not stmt.startswith("--")
        assert stmt.split()[0].upper() in {
            "CREATE", "DROP", "TRAIN", "SELECT", "WITH", "VALIDATE", "INSERT",
        }


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        connect("bogus")
