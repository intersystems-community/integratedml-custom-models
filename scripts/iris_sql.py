"""Run SQL against IRIS over DB-API or embedded Python, with real errors.

Two modes, chosen by `connect(mode="auto")`:

* ``dbapi``: `iris.dbapi.connect(...)` from the `intersystems-irispython`
  package, as documented on PyPI, over the superserver port (default 1972),
  the way a notebook or app connects. Settings come from
  arguments or the IRIS_HOST, IRIS_PORT, IRIS_NAMESPACE, IRIS_USERNAME and
  IRIS_PASSWORD environment variables (defaults localhost, 1972, USER, demo,
  demo).
* ``embedded``: IRIS embedded Python, when running under `irispython` inside
  the IRIS container. The namespace comes from IRISNAMESPACE.

"auto" picks embedded when the embedded `iris` module is loaded, else DB-API.

Both raise `SQLError` on failure. In embedded mode this uses %SQL.Statement
and checks SQLCODE after fetching, because `iris.sql.exec` can drop errors
raised while rows are fetched (for example a failing PREDICT()) and return
an empty result instead.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Sequence, Tuple

Result = Optional[Tuple[List[str], List[list]]]


class SQLError(Exception):
    pass


class DbapiExecutor:
    mode = "dbapi"

    def __init__(self, host=None, port=None, namespace=None, user=None,
                 password=None):
        # DB-API entry point as documented for intersystems-irispython on
        # PyPI: `import iris.dbapi; iris.dbapi.connect(**args)`. The
        # top-level `iris.connect` is the Native SDK connection (for
        # iris.createIRIS), not DB-API.
        import iris.dbapi

        self.dbapi = iris.dbapi
        args = {
            "hostname": host or os.environ.get("IRIS_HOST", "localhost"),
            "port": int(port or os.environ.get("IRIS_PORT", 1972)),
            "namespace": namespace or os.environ.get("IRIS_NAMESPACE", "USER"),
            "username": user or os.environ.get("IRIS_USERNAME", "demo"),
            "password": password or os.environ.get("IRIS_PASSWORD", "demo"),
        }
        self.target = f"{args['hostname']}:{args['port']}/{args['namespace']}"
        self.conn = iris.dbapi.connect(**args)

    def execute(self, sql: str, params: Sequence[Any] = ()) -> Result:
        cur = self.conn.cursor()
        try:
            cur.execute(sql, list(params))
            if not cur.description:
                return None
            columns = [d[0] for d in cur.description]
            return columns, [list(row[:]) for row in cur.fetchall()]
        except self.dbapi.Error as exc:
            raise SQLError(str(exc)) from exc
        finally:
            cur.close()

    def close(self) -> None:
        self.conn.close()


class EmbeddedExecutor:
    mode = "embedded"

    def __init__(self):
        import iris

        self.iris = iris
        self.target = "embedded/" + str(
            iris.cls("%SYSTEM.SYS").NameSpace()
        )

    def _check(self, sc) -> None:
        status = self.iris.cls("%SYSTEM.Status")
        if status.IsError(sc):
            raise SQLError(status.GetErrorText(sc))

    def execute(self, sql: str, params: Sequence[Any] = ()) -> Result:
        stmt = self.iris.cls("%SQL.Statement")._New()
        self._check(stmt._Prepare(sql))
        rs = stmt._Execute(*params)
        if rs._SQLCODE < 0:
            raise SQLError(f"SQLCODE={rs._SQLCODE} {rs._Message}")
        if rs._StatementType != 1:  # not a SELECT
            return None
        meta = rs._GetMetadata()
        n = meta.columnCount
        columns = [meta.columns.GetAt(i).colName for i in range(1, n + 1)]
        rows = []
        while rs._Next():
            rows.append([rs._GetData(i) for i in range(1, n + 1)])
        if rs._SQLCODE < 0:  # error while fetching, e.g. inside PREDICT()
            raise SQLError(f"SQLCODE={rs._SQLCODE} {rs._Message}")
        return columns, rows

    def close(self) -> None:
        pass


def _embedded_available() -> bool:
    try:
        import iris
    except Exception:
        return False
    return hasattr(iris, "cls") and hasattr(iris, "sql")


def connect(mode: str = "auto", **kwargs):
    """Return an executor with `.execute(sql, params)` and `.close()`.

    kwargs (host, port, namespace, user, password) apply to DB-API only;
    None values fall back to the environment.
    """
    if mode == "auto":
        mode = "embedded" if _embedded_available() else "dbapi"
    if mode == "embedded":
        return EmbeddedExecutor()
    if mode == "dbapi":
        return DbapiExecutor(**kwargs)
    raise ValueError(f"unknown mode: {mode}")


def split_statements(text: str) -> List[str]:
    """Split SQL on `;`, skipping `--` comments and semicolons in quotes."""
    statements, buf, i, quote = [], [], 0, None
    while i < len(text):
        ch = text[i]
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
            buf.append(ch)
        elif text.startswith("--", i):
            end = text.find("\n", i)
            i = len(text) if end == -1 else end
            continue
        elif ch == ";":
            statements.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
        i += 1
    statements.append("".join(buf).strip())
    return [s for s in statements if s]
