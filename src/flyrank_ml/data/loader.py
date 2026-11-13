"""DuckDB-over-Hugging-Face access. Token is read from the environment only."""
from __future__ import annotations

from typing import Any

from ..config import BlockedError, get_hf_token


def connect_duckdb(token: str | None = None):
    """Return a DuckDB connection that can read hf:// paths using HF_TOKEN."""
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover
        raise BlockedError("duckdb is not installed. Run: pip install -r requirements.txt") from exc
    token = token or get_hf_token(required=True)
    con = duckdb.connect()
    # Token is passed as a bound parameter and is never printed or logged.
    con.execute("CREATE SECRET hf_secret (TYPE HUGGINGFACE, TOKEN ?)", [token])
    return con


def list_tables(con, hf_glob: str) -> list[str]:
    """List parquet files matching an hf:// glob, e.g. hf://datasets/<org>/<repo>/**/*.parquet."""
    rows = con.execute("SELECT DISTINCT file FROM glob(?) ORDER BY 1", [hf_glob]).fetchall()
    return [r[0] for r in rows]


def describe_table(con, hf_path: str) -> list[dict[str, Any]]:
    """Return column names and types for one parquet path."""
    rows = con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [hf_path]).fetchall()
    return [{"column": r[0], "type": r[1]} for r in rows]


def inspect_schema(con, hf_glob: str) -> dict[str, list[dict[str, Any]]]:
    """Map every matching parquet file to its columns. Run first, then configure the pipeline."""
    return {p: describe_table(con, p) for p in list_tables(con, hf_glob)}


_IDENT = __import__("re").compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_daily_from_warehouse(data_cfg: dict[str, Any], con=None):
    """Aggregate the configured daily table to (page_id, date, clicks, impressions) inside DuckDB.

    Only these four columns leave the warehouse; nothing is written to disk by this function.
    """
    from ..config import require_configured

    require_configured(data_cfg)
    cols = data_cfg["columns"]
    for name in cols.values():
        if not _IDENT.match(str(name)):
            raise ValueError(f"Unsafe column name in config: {name!r}")
    con = con or connect_duckdb()
    sql = (
        f'SELECT "{cols["page_id"]}" AS page_id, CAST("{cols["date"]}" AS DATE) AS date, '
        f'SUM("{cols["clicks"]}") AS clicks, SUM("{cols["impressions"]}") AS impressions '
        "FROM read_parquet(?) GROUP BY 1, 2"
    )
    return con.execute(sql, [data_cfg["daily_table"]]).df()
