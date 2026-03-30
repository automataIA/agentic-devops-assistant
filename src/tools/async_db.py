"""Async wrapper for DuckDB queries to avoid blocking the event loop."""

from __future__ import annotations

import asyncio
from typing import Any

import duckdb
import pandas as pd


async def async_query(
    db: duckdb.DuckDBPyConnection,
    sql: str,
    params: list[Any] | None = None,
) -> pd.DataFrame:
    """Execute a DuckDB query without blocking the event loop.

    Tries to offload to a thread pool for true async behavior. Falls back
    to synchronous execution if thread offloading fails (e.g. in tests
    with mock connections or in-memory databases that aren't thread-safe).

    Args:
        db: DuckDB connection instance.
        sql: SQL query string.
        params: Optional list of query parameters.

    Returns:
        Query results as a pandas DataFrame.
    """
    try:
        return await asyncio.to_thread(_sync_query, db, sql, params)
    except Exception:
        # Fallback: run synchronously (mock objects, thread-unsafe connections)
        return _sync_query(db, sql, params)


def _sync_query(
    db: duckdb.DuckDBPyConnection,
    sql: str,
    params: list[Any] | None = None,
) -> pd.DataFrame:
    return db.execute(sql, params or []).fetchdf()
