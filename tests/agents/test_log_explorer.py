"""Unit tests for the Log Explorer agent tools (mock DuckDB, no LLM)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.agents.log_explorer import (
    count_by_level,
    get_trace,
    list_services,
    search_errors,
)
from src.deps.connections import LogDeps


def _make_deps(fetchdf_return: pd.DataFrame) -> LogDeps:
    mock_db = MagicMock()
    mock_db.execute.return_value.fetchdf.return_value = fetchdf_return
    return LogDeps(db=mock_db, minio_endpoint="http://localhost:9000")


class _FakeCtx:
    def __init__(self, deps_obj: LogDeps) -> None:
        self.deps = deps_obj


# ── search_errors ─────────────────────────────────────────────────────────────

async def test_search_errors_returns_markdown() -> None:
    df = pd.DataFrame({
        "timestamp": ["2026-03-25T10:00:00"],
        "level": ["ERROR"],
        "message": ["OOMKilled"],
        "trace_id": ["abc123"],
    })
    ctx = _FakeCtx(_make_deps(df))
    result = await search_errors(ctx, "api-gateway", hours=1)  # type: ignore[arg-type]
    assert "OOMKilled" in result
    assert "abc123" in result


async def test_search_errors_empty_result() -> None:
    ctx = _FakeCtx(_make_deps(pd.DataFrame()))
    result = await search_errors(ctx, "api-gateway", hours=1)  # type: ignore[arg-type]
    assert "No errors found" in result


# ── count_by_level ────────────────────────────────────────────────────────────

async def test_count_by_level_returns_markdown() -> None:
    df = pd.DataFrame({"level": ["ERROR", "INFO", "WARN"], "count": [42, 1000, 15]})
    ctx = _FakeCtx(_make_deps(df))
    result = await count_by_level(ctx, time_window_minutes=60)  # type: ignore[arg-type]
    assert "ERROR" in result
    assert "42" in result


async def test_count_by_level_empty() -> None:
    ctx = _FakeCtx(_make_deps(pd.DataFrame()))
    result = await count_by_level(ctx, time_window_minutes=60)  # type: ignore[arg-type]
    assert "No log entries" in result


# ── list_services ─────────────────────────────────────────────────────────────

async def test_list_services_returns_markdown() -> None:
    df = pd.DataFrame({
        "service": ["api-gateway", "auth-service"],
        "log_count": [500, 200],
        "last_seen": ["2026-03-25T10:00:00", "2026-03-25T09:00:00"],
    })
    ctx = _FakeCtx(_make_deps(df))
    result = await list_services(ctx, hours=24)  # type: ignore[arg-type]
    assert "api-gateway" in result


# ── get_trace ─────────────────────────────────────────────────────────────────

async def test_get_trace_returns_markdown() -> None:
    df = pd.DataFrame({
        "timestamp": ["2026-03-25T10:00:00", "2026-03-25T10:00:01"],
        "service": ["api-gateway", "auth-service"],
        "level": ["ERROR", "INFO"],
        "message": ["Request failed", "Token issued"],
    })
    ctx = _FakeCtx(_make_deps(df))
    result = await get_trace(ctx, "abc123")  # type: ignore[arg-type]
    assert "Request failed" in result


async def test_get_trace_not_found() -> None:
    ctx = _FakeCtx(_make_deps(pd.DataFrame()))
    result = await get_trace(ctx, "nonexistent")  # type: ignore[arg-type]
    assert "No log entries found" in result
