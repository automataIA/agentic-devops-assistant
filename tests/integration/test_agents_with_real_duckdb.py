"""Integration tests — real DuckDB in-memory, TestModel LLM (no network).

These tests exercise the full pipeline from fixtures → agent tools → output
without requiring live MinIO, Lakekeeper, or Ollama services.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.agents.log_explorer import count_by_level, list_services, search_errors
from src.agents.docu_rag import list_documents, semantic_search
from src.deps.connections import DocuRagDeps, LogDeps


class _FakeCtx:
    def __init__(self, deps: LogDeps | DocuRagDeps) -> None:
        self.deps = deps


# ── Log Explorer with real DuckDB ─────────────────────────────────────────────

async def test_search_errors_with_real_db(test_log_deps: LogDeps) -> None:
    ctx = _FakeCtx(test_log_deps)
    result = await search_errors(ctx, "api-gateway", hours=9999)  # type: ignore[arg-type]
    assert "Upstream timeout" in result
    assert "OOMKilled" in result


async def test_search_errors_no_match(test_log_deps: LogDeps) -> None:
    ctx = _FakeCtx(test_log_deps)
    result = await search_errors(ctx, "nonexistent-service", hours=1)  # type: ignore[arg-type]
    assert "No errors found" in result


async def test_count_by_level_with_real_db(test_log_deps: LogDeps) -> None:
    ctx = _FakeCtx(test_log_deps)
    result = await count_by_level(ctx, time_window_minutes=99999)  # type: ignore[arg-type]
    assert "ERROR" in result
    assert "INFO" in result


async def test_list_services_with_real_db(test_log_deps: LogDeps) -> None:
    ctx = _FakeCtx(test_log_deps)
    result = await list_services(ctx, hours=9999)  # type: ignore[arg-type]
    assert "api-gateway" in result
    assert "auth-service" in result


# ── Docu-RAG with real DuckDB ─────────────────────────────────────────────────

async def test_list_documents_with_real_db(test_docu_rag_deps: DocuRagDeps) -> None:
    ctx = _FakeCtx(test_docu_rag_deps)
    result = await list_documents(ctx)  # type: ignore[arg-type]
    assert "DB Rollback Runbook" in result


async def test_semantic_search_with_real_db(test_docu_rag_deps: DocuRagDeps) -> None:
    ctx = _FakeCtx(test_docu_rag_deps)
    fake_embedding = [0.0] * 768
    with patch("src.agents.docu_rag.embed_text", return_value=fake_embedding):
        result = await semantic_search(ctx, "rollback database", top_k=3)  # type: ignore[arg-type]
    assert "DB Rollback Runbook" in result
