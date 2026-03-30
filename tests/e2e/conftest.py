"""E2E test fixtures — httpx AsyncClient wired to the real FastAPI app.

Strategy:
- `app.state.deps` is injected directly (bypasses the real lifespan so no
  MinIO / Lakekeeper / Ollama are required).
- The supervisore LLM is overridden with TestModel in each test that needs it.
- Specialist-agent sub-calls are patched at the supervisore module level so
  TestModel on the supervisore never reaches a real LLM.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.app import app
from src.deps.connections import (
    DocuRagDeps,
    LogDeps,
    SupervisoreDeps,
    ValidatorDeps,
)


# ── App client fixture ────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def client(test_deps: SupervisoreDeps) -> AsyncGenerator[AsyncClient, None]:
    """httpx AsyncClient pointed at the FastAPI app.

    Injects `test_deps` (in-memory DuckDB + seed data) directly into
    `app.state.deps`, bypassing the real lifespan so no Docker services
    are required.
    """
    app.state.deps = test_deps
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as ac:
        yield ac


# ── Patched sub-agent factory ─────────────────────────────────────────────────


@pytest.fixture
def mock_log_run() -> AsyncMock:
    """Patch log_explorer_agent.run() → returns a fixed log-analysis answer."""
    return AsyncMock(
        return_value=MagicMock(
            output="Found 2 errors for api-gateway: Upstream timeout, OOMKilled."
        )
    )


@pytest.fixture
def mock_docu_run() -> AsyncMock:
    """Patch docu_rag_agent.run() → returns a fixed documentation answer."""
    return AsyncMock(
        return_value=MagicMock(
            output="To rollback, run: helm rollback app 1. Source: DB Rollback Runbook."
        )
    )


@pytest.fixture
def mock_validator_run() -> AsyncMock:
    """Patch config_validator_agent.run() → returns a fixed validation summary."""
    return AsyncMock(
        return_value=MagicMock(output="Valid Deployment manifest.")
    )
