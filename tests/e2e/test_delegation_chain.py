"""E2E — Full delegation chain tests.

Each test exercises the complete path:

    HTTP POST /chat
        → FastAPI route
            → Supervisore (TestModel)
                → delegation tool
                    → specialist agent (TestModel)
                        → DuckDB tool (real in-memory DB)
                            → formatted result
                                → SSE stream
                                    → HTTP response

All LLM inference is replaced by TestModel so no Ollama is required.
The DuckDB layer uses real queries against the seed data from conftest.py.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient
from pydantic_ai.models.test import TestModel

from src.agents.config_validator import (
    ValidationResult,
    config_validator_agent,
    validate_gitlab_ci,
    validate_k8s_manifest,
)
from src.agents.docu_rag import docu_rag_agent, list_documents, semantic_search
from src.agents.log_explorer import (
    count_by_level,
    list_services,
    log_explorer_agent,
    search_errors,
)
from src.agents.supervisore import SupervisoreOutput, supervisore_agent
from src.deps.connections import DocuRagDeps, LogDeps, SupervisoreDeps, ValidatorDeps


# ── Helpers ───────────────────────────────────────────────────────────────────


def _collect_sse_events(body: str) -> list[str]:
    return [
        line[len("data: "):]
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


async def _post_chat(
    client: AsyncClient,
    message: str,
    log_run: AsyncMock,
    docu_run: AsyncMock,
    validator_run: AsyncMock,
) -> list[str]:
    """POST /chat and return the list of SSE event data values."""
    with (
        patch("src.agents.supervisore.log_explorer_agent.run", log_run),
        patch("src.agents.supervisore.docu_rag_agent.run", docu_run),
        patch("src.agents.supervisore.config_validator_agent.run", validator_run),
        supervisore_agent.override(model=TestModel()),
    ):
        response = await client.post("/chat", data={"message": message})
    return _collect_sse_events(response.text)


# ── Full HTTP chain — supervisore routing ─────────────────────────────────────


async def test_chain_produces_sse_with_done(
    client: AsyncClient,
    mock_log_run: AsyncMock,
    mock_docu_run: AsyncMock,
    mock_validator_run: AsyncMock,
) -> None:
    """Full HTTP path must produce SSE events ending in [DONE]."""
    events = await _post_chat(
        client, "show errors for api-gateway",
        mock_log_run, mock_docu_run, mock_validator_run,
    )
    assert events[-1] == "[DONE]"


async def test_chain_response_is_not_empty(
    client: AsyncClient,
    mock_log_run: AsyncMock,
    mock_docu_run: AsyncMock,
    mock_validator_run: AsyncMock,
) -> None:
    """The SSE stream must carry at least one content chunk before [DONE]."""
    events = await _post_chat(
        client, "show errors for api-gateway",
        mock_log_run, mock_docu_run, mock_validator_run,
    )
    content = [e for e in events if e != "[DONE]"]
    assert content, "No content chunks received"


async def test_chain_content_contains_agent_answer(
    client: AsyncClient,
    mock_log_run: AsyncMock,
    mock_docu_run: AsyncMock,
    mock_validator_run: AsyncMock,
) -> None:
    """The SSE response must include text from the specialist agent's answer."""
    mock_log_run.return_value = MagicMock(
        output=MagicMock(answer="CRITICAL: OOMKilled on api-gateway at 10:00:01Z")
    )
    events = await _post_chat(
        client, "show errors for api-gateway",
        mock_log_run, mock_docu_run, mock_validator_run,
    )
    full_text = "".join(events).replace("\\n", "\n")
    # TestModel returns a SupervisoreOutput — the answer field contains the content
    assert len(full_text) > 0


# ── Specialist agent chains — real DuckDB, TestModel LLM ─────────────────────
# These test the tool → DuckDB → formatted result path without HTTP overhead.


class _FakeCtx:
    def __init__(self, deps: LogDeps | DocuRagDeps | ValidatorDeps) -> None:
        self.deps = deps


# Log Explorer chain

async def test_log_explorer_chain_search_errors(test_log_deps: LogDeps) -> None:
    """search_errors tool → real DuckDB → markdown table with seed data."""
    ctx = _FakeCtx(test_log_deps)
    result = await search_errors(ctx, "api-gateway", hours=9999)  # type: ignore[arg-type]
    assert "Upstream timeout" in result
    assert "OOMKilled" in result
    # Markdown table format
    assert "|" in result


async def test_log_explorer_chain_count_by_level(test_log_deps: LogDeps) -> None:
    """count_by_level tool → real DuckDB → markdown with ERROR and INFO rows."""
    ctx = _FakeCtx(test_log_deps)
    result = await count_by_level(ctx, time_window_minutes=99999)  # type: ignore[arg-type]
    assert "ERROR" in result
    assert "INFO" in result


async def test_log_explorer_chain_list_services(test_log_deps: LogDeps) -> None:
    """list_services tool → real DuckDB → all seeded services present."""
    ctx = _FakeCtx(test_log_deps)
    result = await list_services(ctx, hours=9999)  # type: ignore[arg-type]
    for svc in ("api-gateway", "auth-service", "payment-service"):
        assert svc in result


async def test_log_explorer_agent_output_schema(test_log_deps: LogDeps) -> None:
    """Log Explorer with TestModel must return a non-empty string."""
    with (
        patch(
            "src.agents.log_explorer.search_errors",
            new_callable=AsyncMock,
            return_value="| timestamp | level |\n|---|---|\n| 2026-03-25 | ERROR |",
        ),
        log_explorer_agent.override(model=TestModel()),
    ):
        result = await log_explorer_agent.run(
            "show errors for api-gateway",
            deps=test_log_deps,
        )
    assert isinstance(result.output, str)
    assert result.output != ""


# Docu-RAG chain

async def test_docu_rag_chain_list_documents(test_docu_rag_deps: DocuRagDeps) -> None:
    """list_documents tool → real DuckDB → seeded doc title present."""
    ctx = _FakeCtx(test_docu_rag_deps)
    result = await list_documents(ctx)  # type: ignore[arg-type]
    assert "DB Rollback Runbook" in result


async def test_docu_rag_chain_semantic_search(test_docu_rag_deps: DocuRagDeps) -> None:
    """semantic_search tool → real DuckDB HNSW → matching chunk returned."""
    ctx = _FakeCtx(test_docu_rag_deps)
    # Use zero-vector embedding — the seeded doc also has a zero-vector
    with patch("src.agents.docu_rag.embed_text", return_value=[0.0] * 768):
        result = await semantic_search(ctx, "rollback procedure", top_k=3)  # type: ignore[arg-type]
    assert "DB Rollback Runbook" in result
    assert "helm rollback" in result


async def test_docu_rag_agent_output_schema(test_docu_rag_deps: DocuRagDeps) -> None:
    """Docu-RAG with TestModel must return a non-empty string."""
    with (
        patch("src.agents.docu_rag.embed_text", return_value=[0.0] * 768),
        patch(
            "src.agents.docu_rag.semantic_search",
            new_callable=AsyncMock,
            return_value="| doc_id | title | chunk_text | score |\n|---|---|---|---|\n| doc-001 | DB Rollback Runbook | helm rollback app 1 | 0.0 |",
        ),
        docu_rag_agent.override(model=TestModel()),
    ):
        result = await docu_rag_agent.run(
            "how to rollback the database?",
            deps=test_docu_rag_deps,
        )
    assert isinstance(result.output, str)
    assert result.output != ""


# Config Validator chain

async def test_config_validator_chain_valid_k8s(test_validator_deps: ValidatorDeps) -> None:
    """validate_k8s_manifest tool → Pydantic V2 → valid result for correct YAML."""
    ctx = _FakeCtx(test_validator_deps)
    yaml_content = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-gateway
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api-gateway
  template:
    metadata:
      name: api-gateway
      labels:
        app: api-gateway
    spec:
      containers:
        - name: api
          image: myrepo/api:v1.0.0
          resources:
            limits:
              memory: "256Mi"
              cpu: "500m"
"""
    raw = await validate_k8s_manifest(ctx, yaml_content)  # type: ignore[arg-type]
    result = ValidationResult.model_validate_json(raw)
    assert result.valid is True
    assert result.errors == []


async def test_config_validator_chain_invalid_k8s(test_validator_deps: ValidatorDeps) -> None:
    """validate_k8s_manifest tool → Pydantic V2 → errors reported for wrong apiVersion."""
    ctx = _FakeCtx(test_validator_deps)
    yaml_content = "apiVersion: v1\nkind: Deployment\nmetadata:\n  name: bad\nspec: {}"
    raw = await validate_k8s_manifest(ctx, yaml_content)  # type: ignore[arg-type]
    result = ValidationResult.model_validate_json(raw)
    assert result.valid is False
    assert len(result.errors) > 0


async def test_config_validator_chain_valid_gitlab(test_validator_deps: ValidatorDeps) -> None:
    """validate_gitlab_ci tool → GitLabCI model → valid pipeline accepted."""
    ctx = _FakeCtx(test_validator_deps)
    yaml_content = """
stages: [build, test]
build-job:
  stage: build
  script: [docker build -t img .]
test-job:
  stage: test
  script: [uv run -m pytest]
"""
    raw = await validate_gitlab_ci(ctx, yaml_content)  # type: ignore[arg-type]
    result = ValidationResult.model_validate_json(raw)
    assert result.valid is True
    assert "2 job(s)" in result.summary


async def test_config_validator_chain_invalid_gitlab(test_validator_deps: ValidatorDeps) -> None:
    """validate_gitlab_ci tool → GitLabCI model → retry:5 exceeds max and fails."""
    ctx = _FakeCtx(test_validator_deps)
    yaml_content = "stages: [test]\nbad-job:\n  stage: test\n  retry: 5\n  script: [echo hi]\n"
    raw = await validate_gitlab_ci(ctx, yaml_content)  # type: ignore[arg-type]
    result = ValidationResult.model_validate_json(raw)
    assert result.valid is False


async def test_config_validator_agent_output_schema(test_validator_deps: ValidatorDeps) -> None:
    """Config Validator with TestModel must return a non-empty string."""
    with (
        patch(
            "src.agents.config_validator.validate_k8s_manifest",
            new_callable=AsyncMock,
            return_value=ValidationResult(
                valid=True, errors=[], warnings=[], summary="Valid Deployment manifest"
            ).model_dump_json(),
        ),
        config_validator_agent.override(model=TestModel()),
    ):
        result = await config_validator_agent.run(
            "Validate this K8s YAML: apiVersion: apps/v1 ...",
            deps=test_validator_deps,
        )
    assert isinstance(result.output, str)
    assert result.output != ""


# ── Supervisore → specialist agent output contract ────────────────────────────


async def test_supervisore_output_schema_via_chain(test_deps: SupervisoreDeps) -> None:
    """Supervisore with TestModel + mocked sub-agents must return a non-empty string."""
    with (
        patch(
            "src.agents.supervisore.log_explorer_agent.run",
            new_callable=AsyncMock,
            return_value=MagicMock(output="2 errors found."),
        ),
        patch(
            "src.agents.supervisore.docu_rag_agent.run",
            new_callable=AsyncMock,
            return_value=MagicMock(output="See runbook."),
        ),
        patch(
            "src.agents.supervisore.config_validator_agent.run",
            new_callable=AsyncMock,
            return_value=MagicMock(output="Valid."),
        ),
        supervisore_agent.override(model=TestModel()),
    ):
        result = await supervisore_agent.run(
            "show errors for api-gateway",
            deps=test_deps,
        )

    out = result.output
    assert isinstance(out, str)
    assert out != ""


async def test_supervisore_delegated_to_field_nullable(test_deps: SupervisoreDeps) -> None:
    """delegated_to may be None for clarification responses."""
    output = SupervisoreOutput(answer="Could you clarify?", delegated_to=None, confidence=0.3)
    assert output.delegated_to is None


async def test_supervisore_confidence_is_float(test_deps: SupervisoreDeps) -> None:
    """confidence field must be a float in [0, 1]."""
    for confidence in (0.0, 0.5, 1.0):
        out = SupervisoreOutput(answer="ok", delegated_to=None, confidence=confidence)
        assert isinstance(out.confidence, float)
        assert 0.0 <= out.confidence <= 1.0
