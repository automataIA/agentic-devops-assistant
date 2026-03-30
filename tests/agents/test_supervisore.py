"""Unit tests for the Supervisore agent (routing logic, usage propagation)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from pydantic_ai.models.test import TestModel

from src.agents.supervisore import SupervisoreOutput, supervisore_agent
from src.deps.connections import (
    DocuRagDeps,
    LogDeps,
    SupervisoreDeps,
    ValidatorDeps,
)


def _make_test_deps() -> SupervisoreDeps:
    mock_db = MagicMock()
    return SupervisoreDeps(
        db=mock_db,
        log_deps=LogDeps(db=mock_db, minio_endpoint="http://localhost:9000"),
        docu_rag_deps=DocuRagDeps(db=mock_db, embedding_model="nomic-embed-text", embedding_dim=768),
        validator_deps=ValidatorDeps(),
    )


async def test_supervisore_returns_string() -> None:
    """Supervisore with TestModel must return a non-empty string."""
    deps = _make_test_deps()

    with (
        patch(
            "src.agents.supervisore.log_explorer_agent.run",
            new_callable=AsyncMock,
            return_value=MagicMock(output="Found 5 errors."),
        ),
        patch(
            "src.agents.supervisore.docu_rag_agent.run",
            new_callable=AsyncMock,
            return_value=MagicMock(output="Doc answer."),
        ),
        patch(
            "src.agents.supervisore.config_validator_agent.run",
            new_callable=AsyncMock,
            return_value=MagicMock(output="Valid manifest."),
        ),
        supervisore_agent.override(model=TestModel()),
    ):
        result = await supervisore_agent.run(
            "Show me errors for api-gateway in the last hour",
            deps=deps,
        )

    assert isinstance(result.output, str)
    assert result.output != ""


async def test_supervisore_delegation_tools_exist() -> None:
    """All three delegation tools must be registered on the supervisore."""
    tool_names = set(supervisore_agent._function_toolset.tools.keys())  # type: ignore[attr-defined]
    assert "delegate_to_log_explorer" in tool_names
    assert "delegate_to_docu_rag" in tool_names
    assert "delegate_to_config_validator" in tool_names


async def test_supervisore_output_model_fields() -> None:
    """SupervisoreOutput model (used in tests/observability) has correct fields."""
    output = SupervisoreOutput(
        answer="Test answer",
        delegated_to="delegate_to_log_explorer",
        confidence=0.95,
    )
    assert output.delegated_to == "delegate_to_log_explorer"
    assert 0.0 <= output.confidence <= 1.0


async def test_supervisore_delegated_to_can_be_none() -> None:
    """delegated_to is nullable — for ambiguous requests that need clarification."""
    output = SupervisoreOutput(answer="Can you clarify?", delegated_to=None, confidence=0.4)
    assert output.delegated_to is None
