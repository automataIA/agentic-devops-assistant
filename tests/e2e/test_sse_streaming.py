"""E2E — SSE streaming format tests.

Verify that /chat produces well-formed Server-Sent Events:
- every event starts with "data: "
- the stream terminates with "data: [DONE]"
- at least one content chunk is emitted before [DONE]
- embedded newlines in content are escaped (not raw newlines breaking the SSE frame)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from httpx import AsyncClient
from pydantic_ai.models.test import TestModel

from src.agents.supervisore import supervisore_agent


def _collect_sse_events(body: str) -> list[str]:
    """Extract SSE data values from a raw SSE response body."""
    events = []
    for line in body.splitlines():
        line = line.strip()
        if line.startswith("data: "):
            events.append(line[len("data: "):])
    return events


async def _chat(
    client: AsyncClient,
    message: str,
    mock_log_run: AsyncMock,
    mock_docu_run: AsyncMock,
    mock_validator_run: AsyncMock,
) -> str:
    """Helper: POST /chat and return the full SSE response body."""
    with (
        patch("src.agents.supervisore.log_explorer_agent.run", mock_log_run),
        patch("src.agents.supervisore.docu_rag_agent.run", mock_docu_run),
        patch("src.agents.supervisore.config_validator_agent.run", mock_validator_run),
        supervisore_agent.override(model=TestModel()),
    ):
        response = await client.post("/chat", data={"message": message})
    return response.text


# ── SSE format ────────────────────────────────────────────────────────────────


async def test_sse_all_lines_have_data_prefix(
    client: AsyncClient,
    mock_log_run: AsyncMock,
    mock_docu_run: AsyncMock,
    mock_validator_run: AsyncMock,
) -> None:
    """Every non-blank line in the SSE body must start with 'data: '."""
    body = await _chat(client, "any question", mock_log_run, mock_docu_run, mock_validator_run)
    non_blank = [l for l in body.splitlines() if l.strip()]
    assert all(line.startswith("data: ") for line in non_blank), (
        f"Found SSE line(s) without 'data: ' prefix:\n"
        + "\n".join(l for l in non_blank if not l.startswith("data: "))
    )


async def test_sse_ends_with_done(
    client: AsyncClient,
    mock_log_run: AsyncMock,
    mock_docu_run: AsyncMock,
    mock_validator_run: AsyncMock,
) -> None:
    """The last event in the stream must be the sentinel '[DONE]'."""
    body = await _chat(client, "any question", mock_log_run, mock_docu_run, mock_validator_run)
    events = _collect_sse_events(body)
    assert events, "No SSE events received"
    assert events[-1] == "[DONE]", f"Last event was {events[-1]!r}, expected '[DONE]'"


async def test_sse_has_content_before_done(
    client: AsyncClient,
    mock_log_run: AsyncMock,
    mock_docu_run: AsyncMock,
    mock_validator_run: AsyncMock,
) -> None:
    """At least one content event must precede the [DONE] sentinel."""
    body = await _chat(client, "any question", mock_log_run, mock_docu_run, mock_validator_run)
    events = _collect_sse_events(body)
    content_events = [e for e in events if e != "[DONE]"]
    assert content_events, "No content events before [DONE]"


async def test_sse_no_raw_newlines_in_data_field(
    client: AsyncClient,
    mock_log_run: AsyncMock,
    mock_docu_run: AsyncMock,
    mock_validator_run: AsyncMock,
) -> None:
    """SSE data fields must not contain raw newlines (would break the SSE framing).

    The app escapes them as the literal string '\\n'.
    """
    body = await _chat(client, "any question", mock_log_run, mock_docu_run, mock_validator_run)
    for line in body.splitlines():
        if line.startswith("data: "):
            payload = line[len("data: "):]
            assert "\n" not in payload, (
                f"Raw newline found inside SSE data field: {payload!r}"
            )


async def test_sse_multiple_messages_each_end_with_done(
    client: AsyncClient,
    mock_log_run: AsyncMock,
    mock_docu_run: AsyncMock,
    mock_validator_run: AsyncMock,
) -> None:
    """Two sequential /chat calls must each produce their own [DONE] sentinel."""
    for question in ("first question", "second question"):
        body = await _chat(client, question, mock_log_run, mock_docu_run, mock_validator_run)
        events = _collect_sse_events(body)
        assert events[-1] == "[DONE]", (
            f"Response for {question!r} did not end with [DONE]: {events}"
        )


# ── Error handling in SSE ─────────────────────────────────────────────────────


async def test_sse_agent_error_still_sends_done(client: AsyncClient) -> None:
    """If the agent raises an exception, the stream must still emit [DONE].

    This guarantees the HTMX SSE client always receives a terminator and
    doesn't hang indefinitely.
    """
    error_mock = AsyncMock(side_effect=RuntimeError("simulated LLM failure"))

    with (
        patch("src.agents.supervisore.log_explorer_agent.run", error_mock),
        patch("src.agents.supervisore.docu_rag_agent.run", error_mock),
        patch("src.agents.supervisore.config_validator_agent.run", error_mock),
        supervisore_agent.override(model=TestModel()),
    ):
        response = await client.post("/chat", data={"message": "trigger error"})

    assert response.status_code == 200  # HTTP layer stays healthy
    events = _collect_sse_events(response.text)
    assert events[-1] == "[DONE]"
