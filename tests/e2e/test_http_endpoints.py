"""E2E — HTTP layer tests.

Verify status codes, response headers, content types, and error handling
for every public endpoint, without any real LLM or external service.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient
from pydantic_ai.models.test import TestModel

from src.agents.supervisore import supervisore_agent


# ── GET / ─────────────────────────────────────────────────────────────────────


async def test_root_returns_200(client: AsyncClient) -> None:
    response = await client.get("/")
    assert response.status_code == 200


async def test_root_content_type_is_html(client: AsyncClient) -> None:
    response = await client.get("/")
    assert "text/html" in response.headers["content-type"]


async def test_root_contains_htmx_script(client: AsyncClient) -> None:
    response = await client.get("/")
    assert "htmx" in response.text.lower()


async def test_root_contains_chat_form(client: AsyncClient) -> None:
    response = await client.get("/")
    assert 'hx-post="/chat/html"' in response.text


async def test_root_contains_daisyui(client: AsyncClient) -> None:
    response = await client.get("/")
    # DaisyUI is compiled into the static CSS bundle — verify the stylesheet is linked
    assert "/static/css/output.css" in response.text


# ── GET /health ───────────────────────────────────────────────────────────────


async def test_health_returns_200(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200


async def test_health_content_type_is_json(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert "application/json" in response.headers["content-type"]


async def test_health_status_field_is_ok(client: AsyncClient) -> None:
    body = (await client.get("/health")).json()
    assert body["status"] == "ok"


async def test_health_duckdb_field_is_true(client: AsyncClient) -> None:
    body = (await client.get("/health")).json()
    assert body["duckdb"] is True


async def test_health_version_field_present(client: AsyncClient) -> None:
    body = (await client.get("/health")).json()
    assert "version" in body
    assert body["version"] == "0.3.0"


# ── POST /chat — HTTP contract ────────────────────────────────────────────────


async def test_chat_returns_200(
    client: AsyncClient,
    mock_log_run: AsyncMock,
    mock_docu_run: AsyncMock,
    mock_validator_run: AsyncMock,
) -> None:
    """POST /chat with a valid message must return HTTP 200."""
    with (
        patch("src.agents.supervisore.log_explorer_agent.run", mock_log_run),
        patch("src.agents.supervisore.docu_rag_agent.run", mock_docu_run),
        patch("src.agents.supervisore.config_validator_agent.run", mock_validator_run),
        supervisore_agent.override(model=TestModel()),
    ):
        response = await client.post("/chat", data={"message": "show errors"})

    assert response.status_code == 200


async def test_chat_content_type_is_event_stream(
    client: AsyncClient,
    mock_log_run: AsyncMock,
    mock_docu_run: AsyncMock,
    mock_validator_run: AsyncMock,
) -> None:
    """POST /chat must return Content-Type: text/event-stream."""
    with (
        patch("src.agents.supervisore.log_explorer_agent.run", mock_log_run),
        patch("src.agents.supervisore.docu_rag_agent.run", mock_docu_run),
        patch("src.agents.supervisore.config_validator_agent.run", mock_validator_run),
        supervisore_agent.override(model=TestModel()),
    ):
        response = await client.post("/chat", data={"message": "show errors"})

    assert "text/event-stream" in response.headers["content-type"]


async def test_chat_cache_control_header(
    client: AsyncClient,
    mock_log_run: AsyncMock,
    mock_docu_run: AsyncMock,
    mock_validator_run: AsyncMock,
) -> None:
    """SSE response must include Cache-Control: no-cache."""
    with (
        patch("src.agents.supervisore.log_explorer_agent.run", mock_log_run),
        patch("src.agents.supervisore.docu_rag_agent.run", mock_docu_run),
        patch("src.agents.supervisore.config_validator_agent.run", mock_validator_run),
        supervisore_agent.override(model=TestModel()),
    ):
        response = await client.post("/chat", data={"message": "show errors"})

    assert response.headers.get("cache-control") == "no-cache"


async def test_chat_nginx_buffering_disabled(
    client: AsyncClient,
    mock_log_run: AsyncMock,
    mock_docu_run: AsyncMock,
    mock_validator_run: AsyncMock,
) -> None:
    """SSE response must include X-Accel-Buffering: no (disables nginx buffering)."""
    with (
        patch("src.agents.supervisore.log_explorer_agent.run", mock_log_run),
        patch("src.agents.supervisore.docu_rag_agent.run", mock_docu_run),
        patch("src.agents.supervisore.config_validator_agent.run", mock_validator_run),
        supervisore_agent.override(model=TestModel()),
    ):
        response = await client.post("/chat", data={"message": "show errors"})

    assert response.headers.get("x-accel-buffering") == "no"


# ── POST /chat — input validation ─────────────────────────────────────────────


async def test_chat_missing_message_returns_422(client: AsyncClient) -> None:
    """POST /chat without the required `message` field must return HTTP 422."""
    response = await client.post("/chat", data={})
    assert response.status_code == 422


async def test_chat_missing_message_error_detail(client: AsyncClient) -> None:
    """422 response body must describe the missing `message` field."""
    response = await client.post("/chat", data={})
    body = response.json()
    # FastAPI validation errors list the field name in the detail array
    field_names = [err["loc"][-1] for err in body.get("detail", [])]
    assert "message" in field_names


async def test_chat_wrong_content_type_returns_422(client: AsyncClient) -> None:
    """Sending JSON instead of form data must return HTTP 422."""
    response = await client.post("/chat", json={"message": "hello"})
    assert response.status_code == 422


# ── POST /chat/html ───────────────────────────────────────────────────────────


async def test_chat_html_returns_200(
    client: AsyncClient,
    mock_log_run: AsyncMock,
    mock_docu_run: AsyncMock,
    mock_validator_run: AsyncMock,
) -> None:
    """POST /chat/html with a valid message must return HTTP 200."""
    with (
        patch("src.agents.supervisore.log_explorer_agent.run", mock_log_run),
        patch("src.agents.supervisore.docu_rag_agent.run", mock_docu_run),
        patch("src.agents.supervisore.config_validator_agent.run", mock_validator_run),
        supervisore_agent.override(model=TestModel()),
    ):
        response = await client.post("/chat/html", data={"message": "show errors"})

    assert response.status_code == 200


async def test_chat_html_content_type_is_html(
    client: AsyncClient,
    mock_log_run: AsyncMock,
    mock_docu_run: AsyncMock,
    mock_validator_run: AsyncMock,
) -> None:
    """POST /chat/html must return Content-Type: text/html."""
    with (
        patch("src.agents.supervisore.log_explorer_agent.run", mock_log_run),
        patch("src.agents.supervisore.docu_rag_agent.run", mock_docu_run),
        patch("src.agents.supervisore.config_validator_agent.run", mock_validator_run),
        supervisore_agent.override(model=TestModel()),
    ):
        response = await client.post("/chat/html", data={"message": "show errors"})

    assert "text/html" in response.headers["content-type"]


async def test_chat_html_contains_chat_bubble(
    client: AsyncClient,
    mock_log_run: AsyncMock,
    mock_docu_run: AsyncMock,
    mock_validator_run: AsyncMock,
) -> None:
    """Response must contain DaisyUI chat-bubble markup."""
    with (
        patch("src.agents.supervisore.log_explorer_agent.run", mock_log_run),
        patch("src.agents.supervisore.docu_rag_agent.run", mock_docu_run),
        patch("src.agents.supervisore.config_validator_agent.run", mock_validator_run),
        supervisore_agent.override(model=TestModel()),
    ):
        response = await client.post("/chat/html", data={"message": "show errors"})

    assert "chat-bubble" in response.text
    assert "chat chat-start" in response.text


async def test_chat_html_missing_message_returns_empty(client: AsyncClient) -> None:
    """POST /chat/html without message and without file returns an empty 200 response."""
    response = await client.post("/chat/html", data={})
    assert response.status_code == 200
    assert response.text == ""


# ── 404 for unknown routes ────────────────────────────────────────────────────


async def test_unknown_route_returns_404(client: AsyncClient) -> None:
    response = await client.get("/nonexistent")
    assert response.status_code == 404
