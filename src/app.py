"""FastAPI application with HTMX frontend and SSE streaming."""

from __future__ import annotations

import json
import logging
import os
import re
import uuid

# Configure src.* loggers independently of uvicorn's dictConfig.
# We add a dedicated StreamHandler with propagate=False so uvicorn's
# log setup (which resets the root logger) cannot silence our DEBUG output.
_src_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
_src_logger = logging.getLogger("src")
_src_logger.setLevel(_src_log_level)
if not _src_logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s"))
    _src_logger.addHandler(_h)
    _src_logger.propagate = False
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated

import duckdb
import logfire
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Request, Response, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import TypeAdapter
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)

from src.agents.config_validator import config_validator_agent
from src.agents.config_validator_v2 import _OLLAMA_BACKEND, config_validator_v2_agent
from src.tools.ollama_output import ollama_structured_finish
from src.agents.supervisore import supervisore_agent
from src.deps.connections import SupervisoreDeps, ValidatorDeps, build_deps
from src.models.validation_session import FixRecord, ValidationSession, ValidatorV2Output
from src.tools.async_db import async_query
from src.tools.ingest_multi import ingest_source
from src.tools.schema_store_local import delete_schema, get_schema, list_schemas, store_schema
from src.tools.yaml_annotator import annotate_yaml
from src.tools.yaml_repair import repair_yaml

load_dotenv()

logger = logging.getLogger(__name__)

# ── Templates ─────────────────────────────────────────────────────────────────

templates = Jinja2Templates(directory=str(os.path.join(os.path.dirname(__file__), "templates")))

# ── Agent label map ───────────────────────────────────────────────────────────

_AGENT_LABELS: dict[str, str] = {
    "delegate_to_log_explorer": "Log Explorer",
    "delegate_to_docu_rag": "Docu-RAG",
    "delegate_to_config_validator": "Config Validator",
}

# ── Prefix command map ────────────────────────────────────────────────────────

_PREFIX_COMMANDS: dict[str, str] = {
    "/logs": "delegate_to_log_explorer",
    "/docs": "delegate_to_docu_rag",
    "/validate": "delegate_to_config_validator",
}


# ── User identification ───────────────────────────────────────────────────────


def _get_or_create_user(db: duckdb.DuckDBPyConnection, request: Request, response: Response | None = None) -> str:
    """Get user_id from cookie, or create a new anonymous user."""
    user_id = request.cookies.get("sre_user_id")
    if user_id:
        # Verify user exists
        row = db.execute("SELECT user_id FROM users WHERE user_id = ?", [user_id]).fetchone()
        if row:
            return user_id
    # Create new user
    user_id = str(uuid.uuid4())
    db.execute(
        "INSERT INTO users (user_id, display_name) VALUES (?, ?)",
        [user_id, f"User-{user_id[:8]}"],
    )
    return user_id


def _set_user_cookie(response: Response, user_id: str) -> None:
    """Set user identification cookie on response."""
    response.set_cookie(
        "sre_user_id", user_id,
        httponly=True, samesite="lax", max_age=60 * 60 * 24 * 365,
    )


# ── Conversation helpers ──────────────────────────────────────────────────────


def _create_conversation(db: duckdb.DuckDBPyConnection, user_id: str, title: str = "New Chat") -> str:
    """Create a new conversation and return its ID."""
    conv_id = str(uuid.uuid4())
    db.execute(
        "INSERT INTO conversations (conversation_id, user_id, title) VALUES (?, ?, ?)",
        [conv_id, user_id, title],
    )
    return conv_id


def _save_message(
    db: duckdb.DuckDBPyConnection,
    conversation_id: str,
    role: str,
    content: str,
    agent: str | None = None,
    sources: list[str] | None = None,
    chunk_sources: list[dict] | None = None,
    is_error: bool = False,
) -> str:
    """Save a message and return its ID."""
    msg_id = str(uuid.uuid4())
    db.execute(
        """INSERT INTO messages (message_id, conversation_id, role, content, agent, sources, chunk_sources, is_error)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            msg_id, conversation_id, role, content, agent,
            json.dumps(sources or []),
            json.dumps(chunk_sources or []),
            is_error,
        ],
    )
    return msg_id


def _update_conversation_title(db: duckdb.DuckDBPyConnection, conversation_id: str, title: str) -> None:
    """Update conversation title (from first message)."""
    db.execute(
        "UPDATE conversations SET title = ?, updated_at = now() WHERE conversation_id = ?",
        [title[:100], conversation_id],
    )


def _parse_prefix_command(message: str) -> tuple[str | None, str]:
    """Parse prefix command from message. Returns (agent_tool_name, cleaned_message)."""
    stripped = message.strip()
    for prefix, tool_name in _PREFIX_COMMANDS.items():
        if stripped.lower().startswith(prefix):
            remaining = stripped[len(prefix):].strip()
            return tool_name, remaining or stripped  # fallback to original if no remaining text
    return None, message


def _load_conversation_history(
    db: duckdb.DuckDBPyConnection,
    conversation_id: str,
    max_turns: int = 10,
) -> list[ModelMessage]:
    """Load the last *max_turns* user/assistant pairs for a conversation as ModelMessages."""
    rows = db.execute(
        """SELECT role, content FROM messages
           WHERE conversation_id = ?
           ORDER BY created_at DESC
           LIMIT ?""",
        [conversation_id, max_turns * 2],
    ).fetchall()
    # rows come newest-first; reverse to chronological order
    history: list[ModelMessage] = []
    for role, content in reversed(rows):
        if role == "user":
            history.append(ModelRequest(parts=[UserPromptPart(content=content)]))
        else:
            history.append(ModelResponse(parts=[TextPart(content=content)], model_name="history"))
    return history


# ── Message (de)serialisation ─────────────────────────────────────────────────

_msg_adapter: TypeAdapter[list[ModelMessage]] = TypeAdapter(list[ModelMessage])


def _serialize_messages(messages: list[ModelMessage]) -> str:
    return _msg_adapter.dump_json(messages).decode()


def _deserialize_messages(json_str: str) -> list[ModelMessage]:
    return _msg_adapter.validate_json(json_str)


# ── Session helpers (DuckDB) ──────────────────────────────────────────────────


def _load_session(db: duckdb.DuckDBPyConnection, session_id: str) -> ValidationSession | None:
    row = db.execute(
        """SELECT session_id, filename, yaml_original, yaml_current,
                  fixes_applied, message_history, status
           FROM validation_sessions WHERE session_id = ?""",
        [session_id],
    ).fetchone()
    if row is None:
        return None
    return ValidationSession(
        session_id=row[0],
        filename=row[1],
        yaml_original=row[2],
        yaml_current=row[3],
        fixes_applied=[FixRecord(**f) for f in (json.loads(row[4]) if row[4] else [])],
        message_history=row[5] or "",
        status=row[6],
    )


def _create_session(
    db: duckdb.DuckDBPyConnection,
    session_id: str,
    filename: str,
    yaml_original: str,
    output: ValidatorV2Output,
    messages: list[ModelMessage],
) -> None:
    status = "done" if output.is_done else "awaiting_input"
    db.execute(
        """INSERT INTO validation_sessions
           (session_id, filename, yaml_original, yaml_current, fixes_applied, message_history, status)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [
            session_id,
            filename,
            yaml_original,
            output.yaml_current,
            json.dumps([f.model_dump() for f in output.fixes_applied]),
            _serialize_messages(messages),
            status,
        ],
    )


def _update_session(
    db: duckdb.DuckDBPyConnection,
    session: ValidationSession,
    output: ValidatorV2Output,
    messages: list[ModelMessage],
) -> None:
    all_fixes = session.fixes_applied + output.fixes_applied
    status = "done" if output.is_done else "awaiting_input"
    db.execute(
        """UPDATE validation_sessions
           SET yaml_current = ?, fixes_applied = ?, message_history = ?, status = ?
           WHERE session_id = ?""",
        [
            output.yaml_current,
            json.dumps([f.model_dump() for f in all_fixes]),
            _serialize_messages(messages),
            status,
            session.session_id,
        ],
    )


# ── Source extraction helpers ─────────────────────────────────────────────────


def _extract_sources(text: str) -> list[str]:
    """Extract citation sources from agent response text (max 5).

    Three heuristic passes:
    1. Explicit ``Source:`` / ``Sources:`` marker.
    2. Square-bracket references: ``[title]`` (skips pure numerics like ``[1]``).
    3. Inline file references ending in .md / .yaml / .yml / .json.
    """
    sources: list[str] = []

    # Pass 1 — explicit source marker
    marker_match = re.search(r"[Ss]ources?:\s*(.+)", text)
    if marker_match:
        for s in re.split(r"[,;]", marker_match.group(1)):
            s = s.strip().strip(".")
            if s and s not in sources:
                sources.append(s)

    # Pass 2 — [bracketed references] (skip pure numbers)
    for m in re.finditer(r"\[([^\]]+)\]", text):
        label = m.group(1).strip()
        if label and not re.fullmatch(r"\d+", label) and label not in sources:
            sources.append(label)

    # Pass 3 — file extensions
    for m in re.finditer(r"\b([\w\-./]+\.(?:md|yaml|yml|json))\b", text):
        label = m.group(1)
        if label not in sources:
            sources.append(label)

    return sources[:5]


def _detect_agent(messages: list) -> str | None:  # type: ignore[type-arg]
    """Return a human-readable label for the specialist agent that was called."""
    for msg in messages:
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ToolCallPart):
                    return _AGENT_LABELS.get(part.tool_name)
    return None


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Initialise shared resources at startup, release at shutdown."""
    deps = await build_deps()
    app.state.deps = deps

    logfire_token = os.getenv("LOGFIRE_TOKEN")
    if logfire_token:
        logfire.configure(token=logfire_token)
        logfire.instrument_fastapi(app)

    yield

    deps.db.close()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SRE/DevOps Knowledge Copilot",
    description="4-agent assistant for SRE teams: logs, docs, config validation.",
    version="0.2.0",
    lifespan=lifespan,
)

# ── Static files ─────────────────────────────────────────────────────────────

app.mount(
    "/static",
    StaticFiles(directory=str(os.path.join(os.path.dirname(__file__), "static"))),
    name="static",
)


# ── SSE helpers ───────────────────────────────────────────────────────────────


async def _stream_response(
    message: str,
    deps: SupervisoreDeps,
) -> AsyncGenerator[str]:
    """Stream SSE events from the Supervisore agent."""
    try:
        async with supervisore_agent.run_stream(message, deps=deps) as stream:
            async for chunk in stream.stream_text(delta=True):
                safe_chunk = chunk.replace("\n", "\\n")
                yield f"data: {safe_chunk}\n\n"
    except Exception as exc:  # noqa: BLE001
        error_msg = f"Agent error: {exc}"
        yield f"data: {error_msg}\n\n"
    finally:
        yield "data: [DONE]\n\n"


# ── Validation session handlers ───────────────────────────────────────────────


async def _handle_dynamic_validation(
    request: Request,
    filename: str,
    yaml_str: str,
    prompt: str,
) -> HTMLResponse:
    """Validate non-K8s/GitLab YAML using dynamic JSON Schema (SchemaStore + OpenAPI)."""
    try:
        result = await config_validator_agent.run(prompt, deps=ValidatorDeps())
        content = result.output
        is_error = False
    except Exception as exc:  # noqa: BLE001
        content = f"Validation error: {exc}"
        is_error = True

    return templates.TemplateResponse(  # type: ignore[return-value]
        request=request,
        name="_message.html",
        context={
            "content": content,
            "agent": "Config Validator",
            "is_error": is_error,
            "sources": [],
            "chunk_sources": [],
            "pending_question": None,
            "session_id": None,
            "show_export": False,
        },
    )


async def _handle_yaml_upload(
    request: Request,
    file: UploadFile,
    message: str,
    deps: SupervisoreDeps,
) -> HTMLResponse:
    """Handle a YAML file upload: create session, run agent turn 1."""
    raw = await file.read()

    if len(raw) > 512_000:
        response = templates.TemplateResponse(
            request=request,
            name="_message.html",
            context={
                "content": "File too large (max 512 KB). Please reduce the YAML size.",
                "agent": "Config Validator",
                "is_error": True,
                "sources": [],
                "chunk_sources": [],
                "pending_question": None,
                "session_id": None,
                "show_export": False,
            },
        )
        return response  # type: ignore[return-value]

    try:
        yaml_str = raw.decode("utf-8")
        yaml.safe_load(yaml_str)  # YAML bomb / syntax check
    except Exception as exc:
        response = templates.TemplateResponse(
            request=request,
            name="_message.html",
            context={
                "content": f"Invalid YAML file: {exc}",
                "agent": "Config Validator",
                "is_error": True,
                "sources": [],
                "chunk_sources": [],
                "pending_question": None,
                "session_id": None,
                "show_export": False,
            },
        )
        return response  # type: ignore[return-value]

    # ── Active reference schema → static repair path ─────────────────────────
    _active_schema_id = request.cookies.get("active_schema_id", "").strip()
    if _active_schema_id:
        _schema_content = get_schema(deps.db, _active_schema_id)
        if _schema_content is not None:
            return await _handle_static_repair(
                request, file.filename or "manifest.yaml", yaml_str, _schema_content, deps
            )

    # Detect YAML type to route to the right agent.
    # K8s (apiVersion+kind) and GitLab CI (no kind but job structure) → v2 multi-turn agent.
    # Everything else (GitHub Actions, Docker Compose, OpenAPI, Helm, etc.) → dynamic agent.
    try:
        _parsed = yaml.safe_load(yaml_str) or {}
    except Exception:
        _parsed = {}

    _is_k8s = bool(_parsed.get("apiVersion") and _parsed.get("kind"))
    _is_openapi = bool(_parsed.get("openapi") or _parsed.get("swagger"))
    _is_gitlab = not _is_k8s and not _is_openapi and (
        "stages" in _parsed
        or any(isinstance(v, dict) and "script" in v for v in _parsed.values())
    )
    _use_v2 = _is_k8s or _is_gitlab

    context_note = f"\nUser context: {message}" if message.strip() else ""
    prompt = (
        f"Validate this YAML manifest.\n"
        f"Filename: {file.filename}{context_note}\n\n"
        f"YAML content:\n{yaml_str}"
    )

    if not _use_v2:
        # Dynamic schema validation path (GitHub Actions, Docker Compose, OpenAPI, etc.)
        return await _handle_dynamic_validation(request, file.filename or "manifest.yaml", yaml_str, prompt)

    try:
        result = await config_validator_v2_agent.run(
            prompt, message_history=[], deps=ValidatorDeps(db=deps.db)
        )
        raw = result.output
        if isinstance(raw, str):
            # Ollama backend: tool calls completed (output_type=str), now extract
            # ValidatorV2Output via Ollama's native format=schema (grammar-constrained).
            output: ValidatorV2Output = await ollama_structured_finish(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a YAML validation data extractor. "
                            "Given a validation report, extract the structured result as JSON. "
                            "yaml_current must contain the complete final YAML string after all "
                            "fixes applied. If no fixes were applied, use the original YAML."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Original YAML:\n{yaml_str}\n\n"
                            f"Validation report:\n{raw}\n\n"
                            "Extract this as a ValidatorV2Output JSON object."
                        ),
                    },
                ],
                output_type=ValidatorV2Output,
            )
        else:
            output = raw
    except Exception as exc:  # noqa: BLE001
        response = templates.TemplateResponse(
            request=request,
            name="_message.html",
            context={
                "content": f"Validation agent error: {exc}",
                "agent": "Config Validator",
                "is_error": True,
                "sources": [],
                "chunk_sources": [],
                "pending_question": None,
                "session_id": None,
                "show_export": False,
            },
        )
        return response  # type: ignore[return-value]

    session_id = str(uuid.uuid4())
    _create_session(
        deps.db,
        session_id,
        file.filename or "manifest.yaml",
        yaml_str,
        output,
        result.all_messages(),
    )

    response = templates.TemplateResponse(
        request=request,
        name="_message.html",
        context={
            "content": output.message,
            "agent": "Config Validator",
            "is_error": False,
            "sources": [],
            "chunk_sources": [],
            "pending_question": output.pending_question,
            "session_id": session_id,
            "show_export": output.is_done,
        },
    )
    if not output.is_done:
        response.set_cookie("validation_session_id", session_id, httponly=True, samesite="lax")  # type: ignore[union-attr]
    return response  # type: ignore[return-value]


async def _handle_static_repair(
    request: Request,
    filename: str,
    yaml_str: str,
    schema_content: dict,
    deps: SupervisoreDeps,
) -> HTMLResponse:
    """Validate and repair a YAML file against an active reference schema (no LLM)."""
    try:
        result = repair_yaml(yaml_str, schema_content)
        lines: list[str] = [result.summary]
        if result.errors:
            lines.append(f"\nIssues found ({len(result.errors)}):")
            for e in result.errors:
                lines.append(f"  • {e}")
        if result.llm_fields:
            lines.append(f"\nFields requiring manual review: {', '.join(result.llm_fields)}")
        content = "\n".join(lines)

        session_id = str(uuid.uuid4())
        synthetic_output = ValidatorV2Output(
            message=result.summary,
            yaml_current=result.repaired_yaml,
            fixes_applied=[],
            pending_question=None,
            is_done=True,
        )
        # Store errors as a synthetic first message so fix-with-ai can read them
        from pydantic_ai.messages import ModelRequest, UserPromptPart
        error_meta = ModelRequest(parts=[UserPromptPart(content=json.dumps({
            "errors": result.errors,
            "llm_fields": result.llm_fields,
        }))])
        _create_session(deps.db, session_id, filename, yaml_str, synthetic_output, [error_meta])

        # Show download only if fully valid; if errors remain show Fix with AI instead
        show_export = result.valid

        response = templates.TemplateResponse(  # type: ignore[return-value]
            request=request,
            name="_message.html",
            context={
                "content": content,
                "agent": "Config Validator",
                "is_error": not result.valid,
                "sources": [],
                "chunk_sources": [],
                "pending_question": None,
                "session_id": session_id,
                "show_export": show_export,
                "show_fix_with_ai": not result.valid,
            },
        )
        return response
    except Exception as exc:  # noqa: BLE001
        return templates.TemplateResponse(  # type: ignore[return-value]
            request=request,
            name="_message.html",
            context={
                "content": f"Static repair error: {exc}",
                "agent": "Config Validator",
                "is_error": True,
                "sources": [],
                "chunk_sources": [],
                "pending_question": None,
                "session_id": None,
                "show_export": False,
            },
        )


async def _handle_session_turn(
    request: Request,
    session: ValidationSession,
    message: str,
    deps: SupervisoreDeps,
) -> HTMLResponse:
    """Continue an active validation session with the user's answer."""
    history = _deserialize_messages(session.message_history)
    prompt = f"User chose: {message}\n\nCurrent YAML:\n{session.yaml_current}"

    try:
        result = await config_validator_v2_agent.run(
            prompt, message_history=history, deps=ValidatorDeps(db=deps.db)
        )
        raw = result.output
        if isinstance(raw, str):
            # Ollama backend: extract ValidatorV2Output via native format=schema.
            output: ValidatorV2Output = await ollama_structured_finish(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a YAML validation data extractor. "
                            "Given a validation report, extract the structured result as JSON. "
                            "yaml_current must contain the complete final YAML after all fixes."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Current YAML:\n{session.yaml_current}\n\n"
                            f"Validation report:\n{raw}\n\n"
                            "Extract this as a ValidatorV2Output JSON object."
                        ),
                    },
                ],
                output_type=ValidatorV2Output,
            )
        else:
            output = raw
    except Exception as exc:  # noqa: BLE001
        response = templates.TemplateResponse(
            request=request,
            name="_message.html",
            context={
                "content": f"Validation agent error: {exc}",
                "agent": "Config Validator",
                "is_error": True,
                "sources": [],
                "chunk_sources": [],
                "pending_question": None,
                "session_id": session.session_id,
                "show_export": False,
            },
        )
        return response  # type: ignore[return-value]

    _update_session(deps.db, session, output, result.all_messages())

    response = templates.TemplateResponse(
        request=request,
        name="_message.html",
        context={
            "content": output.message,
            "agent": "Config Validator",
            "is_error": False,
            "sources": [],
            "chunk_sources": [],
            "pending_question": output.pending_question,
            "session_id": session.session_id,
            "show_export": output.is_done,
        },
    )
    if output.is_done:
        response.delete_cookie("validation_session_id")  # type: ignore[union-attr]
    return response  # type: ignore[return-value]


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Serve the chat UI."""
    deps: SupervisoreDeps = request.app.state.deps
    user_id = _get_or_create_user(deps.db, request)
    response = templates.TemplateResponse(request=request, name="index.html")
    _set_user_cookie(response, user_id)
    return response


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    """Serve an inline SVG terminal-prompt favicon."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
        '<rect width="32" height="32" rx="4" fill="#1d2433"/>'
        '<text x="4" y="23" font-family="monospace" font-size="18" '
        'font-weight="bold" fill="#7aa2f7">&gt;_</text>'
        "</svg>"
    )
    return Response(content=svg, media_type="image/svg+xml")


@app.post("/chat")
async def chat(
    request: Request,
    message: Annotated[str, Form()],
) -> StreamingResponse:
    """Accept a user message and stream the agent response via SSE."""
    deps: SupervisoreDeps = request.app.state.deps
    return StreamingResponse(
        _stream_response(message, deps),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/chat/html", response_class=HTMLResponse)
async def chat_html(
    request: Request,
    message: Annotated[str, Form()] = "",
    file: Annotated[UploadFile | None, File()] = None,
    conversation_id: Annotated[str, Form()] = "",
) -> HTMLResponse:
    """Run the Supervisore agent (or validation session) and return an HTML fragment."""
    deps: SupervisoreDeps = request.app.state.deps
    user_id = _get_or_create_user(deps.db, request)
    conv_id = conversation_id.strip() if conversation_id else ""

    # ── File upload → start validation session ────────────────────────────────
    if file is not None and file.filename:
        return await _handle_yaml_upload(request, file, message, deps)

    # ── Active validation session → continue clarification ───────────────────
    session_id = request.cookies.get("validation_session_id")
    if session_id and message:
        session = _load_session(deps.db, session_id)
        if session and session.status == "awaiting_input":
            return await _handle_session_turn(request, session, message, deps)

    # ── Normal Supervisore path ───────────────────────────────────────────────
    if not message:
        return HTMLResponse("")

    # Auto-create conversation if not provided
    if not conv_id:
        title = message[:100] if len(message) <= 100 else message[:97] + "…"
        conv_id = _create_conversation(deps.db, user_id, title)
    else:
        # Update conversation timestamp
        db = deps.db
        # Check if conversation has a default title and update it
        row = db.execute(
            "SELECT title FROM conversations WHERE conversation_id = ?", [conv_id]
        ).fetchone()
        if row and row[0] == "New Chat":
            title = message[:100] if len(message) <= 100 else message[:97] + "…"
            _update_conversation_title(db, conv_id, title)
        db.execute(
            "UPDATE conversations SET updated_at = now() WHERE conversation_id = ?",
            [conv_id],
        )

    # Load conversation history before saving current message
    history = _load_conversation_history(deps.db, conv_id) if conv_id else []

    # Save user message
    _save_message(deps.db, conv_id, "user", message)

    # Parse prefix commands (/logs, /docs, /validate)
    forced_agent, cleaned_message = _parse_prefix_command(message)

    deps.docu_rag_deps.retrieved_chunks = []  # reset before each run
    try:
        if forced_agent:
            # Direct delegation — bypass Supervisore routing
            from src.agents.log_explorer import log_explorer_agent
            from src.agents.docu_rag import docu_rag_agent

            agents_map = {
                "delegate_to_log_explorer": (log_explorer_agent, deps.log_deps, "Log Explorer"),
                "delegate_to_docu_rag": (docu_rag_agent, deps.docu_rag_deps, "Docu-RAG"),
                "delegate_to_config_validator": (config_validator_agent, deps.validator_deps, "Config Validator"),
            }
            agent_info = agents_map[forced_agent]
            result = await agent_info[0].run(cleaned_message, deps=agent_info[1])
            content = result.output
            sources = _extract_sources(content)
            chunk_sources = deps.docu_rag_deps.retrieved_chunks
            agent_label = agent_info[2]
            is_error = False
        else:
            result = await supervisore_agent.run(message, message_history=history, deps=deps)
            sources = _extract_sources(result.output)
            chunk_sources = deps.docu_rag_deps.retrieved_chunks
            agent_label = _detect_agent(result.all_messages())
            content = result.output
            is_error = False
    except Exception as exc:  # noqa: BLE001
        content = f"Agent error: {exc}"
        sources = []
        chunk_sources = []
        agent_label = None
        is_error = True

    # Save assistant message
    msg_id = _save_message(
        deps.db, conv_id, "assistant", content,
        agent=agent_label, sources=sources,
        chunk_sources=[{"doc_id": cs["doc_id"], "title": cs["title"], "chunk_index": cs.get("chunk_index", 0)} for cs in chunk_sources],
        is_error=is_error,
    )

    response = templates.TemplateResponse(
        request=request,
        name="_message.html",
        context={
            "content": content,
            "sources": sources,
            "chunk_sources": chunk_sources,
            "agent": agent_label,
            "is_error": is_error,
            "pending_question": None,
            "session_id": None,
            "show_export": False,
            "message_id": msg_id,
            "conversation_id": conv_id,
        },
    )
    _set_user_cookie(response, user_id)
    # Pass conversation_id back via header so JS can track it
    response.headers["X-Conversation-Id"] = conv_id  # type: ignore[union-attr]
    return response


@app.post("/ingest", response_class=HTMLResponse)
async def ingest(
    request: Request,
    file: Annotated[UploadFile | None, File()] = None,
    url: Annotated[str, Form()] = "",
    tags: Annotated[str, Form()] = "",
) -> HTMLResponse:
    """Accept a file or URL, ingest into the docs vector store, return toast fragment."""
    import pathlib
    import tempfile

    deps: SupervisoreDeps = request.app.state.deps
    db = deps.docu_rag_deps.db
    model = deps.docu_rag_deps.embedding_model

    try:
        if file and file.filename:
            raw = await file.read()
            suffix = pathlib.Path(file.filename).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(raw)
                tmp_path = pathlib.Path(tmp.name)
            try:
                n = await ingest_source(
                    db, model, file_path=tmp_path, title=file.filename, tags=tags
                )
            finally:
                tmp_path.unlink(missing_ok=True)
            msg, is_error = f"'{file.filename}' ingested — {n} chunks stored.", False
        elif url.strip():
            n = await ingest_source(db, model, url=url.strip(), tags=tags)
            msg, is_error = f"URL ingested — {n} chunks stored.", False
        else:
            msg, is_error = "Provide a file or a URL.", True
    except Exception as exc:  # noqa: BLE001
        msg, is_error = f"Ingestion error: {exc}", True

    return templates.TemplateResponse(
        request=request,
        name="_ingest_result.html",
        context={"message": msg, "is_error": is_error},
    )


@app.get("/docs/chunk/{doc_id}", response_class=HTMLResponse)
async def chunk_preview(request: Request, doc_id: str) -> HTMLResponse:
    """Return an HTML fragment with the chunk text rendered for the popup."""
    deps: SupervisoreDeps = request.app.state.deps
    result = await async_query(
        deps.docu_rag_deps.db,
        "SELECT title, chunk_index, chunk_text FROM docs WHERE doc_id = ? LIMIT 1",
        [doc_id],
    )
    if result.empty:
        return HTMLResponse(
            "<p class='text-error font-mono text-sm p-4'>Chunk not found.</p>",
            status_code=404,
        )
    row = result.iloc[0]
    return templates.TemplateResponse(
        request=request,
        name="_chunk_preview.html",
        context={
            "title": row["title"],
            "chunk_index": int(row["chunk_index"]),
            "chunk_text": row["chunk_text"],
        },
    )


@app.post("/validate/{session_id}/fix-with-ai", response_class=HTMLResponse)
async def fix_with_ai(request: Request, session_id: str) -> HTMLResponse:
    """Escalate a failed static repair to the LLM agent, injecting known schema errors as context."""
    deps: SupervisoreDeps = request.app.state.deps
    session = _load_session(deps.db, session_id)
    if session is None:
        return templates.TemplateResponse(  # type: ignore[return-value]
            request=request,
            name="_message.html",
            context={
                "content": "Session not found.",
                "agent": "Config Validator",
                "is_error": True,
                "sources": [], "chunk_sources": [],
                "pending_question": None, "session_id": None, "show_export": False,
            },
        )

    # Extract stored schema errors from the synthetic first message
    history = _deserialize_messages(session.message_history)
    error_context = ""
    if history:
        try:
            from pydantic_ai.messages import ModelRequest
            first = history[0]
            if isinstance(first, ModelRequest):
                data = json.loads(first.parts[0].content)  # type: ignore[union-attr]
                errors = data.get("errors", [])
                if errors:
                    error_context = "\n\nKnown schema violations (from static analysis):\n" + "\n".join(
                        f"- {e}" for e in errors
                    )
        except Exception:
            pass

    prompt = (
        f"Validate and fix this YAML manifest.{error_context}\n\n"
        f"Filename: {session.filename}\n\n"
        f"YAML content:\n{session.yaml_original}"
    )

    try:
        result = await config_validator_v2_agent.run(
            prompt, message_history=[], deps=ValidatorDeps(db=deps.db)
        )
        raw = result.output
        if isinstance(raw, str):
            output: ValidatorV2Output = await ollama_structured_finish(
                messages=[
                    {"role": "system", "content": (
                        "You are a YAML validation data extractor. "
                        "Given a validation report, extract the structured result as JSON. "
                        "yaml_current must contain the complete final YAML string after all fixes applied."
                    )},
                    {"role": "user", "content": (
                        f"Original YAML:\n{session.yaml_original}\n\n"
                        f"Validation report:\n{raw}\n\n"
                        "Extract this as a ValidatorV2Output JSON object."
                    )},
                ],
                output_type=ValidatorV2Output,
            )
        else:
            output = raw
    except Exception as exc:  # noqa: BLE001
        return templates.TemplateResponse(  # type: ignore[return-value]
            request=request,
            name="_message.html",
            context={
                "content": f"AI validation error: {exc}",
                "agent": "Config Validator",
                "is_error": True,
                "sources": [], "chunk_sources": [],
                "pending_question": None, "session_id": session_id, "show_export": False,
            },
        )

    new_session_id = str(uuid.uuid4())
    _create_session(deps.db, new_session_id, session.filename, session.yaml_original, output, result.all_messages())

    response = templates.TemplateResponse(  # type: ignore[return-value]
        request=request,
        name="_message.html",
        context={
            "content": output.message,
            "agent": "Config Validator",
            "is_error": False,
            "sources": [], "chunk_sources": [],
            "pending_question": output.pending_question,
            "session_id": new_session_id,
            "show_export": output.is_done,
        },
    )
    if not output.is_done:
        response.set_cookie("validation_session_id", new_session_id, httponly=True, samesite="lax")  # type: ignore[union-attr]
    return response


@app.get("/validate/{session_id}/export")
async def export_validated_yaml(
    request: Request,
    session_id: str,
) -> Response:
    """Download the annotated, corrected YAML for a completed validation session."""
    deps: SupervisoreDeps = request.app.state.deps
    session = _load_session(deps.db, session_id)

    if session is None:
        return Response(content="Session not found.", status_code=404)

    from src.agents.config_validator import _collect_warnings

    try:
        original_data = yaml.safe_load(session.yaml_original)
        warnings = _collect_warnings(original_data) if original_data else []
    except Exception:
        warnings = []

    annotated = annotate_yaml(session.yaml_current, session.fixes_applied, warnings)
    filename = session.filename.replace(".yaml", "_fixed.yaml").replace(".yml", "_fixed.yml")

    return Response(
        content=annotated,
        media_type="application/x-yaml",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/schemas", response_class=HTMLResponse)
async def schemas_upload(
    request: Request,
    file: Annotated[UploadFile | None, File()] = None,
    title: Annotated[str, Form()] = "",
    description: Annotated[str, Form()] = "",
) -> HTMLResponse:
    """Store a reference schema (JSON Schema or OpenAPI) in DuckDB."""
    deps: SupervisoreDeps = request.app.state.deps
    schemas: list[dict] = []
    try:
        if not file or not file.filename:
            raise ValueError("No file provided")
        raw = (await file.read()).decode("utf-8")
        title = title.strip() or file.filename
        schema_id = store_schema(deps.db, raw, title, description.strip())
        schemas = list_schemas(deps.db)
        msg, is_error = f"Schema '{title}' stored (id: {schema_id[:8]}…).", False
    except Exception as exc:  # noqa: BLE001
        schemas = list_schemas(deps.db)
        msg, is_error = f"Upload error: {exc}", True
    return templates.TemplateResponse(  # type: ignore[return-value]
        request=request,
        name="_schema_result.html",
        context={"message": msg, "is_error": is_error, "schemas": schemas, "errors": [], "repaired_yaml": None, "llm_fields": []},
    )


@app.get("/schemas", response_class=HTMLResponse)
async def schemas_list(request: Request) -> HTMLResponse:
    """Return the schema selector <select> fragment."""
    deps: SupervisoreDeps = request.app.state.deps
    schemas = list_schemas(deps.db)
    active_schema_id = request.cookies.get("active_schema_id", "")
    active_schema_title = next((s["title"] for s in schemas if s["schema_id"] == active_schema_id), "")
    return templates.TemplateResponse(  # type: ignore[return-value]
        request=request,
        name="_schema_list.html",
        context={"schemas": schemas, "active_schema_id": active_schema_id, "active_schema_title": active_schema_title},
    )


@app.post("/schemas/activate", response_class=HTMLResponse)
async def schemas_activate(
    request: Request,
    schema_id: Annotated[str, Form()] = "",
) -> HTMLResponse:
    """Set a schema as the active reference for chat-upload validation."""
    deps: SupervisoreDeps = request.app.state.deps
    schemas = list_schemas(deps.db)
    active_schema_id = schema_id.strip()
    active_schema_title = next((s["title"] for s in schemas if s["schema_id"] == active_schema_id), "")
    response = templates.TemplateResponse(  # type: ignore[return-value]
        request=request,
        name="_schema_list.html",
        context={"schemas": schemas, "active_schema_id": active_schema_id, "active_schema_title": active_schema_title},
    )
    if active_schema_id:
        response.set_cookie("active_schema_id", active_schema_id, httponly=True, samesite="lax")
    return response


@app.delete("/schemas/active", response_class=HTMLResponse)
async def schemas_deactivate(request: Request) -> HTMLResponse:
    """Clear the active reference schema."""
    deps: SupervisoreDeps = request.app.state.deps
    schemas = list_schemas(deps.db)
    response = templates.TemplateResponse(  # type: ignore[return-value]
        request=request,
        name="_schema_list.html",
        context={"schemas": schemas, "active_schema_id": "", "active_schema_title": ""},
    )
    response.delete_cookie("active_schema_id")
    return response


@app.delete("/schemas/{schema_id}", response_class=HTMLResponse)
async def schemas_delete(request: Request, schema_id: str) -> HTMLResponse:
    """Delete a stored schema and return refreshed selector."""
    deps: SupervisoreDeps = request.app.state.deps
    delete_schema(deps.db, schema_id)
    schemas = list_schemas(deps.db)
    # If the deleted schema was the active one, clear it
    current_active = request.cookies.get("active_schema_id", "")
    active_schema_id = "" if current_active == schema_id else current_active
    active_schema_title = next((s["title"] for s in schemas if s["schema_id"] == active_schema_id), "")
    response = templates.TemplateResponse(  # type: ignore[return-value]
        request=request,
        name="_schema_list.html",
        context={"schemas": schemas, "active_schema_id": active_schema_id, "active_schema_title": active_schema_title},
    )
    if current_active == schema_id:
        response.delete_cookie("active_schema_id")
    return response


@app.post("/validate/document", response_class=HTMLResponse)
async def validate_document(
    request: Request,
    file: Annotated[UploadFile | None, File()] = None,
    yaml_text: Annotated[str, Form()] = "",
    schema_id: Annotated[str, Form()] = "",
) -> HTMLResponse:
    """Validate and statically repair a YAML document against a stored schema."""
    deps: SupervisoreDeps = request.app.state.deps
    try:
        if file and file.filename:
            content = (await file.read()).decode("utf-8")
        elif yaml_text.strip():
            content = yaml_text
        else:
            raise ValueError("Provide a file or paste YAML content")
        if not schema_id.strip():
            raise ValueError("Select a reference schema")
        schema = get_schema(deps.db, schema_id.strip())
        if schema is None:
            raise ValueError(f"Schema '{schema_id}' not found")
        result = repair_yaml(content, schema)
        return templates.TemplateResponse(  # type: ignore[return-value]
            request=request,
            name="_schema_result.html",
            context={
                "message": result.summary,
                "is_error": not result.valid,
                "schemas": None,
                "errors": result.errors,
                "repaired_yaml": result.repaired_yaml if result.errors or result.llm_fields else None,
                "llm_fields": result.llm_fields,
            },
        )
    except Exception as exc:  # noqa: BLE001
        return templates.TemplateResponse(  # type: ignore[return-value]
            request=request,
            name="_schema_result.html",
            context={"message": f"Error: {exc}", "is_error": True, "schemas": None, "errors": [], "repaired_yaml": None, "llm_fields": []},
        )


@app.get("/health")
async def health(request: Request) -> dict[str, str | bool]:
    """Health check endpoint."""
    deps: SupervisoreDeps = request.app.state.deps
    try:
        deps.db.execute("SELECT 1")
        db_ok = True
    except Exception:  # noqa: BLE001
        db_ok = False

    return {
        "status": "ok" if db_ok else "degraded",
        "duckdb": db_ok,
        "version": "0.3.0",
    }


# ── Status endpoint (enhanced) ────────────────────────────────────────────────


@app.get("/status")
async def status(request: Request) -> dict:
    """Enhanced status endpoint for the UI status panel."""
    deps: SupervisoreDeps = request.app.state.deps
    db = deps.db

    result: dict = {"version": "0.3.0"}

    # DuckDB
    try:
        db.execute("SELECT 1")
        result["duckdb"] = True
    except Exception:  # noqa: BLE001
        result["duckdb"] = False

    # LLM backend
    result["llm_backend"] = os.getenv("AGENT_BACKEND", "ollama")
    result["llm_model"] = os.getenv("AGENT_MODEL", "")

    # Check LLM reachability
    try:
        import httpx
        backend = os.getenv("AGENT_BACKEND", "ollama")
        if backend == "ollama":
            base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1").rstrip("/v1")
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(f"{base}/api/tags")
                result["llm_ok"] = r.status_code == 200
                if r.status_code == 200:
                    models = r.json().get("models", [])
                    result["available_models"] = [m["name"] for m in models][:10]
        else:
            result["llm_ok"] = True  # Cloud APIs assumed available
            result["available_models"] = []
    except Exception:  # noqa: BLE001
        result["llm_ok"] = False
        result["available_models"] = []

    # Data counts
    try:
        row = db.execute("SELECT COUNT(*) FROM logs").fetchone()
        result["logs_count"] = row[0] if row else 0
    except Exception:  # noqa: BLE001
        result["logs_count"] = 0

    try:
        row = db.execute("SELECT COUNT(DISTINCT title) FROM docs").fetchone()
        result["docs_count"] = row[0] if row else 0
    except Exception:  # noqa: BLE001
        result["docs_count"] = 0

    try:
        row = db.execute("SELECT COUNT(*) FROM schemas").fetchone()
        result["schemas_count"] = row[0] if row else 0
    except Exception:  # noqa: BLE001
        result["schemas_count"] = 0

    # Services in logs
    try:
        rows = db.execute(
            "SELECT DISTINCT service FROM logs ORDER BY service LIMIT 20"
        ).fetchall()
        result["services"] = [r[0] for r in rows] if rows else []
    except Exception:  # noqa: BLE001
        result["services"] = []

    return result


# ── Conversation API ──────────────────────────────────────────────────────────


@app.get("/conversations")
async def list_conversations(request: Request) -> JSONResponse:
    """List conversations for the current user."""
    deps: SupervisoreDeps = request.app.state.deps
    user_id = _get_or_create_user(deps.db, request)
    rows = deps.db.execute(
        """SELECT conversation_id, title,
                  CAST(created_at AS VARCHAR), CAST(updated_at AS VARCHAR)
           FROM conversations WHERE user_id = ?
           ORDER BY updated_at DESC LIMIT 50""",
        [user_id],
    ).fetchall()
    conversations = [
        {"id": r[0], "title": r[1], "created_at": r[2], "updated_at": r[3]}
        for r in rows
    ]
    response = JSONResponse(content=conversations)
    _set_user_cookie(response, user_id)
    return response


@app.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(request: Request, conversation_id: str) -> JSONResponse:
    """Load all messages for a conversation."""
    deps: SupervisoreDeps = request.app.state.deps
    user_id = _get_or_create_user(deps.db, request)

    # Verify ownership
    owner = deps.db.execute(
        "SELECT user_id FROM conversations WHERE conversation_id = ?",
        [conversation_id],
    ).fetchone()
    if not owner or owner[0] != user_id:
        return JSONResponse(content={"error": "Not found"}, status_code=404)

    rows = deps.db.execute(
        """SELECT message_id, role, content, agent, sources, chunk_sources, is_error,
                  CAST(created_at AS VARCHAR)
           FROM messages WHERE conversation_id = ?
           ORDER BY created_at ASC""",
        [conversation_id],
    ).fetchall()

    messages = [
        {
            "id": r[0],
            "role": r[1],
            "content": r[2],
            "agent": r[3],
            "sources": json.loads(r[4]) if r[4] else [],
            "chunk_sources": json.loads(r[5]) if r[5] else [],
            "is_error": bool(r[6]),
            "created_at": r[7],
        }
        for r in rows
    ]
    return JSONResponse(content=messages)


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(request: Request, conversation_id: str) -> JSONResponse:
    """Delete a conversation and its messages."""
    deps: SupervisoreDeps = request.app.state.deps
    user_id = _get_or_create_user(deps.db, request)

    # Verify ownership
    owner = deps.db.execute(
        "SELECT user_id FROM conversations WHERE conversation_id = ?",
        [conversation_id],
    ).fetchone()
    if not owner or owner[0] != user_id:
        return JSONResponse(content={"error": "Not found"}, status_code=404)

    deps.db.execute("DELETE FROM messages WHERE conversation_id = ?", [conversation_id])
    deps.db.execute("DELETE FROM conversations WHERE conversation_id = ?", [conversation_id])
    return JSONResponse(content={"ok": True})


# ── Feedback API ──────────────────────────────────────────────────────────────


@app.post("/feedback")
async def submit_feedback(
    request: Request,
    message_id: Annotated[str, Form()],
    rating: Annotated[str, Form()],
    comment: Annotated[str, Form()] = "",
) -> JSONResponse:
    """Submit thumbs up/down feedback on a message."""
    deps: SupervisoreDeps = request.app.state.deps
    user_id = _get_or_create_user(deps.db, request)

    if rating not in ("up", "down"):
        return JSONResponse(content={"error": "Invalid rating"}, status_code=400)

    try:
        deps.db.execute(
            """INSERT INTO feedback (message_id, user_id, rating, comment)
               VALUES (?, ?, ?, ?)
               ON CONFLICT (message_id, user_id) DO UPDATE SET rating = ?, comment = ?""",
            [message_id, user_id, rating, comment.strip() or None, rating, comment.strip() or None],
        )
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(content={"error": str(exc)}, status_code=500)

    return JSONResponse(content={"ok": True})


# ── Settings API ──────────────────────────────────────────────────────────────


@app.get("/settings")
async def get_settings(request: Request) -> JSONResponse:
    """Get current user settings."""
    deps: SupervisoreDeps = request.app.state.deps
    user_id = _get_or_create_user(deps.db, request)

    row = deps.db.execute(
        "SELECT theme, language, agent_backend, agent_model FROM user_settings WHERE user_id = ?",
        [user_id],
    ).fetchone()

    if row:
        settings = {
            "theme": row[0],
            "language": row[1],
            "agent_backend": row[2],
            "agent_model": row[3],
        }
    else:
        settings = {
            "theme": "business",
            "language": "en",
            "agent_backend": os.getenv("AGENT_BACKEND", "ollama"),
            "agent_model": os.getenv("AGENT_MODEL", ""),
        }

    return JSONResponse(content=settings)


@app.put("/settings")
async def update_settings(request: Request) -> JSONResponse:
    """Update user settings."""
    deps: SupervisoreDeps = request.app.state.deps
    user_id = _get_or_create_user(deps.db, request)
    body = await request.json()

    theme = body.get("theme", "business")
    language = body.get("language", "en")
    agent_backend = body.get("agent_backend")
    agent_model = body.get("agent_model")

    deps.db.execute(
        """INSERT INTO user_settings (user_id, theme, language, agent_backend, agent_model)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT (user_id) DO UPDATE
           SET theme = ?, language = ?, agent_backend = ?, agent_model = ?""",
        [user_id, theme, language, agent_backend, agent_model,
         theme, language, agent_backend, agent_model],
    )

    return JSONResponse(content={"ok": True})
