"""Supervisore — router agent that delegates to specialist sub-agents."""

from __future__ import annotations

import asyncio
import logging
import time

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

from src.agents.config_validator import config_validator_agent
from src.agents.docu_rag import docu_rag_agent
from src.agents.log_explorer import log_explorer_agent
from src.deps.connections import SupervisoreDeps, build_model

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3


# ── Output model ──────────────────────────────────────────────────────────────


class SupervisoreOutput(BaseModel):
    """Structured metadata about the routing decision.

    Used in tests and observability — the actual user-facing answer
    is streamed as plain text via run_stream() in the FastAPI endpoint.
    """

    answer: str
    delegated_to: str | None  # which specialist agent handled the request
    confidence: float  # 0.0 – 1.0 self-assessed routing confidence


# ── Agent ─────────────────────────────────────────────────────────────────────

# output_type=str: local Ollama models (llama3.1, granite, mistral…) reliably
# return free text after a tool call. Structured JSON output after tool use
# is not consistently supported by most 7B-class models on Ollama.
# The FastAPI /chat endpoint uses run_stream() which streams text chunks,
# so str output is the correct type for the chat UI.
# SupervisoreOutput is kept for tests that use TestModel (which handles
# structured output deterministically).
supervisore_agent: Agent[SupervisoreDeps, str] = Agent(
    build_model(),
    deps_type=SupervisoreDeps,
    output_type=str,
    system_prompt=(
        "You are a Tech Lead and the entry point for all SRE/DevOps team queries. "
        "Analyse the user's request and route it to exactly ONE specialist agent via a tool call. "
        "After receiving the tool result, summarise it clearly for the user. "
        "Never answer specialist questions directly — always delegate first. "
        "If the request is ambiguous, ask ONE clarifying question before delegating. "
        "Rules:\n"
        "- Logs / incidents / traces / service health → delegate_to_log_explorer\n"
        "- Architecture docs / runbooks / design decisions → delegate_to_docu_rag\n"
        "- YAML manifest / K8s config / GitLab CI → delegate_to_config_validator\n"
    ),
)


# ── Delegation tools ──────────────────────────────────────────────────────────


@supervisore_agent.tool
async def delegate_to_log_explorer(
    ctx: RunContext[SupervisoreDeps],
    question: str,
) -> str:
    """Delegate to the Log Explorer agent for log analysis, incident investigation,
    service health checks, error searches, or distributed trace lookups.

    Use this when the user asks about:
    - Log errors or exceptions for a specific service
    - Incident timelines and root-cause analysis
    - Trace IDs and request flows
    - Log volume or error rate trends

    Args:
        question: The user's question in natural language.
    """
    for attempt in range(_MAX_RETRIES):
        try:
            t0 = time.monotonic()
            result = await log_explorer_agent.run(
                question,
                deps=ctx.deps.log_deps,
                usage=ctx.usage,
            )
            elapsed = time.monotonic() - t0
            logger.info("log_explorer completed in %.2fs", elapsed)
            return result.output
        except Exception as exc:
            logger.warning("log_explorer attempt %d failed: %s", attempt + 1, exc)
            if attempt == _MAX_RETRIES - 1:
                return f"Log Explorer is currently unavailable: {exc}. Please try again."
            await asyncio.sleep(2**attempt)
    return "Log Explorer failed after retries."  # unreachable but satisfies type checker


@supervisore_agent.tool
async def delegate_to_docu_rag(
    ctx: RunContext[SupervisoreDeps],
    question: str,
) -> str:
    """Delegate to the Docu-RAG agent for questions about internal documentation,
    architecture decisions, runbooks, procedures, or design rationale.

    Use this when the user asks about:
    - How a system component works
    - The correct procedure for an operational task (e.g., database rollback)
    - Architecture or design decisions
    - Internal policies or runbooks

    Args:
        question: The user's question in natural language.
    """
    for attempt in range(_MAX_RETRIES):
        try:
            t0 = time.monotonic()
            result = await docu_rag_agent.run(
                question,
                deps=ctx.deps.docu_rag_deps,
                usage=ctx.usage,
            )
            elapsed = time.monotonic() - t0
            logger.info("docu_rag completed in %.2fs", elapsed)
            return result.output
        except Exception as exc:
            logger.warning("docu_rag attempt %d failed: %s", attempt + 1, exc)
            if attempt == _MAX_RETRIES - 1:
                return f"Docu-RAG is currently unavailable: {exc}. Please try again."
            await asyncio.sleep(2**attempt)
    return "Docu-RAG failed after retries."


@supervisore_agent.tool
async def delegate_to_config_validator(
    ctx: RunContext[SupervisoreDeps],
    question: str,
) -> str:
    """Delegate to the Config Validator agent for YAML validation or configuration review.

    Use this when the user:
    - Pastes a Kubernetes manifest (Deployment, Service, ConfigMap, Secret)
    - Pastes a GitLab CI pipeline (.gitlab-ci.yml content)
    - Asks whether a configuration is valid or compliant
    - Wants to check for misconfigurations or missing required fields

    Args:
        question: The user's question or YAML content to validate.
    """
    for attempt in range(_MAX_RETRIES):
        try:
            t0 = time.monotonic()
            result = await config_validator_agent.run(
                question,
                deps=ctx.deps.validator_deps,
                usage=ctx.usage,
            )
            elapsed = time.monotonic() - t0
            logger.info("config_validator completed in %.2fs", elapsed)
            return result.output
        except Exception as exc:
            logger.warning("config_validator attempt %d failed: %s", attempt + 1, exc)
            if attempt == _MAX_RETRIES - 1:
                return f"Config Validator is currently unavailable: {exc}. Please try again."
            await asyncio.sleep(2**attempt)
    return "Config Validator failed after retries."
