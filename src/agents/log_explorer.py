"""Log Explorer agent — queries structured logs via DuckDB / Apache Iceberg on MinIO."""

from __future__ import annotations

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

from src.deps.connections import LogDeps, build_model
from src.tools.async_db import async_query

# ── Output model ──────────────────────────────────────────────────────────────


class LogQueryResult(BaseModel):
    answer: str
    rows_returned: int
    query_used: str


# ── Agent ─────────────────────────────────────────────────────────────────────

log_explorer_agent: Agent[LogDeps, str] = Agent(
    build_model(),
    deps_type=LogDeps,
    output_type=str,
    system_prompt=(
        "You are an SRE log analysis expert with access to structured log tables via DuckDB. "
        "Translate user questions into SQL queries using the provided tools. "
        "IMPORTANT: Only SELECT queries are permitted — never INSERT, UPDATE, DELETE, or DDL. "
        "Always include a time filter to avoid full-table scans. "
        "After calling a tool, summarise the results clearly in plain text."
    ),
)


# ── Tools ─────────────────────────────────────────────────────────────────────


@log_explorer_agent.tool
async def search_errors(
    ctx: RunContext[LogDeps],
    service: str,
    hours: int = 1,
) -> str:
    """Search for ERROR and CRITICAL log entries for a given service.

    Args:
        service: The service name to filter (e.g. 'api-gateway', 'auth-service').
        hours: Time window in hours to look back from now. Default: 1.

    Returns:
        Markdown table of matching log rows, or a no-results message.
    """
    result = await async_query(
        ctx.deps.db,
        """
        SELECT timestamp, level, message, trace_id
        FROM logs
        WHERE service = ?
          AND level IN ('ERROR', 'CRITICAL')
          AND TRY_CAST(timestamp AS TIMESTAMPTZ) >= NOW() - INTERVAL (? || ' hours')
        ORDER BY timestamp DESC
        LIMIT 100
        """,
        [service, str(hours)],
    )

    if result.empty:
        return f"No errors found for service '{service}' in the last {hours}h."
    return result.to_markdown(index=False)


@log_explorer_agent.tool
async def count_by_level(
    ctx: RunContext[LogDeps],
    time_window_minutes: int = 60,
) -> str:
    """Count log entries grouped by severity level over a time window.

    Args:
        time_window_minutes: Time window in minutes to look back from now. Default: 60.

    Returns:
        Markdown table with columns: level, count.
    """
    result = await async_query(
        ctx.deps.db,
        """
        SELECT level, COUNT(*) AS count
        FROM logs
        WHERE TRY_CAST(timestamp AS TIMESTAMPTZ) >= NOW() - INTERVAL (? || ' minutes')
        GROUP BY level
        ORDER BY count DESC
        """,
        [str(time_window_minutes)],
    )

    if result.empty:
        return f"No log entries found in the last {time_window_minutes} minutes."
    return result.to_markdown(index=False)


@log_explorer_agent.tool
async def list_services(
    ctx: RunContext[LogDeps],
    hours: int = 24,
) -> str:
    """List all distinct service names that have logged entries recently.

    Args:
        hours: Time window in hours to look back. Default: 24.

    Returns:
        Markdown table with columns: service, log_count, last_seen.
    """
    result = await async_query(
        ctx.deps.db,
        """
        SELECT service,
               COUNT(*)       AS log_count,
               MAX(timestamp) AS last_seen
        FROM logs
        WHERE TRY_CAST(timestamp AS TIMESTAMPTZ) >= NOW() - INTERVAL (? || ' hours')
        GROUP BY service
        ORDER BY log_count DESC
        """,
        [str(hours)],
    )

    if result.empty:
        return f"No services found with log entries in the last {hours}h."
    return result.to_markdown(index=False)


@log_explorer_agent.tool
async def get_trace(
    ctx: RunContext[LogDeps],
    trace_id: str,
) -> str:
    """Retrieve all log lines associated with a distributed trace ID.

    Args:
        trace_id: The trace ID to look up.

    Returns:
        Markdown table of all log entries for the trace, ordered by timestamp.
    """
    result = await async_query(
        ctx.deps.db,
        """
        SELECT timestamp, service, level, message
        FROM logs
        WHERE trace_id = ?
        ORDER BY timestamp ASC
        """,
        [trace_id],
    )

    if result.empty:
        return f"No log entries found for trace_id='{trace_id}'."
    return result.to_markdown(index=False)
