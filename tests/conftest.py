"""Shared pytest fixtures for all test layers."""

from __future__ import annotations

import pytest
import duckdb

from src.deps.connections import (
    DocuRagDeps,
    LogDeps,
    SupervisoreDeps,
    ValidatorDeps,
)


@pytest.fixture
def in_memory_db() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB connection with all required tables and test data pre-loaded.

    Extensions (vss, httpfs, iceberg) are installed but not configured against
    real MinIO/Lakekeeper — tests use only local DuckDB tables.
    """
    conn = duckdb.connect(":memory:")

    # Load extensions (best-effort — vss may fail on some CI environments)
    for ext in ("vss", "httpfs"):
        try:
            conn.execute(f"INSTALL {ext}")
            conn.execute(f"LOAD {ext}")
        except Exception:  # noqa: BLE001
            pass

    # ── logs table ────────────────────────────────────────────────────────────
    conn.execute("""
        CREATE TABLE logs (
            timestamp VARCHAR,
            service   VARCHAR,
            level     VARCHAR,
            message   VARCHAR,
            trace_id  VARCHAR
        )
    """)
    conn.executemany(
        "INSERT INTO logs VALUES (?, ?, ?, ?, ?)",
        [
            ("2026-03-25T10:00:00Z", "api-gateway",   "ERROR",  "Upstream timeout",        "trace001"),
            ("2026-03-25T10:00:01Z", "api-gateway",   "CRITICAL","OOMKilled",               "trace001"),
            ("2026-03-25T10:00:02Z", "auth-service",  "INFO",   "Request processed",       "trace002"),
            ("2026-03-25T10:00:03Z", "auth-service",  "WARN",   "High memory usage: 85%",  "trace003"),
            ("2026-03-25T10:00:04Z", "payment-service","ERROR", "Stripe API returned 502", "trace004"),
        ],
    )

    # ── docs table ────────────────────────────────────────────────────────────
    conn.execute("""
        CREATE TABLE docs (
            doc_id     VARCHAR PRIMARY KEY,
            title      VARCHAR NOT NULL,
            chunk_text VARCHAR NOT NULL,
            embedding  FLOAT[768]
        )
    """)
    # Insert a test doc with a trivial zero-vector embedding
    zero_vec = [0.0] * 768
    conn.execute(
        "INSERT INTO docs VALUES (?, ?, ?, ?)",
        ["doc-001", "DB Rollback Runbook", "To rollback: helm rollback app 1", zero_vec],
    )

    # ── multi-tenant tables ───────────────────────────────────────────────────
    conn.execute("""
        CREATE TABLE users (
            user_id      VARCHAR PRIMARY KEY,
            display_name VARCHAR DEFAULT 'Anonymous',
            created_at   TIMESTAMPTZ DEFAULT now()
        )
    """)
    conn.execute("""
        CREATE TABLE conversations (
            conversation_id VARCHAR PRIMARY KEY,
            user_id         VARCHAR NOT NULL,
            title           VARCHAR DEFAULT 'New Chat',
            created_at      TIMESTAMPTZ DEFAULT now(),
            updated_at      TIMESTAMPTZ DEFAULT now()
        )
    """)
    conn.execute("""
        CREATE TABLE messages (
            message_id      VARCHAR PRIMARY KEY,
            conversation_id VARCHAR NOT NULL,
            role            VARCHAR NOT NULL,
            content         TEXT NOT NULL,
            agent           VARCHAR,
            sources         TEXT DEFAULT '[]',
            chunk_sources   TEXT DEFAULT '[]',
            is_error        BOOLEAN DEFAULT FALSE,
            created_at      TIMESTAMPTZ DEFAULT now()
        )
    """)
    conn.execute("""
        CREATE TABLE feedback (
            message_id VARCHAR NOT NULL,
            user_id    VARCHAR NOT NULL,
            rating     VARCHAR NOT NULL,
            comment    TEXT,
            created_at TIMESTAMPTZ DEFAULT now(),
            PRIMARY KEY (message_id, user_id)
        )
    """)
    conn.execute("""
        CREATE TABLE schemas (
            schema_id   VARCHAR PRIMARY KEY,
            title       VARCHAR NOT NULL,
            description VARCHAR DEFAULT '',
            format      VARCHAR DEFAULT 'json_schema',
            schema_json VARCHAR NOT NULL,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE validation_sessions (
            session_id      VARCHAR PRIMARY KEY,
            created_at      TIMESTAMPTZ DEFAULT now(),
            filename        VARCHAR,
            yaml_original   TEXT,
            yaml_current    TEXT,
            fixes_applied   TEXT,
            message_history TEXT,
            status          VARCHAR DEFAULT 'in_progress'
        )
    """)

    return conn


@pytest.fixture
def test_log_deps(in_memory_db: duckdb.DuckDBPyConnection) -> LogDeps:
    return LogDeps(
        db=in_memory_db,
        minio_endpoint="http://localhost:9000",
    )


@pytest.fixture
def test_docu_rag_deps(in_memory_db: duckdb.DuckDBPyConnection) -> DocuRagDeps:
    return DocuRagDeps(
        db=in_memory_db,
        embedding_model="nomic-embed-text",
        embedding_dim=768,
    )


@pytest.fixture
def test_validator_deps() -> ValidatorDeps:
    return ValidatorDeps()


@pytest.fixture
def test_deps(
    in_memory_db: duckdb.DuckDBPyConnection,
    test_log_deps: LogDeps,
    test_docu_rag_deps: DocuRagDeps,
    test_validator_deps: ValidatorDeps,
) -> SupervisoreDeps:
    return SupervisoreDeps(
        db=in_memory_db,
        log_deps=test_log_deps,
        docu_rag_deps=test_docu_rag_deps,
        validator_deps=test_validator_deps,
    )
