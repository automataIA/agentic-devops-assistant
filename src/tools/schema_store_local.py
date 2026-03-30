"""CRUD for locally stored reference schemas (DuckDB table: schemas)."""

from __future__ import annotations

import hashlib
import json

import duckdb
import yaml


def store_schema(
    db: duckdb.DuckDBPyConnection,
    raw_content: str,
    title: str,
    description: str = "",
) -> str:
    """Parse YAML/JSON, detect format, store in DuckDB. Returns schema_id (sha256[:16]).

    Idempotent: same content → same schema_id (INSERT OR REPLACE).
    """
    raw_stripped = raw_content.lstrip()
    if raw_stripped.startswith("{") or raw_stripped.startswith("["):
        data: dict = json.loads(raw_content)
    else:
        data = yaml.safe_load(raw_content)

    if not isinstance(data, dict):
        raise ValueError("Schema must be a YAML/JSON mapping")

    if "openapi" in data or "swagger" in data:
        fmt = "openapi"
    elif "$schema" in data or "properties" in data or "definitions" in data or "type" in data:
        fmt = "json_schema"
    else:
        fmt = "sample_yaml"

    schema_id = hashlib.sha256(raw_content.encode()).hexdigest()[:16]
    schema_json = json.dumps(data)

    db.execute(
        """
        INSERT INTO schemas (schema_id, title, description, format, schema_json)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT (schema_id) DO UPDATE SET
            title = excluded.title,
            description = excluded.description
        """,
        [schema_id, title, description, fmt, schema_json],
    )
    return schema_id


def list_schemas(db: duckdb.DuckDBPyConnection) -> list[dict]:
    """Return all stored schemas ordered newest first."""
    rows = db.execute(
        "SELECT schema_id, title, description, format, created_at FROM schemas ORDER BY created_at DESC"
    ).fetchall()
    cols = ("schema_id", "title", "description", "format", "created_at")
    return [dict(zip(cols, r)) for r in rows]


def get_schema(db: duckdb.DuckDBPyConnection, schema_id: str) -> dict | None:
    """Return the parsed schema dict for schema_id, or None if not found."""
    row = db.execute(
        "SELECT schema_json FROM schemas WHERE schema_id = ?", [schema_id]
    ).fetchone()
    if row is None:
        return None
    return json.loads(row[0])  # type: ignore[no-any-return]


def delete_schema(db: duckdb.DuckDBPyConnection, schema_id: str) -> None:
    """Remove a schema by id (no-op if not found)."""
    db.execute("DELETE FROM schemas WHERE schema_id = ?", [schema_id])
