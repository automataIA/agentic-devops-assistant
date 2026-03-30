"""Shared dependency dataclasses and factory functions for all agents."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import boto3
import duckdb
import logfire
from botocore.client import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider

load_dotenv()


# ── Dependency dataclasses ────────────────────────────────────────────────────


@dataclass
class LogDeps:
    db: duckdb.DuckDBPyConnection
    minio_endpoint: str


@dataclass
class DocuRagDeps:
    db: duckdb.DuckDBPyConnection
    embedding_model: str
    embedding_dim: int
    # Populated by semantic_search on each tool call; read by chat_html to render source pills.
    retrieved_chunks: list[dict] = field(default_factory=list)


@dataclass
class ValidatorDeps:
    db: duckdb.DuckDBPyConnection | None = None


@dataclass
class SupervisoreDeps:
    db: duckdb.DuckDBPyConnection
    log_deps: LogDeps
    docu_rag_deps: DocuRagDeps
    validator_deps: ValidatorDeps
    user_id: str | None = None  # set per-request by the route handler


# ── Model factory ─────────────────────────────────────────────────────────────


def build_model() -> OpenAIChatModel | AnthropicModel:
    """Return the configured LLM model based on AGENT_BACKEND env var.

    Supported backends:
    - ``ollama``    (default) — local Ollama via OllamaProvider; default model: mistral-nemo
    - ``anthropic``           — Anthropic API; requires ANTHROPIC_API_KEY
    - ``openai``              — OpenAI API; requires OPENAI_API_KEY
    - ``mistral``             — Mistral cloud (OpenAI-compat endpoint); requires MISTRAL_API_KEY
    """
    backend = os.getenv("AGENT_BACKEND", "ollama").lower()
    model_name = os.getenv("AGENT_MODEL", "")

    if backend == "ollama":
        # OllamaProvider reads OLLAMA_BASE_URL from env automatically.
        # The env var must include /v1 (e.g. http://localhost:11434/v1) because
        # OllamaProvider passes it directly to AsyncOpenAI which requires it.
        # embed_text() strips /v1 separately for the native /api/embeddings call.
        provider = OllamaProvider()
        return OpenAIChatModel(
            model_name=model_name or "mistral-nemo",
            provider=provider,
        )

    if backend == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("AGENT_BACKEND=anthropic but ANTHROPIC_API_KEY is not set")
        return AnthropicModel(model_name or "claude-sonnet-4-6")

    if backend == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("AGENT_BACKEND=openai but OPENAI_API_KEY is not set")
        return OpenAIChatModel(
            model_name=model_name or "gpt-4o",
            provider=OpenAIProvider(api_key=api_key),
        )

    if backend == "mistral":
        # Mistral cloud via its OpenAI-compatible endpoint — no extra package required
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("AGENT_BACKEND=mistral but MISTRAL_API_KEY is not set")
        return OpenAIChatModel(
            model_name=model_name or "mistral-large-latest",
            provider=OpenAIProvider(
                base_url="https://api.mistral.ai/v1",
                api_key=api_key,
            ),
        )

    raise RuntimeError(
        f"Unknown AGENT_BACKEND='{backend}'. Valid options: ollama, anthropic, openai, mistral"
    )


# ── MinIO bucket provisioning ─────────────────────────────────────────────────

_REQUIRED_BUCKETS = ("warehouse", "logs", "docs", "schemas")


def ensure_minio_buckets() -> None:
    """Create required MinIO buckets if they don't exist.

    All connection parameters are read from environment variables:
    MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY.
    """
    endpoint = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
    access_key = os.getenv("MINIO_ACCESS_KEY", "")
    secret_key = os.getenv("MINIO_SECRET_KEY", "")

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )

    for bucket in _REQUIRED_BUCKETS:
        try:
            s3.head_bucket(Bucket=bucket)
            print(f"[minio] bucket '{bucket}' already exists")
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("404", "NoSuchBucket"):
                s3.create_bucket(Bucket=bucket)
                print(f"[minio] created bucket '{bucket}'")
            else:
                logfire.warn("MinIO bucket check failed", bucket=bucket, error=str(exc))


# ── DuckDB setup ──────────────────────────────────────────────────────────────


def _setup_duckdb(conn: duckdb.DuckDBPyConnection) -> None:
    """Install and load required DuckDB extensions, configure S3 credentials."""
    for ext in ("iceberg", "httpfs", "vss"):
        conn.execute(f"INSTALL {ext}")
        conn.execute(f"LOAD {ext}")

    minio_endpoint = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
    # Strip scheme — DuckDB expects host:port format
    endpoint_host = minio_endpoint.replace("http://", "").replace("https://", "")

    conn.execute(f"SET s3_endpoint='{endpoint_host}'")
    conn.execute(f"SET s3_access_key_id='{os.getenv('MINIO_ACCESS_KEY', 'minioadmin')}'")
    conn.execute(f"SET s3_secret_access_key='{os.getenv('MINIO_SECRET_KEY', 'minioadmin')}'")
    conn.execute("SET s3_use_ssl=false")
    conn.execute("SET s3_url_style='path'")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS schemas (
            schema_id   VARCHAR PRIMARY KEY,
            title       VARCHAR NOT NULL,
            description VARCHAR DEFAULT '',
            format      VARCHAR DEFAULT 'json_schema',
            schema_json VARCHAR NOT NULL,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)


def _setup_iceberg_views(conn: duckdb.DuckDBPyConnection) -> bool:
    """Open PyIceberg SQLite catalog and create a DuckDB VIEW for each Iceberg table.

    Returns True when at least the ``logs`` view was successfully created.
    Falls back silently so the demo-data path can take over.
    """
    catalog_uri = os.getenv("ICEBERG_CATALOG_URI", "")
    if not catalog_uri:
        return False

    warehouse = os.getenv("ICEBERG_WAREHOUSE", "s3://warehouse/iceberg")
    minio_endpoint = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
    access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")

    try:
        from pyiceberg.catalog.sql import SqlCatalog
        from pyiceberg.exceptions import NoSuchTableError

        catalog = SqlCatalog(
            "local",
            **{
                "uri": catalog_uri,
                "warehouse": warehouse,
                "s3.endpoint": minio_endpoint,
                "s3.access-key-id": access_key,
                "s3.secret-access-key": secret_key,
                "s3.region": "us-east-1",
                "s3.path-style-access": "true",
            },
        )
    except Exception as exc:  # noqa: BLE001
        logfire.warn("Could not open PyIceberg SQLite catalog", error=str(exc))
        return False

    try:
        logs_table = catalog.load_table("logs.app_logs")
        meta_loc = logs_table.metadata_location
        conn.execute(f"CREATE OR REPLACE VIEW logs AS SELECT * FROM iceberg_scan('{meta_loc}')")
        count = conn.execute("SELECT COUNT(*) FROM logs").fetchone()[0]  # type: ignore[index]
        print(f"[iceberg] logs view → {meta_loc} ({count:,} rows)")
        return True
    except NoSuchTableError:
        print("[iceberg] Table 'logs.app_logs' not found — run: uv run -m src.tools.seed_iceberg")
        return False
    except Exception as exc:  # noqa: BLE001
        logfire.warn("Could not create DuckDB view for logs.app_logs", error=str(exc))
        return False


def _maybe_load_demo_data(conn: duckdb.DuckDBPyConnection, *, skip: bool = False) -> None:
    """Load demo parquet as a DuckDB TABLE when no Iceberg catalog is configured.

    Activated when ``skip=False`` (Iceberg not available) and the parquet exists.
    Makes the app fully functional without Docker services.
    """
    if skip:
        return  # Iceberg views already set up — no demo data needed

    logs_parquet = os.getenv("DEMO_LOGS_PATH", "./data/logs_seed.parquet")
    if os.path.exists(logs_parquet):
        try:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS logs AS
                SELECT * FROM read_parquet('{logs_parquet}')
            """)
            count = conn.execute("SELECT COUNT(*) FROM logs").fetchone()[0]  # type: ignore[index]
            print(f"[demo] Loaded {count:,} log rows from {logs_parquet}")
        except Exception as exc:  # noqa: BLE001
            print(f"[demo] Could not load logs parquet: {exc}")
    else:
        # Create empty logs table so queries don't fail with "table not found"
        conn.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                timestamp VARCHAR,
                service   VARCHAR,
                level     VARCHAR,
                message   VARCHAR,
                trace_id  VARCHAR
            )
        """)
        print("[demo] No logs parquet found — logs table is empty. Run: uv run -m src.tools.seed_logs --rows 5000 --output ./data/logs_seed.parquet")


def _ensure_validation_sessions_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create validation_sessions table and purge sessions older than 24 hours."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS validation_sessions (
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
    conn.execute("DELETE FROM validation_sessions WHERE created_at < now() - INTERVAL '24 hours'")


def _ensure_docs_table(conn: duckdb.DuckDBPyConnection, embedding_dim: int) -> None:
    """Create the docs vector table and HNSW index if they don't exist."""
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS docs (
            doc_id  VARCHAR PRIMARY KEY,
            title   VARCHAR NOT NULL,
            chunk_text VARCHAR NOT NULL,
            embedding FLOAT[{embedding_dim}]
        )
    """)
    conn.execute("ALTER TABLE docs ADD COLUMN IF NOT EXISTS chunk_index INTEGER DEFAULT 0")
    # HNSW index requires hnsw_enable_experimental_persistence for file-based DBs.
    # For :memory: it works out of the box.
    try:
        # Persistence required for file-based DBs (no-op for :memory:)
        conn.execute("SET hnsw_enable_experimental_persistence = true")
        conn.execute("""
            CREATE INDEX IF NOT EXISTS docs_embedding_idx
            ON docs USING HNSW (embedding)
            WITH (metric = 'cosine')
        """)
    except Exception as exc:  # noqa: BLE001
        logfire.warn("Could not create HNSW index", error=str(exc))


# ── Multi-tenant tables ───────────────────────────────────────────────────────


def _ensure_users_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create users table for multi-tenant support."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id      VARCHAR PRIMARY KEY,
            display_name VARCHAR DEFAULT 'Anonymous',
            created_at   TIMESTAMPTZ DEFAULT now()
        )
    """)


def _ensure_conversations_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create conversations table for persistent chat history."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id VARCHAR PRIMARY KEY,
            user_id         VARCHAR NOT NULL,
            title           VARCHAR DEFAULT 'New Chat',
            created_at      TIMESTAMPTZ DEFAULT now(),
            updated_at      TIMESTAMPTZ DEFAULT now()
        )
    """)


def _ensure_messages_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create messages table for conversation persistence."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
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


def _ensure_feedback_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create feedback table for message ratings."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            message_id VARCHAR NOT NULL,
            user_id    VARCHAR NOT NULL,
            rating     VARCHAR NOT NULL,
            comment    TEXT,
            created_at TIMESTAMPTZ DEFAULT now(),
            PRIMARY KEY (message_id, user_id)
        )
    """)


def _ensure_user_settings_table(conn: duckdb.DuckDBPyConnection) -> None:
    """Create user_settings table for per-user preferences."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_settings (
            user_id       VARCHAR PRIMARY KEY,
            theme         VARCHAR DEFAULT 'business',
            language      VARCHAR DEFAULT 'en',
            agent_backend VARCHAR,
            agent_model   VARCHAR
        )
    """)


# ── Public factory ────────────────────────────────────────────────────────────


async def build_deps() -> SupervisoreDeps:
    """Build all agent dependencies. Call once at application startup."""
    ensure_minio_buckets()

    db_path = os.getenv("DUCKDB_DATA_PATH", ":memory:")
    conn = duckdb.connect(db_path)

    _setup_duckdb(conn)
    iceberg_ready = _setup_iceberg_views(conn)
    _maybe_load_demo_data(conn, skip=iceberg_ready)

    embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    embedding_dim = int(os.getenv("EMBEDDING_DIM", "768"))
    _ensure_docs_table(conn, embedding_dim)
    _ensure_validation_sessions_table(conn)

    # Multi-tenant tables
    _ensure_users_table(conn)
    _ensure_conversations_table(conn)
    _ensure_messages_table(conn)
    _ensure_feedback_table(conn)
    _ensure_user_settings_table(conn)

    minio_endpoint = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")

    log_deps = LogDeps(
        db=conn,
        minio_endpoint=minio_endpoint,
    )
    docu_rag_deps = DocuRagDeps(
        db=conn,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
    )
    validator_deps = ValidatorDeps(db=conn)

    return SupervisoreDeps(
        db=conn,
        log_deps=log_deps,
        docu_rag_deps=docu_rag_deps,
        validator_deps=validator_deps,
    )
