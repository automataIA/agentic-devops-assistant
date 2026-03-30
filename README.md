# SRE / DevOps Knowledge Copilot

An agentic assistant for SRE and DevOps teams that routes questions to specialist AI agents: structured log analysis, semantic search on internal docs, and strict YAML validation for Kubernetes and GitLab CI.

## Architecture

```
User → Supervisore (Router)
         ├── Log Explorer      → DuckDB → Apache Iceberg on MinIO
         ├── Docu-RAG          → DuckDB vss (HNSW) → vector embeddings on MinIO
         └── Config Validator  → Pydantic V2 strict models → YAML manifests
```

| Agent | File | Responsibility |
|---|---|---|
| Supervisore | `src/agents/supervisore.py` | Parse intent, delegate to specialist via tool call |
| Log Explorer | `src/agents/log_explorer.py` | SQL queries on Iceberg log tables |
| Docu-RAG | `src/agents/docu_rag.py` | Semantic search on markdown/PDF docs |
| Config Validator | `src/agents/config_validator.py` | YAML validation via Pydantic V2 strict models |

## Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | PydanticAI |
| LLM inference | Ollama (`OllamaProvider`) · Anthropic · OpenAI · Mistral cloud |
| Web API | FastAPI + HTMX (SSE streaming) |
| Analytics engine | DuckDB (extensions: `iceberg`, `vss`, `httpfs`) |
| Table format | Apache Iceberg (REST Catalog: Lakekeeper) |
| Object storage | MinIO (S3-compatible) |
| Persistence | DuckDB — conversations, messages, feedback, user settings |
| Observability | Pydantic Logfire (OpenTelemetry) |
| Validation | Pydantic V2 |
| Runtime | Python 3.13 + uv |

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- [Ollama](https://ollama.com/download) running locally
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

## Quick Start

**1. Clone and install dependencies**

```bash
git clone <repo-url>
cd agentic-devops-assistant
uv sync
```

**2. Pull Ollama models**

```bash
ollama pull qwen3:8b-q4_k_m   # LLM — recommended for reliable tool calling
ollama pull nomic-embed-text   # Embedding model for Docu-RAG
```

**3. Configure environment**

```bash
cp .env.example .env
# Edit .env to pick backend, model, and optional services
```

**4. Seed demo data (optional but recommended)**

```bash
uv run -m src.tools.seed_logs --rows 5000 --output ./data/logs_seed.parquet
uv run -m src.tools.ingest_docs --path ./docs/
```

**5. Start the application**

```bash
uv run uvicorn src.app:app --reload
```

Open `http://localhost:8000` in your browser.

> **Docker is optional.** The app runs fully without Docker using local DuckDB + Ollama. Docker Compose adds MinIO for production-grade Iceberg storage:
> ```bash
> docker compose -f docker/docker-compose.yml up -d
> ```
>
> The `docker/` folder contains a `docker/.env` symlink pointing to the root `.env`. This is required because Docker Compose V2 uses the compose file's directory as the project directory and resolves `.env` from there. The symlink is already listed in `.gitignore` and is not committed to the repository.

## Configuration

All configuration is via environment variables (`.env` file):

### LLM backend

| `AGENT_BACKEND` | Provider | Default `AGENT_MODEL` | Requirements |
|---|---|---|---|
| `ollama` (default) | `OllamaProvider` (local) | `qwen3:8b-q4_k_m` | Ollama running locally |
| `anthropic` | `AnthropicModel` | `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` |
| `openai` | `OpenAIChatModel` | `gpt-4o` | `OPENAI_API_KEY` |
| `mistral` | OpenAI-compat endpoint | `mistral-large-latest` | `MISTRAL_API_KEY` |

Tested local models (Ollama tool-calling reliability):

| Model | Log Explorer | Docu-RAG | Config Validator |
|---|---|---|---|
| `qwen3:8b-q4_k_m` | ✅ | ✅ | ✅ |
| `mistral-nemo` | ✅ | ⚠️ inconsistent | ❌ YAML 400 error |
| `llama3.1:8b` | ✅ | ⚠️ inconsistent | ⚠️ inconsistent |

### All variables

| Variable | Default | Description |
|---|---|---|
| `AGENT_BACKEND` | `ollama` | LLM backend: `ollama`, `anthropic`, `openai`, `mistral` |
| `AGENT_MODEL` | _(per-backend default)_ | Override model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Must include `/v1`; `OllamaProvider` passes it to `AsyncOpenAI` |
| `ANTHROPIC_API_KEY` | — | Required when `AGENT_BACKEND=anthropic` |
| `OPENAI_API_KEY` | — | Required when `AGENT_BACKEND=openai` |
| `MISTRAL_API_KEY` | — | Required when `AGENT_BACKEND=mistral` |
| `MINIO_ENDPOINT` | `http://localhost:9000` | MinIO S3 endpoint |
| `MINIO_ACCESS_KEY` | `minioadmin` | MinIO access key |
| `MINIO_SECRET_KEY` | `minioadmin` | MinIO secret key |
| `ICEBERG_CATALOG_URI` | `sqlite:///./data/iceberg_catalog.db` | PyIceberg SQLite catalog location; leave empty for parquet fallback |
| `ICEBERG_WAREHOUSE` | `s3://warehouse/iceberg` | Iceberg table root on MinIO (S3) |
| `DUCKDB_DATA_PATH` | `./data/local.duckdb` | DuckDB file path |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `EMBEDDING_DIM` | `768` | Vector dimension (768 for `nomic-embed-text`) |
| `LOGFIRE_TOKEN` | — | Pydantic Logfire token (optional) |

## Ingesting Documents

Place markdown files in a directory and run:

```bash
uv run -m src.tools.ingest_docs --path ./docs/
```

This chunks the documents, generates embeddings via Ollama, and stores them in the DuckDB `docs` table for semantic search.

## Seeding Synthetic Logs (Development)

```bash
uv run -m src.tools.seed_logs --rows 10000
uv run -m src.tools.seed_logs --rows 5000 --output ./data/logs.parquet
```

## Project Structure

```
agentic-devops-assistant/
├── src/
│   ├── agents/
│   │   ├── supervisore.py       # Router / orchestrator agent
│   │   ├── log_explorer.py      # DuckDB/Iceberg log queries
│   │   ├── docu_rag.py          # Vector search on docs
│   │   └── config_validator.py  # K8s/GitLab YAML validation
│   ├── deps/
│   │   └── connections.py       # Shared deps: DuckDB, model factory, build_deps()
│   ├── models/
│   │   ├── k8s.py               # Pydantic strict models for Kubernetes manifests
│   │   ├── gitlab.py            # Pydantic strict models for GitLab CI configs
│   │   ├── user.py              # User and UserSettings models
│   │   └── conversation.py      # Conversation, Message, Feedback models
│   ├── tools/
│   │   ├── embeddings.py        # embed_text() via Ollama /api/embeddings
│   │   ├── ingest_docs.py       # CLI: markdown ingestion with chunking
│   │   └── seed_logs.py         # CLI: synthetic log generation
│   └── app.py                   # FastAPI app with SSE streaming + HTMX UI
├── tests/
│   ├── agents/                  # Agent unit tests (TestModel, mock DuckDB)
│   ├── models/                  # Pydantic model unit tests (zero I/O)
│   ├── integration/             # Integration tests (real DuckDB in-memory, TestModel)
│   └── conftest.py              # Shared fixtures
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml       # MinIO + minio-init
│   └── docker-compose.dev.yml   # Dev overrides (LOG_LEVEL=DEBUG, hot-reload)
├── .env.example                 # Environment variable template
├── pyproject.toml
└── .python-version              # 3.13
```

## Development

```bash
# Run all tests
uv run -m pytest tests/ -v

# Run only unit tests (no Docker needed)
uv run -m pytest tests/ -v -m "not integration"

# Lint and format
uvx ruff check . --fix
uvx ruff format .

# Type check
uv run -m mypy src/

# Start dev server with reload
uv run uvicorn src.app:app --reload
```

## Conversation Features

- **Persistent history** — every chat is stored in DuckDB and restored when you revisit; the sidebar lists all your past conversations
- **Multi-turn context** — each message carries the last 10 turns as history so follow-up questions work without restating context
- **Feedback** — thumbs-up / thumbs-down on any assistant response; stored per user
- **Prefix commands** — `/logs`, `/docs`, `/validate` at the start of a message bypass the router and go directly to the specialist agent
- **Keyboard shortcuts** — `Ctrl+K` new chat · `Ctrl+/` focus textarea with `/`
- **Settings** — per-user theme, language, and LLM backend overrides via `GET/PUT /settings`

## What Each Agent Can Do

### Log Explorer
Ask questions about your logs in natural language:
- *"Show me all errors for api-gateway in the last hour"*
- *"What's the error rate breakdown by severity in the last 30 minutes?"*
- *"Find all log lines for trace ID abc123def456"*

### Docu-RAG
Search your internal architecture documentation:
- *"What's the procedure for rolling back the production database?"*
- *"How does the payment service handle failed Stripe calls?"*
- *"What are the alerting thresholds for the API gateway?"*

### Config Validator
Validate any DevOps/SRE YAML configuration by attaching a `.yaml` / `.yml` file:

**Kubernetes** (multi-turn, auto-fix, YAML export):
- Deployment, Service, ConfigMap, Secret — strict Pydantic V2 validation
- Asks clarifying questions for replicas, namespace, resource limits
- Exports the corrected YAML with inline `# FIXED:` comments

**GitLab CI** (multi-turn):
- Full `.gitlab-ci.yml` pipeline validation — jobs, stages, rules, artifacts, retry bounds

**Dynamic schema validation** — auto-detects format from filename using [SchemaStore.org](https://www.schemastore.org) (1 240+ schemas):

| File | Format |
|---|---|
| `docker-compose.yml` | Docker Compose |
| `.github/workflows/ci.yml` | GitHub Actions |
| `Chart.yaml` | Helm Chart |
| `.circleci/config.yml` | CircleCI |
| `azure-pipelines.yml` | Azure Pipelines |
| `.drone.yml` | Drone CI |
| `renovate.json` | Renovate |
| `.github/dependabot.yml` | Dependabot |
| `.pre-commit-config.yaml` | pre-commit |
| `openapi.yml` (has `openapi:` key) | OpenAPI 3.0 / 3.1 |

Any filename not matched by SchemaStore falls back to LLM semantic validation with an explicit note that structural validation was skipped.

### Schema Management + Static YAML Repair

The "Schemas" button (amber) in the sidebar opens a two-tab modal for schema-guided validation with minimal LLM involvement:

**Flow A — Upload a reference schema**
Store any JSON Schema, OpenAPI spec, or YAML template as a named reference. Schemas are content-hashed for idempotent re-uploads.

**Flow B — Validate a document against a stored schema**
Upload or paste a YAML document, select a stored schema, and run the static repair pipeline:

1. `yamlfix` — syntax normalisation (truthy strings, indentation), comment-safe
2. `ruamel.yaml` — roundtrip parse preserving comments
3. Default injection — fills `default` values for missing required keys
4. Type coercion — `replicas: "3"` → `replicas: 3`, `enabled: yes` → `enabled: true`
5. Extra key stripping — removes keys not present in `additionalProperties: false` schemas
6. Ambiguous fields flagged for manual review (no hallucination risk)

The result shows the repaired YAML, remaining errors, and an amber audit trail of fields that could not be fixed deterministically.

**Flow C — Active reference schema for chat uploads**
Click **"Set active"** below the schema selector to pin a schema as the active reference. An `active_schema_id` cookie is set. Any `.yaml` file subsequently uploaded via the chat attachment button is automatically routed through the static repair pipeline using that schema — no need to open the modal each time.

If the static repair finds errors it cannot fix deterministically (semantic violations: wrong enum values, invalid image names, out-of-range ports), a **"Fix with AI"** button appears in the chat bubble. Clicking it calls `POST /validate/{session_id}/fix-with-ai`, which reads the stored schema error list and forwards the YAML + known violations as context to `config_validator_v2_agent`. The LLM starts already knowing what to fix, skips re-analysis, and produces a corrected YAML with inline `# FIXED:` comments available for download.

## Architecture Notes

- **Single DuckDB connection**: One `duckdb.connect()` per application instance, shared across all agents. Log Explorer and Docu-RAG are read-only; ingestion scripts run as separate CLI processes.
- **Docu-RAG retrieval pipeline**: `semantic_search` uses a three-stage pipeline — (1) HNSW cosine over-fetch (`top_k × 4` candidates) with a per-document SQL cap (`QUALIFY ROW_NUMBER() ≤ 2`) to prevent source-dominant documents from flooding results; (2) FlashRank cross-encoder reranking (`ms-marco-MiniLM-L-12-v2`, ONNX, no PyTorch) for discriminative `(query, chunk)` scoring; (3) sandwich ordering (best chunk first, second-best last) with score annotations injected into each chunk header. Controlled by `LOG_LEVEL=DEBUG` for raw retrieval visibility.
- **Ollama embeddings**: Uses Ollama's native `/api/embeddings` endpoint (not `/v1/embeddings` — different schema). `embed_text()` strips `/v1` from `OLLAMA_BASE_URL` automatically. The embedding dimension is fixed at table creation time; changing it requires re-ingesting all documents.
- **OllamaProvider**: PydanticAI's `OllamaProvider` applies model-specific profiles (llama, mistral, qwen, deepseek, gemma…) and built-in workarounds for Ollama quirks (missing `finish_reason`, json schema handling). It passes `OLLAMA_BASE_URL` directly to `AsyncOpenAI`, so the URL must include `/v1`.
- **`output_type=str` on all agents**: Local 7B-class models reliably return free text after tool calls but fail to produce structured JSON output. All four agents use `output_type=str`; structured result classes (`LogQueryResult`, `DocSearchResult`, `ValidationResult`) are used only in tool-level tests and the validator's tool functions.
- **PyIceberg + SQLite catalog**: Iceberg tables are written to MinIO via `pyiceberg` with a local SQLite catalog (`ICEBERG_CATALOG_URI`). No REST catalog server (Lakekeeper/Polaris/Nessie) is required. DuckDB reads the tables directly via `iceberg_scan(metadata_location)` as a transparent `logs_view`. Set `ICEBERG_CATALOG_URI` to empty to fall back to a local demo parquet file.
- **Config Validator routing**: `_handle_yaml_upload` checks for an `active_schema_id` cookie first — if set and the schema exists, routes directly to `_handle_static_repair()` (deterministic, no LLM). Otherwise inspects the YAML: `apiVersion`+`kind` → K8s multi-turn agent; `stages`/`script` → GitLab multi-turn agent; `openapi`/`swagger` → `validate_openapi_spec`; anything else → `validate_generic_yaml`.
- **Schema management**: `schemas` DuckDB table (created at startup alongside `docs`) stores reference schemas with `sha256[:16]` content-hash IDs. `schemas` MinIO bucket is provisioned alongside `warehouse`, `logs`, `docs`. `src/tools/schema_store_local.py` provides the CRUD layer; `ValidatorDeps.db` carries the shared connection. The active schema is tracked via `active_schema_id` cookie (`POST /schemas/activate`, `DELETE /schemas/active`).
- **Static repair pipeline**: `src/tools/yaml_repair.py` — `repair_yaml(yaml_content, schema)` is purely deterministic: no LLM, no network. Uses `jsonschema.validators.extend()` to inject defaults in-place (a standard jsonschema pattern), then applies type coercion and extra-key removal. `ruamel.yaml` is used for all parse/dump to preserve comments. Failed coercions (e.g. `replicas: "three"`) are surfaced in `llm_fields`. When errors remain, the response includes a **"Fix with AI"** button that escalates to `config_validator_v2_agent` via `POST /validate/{session_id}/fix-with-ai`, forwarding the exact schema error list as context.
- **Tool delegation pattern**: The Supervisore selects specialist agents dynamically via LLM tool calls. The tool docstrings are the routing instructions — keep them precise.
- **Demo data fallback**: When `ICEBERG_CATALOG_URI` is empty, `_maybe_load_demo_data()` auto-loads `./data/logs_seed.parquet` at startup, making the app fully functional without any Docker services.
