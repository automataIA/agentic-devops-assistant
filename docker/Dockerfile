# syntax=docker/dockerfile:1.7
# Multi-stage build — non-root user, uv-managed venv, minimal runtime image

# ── Stage 0: CSS builder (Tailwind CSS 4 + DaisyUI 5) ───────────────────────
FROM node:22-slim AS css-builder
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci --ignore-scripts 2>/dev/null || npm install --ignore-scripts
COPY src/static/css/input.css ./src/static/css/input.css
COPY src/templates/ ./src/templates/
RUN npx @tailwindcss/cli -i src/static/css/input.css -o src/static/css/output.css --minify


# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.13-slim AS builder

# Install uv from the official distroless image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy files into the venv instead of symlinks (required for cross-stage copy)
# Pre-compile .pyc files at install time for faster cold-start
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# ── Dependency layer (cached unless pyproject.toml / uv.lock change) ─────────
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# ── Application layer ─────────────────────────────────────────────────────────
COPY src/ ./src/

# Copy the built CSS from css-builder stage
COPY --from=css-builder /app/src/static/css/output.css ./src/static/css/output.css

RUN uv sync --frozen --no-dev

# Pre-install DuckDB extensions so the container starts without egress
RUN uv run --no-sync python -c "\
import duckdb; \
conn = duckdb.connect(':memory:'); \
[conn.execute(f'INSTALL {e}') for e in ('iceberg', 'httpfs', 'vss')]; \
print('DuckDB extensions pre-installed.')"

# Pre-download Playwright Chromium for Crawl4AI JS-fallback
RUN uv run --no-sync playwright install chromium


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

# Non-root system user
RUN groupadd --system app && \
    useradd  --system --gid app --no-create-home --shell /sbin/nologin app

# uv binary — needed for `uv run` in CMD
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy the pre-built venv and application source from builder
COPY --from=builder --chown=app:app /app/.venv  ./.venv
COPY --from=builder --chown=app:app /app/src     ./src

# DuckDB extension cache pre-installed in builder's home → bring it along
COPY --from=builder --chown=app:app /root/.duckdb /home/app/.duckdb

# Playwright Chromium browser pre-downloaded in builder
COPY --from=builder --chown=app:app /root/.cache/ms-playwright /home/app/.cache/ms-playwright

# Install Chromium OS-level deps (X11, fonts, NSS) — must run as root before USER app
RUN /app/.venv/bin/playwright install-deps chromium \
    && rm -rf /var/lib/apt/lists/*

# Persistent volume for SQLite catalog, DuckDB state, parquet seed files
# Create dir as root before switching user so Docker preserves ownership on volume mount
RUN mkdir -p /app/data && chown app:app /app/data
VOLUME ["/app/data"]

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_NO_SYNC=1 \
    # uv must find the venv we copied
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PORT=8000

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')"

CMD ["uv", "run", "--no-sync", "python", "-m", "src.main"]
