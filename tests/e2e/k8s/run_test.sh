#!/usr/bin/env bash
# =============================================================================
# K8s Documentation Ingestion + YAML Validation — Playwright CLI E2E Test
# =============================================================================
# Steps:
#   1. Ingest K8s docs: 5 MD files + 2 official URLs
#   2. Upload K8s Deployment JSON Schema
#   3. Chat: attach broken-deployment.yaml → ask Config Validator to fix it
#
# Prerequisites:
#   - App running at http://localhost:8000
#   - Ollama running with qwen3:8b-q4_k_m + nomic-embed-text
#   - playwright-cli available in PATH
#
# Usage:
#   cd /home/dio/agentic-devops-assistant
#   bash tests/e2e/k8s/run_test.sh
# =============================================================================

set -euo pipefail

ASSETS="$(cd "$(dirname "$0")/assets" && pwd)"
APP_URL="http://localhost:8000"
SCREENSHOTS="$(cd "$(dirname "$0")" && pwd)/screenshots"
mkdir -p "$SCREENSHOTS"

echo "==> Assets: $ASSETS"
echo "==> Screenshots: $SCREENSHOTS"
echo ""

# ── 1. Health check ────────────────────────────────────────────────────────────
echo "[1/6] Checking app health..."
curl -sf "$APP_URL/status" | uv run python -c \
  "import sys,json; s=json.load(sys.stdin); print('  version:', s['version'], '| llm_ok:', s['llm_ok'], '| docs:', s['docs_count'])"
echo ""

# ── 2. Ingest MD docs via API (faster than browser for bulk) ───────────────────
echo "[2/6] Ingesting K8s markdown documentation..."
for f in "$ASSETS"/*.md; do
  name=$(basename "$f")
  echo "  → Uploading $name..."
  curl -sf -X POST "$APP_URL/ingest" \
    -F "file=@$f" \
    -F "tags=kubernetes,k8s,docs" \
    | uv run python -c "import sys,json; r=json.load(sys.stdin); print('     ✅', r.get('message','ok'))" \
    2>/dev/null || echo "     ⚠️  response not JSON (check app logs)"
done
echo ""

# ── 3. Ingest via URL ──────────────────────────────────────────────────────────
echo "[3/6] Ingesting K8s docs via URLs..."
for url in \
  "https://kubernetes.io/docs/concepts/workloads/controllers/deployment/" \
  "https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/"; do
  echo "  → Crawling $url..."
  curl -sf -X POST "$APP_URL/ingest" \
    -F "url=$url" \
    -F "tags=kubernetes,k8s,official-docs" \
    | uv run python -c "import sys,json; r=json.load(sys.stdin); print('     ✅', r.get('message','ok'))" \
    2>/dev/null || echo "     ⚠️  crawl may be slow — continuing"
done
echo ""

# ── 4. Upload schema ───────────────────────────────────────────────────────────
echo "[4/6] Uploading K8s Deployment JSON Schema..."
curl -sf -X POST "$APP_URL/schemas" \
  -F "file=@$ASSETS/k8s-deployment-schema.json" \
  -F "title=Kubernetes Deployment (apps/v1)" \
  -F "description=Strict JSON Schema for K8s Deployment manifests" \
  | uv run python -c "import sys,json; r=json.load(sys.stdin); print('  ✅ schema_id:', r.get('schema_id','?'))" \
  2>/dev/null || echo "  ⚠️  schema upload response not JSON"
echo ""

# ── 5. Check docs count ────────────────────────────────────────────────────────
echo "[5/6] Verifying docs count..."
curl -sf "$APP_URL/status" | uv run python -c \
  "import sys,json; s=json.load(sys.stdin); print('  docs:', s['docs_count'], '| schemas:', s['schemas_count'])"
echo ""

# ── 6. Browser: attach broken YAML and validate ────────────────────────────────
echo "[6/6] Launching Playwright browser session..."
echo "  (browser steps are run interactively via playwright-cli)"
