"""Unit tests for the Docu-RAG agent tools (mock DuckDB + mock embeddings, no LLM)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.agents.docu_rag import _format_chunks_for_llm, list_documents, semantic_search
from src.deps.connections import DocuRagDeps


def _make_deps(fetchdf_return: pd.DataFrame) -> DocuRagDeps:
    mock_db = MagicMock()
    mock_db.execute.return_value.fetchdf.return_value = fetchdf_return
    return DocuRagDeps(db=mock_db, embedding_model="nomic-embed-text", embedding_dim=768)


class _FakeCtx:
    def __init__(self, deps_obj: DocuRagDeps) -> None:
        self.deps = deps_obj


def _make_candidates(
    n_large_doc: int = 4,
    n_small_doc: int = 1,
) -> pd.DataFrame:
    """Build a realistic source-domination scenario.

    Returns a DataFrame simulating HNSW results where a large doc (K8s, 4 chunks)
    dominates a small doc (runbook, 1 chunk) — the original bug.
    """
    rows = []
    # Small doc (most relevant): 1 chunk, lowest cosine distance
    for i in range(n_small_doc):
        rows.append({
            "doc_id": f"runbook-abc-{i:08x}",
            "title": "test-doc.md",
            "chunk_text": f"VACUUM ANALYZE; Run during Sunday 02:00 UTC maintenance. Chunk {i}",
            "score": 0.308 + i * 0.001,
        })
    # Large doc (less relevant): many chunks, slightly worse scores
    for i in range(n_large_doc):
        rows.append({
            "doc_id": f"k8s-xyz-{i:08x}",
            "title": "kubernetes.io/deployment",
            "chunk_text": f"kubectl rollout status deployment/nginx. Chunk {i}",
            "score": 0.325 + i * 0.01,
        })
    return pd.DataFrame(rows).sort_values("score").reset_index(drop=True)


# ── _format_chunks_for_llm ─────────────────────────────────────────────────────

def test_format_chunks_injects_scores() -> None:
    df = pd.DataFrame({
        "doc_id": ["doc-aaa-00000001", "doc-bbb-00000002"],
        "title": ["Runbook", "K8s Guide"],
        "chunk_text": ["VACUUM ANALYZE", "kubectl rollout"],
        "rerank_score": [0.95, 0.40],
    })
    output = _format_chunks_for_llm(df, top_k=2)
    assert "Score: 0.9500" in output
    assert "Score: 0.4000" in output


def test_format_chunks_marks_most_relevant() -> None:
    df = pd.DataFrame({
        "doc_id": ["doc-aaa-00000001", "doc-bbb-00000002", "doc-ccc-00000003"],
        "title": ["Best Doc", "Other Doc", "Another Doc"],
        "chunk_text": ["best content", "other content", "another content"],
        "rerank_score": [0.99, 0.60, 0.50],
    })
    output = _format_chunks_for_llm(df, top_k=3)
    assert "← most relevant" in output
    # The most relevant label must appear on the first-listed chunk (Chunk 1) with a markdown link
    assert "[Chunk 1 | [Source: Best Doc](" in output


def test_format_chunks_sandwich_ordering() -> None:
    """Best chunk must appear first, second-best last (sandwich pattern)."""
    df = pd.DataFrame({
        "doc_id": [f"doc-{i:08x}-aabbccdd" for i in range(4)],
        "title": [f"Doc {i}" for i in range(4)],
        "chunk_text": [f"content {i}" for i in range(4)],
        "rerank_score": [0.99, 0.80, 0.70, 0.60],
    })
    output = _format_chunks_for_llm(df, top_k=4)
    lines = output.split("---")
    # First block = rank 0 (score 0.99), last block = rank 1 (score 0.80)
    assert "Doc 0" in lines[0]
    assert "Doc 1" in lines[-1]


def test_format_chunks_top_k_respected() -> None:
    df = pd.DataFrame({
        "doc_id": [f"doc-{i:08x}-aabbccdd" for i in range(10)],
        "title": [f"Doc {i}" for i in range(10)],
        "chunk_text": [f"content {i}" for i in range(10)],
        "rerank_score": [1.0 - i * 0.05 for i in range(10)],
    })
    output = _format_chunks_for_llm(df, top_k=3)
    # Only 3 chunks — Chunk 1, 2, 3 present; Chunk 4 absent
    assert "Chunk 3" in output
    assert "Chunk 4" not in output


# ── semantic_search (with mocked FlashRank + embed_text) ──────────────────────

def _fake_rerank(req):
    """Simulate FlashRank: returns dicts (matching real FlashRank 0.2.x API).

    Assigns higher score to passages containing 'VACUUM' (the relevant chunk).
    """
    items = []
    for p in req.passages:
        score = 0.95 if "VACUUM" in p["text"] else 0.30 - p["id"] * 0.01
        items.append({"id": p["id"], "text": p["text"], "score": score})
    return sorted(items, key=lambda x: x["score"], reverse=True)


async def test_semantic_search_reranker_promotes_relevant_chunk() -> None:
    """FlashRank must promote the small-doc relevant chunk above K8s noise."""
    candidates = _make_candidates(n_large_doc=4, n_small_doc=1)
    ctx = _FakeCtx(_make_deps(candidates))
    fake_embedding = [0.1] * 768

    with (
        patch("src.agents.docu_rag.embed_text", return_value=fake_embedding),
        patch("src.agents.docu_rag._get_ranker") as mock_ranker,
    ):
        ranker_instance = MagicMock()
        ranker_instance.rerank.side_effect = _fake_rerank
        mock_ranker.return_value = ranker_instance

        result = await semantic_search(ctx, "VACUUM PostgreSQL maintenance")  # type: ignore[arg-type]

    # The most relevant chunk (VACUUM runbook) must appear first and be marked
    assert "← most relevant" in result
    assert "test-doc.md" in result
    first_chunk_end = result.index("---") if "---" in result else len(result)
    assert "test-doc.md" in result[:first_chunk_end], (
        "The runbook chunk must be the FIRST chunk (sandwich position 0)"
    )


async def test_semantic_search_source_labels_present() -> None:
    """Each chunk block must contain Source: and Score: headers."""
    candidates = _make_candidates(n_large_doc=2, n_small_doc=1)
    ctx = _FakeCtx(_make_deps(candidates))

    with (
        patch("src.agents.docu_rag.embed_text", return_value=[0.1] * 768),
        patch("src.agents.docu_rag._get_ranker") as mock_ranker,
    ):
        ranker_instance = MagicMock()
        ranker_instance.rerank.side_effect = _fake_rerank
        mock_ranker.return_value = ranker_instance

        result = await semantic_search(ctx, "VACUUM")  # type: ignore[arg-type]

    assert "Source:" in result
    assert "Score:" in result


async def test_semantic_search_empty_result() -> None:
    ctx = _FakeCtx(_make_deps(pd.DataFrame()))
    fake_embedding = [0.0] * 768

    with patch("src.agents.docu_rag.embed_text", return_value=fake_embedding):
        result = await semantic_search(ctx, "unknown query")  # type: ignore[arg-type]

    assert "No documentation chunks found" in result


# ── list_documents ────────────────────────────────────────────────────────────

async def test_list_documents_returns_markdown() -> None:
    df = pd.DataFrame({
        "doc_id": ["doc-1", "doc-2"],
        "title": ["Architecture Overview", "Incident Runbook"],
        "chunk_count": [10, 5],
    })
    ctx = _FakeCtx(_make_deps(df))
    result = await list_documents(ctx)  # type: ignore[arg-type]
    assert "Architecture Overview" in result
    assert "Incident Runbook" in result


async def test_list_documents_empty() -> None:
    ctx = _FakeCtx(_make_deps(pd.DataFrame()))
    result = await list_documents(ctx)  # type: ignore[arg-type]
    assert "No documents" in result
    assert "ingest_docs" in result
