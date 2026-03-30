"""Docu-RAG agent — semantic search over architecture docs via DuckDB vss HNSW."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

logger = logging.getLogger(__name__)

from src.deps.connections import DocuRagDeps, build_model
from src.tools.async_db import async_query
from src.tools.embeddings import embed_text

# ── Output model ──────────────────────────────────────────────────────────────


class DocSearchResult(BaseModel):
    answer: str
    sources: list[str]
    chunks_used: int


# ── Agent ─────────────────────────────────────────────────────────────────────

docu_rag_agent: Agent[DocuRagDeps, str] = Agent(
    build_model(),
    deps_type=DocuRagDeps,
    output_type=str,
    system_prompt=(
        "You are an expert SRE documentation assistant. "
        "Use the semantic_search tool to find relevant documentation chunks, "
        "then synthesise a concise answer citing the source documents.\n"
        "The chunks are ranked by relevance score (lower score = more relevant). "
        "ALWAYS prioritize information from the chunk marked '← most relevant'. "
        "If the top-ranked chunk contradicts the others, trust it over the majority. "
        "Do not synthesize across sources unless they explicitly agree.\n"
        "After calling a tool, provide a clear plain-text answer with the document title(s) as sources. "
        "If no relevant documentation is found, say so clearly — do not hallucinate.\n"
        "When citing sources, reproduce the markdown link from the chunk header exactly as written, "
        "in the format [Source: title](/docs/chunk/doc_id), so the user can click it to preview the chunk."
    ),
)


# ── Helpers ───────────────────────────────────────────────────────────────────

# Module-level FlashRank ranker — avoids reloading the ONNX model on every call.
_ranker = None


def _get_ranker():
    global _ranker
    if _ranker is None:
        from flashrank import Ranker
        _ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
    return _ranker


def _format_chunks_for_llm(df, top_k: int) -> str:
    """Format top_k chunks with score annotations and sandwich ordering.

    Sandwich ordering: best chunk first, second-best last, rest in middle.
    This counteracts the 'lost in the middle' LLM attention bias.
    """
    rows = df.head(top_k).reset_index(drop=True)
    if rows.empty:
        return ""

    n = len(rows)
    order = [0] + list(range(2, n)) + [1] if n > 2 else list(range(n))

    lines: list[str] = []
    for idx in order:
        row = rows.iloc[idx]
        score = row.get("rerank_score", row.get("score", 0.0))
        hint = " ← most relevant" if idx == 0 else ""
        lines.append(
            f"[Chunk {idx + 1} | [Source: {row['title']}](/docs/chunk/{row['doc_id']}) | Score: {score:.4f}{hint}]\n"
            f"{row['chunk_text']}"
        )

    return "\n\n---\n\n".join(lines)


# ── Constants ─────────────────────────────────────────────────────────────────

# Over-fetch: retrieve FETCH_MULTIPLIER * top_k HNSW candidates before reranking.
_FETCH_MULTIPLIER = 4
# Cap: at most this many chunks from any single document in the candidate set.
_MAX_CHUNKS_PER_DOC = 2


# ── Tools ─────────────────────────────────────────────────────────────────────


@docu_rag_agent.tool
async def semantic_search(
    ctx: RunContext[DocuRagDeps],
    query: str,
    top_k: int = 5,
) -> str:
    """Search the internal documentation using semantic similarity.

    Pipeline:
      1. HNSW cosine search — over-fetches top (top_k * 4) candidates.
      2. Per-document cap — at most 2 chunks per source to prevent source domination.
      3. FlashRank cross-encoder reranking — discriminative (query, chunk) scoring.
      4. Sandwich ordering + score injection — LLM receives best chunk first.

    Args:
        query: The natural-language search query.
        top_k: Number of top-matching chunks to return to the LLM. Default: 5.

    Returns:
        Formatted text blocks with relevance scores for the LLM to synthesise.
    """
    embedding = await embed_text(query, ctx.deps.embedding_model)
    dim = ctx.deps.embedding_dim
    fetch_n = top_k * _FETCH_MULTIPLIER

    # ── Step 1: HNSW over-fetch + per-doc cap (SQL QUALIFY) ──────────────────
    result = await async_query(
        ctx.deps.db,
        f"""
        SELECT doc_id, title, chunk_text, score
        FROM (
            SELECT doc_id,
                   title,
                   chunk_text,
                   array_cosine_distance(embedding, $1::FLOAT[{dim}]) AS score,
                   ROW_NUMBER() OVER (
                       PARTITION BY REGEXP_REPLACE(doc_id, '-[a-f0-9]{{8}}$', '')
                       ORDER BY array_cosine_distance(embedding, $1::FLOAT[{dim}]) ASC
                   ) AS rn
            FROM docs
            ORDER BY array_cosine_distance(embedding, $1::FLOAT[{dim}]) ASC
            LIMIT {fetch_n}
        )
        WHERE rn <= {_MAX_CHUNKS_PER_DOC}
        ORDER BY score ASC
        """,
        [embedding],
    )

    if result.empty:
        logger.debug("semantic_search: no chunks found for query=%r", query)
        return "No documentation chunks found. Have you run the ingest_docs script?"

    logger.debug(
        "semantic_search [after cap]: query=%r fetch_n=%d candidates=%d\n%s",
        query,
        fetch_n,
        len(result),
        result[["doc_id", "title", "score"]].to_markdown(index=False),
    )

    # ── Step 2: FlashRank cross-encoder reranking ─────────────────────────────
    import pandas as pd
    from flashrank import RerankRequest

    ranker = _get_ranker()
    result_indexed = result.reset_index(drop=True)
    passages = [
        {"id": i, "text": row["chunk_text"]}
        for i, row in result_indexed.iterrows()
    ]
    req = RerankRequest(query=query, passages=passages)

    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        reranked = await loop.run_in_executor(pool, ranker.rerank, req)

    reranked_rows = []
    for item in reranked:
        row = result_indexed.iloc[item["id"]].to_dict()
        row["rerank_score"] = item["score"]
        reranked_rows.append(row)

    reranked_df = pd.DataFrame(reranked_rows).sort_values(
        "rerank_score", ascending=False
    )

    logger.debug(
        "semantic_search [after rerank]: query=%r top_k=%d\n%s",
        query,
        top_k,
        reranked_df[["title", "score", "rerank_score"]].head(top_k).to_markdown(
            index=False
        ),
    )

    # ── Step 3: Persist retrieved chunks for structured source pills in the UI ─
    ctx.deps.retrieved_chunks = [
        {
            "doc_id": row["doc_id"],
            "title": row["title"],
            "chunk_index": int(row.get("chunk_index", 0)),
        }
        for _, row in reranked_df.head(top_k).iterrows()
    ]

    # ── Step 4: Format with sandwich ordering + score injection ───────────────
    return _format_chunks_for_llm(reranked_df, top_k)


@docu_rag_agent.tool
async def list_documents(ctx: RunContext[DocuRagDeps]) -> str:
    """List all documents currently ingested in the vector store.

    Returns:
        Markdown table with columns: doc_id, title, chunk_count.
    """
    result = await async_query(
        ctx.deps.db,
        """
        SELECT doc_id, title, COUNT(*) AS chunk_count
        FROM docs
        GROUP BY doc_id, title
        ORDER BY title
        """,
    )

    if result.empty:
        return "No documents have been ingested yet. Run: uv run -m src.tools.ingest_docs --path ./docs/"
    return result.to_markdown(index=False)
