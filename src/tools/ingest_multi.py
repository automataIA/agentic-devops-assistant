"""Multi-source document ingestion: file (MarkItDown) or URL (Crawl4AI) → DuckDB docs."""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from pathlib import Path

import duckdb

from src.tools.embeddings import embed_text
from src.tools.ingest_docs import _split_into_chunks


async def _file_to_markdown(path: Path) -> str:
    """PDF/DOCX/PPTX/XLSX/image/.md/.txt → Markdown string via MarkItDown (in thread pool)."""
    if path.suffix.lower() in (".md", ".txt"):
        return path.read_text(encoding="utf-8")
    from markitdown import MarkItDown
    result = await asyncio.to_thread(lambda: MarkItDown().convert(str(path)))
    return result.text_content or ""


async def _url_to_markdown(url: str) -> str:
    """Web page → Markdown via Crawl4AI AsyncWebCrawler (headless Chromium).

    Falls back to MarkItDown (plain httpx) if Crawl4AI fails.
    """
    try:
        from crawl4ai import AsyncWebCrawler
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
        markdown = result.markdown or ""
        if markdown.strip():
            return markdown
    except Exception:  # noqa: BLE001
        pass
    # Fallback: MarkItDown plain HTTP fetch (no browser)
    from markitdown import MarkItDown
    result = await asyncio.to_thread(lambda: MarkItDown().convert(url))
    return result.text_content or ""


def _apply_tags(chunks: list[str], tags: str) -> list[str]:
    """Prepend '[Tags: ...]' to each chunk for contextual retrieval (no-op if empty)."""
    if not tags.strip():
        return chunks
    prefix = f"[Tags: {tags.strip()}]\n"
    return [prefix + c for c in chunks]


async def ingest_source(
    db: duckdb.DuckDBPyConnection,
    embedding_model: str,
    *,
    file_path: Path | None = None,
    url: str | None = None,
    title: str = "",
    tags: str = "",
) -> int:
    """Convert source → Markdown → chunks → embeddings → DuckDB docs table.

    Args:
        db: Shared DuckDB connection (from app.state.deps).
        embedding_model: Ollama model name for embeddings.
        file_path: Local file path (mutually exclusive with url).
        url: Web URL to crawl (mutually exclusive with file_path).
        title: Human-readable title (defaults to filename stem or URL).
        tags: Comma-separated tags prepended to each chunk before embedding.

    Returns:
        Number of chunks stored.
    """
    if file_path is not None:
        markdown = await _file_to_markdown(file_path)
        doc_id = hashlib.sha256(file_path.name.encode()).hexdigest()[:16]
        title = title or file_path.stem.replace("-", " ").replace("_", " ").title()
    elif url:
        markdown = await _url_to_markdown(url)
        doc_id = hashlib.sha256(url.encode()).hexdigest()[:16]
        title = title or url
    else:
        raise ValueError("Provide file_path or url")

    # Hard limit: nomic-embed-text supports ~8192 tokens (~6000 chars).
    # Chunks from web pages may contain unsplit sections far exceeding this.
    MAX_CHUNK_CHARS = 6000
    raw_chunks = _split_into_chunks(markdown)
    safe_chunks = [c[:MAX_CHUNK_CHARS] for c in raw_chunks if c.strip()]
    chunks = _apply_tags(safe_chunks, tags)

    # Idempotent re-ingest: delete existing chunks for this doc
    db.execute("DELETE FROM docs WHERE doc_id LIKE ?", [f"{doc_id}%"])

    for chunk_index, chunk in enumerate(chunks):
        embedding = await embed_text(chunk, model=embedding_model)
        db.execute(
            "INSERT INTO docs (doc_id, title, chunk_text, chunk_index, embedding) VALUES (?, ?, ?, ?, ?)",
            [f"{doc_id}-{uuid.uuid4().hex[:8]}", title, chunk, chunk_index, embedding],
        )

    return len(chunks)
