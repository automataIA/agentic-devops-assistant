"""Document ingestion script — chunks markdown files and stores embeddings in DuckDB/Iceberg.

Usage:
    uv run -m src.tools.ingest_docs --path ./docs/
    uv run -m src.tools.ingest_docs --path ./docs/ --clear
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import re
import uuid
from pathlib import Path

from src.deps.connections import build_deps
from src.tools.embeddings import embed_text

# ── Chunking parameters ───────────────────────────────────────────────────────

CHUNK_SIZE = 512       # approximate characters per chunk
CHUNK_OVERLAP = 64    # overlap between consecutive chunks


def _split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph boundaries."""
    paragraphs = re.split(r"\n{2,}", text.strip())
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(current)
            # If a single paragraph exceeds chunk_size, split it by sentences
            if len(para) > chunk_size:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                buf = ""
                for sent in sentences:
                    if len(buf) + len(sent) + 1 <= chunk_size:
                        buf = (buf + " " + sent).strip()
                    else:
                        if buf:
                            chunks.append(buf)
                        buf = sent[-overlap:] + " " + sent if overlap else sent
                if buf:
                    chunks.append(buf)
                current = ""
            else:
                # Keep overlap from previous chunk
                last = chunks[-1][-overlap:] if chunks and overlap else ""
                current = (last + "\n\n" + para).strip() if last else para

    if current:
        chunks.append(current)

    return [c for c in chunks if c.strip()]


def _doc_id(path: Path) -> str:
    """Deterministic doc_id from file path hash."""
    return hashlib.sha256(str(path).encode()).hexdigest()[:16]


async def ingest_markdown_dir(
    path: str,
    clear: bool = False,
) -> int:
    """Ingest all markdown files in *path* into the DuckDB docs table.

    Args:
        path: Directory containing .md files.
        clear: If True, delete all existing docs before ingesting.

    Returns:
        Number of chunks inserted.
    """
    deps = await build_deps()
    db = deps.docu_rag_deps.db
    embedding_model = deps.docu_rag_deps.embedding_model

    if clear:
        db.execute("DELETE FROM docs")
        print("Cleared existing docs.")

    docs_dir = Path(path)
    if not docs_dir.is_dir():
        raise ValueError(f"Not a directory: {path}")

    md_files = sorted(docs_dir.rglob("*.md"))
    if not md_files:
        print(f"No .md files found in {path}")
        return 0

    total_chunks = 0

    for md_file in md_files:
        doc_id = _doc_id(md_file)
        title = md_file.stem.replace("-", " ").replace("_", " ").title()
        content = md_file.read_text(encoding="utf-8")
        chunks = _split_into_chunks(content)

        print(f"Ingesting '{md_file.name}' → {len(chunks)} chunks …", end=" ", flush=True)

        # Delete existing chunks for this doc (idempotent re-ingest)
        db.execute("DELETE FROM docs WHERE doc_id = ?", [doc_id])

        for chunk in chunks:
            embedding = await embed_text(chunk, model=embedding_model)
            chunk_uid = f"{doc_id}-{uuid.uuid4().hex[:8]}"
            db.execute(
                "INSERT INTO docs (doc_id, title, chunk_text, embedding) VALUES (?, ?, ?, ?)",
                [chunk_uid, title, chunk, embedding],
            )
            total_chunks += 1

        print("✓")

    print(f"\nDone. Total chunks ingested: {total_chunks}")
    db.close()
    return total_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest markdown docs into the vector store.")
    parser.add_argument("--path", required=True, help="Directory containing .md files")
    parser.add_argument("--clear", action="store_true", help="Clear existing docs before ingesting")
    args = parser.parse_args()
    asyncio.run(ingest_markdown_dir(args.path, clear=args.clear))


if __name__ == "__main__":
    main()
