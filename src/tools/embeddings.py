"""Embedding utility — calls Ollama's native /api/embeddings endpoint."""

from __future__ import annotations

import os

import httpx


async def embed_text(
    text: str,
    model: str | None = None,
) -> list[float]:
    """Generate an embedding vector for *text* using Ollama.

    Uses Ollama's native ``/api/embeddings`` endpoint (NOT the OpenAI-compatible
    ``/v1/embeddings`` — they have different request/response schemas).

    Args:
        text: The text to embed.
        model: Ollama embedding model name. Defaults to ``EMBEDDING_MODEL`` env var
               (fallback: ``nomic-embed-text``).

    Returns:
        A list of floats representing the embedding vector.
    """
    embedding_model = model or os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    # OLLAMA_BASE_URL may or may not have a trailing /v1 — strip it either way
    ollama_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_base = ollama_base.rstrip("/").removesuffix("/v1")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{ollama_base}/api/embeddings",
            json={"model": embedding_model, "prompt": text},
        )
        response.raise_for_status()
        return response.json()["embedding"]  # type: ignore[no-any-return]
