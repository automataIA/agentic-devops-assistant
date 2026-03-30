"""Ollama native structured output — grammar-constrained JSON via /api/chat.

WHY THIS MODULE EXISTS
----------------------
PydanticAI's OllamaProvider wraps AsyncOpenAI and uses OpenAI's
``response_format`` parameter for structured output. Ollama's /v1/chat/completions
endpoint accepts this parameter but implements it as a soft hint — 7-8B models
(qwen3, mistral-nemo) frequently produce malformed JSON after multi-step tool
calls, causing PydanticAI to raise ``UnexpectedModelBehavior`` ("Exceeded maximum
retries for output validation").

Ollama's native /api/chat endpoint supports ``format: {json_schema}`` (since
v0.5, December 2024), which uses grammar-based *constrained sampling*: the model's
token sampling is restricted at the decode level so the output is structurally
guaranteed to match the schema. This is a hard guarantee, not a soft hint.

This module provides a single async function that calls the native Ollama API
and returns a validated Pydantic model instance.
"""
from __future__ import annotations

import os
from typing import Type, TypeVar

import httpx
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def _ollama_native_url() -> str:
    """Return the Ollama native API base URL, stripping the /v1 suffix.

    OLLAMA_BASE_URL is documented as requiring /v1 for OllamaProvider/AsyncOpenAI.
    The native /api/chat endpoint lives at the root, so we strip it here.
    """
    url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1").rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    return url


async def ollama_structured_finish(
    messages: list[dict[str, str]],
    output_type: Type[T],
    *,
    model: str | None = None,
    temperature: float = 0.0,
) -> T:
    """Call Ollama /api/chat with format=schema for grammar-constrained output.

    The model is forced at the decode level to produce JSON that matches
    ``output_type``'s JSON Schema. Works reliably with qwen3 and mistral-nemo
    even after complex tool-call histories, because constrained sampling bypasses
    the model's tendency to add prose around JSON.

    Args:
        messages: Conversation as ``[{"role": "...", "content": "..."}]`` dicts.
        output_type: Pydantic model class to extract and validate.
        model: Override model name (defaults to ``AGENT_MODEL`` env var).
        temperature: Sampling temperature. 0 = fully deterministic (recommended
            for extraction tasks where creativity is undesirable).

    Returns:
        Validated instance of ``output_type``.

    Raises:
        httpx.HTTPStatusError: Ollama returned a non-2xx HTTP response.
        pydantic.ValidationError: Response content doesn't match ``output_type``
            (should not happen with constrained sampling, but guard exists).
    """
    base = _ollama_native_url()
    model_name = model or os.getenv("AGENT_MODEL", "mistral-nemo")

    payload: dict = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "format": output_type.model_json_schema(),
        "options": {"temperature": temperature},
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{base}/api/chat", json=payload)
        resp.raise_for_status()

    content: str = resp.json()["message"]["content"]
    return output_type.model_validate_json(content)
