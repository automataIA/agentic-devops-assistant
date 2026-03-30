"""Conversation and message models for persistent chat history."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class Conversation(BaseModel):
    """A chat conversation belonging to a user."""

    conversation_id: str
    user_id: str
    title: str = "New Chat"
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class Message(BaseModel):
    """A single message in a conversation."""

    message_id: str
    conversation_id: str
    role: Literal["user", "assistant"]
    content: str
    agent: str | None = None  # which specialist answered (Log Explorer, Docu-RAG, etc.)
    sources: list[str] = Field(default_factory=list)
    chunk_sources: list[dict] = Field(default_factory=list)
    is_error: bool = False
    created_at: datetime = Field(default_factory=datetime.now)


class Feedback(BaseModel):
    """User feedback on an assistant message."""

    message_id: str
    user_id: str
    rating: Literal["up", "down"]
    comment: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)
