"""User and settings models for multi-tenant support."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class User(BaseModel):
    """A registered user / tenant."""

    user_id: str
    display_name: str = "Anonymous"
    created_at: datetime = Field(default_factory=datetime.now)


class UserSettings(BaseModel):
    """Per-user preferences. Overrides can be None to fall back to global defaults."""

    user_id: str
    theme: str = "business"
    language: str = "en"
    agent_backend: str | None = None  # override AGENT_BACKEND
    agent_model: str | None = None  # override AGENT_MODEL
