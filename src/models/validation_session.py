"""Pydantic models for YAML validation session state."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class FixRecord(BaseModel):
    field_path: str
    original_value: str
    fixed_value: str
    reason: str


class AmbiguityQuestion(BaseModel):
    field_path: str
    question: str
    options: list[str]  # 2-4 short strings shown as clickable buttons
    context: str


class ValidatorV2Output(BaseModel):
    message: str                          # human-readable explanation for the user
    yaml_current: str                     # complete YAML after all fixes in this turn
    fixes_applied: list[FixRecord]        # fixes applied IN THIS turn only
    pending_question: AmbiguityQuestion | None  # next clarification (None = none left)
    is_done: bool                         # True when all ambiguities resolved


class ValidationSession(BaseModel):
    session_id: str
    filename: str
    yaml_original: str
    yaml_current: str
    fixes_applied: list[FixRecord]
    message_history: str                  # JSON-encoded list[ModelMessage]
    status: Literal["in_progress", "awaiting_input", "done", "error"]
