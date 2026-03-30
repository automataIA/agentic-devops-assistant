"""Pydantic V2 strict models for GitLab CI configuration."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


# ── Job-level models ──────────────────────────────────────────────────────────


class GitLabArtifacts(_StrictModel):
    paths: list[str] = Field(default_factory=list)
    reports: dict[str, Any] = Field(default_factory=dict)
    expire_in: str = ""
    when: str = "on_success"


class GitLabCache(_StrictModel):
    key: str = ""
    paths: list[str] = Field(default_factory=list)
    policy: str = "pull-push"


class GitLabRule(_StrictModel):
    if_: str | None = Field(default=None, alias="if")
    when: str = "on_success"
    allow_failure: bool = False

    model_config = ConfigDict(strict=True, extra="forbid", populate_by_name=True)


class GitLabJob(_StrictModel):
    stage: str = "test"
    image: str = ""
    script: list[str]
    before_script: list[str] = Field(default_factory=list)
    after_script: list[str] = Field(default_factory=list)
    variables: dict[str, str] = Field(default_factory=dict)
    rules: list[GitLabRule] = Field(default_factory=list)
    artifacts: GitLabArtifacts | None = None
    cache: GitLabCache | None = None
    tags: list[str] = Field(default_factory=list)
    allow_failure: bool = False
    needs: list[str] = Field(default_factory=list)
    environment: str | dict[str, Any] = ""
    timeout: str = ""
    retry: int = Field(default=0, ge=0, le=2)


# ── Top-level pipeline model ──────────────────────────────────────────────────

# GitLab CI top-level reserved keywords (not job names)
_RESERVED_KEYWORDS: frozenset[str] = frozenset(
    {
        "stages",
        "variables",
        "cache",
        "default",
        "workflow",
        "include",
        "image",
        "services",
        "before_script",
        "after_script",
    }
)


class GitLabCI(BaseModel):
    """Top-level GitLab CI pipeline configuration.

    Uses a permissive config to allow arbitrary job names as top-level keys,
    then validates them in the model_validator.
    """

    model_config = ConfigDict(strict=False, extra="allow")

    stages: list[str] = Field(default_factory=list)
    variables: dict[str, str] = Field(default_factory=dict)
    jobs: dict[str, GitLabJob] = Field(default_factory=dict, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def extract_jobs(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Parse top-level keys that are not reserved keywords as GitLab CI jobs."""
        if not isinstance(data, dict):
            return data

        jobs: dict[str, GitLabJob] = {}
        for key, value in data.items():
            if key in _RESERVED_KEYWORDS:
                continue
            if isinstance(value, dict) and "script" in value:
                jobs[key] = GitLabJob.model_validate(value)

        data["jobs"] = jobs
        return data
