"""Unit tests for GitLab CI Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models.gitlab import GitLabCI, GitLabJob


VALID_PIPELINE: dict = {
    "stages": ["build", "test", "deploy"],
    "variables": {"DOCKER_DRIVER": "overlay2"},
    "build-image": {
        "stage": "build",
        "image": "docker:24",
        "script": ["docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA ."],
        "tags": ["docker"],
    },
    "unit-tests": {
        "stage": "test",
        "image": "python:3.13-slim",
        "script": ["uv run -m pytest tests/ -v"],
        "artifacts": {
            "reports": {"junit": "report.xml"},
            "expire_in": "1 week",
        },
    },
    "deploy-staging": {
        "stage": "deploy",
        "script": ["helm upgrade --install app ./chart"],
        "rules": [{"if": "$CI_COMMIT_BRANCH == 'main'", "when": "manual"}],
    },
}


def test_valid_pipeline_parses_jobs() -> None:
    ci = GitLabCI.model_validate(VALID_PIPELINE)
    assert "build-image" in ci.jobs
    assert "unit-tests" in ci.jobs
    assert "deploy-staging" in ci.jobs
    assert ci.stages == ["build", "test", "deploy"]


def test_pipeline_job_script_required() -> None:
    data = {
        "stages": ["test"],
        "bad-job": {"stage": "test"},  # missing script
    }
    # A dict without "script" is not treated as a job — it's silently skipped
    ci = GitLabCI.model_validate(data)
    assert "bad-job" not in ci.jobs


def test_pipeline_reserved_keywords_not_jobs() -> None:
    ci = GitLabCI.model_validate(VALID_PIPELINE)
    assert "stages" not in ci.jobs
    assert "variables" not in ci.jobs


def test_job_retry_bounds() -> None:
    with pytest.raises(ValidationError):
        GitLabJob.model_validate({"script": ["echo hi"], "retry": 3})


def test_job_valid_retry() -> None:
    job = GitLabJob.model_validate({"script": ["echo hi"], "retry": 2})
    assert job.retry == 2


def test_job_rule_with_if() -> None:
    ci = GitLabCI.model_validate(VALID_PIPELINE)
    deploy_job = ci.jobs["deploy-staging"]
    assert deploy_job.rules[0].if_ == "$CI_COMMIT_BRANCH == 'main'"


def test_empty_pipeline() -> None:
    ci = GitLabCI.model_validate({"stages": []})
    assert ci.jobs == {}
