"""Unit tests for the Config Validator agent tools (no LLM, no network)."""

from __future__ import annotations

import pytest

from src.agents.config_validator import (
    ValidationResult,
    validate_gitlab_ci,
    validate_k8s_manifest,
)
from src.deps.connections import ValidatorDeps


# We test the tool functions directly (without going through the agent LLM).

@pytest.fixture
def deps() -> ValidatorDeps:
    return ValidatorDeps()


# ── K8s validation ────────────────────────────────────────────────────────────

VALID_DEPLOYMENT_YAML = """
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-gateway
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api-gateway
  template:
    metadata:
      name: api-gateway
      labels:
        app: api-gateway
    spec:
      containers:
        - name: api
          image: myrepo/api:v1.2.3
          resources:
            limits:
              memory: "128Mi"
              cpu: "500m"
"""

INVALID_DEPLOYMENT_YAML = """
apiVersion: v1
kind: Deployment
metadata:
  name: bad-deploy
spec:
  replicas: 3
  selector:
    matchLabels:
      app: bad
  template:
    metadata:
      name: bad
    spec:
      containers:
        - name: c
          image: img:latest
"""


class _FakeCtx:
    """Minimal RunContext-like object for direct tool testing."""
    def __init__(self, deps_obj: ValidatorDeps) -> None:
        self.deps = deps_obj


async def test_validate_k8s_valid_deployment(deps: ValidatorDeps) -> None:
    ctx = _FakeCtx(deps)
    raw = await validate_k8s_manifest(ctx, VALID_DEPLOYMENT_YAML)  # type: ignore[arg-type]
    result = ValidationResult.model_validate_json(raw)
    assert result.valid is True
    assert result.errors == []


async def test_validate_k8s_invalid_deployment(deps: ValidatorDeps) -> None:
    ctx = _FakeCtx(deps)
    raw = await validate_k8s_manifest(ctx, INVALID_DEPLOYMENT_YAML)  # type: ignore[arg-type]
    result = ValidationResult.model_validate_json(raw)
    assert result.valid is False
    assert len(result.errors) > 0


async def test_validate_k8s_unsupported_kind(deps: ValidatorDeps) -> None:
    yaml_content = "apiVersion: apps/v1\nkind: StatefulSet\nmetadata:\n  name: db\nspec: {}"
    ctx = _FakeCtx(deps)
    raw = await validate_k8s_manifest(ctx, yaml_content)  # type: ignore[arg-type]
    result = ValidationResult.model_validate_json(raw)
    assert result.valid is False
    assert "Unsupported kind" in result.errors[0]


async def test_validate_k8s_yaml_parse_error(deps: ValidatorDeps) -> None:
    ctx = _FakeCtx(deps)
    raw = await validate_k8s_manifest(ctx, ":::invalid yaml:::")  # type: ignore[arg-type]
    result = ValidationResult.model_validate_json(raw)
    assert result.valid is False


async def test_validate_k8s_warning_single_replica(deps: ValidatorDeps) -> None:
    yaml_single = VALID_DEPLOYMENT_YAML.replace("replicas: 3", "replicas: 1")
    ctx = _FakeCtx(deps)
    raw = await validate_k8s_manifest(ctx, yaml_single)  # type: ignore[arg-type]
    result = ValidationResult.model_validate_json(raw)
    assert result.valid is True
    assert any("replicas=1" in w for w in result.warnings)


# ── GitLab CI validation ──────────────────────────────────────────────────────

VALID_CI_YAML = """
stages:
  - build
  - test

build-job:
  stage: build
  script:
    - docker build -t myimage .

test-job:
  stage: test
  script:
    - uv run -m pytest
"""

INVALID_CI_YAML = """
stages:
  - test

bad-job:
  stage: test
  retry: 5
  script:
    - echo hi
"""


async def test_validate_gitlab_valid(deps: ValidatorDeps) -> None:
    ctx = _FakeCtx(deps)
    raw = await validate_gitlab_ci(ctx, VALID_CI_YAML)  # type: ignore[arg-type]
    result = ValidationResult.model_validate_json(raw)
    assert result.valid is True
    assert "2 job(s)" in result.summary


async def test_validate_gitlab_invalid_retry(deps: ValidatorDeps) -> None:
    ctx = _FakeCtx(deps)
    raw = await validate_gitlab_ci(ctx, INVALID_CI_YAML)  # type: ignore[arg-type]
    result = ValidationResult.model_validate_json(raw)
    assert result.valid is False


async def test_validate_gitlab_empty(deps: ValidatorDeps) -> None:
    ctx = _FakeCtx(deps)
    raw = await validate_gitlab_ci(ctx, "stages: []")  # type: ignore[arg-type]
    result = ValidationResult.model_validate_json(raw)
    assert result.valid is True
    assert any("No jobs" in w for w in result.warnings)
