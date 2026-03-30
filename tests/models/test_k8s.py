"""Unit tests for Kubernetes Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models.k8s import K8sDeployment, K8sService, K8sConfigMap, K8sSecret


# ── K8sDeployment ─────────────────────────────────────────────────────────────

VALID_DEPLOYMENT: dict = {
    "apiVersion": "apps/v1",
    "kind": "Deployment",
    "metadata": {"name": "api-gateway", "namespace": "production"},
    "spec": {
        "replicas": 3,
        "selector": {"matchLabels": {"app": "api-gateway"}},
        "template": {
            "metadata": {"name": "api-gateway", "labels": {"app": "api-gateway"}},
            "spec": {
                "containers": [
                    {
                        "name": "api",
                        "image": "myrepo/api:v1.2.3",
                        "ports": [{"containerPort": 8080}],
                    }
                ]
            },
        },
    },
}


def test_valid_deployment() -> None:
    dep = K8sDeployment.model_validate(VALID_DEPLOYMENT)
    assert dep.kind == "Deployment"
    assert dep.spec.replicas == 3


@pytest.mark.parametrize(
    "override,expected_field",
    [
        ({"apiVersion": "v1"}, "apiVersion"),            # wrong apiVersion
        ({"kind": "StatefulSet"}, "kind"),               # wrong kind
        ({"metadata": {}}, "metadata"),                  # missing name
    ],
)
def test_invalid_deployment(override: dict, expected_field: str) -> None:
    data = {**VALID_DEPLOYMENT, **override}
    with pytest.raises(ValidationError) as exc_info:
        K8sDeployment.model_validate(data)
    assert expected_field in str(exc_info.value)


def test_deployment_replicas_bounds() -> None:
    # replicas must be >= 1
    data = {**VALID_DEPLOYMENT}
    data["spec"] = {**VALID_DEPLOYMENT["spec"], "replicas": 0}
    with pytest.raises(ValidationError):
        K8sDeployment.model_validate(data)

    # replicas must be <= 100
    data["spec"] = {**VALID_DEPLOYMENT["spec"], "replicas": 101}
    with pytest.raises(ValidationError):
        K8sDeployment.model_validate(data)


def test_deployment_extra_fields_forbidden() -> None:
    data = {**VALID_DEPLOYMENT, "unknownField": "value"}
    with pytest.raises(ValidationError):
        K8sDeployment.model_validate(data)


# ── K8sService ────────────────────────────────────────────────────────────────

VALID_SERVICE: dict = {
    "apiVersion": "v1",
    "kind": "Service",
    "metadata": {"name": "api-gateway-svc"},
    "spec": {
        "selector": {"app": "api-gateway"},
        "ports": [{"port": 80, "targetPort": 8080}],
        "type": "ClusterIP",
    },
}


def test_valid_service() -> None:
    svc = K8sService.model_validate(VALID_SERVICE)
    assert svc.spec.type == "ClusterIP"


def test_invalid_service_type() -> None:
    data = {**VALID_SERVICE, "spec": {**VALID_SERVICE["spec"], "type": "InvalidType"}}
    with pytest.raises(ValidationError):
        K8sService.model_validate(data)


# ── K8sConfigMap ──────────────────────────────────────────────────────────────

def test_valid_configmap() -> None:
    cm = K8sConfigMap.model_validate({
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": "app-config"},
        "data": {"LOG_LEVEL": "info"},
    })
    assert cm.data["LOG_LEVEL"] == "info"


# ── K8sSecret ─────────────────────────────────────────────────────────────────

def test_valid_secret() -> None:
    secret = K8sSecret.model_validate({
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": "db-creds"},
        "type": "Opaque",
        "data": {"password": "c2VjcmV0"},
    })
    assert secret.type == "Opaque"
