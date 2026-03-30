"""Pydantic V2 strict models for Kubernetes manifests."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


# ── Metadata ──────────────────────────────────────────────────────────────────


class K8sMetadata(_StrictModel):
    name: str
    namespace: str = "default"
    labels: dict[str, str] = Field(default_factory=dict)
    annotations: dict[str, str] = Field(default_factory=dict)


# ── Deployment ────────────────────────────────────────────────────────────────


class K8sResourceRequirements(_StrictModel):
    requests: dict[str, str] = Field(default_factory=dict)
    limits: dict[str, str] = Field(default_factory=dict)


class K8sContainer(_StrictModel):
    name: str
    image: str
    command: list[str] = Field(default_factory=list)
    args: list[str] = Field(default_factory=list)
    env: list[dict[str, Any]] = Field(default_factory=list)
    ports: list[dict[str, Any]] = Field(default_factory=list)
    resources: K8sResourceRequirements = Field(default_factory=K8sResourceRequirements)


class K8sPodSpec(_StrictModel):
    containers: list[K8sContainer]
    initContainers: list[K8sContainer] = Field(default_factory=list)  # noqa: N815
    volumes: list[dict[str, Any]] = Field(default_factory=list)
    serviceAccountName: str = "default"  # noqa: N815
    restartPolicy: Literal["Always", "OnFailure", "Never"] = "Always"  # noqa: N815


class K8sPodTemplate(_StrictModel):
    metadata: K8sMetadata
    spec: K8sPodSpec


class K8sLabelSelector(_StrictModel):
    matchLabels: dict[str, str] = Field(default_factory=dict)  # noqa: N815


class K8sDeploymentSpec(_StrictModel):
    replicas: int = Field(default=1, ge=1, le=100)
    selector: K8sLabelSelector
    template: K8sPodTemplate


class K8sDeployment(_StrictModel):
    apiVersion: Literal["apps/v1"]  # noqa: N815
    kind: Literal["Deployment"]
    metadata: K8sMetadata
    spec: K8sDeploymentSpec


# ── Service ───────────────────────────────────────────────────────────────────


class K8sServicePort(_StrictModel):
    port: int = Field(ge=1, le=65535)
    targetPort: int | str = Field(ge=1)  # noqa: N815
    protocol: Literal["TCP", "UDP", "SCTP"] = "TCP"
    name: str = ""


class K8sServiceSpec(_StrictModel):
    selector: dict[str, str] = Field(default_factory=dict)
    ports: list[K8sServicePort]
    type: Literal["ClusterIP", "NodePort", "LoadBalancer", "ExternalName"] = "ClusterIP"


class K8sService(_StrictModel):
    apiVersion: Literal["v1"]  # noqa: N815
    kind: Literal["Service"]
    metadata: K8sMetadata
    spec: K8sServiceSpec


# ── ConfigMap ─────────────────────────────────────────────────────────────────


class K8sConfigMap(_StrictModel):
    apiVersion: Literal["v1"]  # noqa: N815
    kind: Literal["ConfigMap"]
    metadata: K8sMetadata
    data: dict[str, str] = Field(default_factory=dict)
    binaryData: dict[str, str] = Field(default_factory=dict)  # noqa: N815


# ── Secret ────────────────────────────────────────────────────────────────────


class K8sSecret(_StrictModel):
    apiVersion: Literal["v1"]  # noqa: N815
    kind: Literal["Secret"]
    metadata: K8sMetadata
    type: str = "Opaque"
    data: dict[str, str] = Field(default_factory=dict)
    stringData: dict[str, str] = Field(default_factory=dict)  # noqa: N815
