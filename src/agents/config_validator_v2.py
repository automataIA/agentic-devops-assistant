"""Config Validator v2 — multi-turn YAML validation with auto-fix and clarification."""
from __future__ import annotations

import io
import json
import os
from typing import TypedDict

import yaml
from pydantic import ValidationError
from pydantic_ai import Agent, RunContext

from src.agents.config_validator import _collect_warnings, _parse_yaml
from src.deps.connections import ValidatorDeps, build_model
from src.models.gitlab import GitLabCI
from src.models.k8s import K8sConfigMap, K8sDeployment, K8sSecret, K8sService
from src.models.validation_session import ValidatorV2Output

# When the backend is Ollama, use output_type=str for the agent: tool calls
# work reliably, but 7-8B models cannot produce valid ValidatorV2Output JSON
# after multi-step tool chains via the OpenAI-compat response_format parameter.
# The structured extraction is handled separately via ollama_structured_finish()
# (native /api/chat format=schema — grammar-constrained, guaranteed match).
# For Anthropic / OpenAI / Mistral cloud the output_type=ValidatorV2Output path
# works fine and is kept as-is.
_OLLAMA_BACKEND: bool = os.getenv("AGENT_BACKEND", "ollama").lower() == "ollama"


class _AmbiguityDict(TypedDict):
    field: str
    question: str
    options: list[str]
    context: str


_K8S_MODEL_MAP: dict[str, type] = {
    "Deployment": K8sDeployment,
    "Service": K8sService,
    "ConfigMap": K8sConfigMap,
    "Secret": K8sSecret,
}

config_validator_v2_agent: Agent[ValidatorDeps, ValidatorV2Output | str] = Agent(
    build_model(),
    deps_type=ValidatorDeps,
    output_type=str if _OLLAMA_BACKEND else ValidatorV2Output,
    system_prompt=(
        "You are a Kubernetes and GitLab CI YAML expert. "
        "Follow this exact workflow:\n\n"
        "TURN 1 — first validation:\n"
        "1. Call detect_issues() on the provided YAML.\n"
        "2. For each AUTO-FIXABLE error, call apply_fix(). Chain calls: pass the output "
        "of each apply_fix() as the yaml_content for the next.\n"
        "3. Set yaml_current to the final YAML after all auto-fixes.\n"
        "4. If suggested_ambiguities exist, set pending_question to the FIRST one and "
        "is_done=False.\n"
        "5. If no ambiguities remain, set is_done=True.\n\n"
        "SUBSEQUENT TURNS — user answered a clarification:\n"
        "1. The prompt includes the user's choice AND the current YAML.\n"
        "2. Call apply_fix() to apply the choice to yaml_current.\n"
        "3. If more ambiguities remain (from the previous detect_issues result in history), "
        "set the next one as pending_question.\n"
        "4. When all ambiguities are resolved, set is_done=True with a summary message.\n\n"
        "AUTO-FIX silently (no question):\n"
        "- Wrong apiVersion format or typo\n"
        "- Missing required metadata.name (use filename-derived default)\n"
        "- Trailing whitespace in string fields\n"
        "- Missing selector that matches template labels\n\n"
        "ALWAYS ASK (set as pending_question, NEVER auto-fix):\n"
        "- spec.replicas (affects high availability)\n"
        "- resources.limits / resources.requests (affects scheduling)\n"
        "- metadata.namespace (environment-specific)\n"
        "- image tag (version-sensitive)\n"
        "- Unsupported kind: ask to export as-is or abort\n\n"
        "OUTPUT RULES:\n"
        "- yaml_current: ALWAYS the COMPLETE YAML string, never truncated.\n"
        "- fixes_applied: ONLY fixes applied in THIS turn (not previous turns).\n"
        "- pending_question.options: 2-4 SHORT strings the user can click as buttons.\n"
        "- message: concise explanation of what was done and what is being asked.\n"
    ),
)


@config_validator_v2_agent.tool
async def detect_issues(
    ctx: RunContext[ValidatorDeps],
    yaml_content: str,
) -> str:
    """Parse and validate a YAML manifest, returning all issues and suggested ambiguities.

    Args:
        yaml_content: Raw YAML string to inspect.

    Returns:
        JSON string with: parse_error, kind, validation_errors (list[{field, message}]),
        warnings (list[str]), suggested_ambiguities (list[{field, question, options, context}]).
    """
    data, parse_error = _parse_yaml(yaml_content)
    if data is None:
        return json.dumps({
            "parse_error": parse_error,
            "kind": None,
            "validation_errors": [],
            "warnings": [],
            "suggested_ambiguities": [],
        })

    kind: str = data.get("kind", "")
    validation_errors: list[dict[str, str]] = []
    warnings: list[str] = []

    if not kind:
        # Likely GitLab CI (no 'kind' field)
        kind = "GitLabCI"
        try:
            ci = GitLabCI.model_validate(data)
            if not ci.stages:
                warnings.append("No 'stages' defined — GitLab will use default stage ordering")
            if not ci.jobs:
                warnings.append("No jobs detected in the pipeline")
        except ValidationError as exc:
            validation_errors = [{"field": str(e["loc"]), "message": e["msg"]} for e in exc.errors()]
    elif kind in _K8S_MODEL_MAP:
        try:
            _K8S_MODEL_MAP[kind].model_validate(data)
        except ValidationError as exc:
            validation_errors = [{"field": str(e["loc"]), "message": e["msg"]} for e in exc.errors()]
        warnings = _collect_warnings(data)
    else:
        validation_errors = [{
            "field": "kind",
            "message": (
                f"Unsupported kind '{kind}'. "
                f"Supported K8s: {', '.join(_K8S_MODEL_MAP)}. "
                "For GitLab CI, omit the 'kind' field."
            ),
        }]

    # Build ambiguity suggestions for Deployment manifests
    ambiguities: list[_AmbiguityDict] = []
    if kind == "Deployment":
        spec = data.get("spec", {})
        replicas = spec.get("replicas")
        if replicas is None or replicas == 1:
            ambiguities.append({
                "field": "spec.replicas",
                "question": "How many replicas do you need?",
                "options": ["1 — dev/test", "2 — production", "3 — high availability"],
                "context": f"Current value: {replicas}. Affects availability and resilience.",
            })

        namespace = data.get("metadata", {}).get("namespace")
        if not namespace:
            ambiguities.append({
                "field": "metadata.namespace",
                "question": "Which namespace should this be deployed to?",
                "options": ["default", "production", "staging", "Leave empty"],
                "context": "No namespace set in metadata. Defaults to 'default' if omitted.",
            })

        containers = spec.get("template", {}).get("spec", {}).get("containers", [])
        for container in containers:
            name = container.get("name", "?")
            if not container.get("resources", {}).get("limits"):
                ambiguities.append({
                    "field": f"spec.template.spec.containers.{name}.resources.limits",
                    "question": f"Set resource limits for container '{name}'?",
                    "options": [
                        "Skip — add manually later",
                        "Small: cpu=250m mem=256Mi",
                        "Medium: cpu=500m mem=512Mi",
                        "Large: cpu=1000m mem=1Gi",
                    ],
                    "context": (
                        f"Container '{name}' has no resource limits. "
                        "Without limits it can consume unbounded CPU/memory."
                    ),
                })

    return json.dumps({
        "parse_error": None,
        "kind": kind,
        "validation_errors": validation_errors,
        "warnings": warnings,
        "suggested_ambiguities": ambiguities,
    })


@config_validator_v2_agent.tool
async def apply_fix(
    ctx: RunContext[ValidatorDeps],
    yaml_content: str,
    field_path: str,
    new_value: str,
) -> str:
    """Apply a single field fix to a YAML string and return the updated YAML.

    Preserves original formatting, ordering, and existing comments via ruamel.yaml.

    Args:
        yaml_content: Current YAML string (full document).
        field_path: Dotted field path, e.g. "spec.replicas" or "metadata.namespace".
        new_value: New value as a YAML scalar string ("2", "true", "production", "null").

    Returns:
        Updated YAML string, or a string starting with "ERROR:" on failure.
    """
    from ruamel.yaml import YAML as RuamelYAML

    ryaml = RuamelYAML()
    ryaml.preserve_quotes = True
    try:
        doc = ryaml.load(yaml_content)
    except Exception as exc:
        return f"ERROR: cannot parse YAML: {exc}"

    if doc is None:
        return "ERROR: empty YAML document"

    parts = field_path.split(".")
    node: Any = doc  # ruamel.yaml CommentedMap has no public stubs
    for part in parts[:-1]:
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, dict):
            # Create intermediate mapping if missing
            node[part] = {}
            node = node[part]
        else:
            return f"ERROR: path segment '{part}' not found in '{field_path}'"

    key = parts[-1]
    try:
        parsed_value: Any = yaml.safe_load(new_value)
    except Exception:
        parsed_value = new_value

    if isinstance(node, dict):
        node[key] = parsed_value
    else:
        return f"ERROR: parent of '{key}' is not a mapping"

    stream = io.StringIO()
    ryaml.dump(doc, stream)
    return stream.getvalue()
