"""Config Validator agent — validates YAML configs via JSON Schema (dynamic) or Pydantic V2."""

from __future__ import annotations

import fnmatch
import json
import logging

import httpx
import jsonschema
import yaml
from pydantic import BaseModel, ValidationError
from pydantic_ai import Agent, RunContext

from src.deps.connections import ValidatorDeps, build_model
from src.models.gitlab import GitLabCI
from src.models.k8s import K8sConfigMap, K8sDeployment, K8sSecret, K8sService

logger = logging.getLogger(__name__)

# ── Output model ──────────────────────────────────────────────────────────────


class ValidationResult(BaseModel):
    valid: bool
    errors: list[str]
    warnings: list[str]
    summary: str


# ── Agent ─────────────────────────────────────────────────────────────────────

config_validator_agent: Agent[ValidatorDeps, str] = Agent(
    build_model(),
    deps_type=ValidatorDeps,
    output_type=str,
    system_prompt=(
        "You are a DevOps/SRE configuration expert. "
        "When the user provides YAML content and a filename, call exactly ONE tool:\n\n"
        "1. Kubernetes manifest (has 'apiVersion' AND 'kind') → validate_k8s_manifest(yaml_content).\n"
        "2. GitLab CI pipeline (has 'stages' key or jobs with 'script', no 'apiVersion') → validate_gitlab_ci(yaml_content).\n"
        "3. OpenAPI/Swagger spec (has 'openapi:' or 'swagger:' key) → validate_openapi_spec(yaml_content).\n"
        "4. Everything else (GitHub Actions, Docker Compose, Helm, CircleCI, Azure Pipelines, etc.) "
        "→ validate_generic_yaml(filename, yaml_content). This tool auto-fetches the schema and validates.\n\n"
        "After the tool returns, explain the result clearly: list every error with its field path, "
        "list warnings, and summarise whether the file is valid. "
        "Never skip calling a tool."
    ),
)


# ── Shared helpers ─────────────────────────────────────────────────────────────


def _parse_yaml(yaml_content: str) -> tuple[dict | None, str]:
    """Parse YAML string → (data, error_message)."""
    try:
        data = yaml.safe_load(yaml_content)
        if not isinstance(data, dict):
            return None, "YAML must be a mapping (dict), got: " + type(data).__name__
        return data, ""
    except yaml.YAMLError as exc:
        return None, f"YAML parse error: {exc}"


def _collect_warnings(data: dict) -> list[str]:
    """Heuristic warnings for valid-but-risky Kubernetes configs."""
    warnings: list[str] = []
    spec = data.get("spec", {})
    if data.get("kind") == "Deployment":
        if spec.get("replicas", 1) == 1:
            ns = data.get("metadata", {}).get("namespace", "default")
            warnings.append(f"replicas=1 in namespace '{ns}' — consider >=2 for high availability")
        for c in spec.get("template", {}).get("spec", {}).get("containers", []):
            if not c.get("resources", {}).get("limits"):
                warnings.append(
                    f"Container '{c.get('name', '?')}' has no resource limits — "
                    "may cause OOM kills or noisy-neighbour issues"
                )
    return warnings


# ── Tools — Kubernetes + GitLab (existing, Pydantic V2) ───────────────────────


@config_validator_agent.tool
async def validate_k8s_manifest(ctx: RunContext[ValidatorDeps], yaml_content: str) -> str:
    """Validate a Kubernetes manifest (Deployment, Service, ConfigMap, Secret) via Pydantic V2.

    Args:
        yaml_content: Raw YAML string of the Kubernetes manifest.

    Returns:
        JSON-encoded ValidationResult.
    """
    data, parse_error = _parse_yaml(yaml_content)
    if data is None:
        return ValidationResult(valid=False, errors=[parse_error], warnings=[], summary=parse_error).model_dump_json()

    kind = data.get("kind", "")
    model_map: dict[str, type[BaseModel]] = {
        "Deployment": K8sDeployment,
        "Service": K8sService,
        "ConfigMap": K8sConfigMap,
        "Secret": K8sSecret,
    }

    if kind not in model_map:
        msg = f"Unsupported kind '{kind}'. Supported: {', '.join(model_map)}"
        return ValidationResult(valid=False, errors=[msg], warnings=[], summary=msg).model_dump_json()

    try:
        model_map[kind].model_validate(data)
        warnings = _collect_warnings(data)
        return ValidationResult(
            valid=True,
            errors=[],
            warnings=warnings,
            summary=f"Valid {kind}" + (f" ({len(warnings)} warnings)" if warnings else ""),
        ).model_dump_json()
    except ValidationError as exc:
        errors = [f"{e['loc']}: {e['msg']}" for e in exc.errors()]
        return ValidationResult(
            valid=False, errors=errors, warnings=[], summary=f"Invalid {kind}: {len(errors)} error(s)"
        ).model_dump_json()


@config_validator_agent.tool
async def validate_gitlab_ci(ctx: RunContext[ValidatorDeps], yaml_content: str) -> str:
    """Validate a GitLab CI pipeline YAML (.gitlab-ci.yml) via Pydantic V2.

    Args:
        yaml_content: Raw YAML string of the GitLab CI configuration.

    Returns:
        JSON-encoded ValidationResult.
    """
    data, parse_error = _parse_yaml(yaml_content)
    if data is None:
        return ValidationResult(valid=False, errors=[parse_error], warnings=[], summary=parse_error).model_dump_json()

    try:
        ci = GitLabCI.model_validate(data)
        warnings: list[str] = []
        if not ci.stages:
            warnings.append("No 'stages' defined — GitLab will use default ordering")
        if not ci.jobs:
            warnings.append("No jobs detected in the pipeline")
        return ValidationResult(
            valid=True,
            errors=[],
            warnings=warnings,
            summary=f"Valid GitLab CI with {len(ci.jobs)} job(s)" + (f" ({len(warnings)} warnings)" if warnings else ""),
        ).model_dump_json()
    except ValidationError as exc:
        errors = [f"{e['loc']}: {e['msg']}" for e in exc.errors()]
        return ValidationResult(
            valid=False, errors=errors, warnings=[], summary=f"Invalid GitLab CI: {len(errors)} error(s)"
        ).model_dump_json()


# ── Tools — Dynamic schema validation ─────────────────────────────────────────

# Cache the SchemaStore catalog in memory for the container lifetime.
_schemastore_catalog: list[dict] | None = None


@config_validator_agent.tool
async def fetch_schema_for_file(ctx: RunContext[ValidatorDeps], filename: str) -> str:
    """Fetch the JSON Schema for a config file from SchemaStore.org using the filename.

    Matches the filename against SchemaStore glob patterns (fileMatch) and returns
    the JSON Schema as a string. Use this before calling validate_with_json_schema
    for formats like GitHub Actions, Docker Compose, Helm, CircleCI, Azure Pipelines, etc.

    Args:
        filename: The filename of the config file (e.g. 'docker-compose.yml',
                  '.github/workflows/ci.yml', 'Chart.yaml').

    Returns:
        The JSON Schema as a JSON string, or an error message if not found.
    """
    global _schemastore_catalog

    basename = filename.split("/")[-1]

    if _schemastore_catalog is None:
        logger.debug("fetch_schema_for_file: fetching SchemaStore catalog")
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get("https://www.schemastore.org/api/json/catalog.json")
                resp.raise_for_status()
                _schemastore_catalog = resp.json().get("schemas", [])
            logger.debug("fetch_schema_for_file: catalog loaded (%d schemas)", len(_schemastore_catalog))
        except Exception as exc:
            return f"Could not fetch SchemaStore catalog: {exc}"

    matched_url: str | None = None
    matched_title: str = ""
    for entry in _schemastore_catalog:
        for pattern in entry.get("fileMatch", []):
            # Strip **/  prefix — fnmatch doesn't support recursive glob.
            # '**/docker-compose.yml' → 'docker-compose.yml' for basename match.
            bare = pattern.lstrip("*").lstrip("/")
            if (
                fnmatch.fnmatch(filename, pattern)
                or fnmatch.fnmatch(basename, pattern)
                or fnmatch.fnmatch(basename, bare)
            ):
                matched_url = entry["url"]
                matched_title = entry.get("name", entry.get("title", filename))
                break
        if matched_url:
            break

    if not matched_url:
        return f"No schema found in SchemaStore for filename '{filename}'. Cannot auto-validate."

    logger.debug("fetch_schema_for_file: matched '%s' → %s", filename, matched_url)
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(matched_url)
            resp.raise_for_status()
            schema_text = resp.text
        return f"SCHEMA_TITLE:{matched_title}\n{schema_text}"
    except Exception as exc:
        return f"Could not fetch schema from {matched_url}: {exc}"


@config_validator_agent.tool
async def validate_with_json_schema(
    ctx: RunContext[ValidatorDeps],
    yaml_content: str,
    schema_json: str,
) -> str:
    """Validate YAML content against a JSON Schema (Draft 4/6/7/2019-09/2020-12).

    Use after fetch_schema_for_file returns a schema. Pass the full schema JSON string.
    Strip the 'SCHEMA_TITLE:...' prefix line if present before passing schema_json.

    Args:
        yaml_content: Raw YAML string to validate.
        schema_json: The JSON Schema as a JSON string.

    Returns:
        JSON-encoded ValidationResult.
    """
    data, parse_error = _parse_yaml(yaml_content)
    if data is None:
        return ValidationResult(valid=False, errors=[parse_error], warnings=[], summary=parse_error).model_dump_json()

    # Strip optional SCHEMA_TITLE prefix
    if schema_json.startswith("SCHEMA_TITLE:"):
        schema_json = "\n".join(schema_json.splitlines()[1:])

    try:
        schema = json.loads(schema_json)
    except json.JSONDecodeError as exc:
        return ValidationResult(
            valid=False, errors=[f"Schema JSON parse error: {exc}"], warnings=[], summary="Schema parse error"
        ).model_dump_json()

    try:
        validator_cls = jsonschema.validators.validator_for(schema)
        validator_cls.check_schema(schema)
        validator = validator_cls(schema)
        errors = sorted(validator.iter_errors(data), key=lambda e: list(e.absolute_path))
    except Exception as exc:
        return ValidationResult(
            valid=False, errors=[f"Validation error: {exc}"], warnings=[], summary="Validation failed"
        ).model_dump_json()

    if not errors:
        warnings = _collect_warnings(data)
        return ValidationResult(
            valid=True,
            errors=[],
            warnings=warnings,
            summary="Valid — 0 schema errors" + (f", {len(warnings)} warnings" if warnings else ""),
        ).model_dump_json()

    error_msgs = []
    for e in errors[:20]:
        path = " > ".join(str(p) for p in e.absolute_path) or "root"
        error_msgs.append(f"{path}: {e.message}")

    logger.debug("validate_with_json_schema: %d error(s)", len(errors))
    return ValidationResult(
        valid=False,
        errors=error_msgs,
        warnings=[],
        summary=f"Invalid — {len(errors)} schema error(s)" + (" (showing first 20)" if len(errors) > 20 else ""),
    ).model_dump_json()


@config_validator_agent.tool
async def validate_generic_yaml(
    ctx: RunContext[ValidatorDeps],
    filename: str,
    yaml_content: str,
) -> str:
    """Validate any YAML config file by auto-fetching its JSON Schema from SchemaStore.org.

    Use for: GitHub Actions, Docker Compose, Helm Chart.yaml, CircleCI, Azure Pipelines,
    Renovate, Dependabot, pre-commit, and any other format not handled by the other tools.

    This tool combines schema lookup + validation in a single call.

    Args:
        filename: The filename (e.g. 'docker-compose.yml', 'chart.yaml').
        yaml_content: Raw YAML string to validate.

    Returns:
        JSON-encoded ValidationResult with errors from JSON Schema validation,
        or a note that no schema was found if SchemaStore has no match.
    """
    global _schemastore_catalog

    data, parse_error = _parse_yaml(yaml_content)
    if data is None:
        return ValidationResult(valid=False, errors=[parse_error], warnings=[], summary=parse_error).model_dump_json()

    basename = filename.split("/")[-1]

    # ── Step 1: Fetch and cache SchemaStore catalog ───────────────────────────
    if _schemastore_catalog is None:
        logger.debug("validate_generic_yaml: fetching SchemaStore catalog")
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get("https://www.schemastore.org/api/json/catalog.json")
                resp.raise_for_status()
                _schemastore_catalog = resp.json().get("schemas", [])
            logger.debug("validate_generic_yaml: catalog loaded (%d schemas)", len(_schemastore_catalog))
        except Exception as exc:
            return ValidationResult(
                valid=False, errors=[f"Could not fetch SchemaStore catalog: {exc}"], warnings=[],
                summary="Schema lookup failed"
            ).model_dump_json()

    # ── Step 2: Match filename against catalog ────────────────────────────────
    matched_url: str | None = None
    matched_title: str = ""
    for entry in _schemastore_catalog:
        for pattern in entry.get("fileMatch", []):
            bare = pattern.lstrip("*").lstrip("/")
            if (
                fnmatch.fnmatch(filename, pattern)
                or fnmatch.fnmatch(basename, pattern)
                or fnmatch.fnmatch(basename, bare)
            ):
                matched_url = entry["url"]
                matched_title = entry.get("name", entry.get("title", filename))
                break
        if matched_url:
            break

    if not matched_url:
        logger.debug("validate_generic_yaml: no schema found for '%s' — semantic only", filename)
        return ValidationResult(
            valid=None,  # type: ignore[arg-type]
            errors=[],
            warnings=[f"No JSON Schema found in SchemaStore for '{filename}' — structural validation skipped."],
            summary=f"No schema available for '{filename}'. Semantic review only.",
        ).model_dump_json()

    # ── Step 3: Fetch schema ──────────────────────────────────────────────────
    logger.debug("validate_generic_yaml: matched '%s' → %s", filename, matched_url)
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            resp = await client.get(matched_url)
            resp.raise_for_status()
            schema = json.loads(resp.text)
    except Exception as exc:
        return ValidationResult(
            valid=False, errors=[f"Could not fetch schema: {exc}"], warnings=[],
            summary="Schema fetch failed"
        ).model_dump_json()

    # ── Step 4: Validate ──────────────────────────────────────────────────────
    try:
        validator_cls = jsonschema.validators.validator_for(schema)
        validator = validator_cls(schema)
        errors = sorted(validator.iter_errors(data), key=lambda e: list(e.absolute_path))
    except Exception as exc:
        return ValidationResult(
            valid=False, errors=[f"Validation engine error: {exc}"], warnings=[],
            summary="Validation failed"
        ).model_dump_json()

    if not errors:
        return ValidationResult(
            valid=True, errors=[], warnings=[],
            summary=f"Valid {matched_title} — 0 schema errors",
        ).model_dump_json()

    error_msgs = [
        f"{' > '.join(str(p) for p in e.absolute_path) or 'root'}: {e.message}"
        for e in errors[:20]
    ]
    logger.debug("validate_generic_yaml: %d error(s) for '%s'", len(errors), filename)
    return ValidationResult(
        valid=False,
        errors=error_msgs,
        warnings=[],
        summary=f"Invalid {matched_title} — {len(errors)} error(s)" + (" (showing first 20)" if len(errors) > 20 else ""),
    ).model_dump_json()


@config_validator_agent.tool
async def validate_openapi_spec(ctx: RunContext[ValidatorDeps], yaml_content: str) -> str:
    """Validate an OpenAPI 3.0 or 3.1 specification document.

    Args:
        yaml_content: Raw YAML (or JSON) string of the OpenAPI spec.

    Returns:
        JSON-encoded ValidationResult.
    """
    import yaml as _yaml
    from openapi_spec_validator import validate as _oas_validate

    try:
        spec = _yaml.safe_load(yaml_content)
        if not isinstance(spec, dict):
            raise ValueError("OpenAPI spec must be a YAML/JSON mapping")
        _oas_validate(spec)
        return ValidationResult(
            valid=True, errors=[], warnings=[], summary="Valid OpenAPI specification"
        ).model_dump_json()
    except Exception as exc:
        msg = str(exc)
        return ValidationResult(
            valid=False, errors=[msg], warnings=[], summary="Invalid OpenAPI spec"
        ).model_dump_json()
