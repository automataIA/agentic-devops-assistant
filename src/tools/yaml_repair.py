"""Static YAML repair pipeline.

Stages:
  1. yamlfix  — syntax normalisation (truthy strings, indentation, trailing newlines)
  2. ruamel.yaml — roundtrip parse, preserving comments and key ordering
  3. extend_with_default — jsonschema validator that fills defaults in-place
  4. jsonschema iter_errors — collect all remaining violations
  5. Categorise — type/enum/additionalProperties → deterministic; rest → ambiguous
  6. Apply deterministic fixes — type coercion, strip extra keys
  7. Re-validate — serialise back via ruamel.yaml dump (preserves comments)

The LLM is NOT called here. llm_fields in the result is an audit trail of paths
that could not be fixed deterministically; callers decide whether to involve the LLM.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

import jsonschema
import jsonschema.validators
import ruamel.yaml
import yamlfix


@dataclass
class RepairResult:
    valid: bool
    repaired_yaml: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    llm_fields: list[str] = field(default_factory=list)
    summary: str = ""


# ── Helpers ───────────────────────────────────────────────────────────────────


def _extend_with_default(
    validator_class: type,
) -> type:
    """Return a jsonschema validator subclass that fills 'default' values in-place."""
    orig_properties = validator_class.VALIDATORS.get("properties", lambda *a, **kw: iter(()))

    def set_defaults(  # type: ignore[no-untyped-def]
        validator, properties, instance, schema
    ):
        for prop, subschema in properties.items():
            if "default" in subschema and prop not in instance:
                instance.setdefault(prop, subschema["default"])
        yield from orig_properties(validator, properties, instance, schema)

    return jsonschema.validators.extend(validator_class, {"properties": set_defaults})


def _coerce_type(value: Any, target_type: str) -> Any:
    """Best-effort type coercion. Returns original value on failure."""
    try:
        if target_type == "integer":
            return int(float(str(value)))
        if target_type == "number":
            return float(str(value))
        if target_type == "string":
            return str(value)
        if target_type == "boolean":
            if isinstance(value, bool):
                return value
            return str(value).lower() in ("true", "yes", "1", "on")
    except (ValueError, TypeError):
        pass
    return value


def _path_str(error: jsonschema.exceptions.ValidationError) -> str:
    return " > ".join(str(p) for p in error.absolute_path) or "root"


# ── Main pipeline ─────────────────────────────────────────────────────────────


def repair_yaml(yaml_content: str, schema: dict) -> RepairResult:
    """Run the static repair pipeline. Never calls the LLM.

    Args:
        yaml_content: Raw YAML string to validate and repair.
        schema: Parsed JSON Schema dict (Draft 4 / 7 / 2019-09 / 2020-12).

    Returns:
        RepairResult with repaired YAML, error list, and llm_fields audit trail.
    """
    ry = ruamel.yaml.YAML()
    ry.preserve_quotes = True
    ry.width = 4096  # prevent unwanted line-wrapping

    # Stage 1 — yamlfix: syntax normalisation
    try:
        fixed_str = yamlfix.fix_code(yaml_content)
    except Exception:
        fixed_str = yaml_content  # yamlfix failed — proceed with original

    # Stage 2 — ruamel.yaml: roundtrip parse (CommentedMap preserves comments)
    try:
        data = ry.load(fixed_str)
        if not isinstance(data, dict):
            return RepairResult(
                valid=False,
                repaired_yaml=yaml_content,
                errors=["YAML must be a mapping (dict), got: " + type(data).__name__],
                summary="Parse failed",
            )
    except Exception as exc:
        return RepairResult(
            valid=False,
            repaired_yaml=yaml_content,
            errors=[f"YAML parse error: {exc}"],
            summary="Parse failed",
        )

    # Stage 3 — extend_with_default: fill defaults in-place
    try:
        validator_cls = jsonschema.validators.validator_for(schema)
        DefaultFilling = _extend_with_default(validator_cls)
        filler = DefaultFilling(schema)
        list(filler.iter_errors(data))  # side-effect: fills defaults
    except Exception:
        pass  # schema may lack defaults — continue

    # Stage 4 — collect all remaining errors
    try:
        validator_cls = jsonschema.validators.validator_for(schema)
        validator = validator_cls(schema)
        raw_errors = list(validator.iter_errors(data))
    except Exception as exc:
        return RepairResult(
            valid=False,
            repaired_yaml=yaml_content,
            errors=[f"Schema engine error: {exc}"],
            summary="Validation failed",
        )

    if not raw_errors:
        buf = io.StringIO()
        ry.dump(data, buf)
        return RepairResult(
            valid=True,
            repaired_yaml=buf.getvalue(),
            summary="Valid — 0 errors",
        )

    # Stage 5 — categorise errors
    deterministic: list[jsonschema.exceptions.ValidationError] = []
    ambiguous: list[jsonschema.exceptions.ValidationError] = []
    for e in raw_errors:
        if e.validator in ("type", "enum", "additionalProperties"):
            deterministic.append(e)
        else:
            ambiguous.append(e)

    # Stage 6 — apply deterministic fixes
    failed_fix_paths: list[str] = []
    for e in deterministic:
        path = list(e.absolute_path)

        if e.validator == "additionalProperties":
            # Navigate to the node that contains the extra keys (may be root)
            node: Any = data
            for key in path:
                try:
                    node = node[key]
                except (KeyError, IndexError, TypeError):
                    node = None
                    break
            if node is not None and isinstance(node, dict):
                allowed = set(e.schema.get("properties", {}).keys())
                for k in list(node.keys()):
                    if k not in allowed:
                        del node[k]
            continue

        if not path:
            continue

        # Navigate to the parent node
        node = data
        for key in path[:-1]:
            try:
                node = node[key]
            except (KeyError, IndexError, TypeError):
                node = None
                break
        if node is None:
            continue
        leaf_key = path[-1]

        if e.validator == "type":
            target = e.schema.get("type", "")
            if isinstance(target, str) and leaf_key in node:
                original = node[leaf_key]
                coerced = _coerce_type(original, target)
                if coerced == original:
                    failed_fix_paths.append(_path_str(e))
                else:
                    node[leaf_key] = coerced

    # Stage 7 — re-validate after fixes
    remaining_errors = sorted(
        validator.iter_errors(data), key=lambda e: list(e.absolute_path)
    )
    error_msgs = [
        f"{_path_str(e)}: {e.message}" for e in remaining_errors[:20]
    ]

    buf = io.StringIO()
    ry.dump(data, buf)
    repaired_yaml = buf.getvalue()

    llm_fields = [_path_str(e) for e in ambiguous] + failed_fix_paths

    valid = len(remaining_errors) == 0
    n = len(remaining_errors)
    summary = (
        "Valid after static repair"
        if valid
        else f"{n} error(s) remaining after static repair"
        + (" (showing first 20)" if n > 20 else "")
    )

    return RepairResult(
        valid=valid,
        repaired_yaml=repaired_yaml,
        errors=error_msgs,
        llm_fields=llm_fields,
        summary=summary,
    )
