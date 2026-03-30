"""Produce an annotated YAML string with # FIXED: / # WARNING: inline comments."""
from __future__ import annotations

import io
from typing import (
    Any,  # unavoidable: ruamel.yaml has no public type stubs for CommentedMap/CommentedSeq
)

from ruamel.yaml import YAML

from src.models.validation_session import FixRecord


def _navigate(doc: Any, path: str) -> tuple[Any, str] | tuple[None, None]:
    """Return (parent_mapping, leaf_key) for a dotted field path, or (None, None)."""
    parts = path.split(".")
    node = doc
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            return None, None
        node = node[part]
    return node, parts[-1]


def annotate_yaml(
    yaml_content: str,
    fixes: list[FixRecord],
    warnings: list[str],
) -> str:
    """Return yaml_content with FIXED/WARNING inline comments added via ruamel.yaml.

    Args:
        yaml_content: The corrected YAML string (after all fixes applied).
        fixes: FixRecord list — each one maps a field_path to an EOL comment.
        warnings: List of warning strings added as a block comment at the top.

    Returns:
        Annotated YAML string preserving original formatting and order.
    """
    ryaml = YAML()
    ryaml.preserve_quotes = True
    doc = ryaml.load(yaml_content)
    if doc is None:
        return yaml_content

    # Add EOL comments for each fix
    for fix in fixes:
        parent, key = _navigate(doc, fix.field_path)
        if parent is not None and key is not None and isinstance(parent, dict) and key in parent:
            try:
                comment = (
                    f"FIXED: was {fix.original_value!r} → {fix.fixed_value!r}"
                    f" — {fix.reason}"
                )
                parent.yaml_add_eol_comment(comment, key)
            except Exception:
                pass  # best-effort: skip if node type doesn't support comments

    # Add block comment for warnings at the top of the document
    if warnings:
        try:
            warning_block = "\n".join(f"WARNING: {w}" for w in warnings)
            doc.yaml_set_start_comment(warning_block)
        except Exception:
            # Fallback: prepend raw comment lines
            stream = io.StringIO()
            ryaml.dump(doc, stream)
            header = "\n".join(f"# WARNING: {w}" for w in warnings)
            return header + "\n" + stream.getvalue()

    stream = io.StringIO()
    ryaml.dump(doc, stream)
    return stream.getvalue()
