"""YAML rule loading and version-metadata extraction for the explainer.

The logger uses ``"phonology.explainer"`` so ``caplog`` tests continue to
capture diagnostics from this module after the split.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
import logging
import math
from pathlib import Path
from typing import Any, cast

import yaml

from ._explainer_rule_paths import _resolve_and_validate_rules_dir

logger = logging.getLogger("phonology.explainer")


def load_rules(rules_dir: Path | str) -> dict[str, dict[str, Any]]:
    """Load all YAML rule files from a directory.

    Args:
        rules_dir: Path to the rules directory. Relative inputs are resolved
            from the packaged rules base, so both ``"ancient_greek"`` and
            ``"data/rules/ancient_greek"`` resolve to the same runtime asset.

    Returns:
        Dict mapping rule_id -> rule dict.
    """
    resolved_rules_dir = _resolve_and_validate_rules_dir(rules_dir, "load_rules")

    rules: dict[str, dict[str, Any]] = {}
    rule_sources: dict[str, Path] = {}
    for rule_file in sorted(resolved_rules_dir.iterdir()):
        if not rule_file.is_file() or rule_file.suffix.lower() not in {".yaml", ".yml"}:
            continue

        document = yaml.safe_load(rule_file.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError(f"Rule file {rule_file} must contain a top-level mapping")

        raw_rules = document.get("rules")
        if not isinstance(raw_rules, list):
            raise ValueError(f"Rule file {rule_file} must define a list under 'rules'")

        for index, raw_rule in enumerate(raw_rules):
            if not isinstance(raw_rule, dict):
                raise ValueError(f"Rule entry {index} in {rule_file} must be a mapping")

            rule_id = raw_rule.get("id")
            if not isinstance(rule_id, str) or not rule_id.strip():
                raise ValueError(
                    f"Rule entry {index} in {rule_file} must define a non-empty id"
                )
            if rule_id in rules:
                first_defined_in = rule_sources[rule_id]
                raise ValueError(
                    f"Duplicate rule id {rule_id!r} found in {rule_file}; "
                    f"first defined in {first_defined_in}"
                )
            rules[rule_id] = raw_rule
            rule_sources[rule_id] = rule_file

    return rules


def get_rules_version(rules_dir: Path | str) -> dict[str, str]:
    """Load version metadata from all YAML rule files in a directory.

    Args:
        rules_dir: Path to the rules directory. Relative inputs are resolved
            from the packaged rules base.

    Returns:
        Dict mapping rule file stem -> version string.
        Example: {"vowel_shifts": "1.0.0", "consonant_changes": "1.0.0"}

        Version parsing semantics:
        - non-empty YAML strings are returned as-is after trimming whitespace
        - YAML integers are converted to decimal strings
        - YAML floats are converted through ``Decimal(str(value))`` to avoid
          binary float artifacts while preserving the parsed decimal form
        - missing, invalid, or non-finite values are skipped with debug logs

    Raises:
        Same exceptions as load_rules().
    """
    resolved_rules_dir = _resolve_and_validate_rules_dir(rules_dir, "get_rules_version")

    versions: dict[str, str] = {}
    for rule_file in sorted(resolved_rules_dir.iterdir()):
        if not rule_file.is_file() or rule_file.suffix.lower() not in {".yaml", ".yml"}:
            continue

        content = rule_file.read_text(encoding="utf-8")
        document = yaml.safe_load(content)
        if not isinstance(document, dict):
            logger.debug(
                "Skipping rule version metadata in %s: top-level YAML is not a mapping",
                rule_file,
            )
            continue

        version = _extract_rule_file_version(document, rule_file, content)
        if version is None:
            logger.debug(
                "Skipping rule version metadata in %s: version is missing or invalid",
                rule_file,
            )
            continue
        versions[rule_file.stem] = version
    return versions


def _extract_scalar_node_value(
    version_node: yaml.nodes.Node | None,
) -> str | int | float | Decimal | None:
    """Return a parsed version value while preserving decimal text for floats."""
    if not isinstance(version_node, yaml.ScalarNode):
        return None

    tag = version_node.tag
    value = version_node.value
    if tag == "tag:yaml.org,2002:str":
        return cast(str, value)
    if tag == "tag:yaml.org,2002:int":
        try:
            return int(value)
        except ValueError:
            return None
    if tag == "tag:yaml.org,2002:float":
        try:
            numeric = float(value)
        except ValueError:
            return None
        if not math.isfinite(numeric):
            return numeric
        return Decimal(value)
    return None


def _is_valid_rule_version_value(version: object) -> bool:
    """Return whether a parsed YAML value is acceptable as rule version metadata."""
    if isinstance(version, str):
        return bool(version.strip())
    if isinstance(version, bool):
        return False
    if isinstance(version, int):
        return True
    if isinstance(version, Decimal):
        return True
    if isinstance(version, float):
        return math.isfinite(version)
    return False


def _extract_rule_file_version(
    document: object,
    rule_file: Path,
    content: str,
) -> str | None:
    """Extract a normalized version string from a parsed YAML rule file.

    Args:
        document: Parsed YAML document loaded from ``content``.
        rule_file: Source path used for logging when version parsing fails.
        content: Preloaded YAML text for ``rule_file``. Callers that already
            read the file should pass the original text here so this helper can
            inspect the scalar YAML node without re-reading from disk.
    """
    if not isinstance(document, dict):
        return None

    version_node: yaml.nodes.Node | None = None
    version_node_from_meta = False
    try:
        composed = yaml.compose(content)
    except yaml.YAMLError:
        composed = None
    # yaml.compose exposes low-level ScalarNode tags so version_node and
    # version_node_from_meta can be passed through _extract_scalar_node_value.
    # If yaml.compose cannot provide that node-level view, fall back to the
    # safe_load-derived document mapping and validate values with
    # _is_valid_rule_version_value.
    if isinstance(composed, yaml.MappingNode):
        for key_node, value_node in composed.value:
            if (
                isinstance(key_node, yaml.ScalarNode)
                and key_node.tag == "tag:yaml.org,2002:str"
                and key_node.value == "meta"
                and isinstance(value_node, yaml.MappingNode)
            ):
                for meta_key, meta_val in value_node.value:
                    if (
                        isinstance(meta_key, yaml.ScalarNode)
                        and meta_key.tag == "tag:yaml.org,2002:str"
                        and meta_key.value == "version"
                    ):
                        version_node = meta_val
                        version_node_from_meta = True
                        break
                break
        if version_node is None:
            for key_node, value_node in composed.value:
                if (
                    isinstance(key_node, yaml.ScalarNode)
                    and key_node.tag == "tag:yaml.org,2002:str"
                    and key_node.value == "version"
                ):
                    version_node = value_node
                    break

    version = _extract_scalar_node_value(version_node)
    if version_node_from_meta and not _is_valid_rule_version_value(version):
        version = None
    if version is None:
        meta = document.get("meta")
        if isinstance(meta, Mapping):
            meta_version = meta.get("version")
            if _is_valid_rule_version_value(meta_version):
                version = meta_version
        if version is None:
            version = document.get("version")
            # Note: top-level version is validated by the conversion logic below.

    if isinstance(version, str) and version.strip():
        return version.strip()
    if isinstance(version, int):
        return str(version)
    if isinstance(version, Decimal):
        return str(version)
    if isinstance(version, float):
        if not math.isfinite(version):
            logger.debug(
                "Skipping rule version metadata in %s: version is non-finite (%s)",
                rule_file,
                version,
            )
            return None
        return str(Decimal(str(version)))
    return None
