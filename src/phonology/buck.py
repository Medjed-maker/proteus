"""Read-only loader for Buck-normalized Ancient Greek reference data."""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict

import yaml  # type: ignore[import-untyped]

from ._paths import resolve_repo_data_dir

__all__ = ["BuckData", "load_buck_data"]


class BuckData(TypedDict):
    """Validated Buck reference data keyed by document type."""

    grammar_rules: dict[str, Any]
    dialects: dict[str, Any]
    glossary: dict[str, Any]


@dataclass(frozen=True)
class _BuckPaths:
    """Filesystem paths for the three Buck data files."""

    grammar: Path
    dialects: Path
    glossary: Path


_TRUSTED_BUCK_DIR_ENV_VAR = "PROTEUS_TRUSTED_BUCK_DIR"


def _get_buck_rules_dir() -> Path:
    """Resolve the Buck data directory from env var or default package path.

    When the environment variable ``PROTEUS_TRUSTED_BUCK_DIR`` is set, it
    allows specifying an arbitrary filesystem path as the Buck data
    directory.  Basic safety checks (``is_symlink``, ``exists``,
    ``is_dir``) are performed, but the variable ultimately grants full
    filesystem access and must only be used in trusted environments.
    Operators should avoid setting this env var in untrusted or
    user-supplied contexts.
    """
    override = os.environ.get(_TRUSTED_BUCK_DIR_ENV_VAR)
    if override:
        override_path = Path(override).expanduser()
        for part in [override_path, *override_path.parents]:
            if part == part.parent:
                # filesystem root — skip
                break
            if part.is_symlink():
                raise ValueError(
                    f"{_TRUSTED_BUCK_DIR_ENV_VAR} must not contain a symlink, "
                    f"got {override} (symlink at {part})"
                )
        resolved = override_path.resolve(strict=False)
        if not resolved.exists() or not resolved.is_dir():
            raise FileNotFoundError(
                f"Could not find Buck data directory {resolved}"
            )
        return resolved
    return resolve_repo_data_dir("rules") / "ancient_greek" / "buck"


def _load_yaml_mapping(path: Path, *, required_list_key: str) -> dict[str, Any]:
    """Load a YAML document and require a top-level mapping plus one list key."""
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(
            f"Failed to parse YAML file {path}: {exc}"
        ) from exc
    if not isinstance(document, dict):
        raise ValueError(f"Buck data file {path} must contain a top-level mapping")
    raw_items = document.get(required_list_key)
    if not isinstance(raw_items, list):
        raise ValueError(
            f"Buck data file {path} must define a list under {required_list_key!r}"
        )
    return document


def _validate_grammar_rules(document: dict[str, Any], path: Path) -> set[str]:
    """Validate Buck grammar rules and return the discovered rule ids."""
    raw_rules = document["rules"]
    rule_ids: set[str] = set()
    for index, raw_rule in enumerate(raw_rules):
        if not isinstance(raw_rule, dict):
            raise ValueError(f"Buck rule entry {index} in {path} must be a mapping")
        rule_id = raw_rule.get("id")
        if not isinstance(rule_id, str) or not rule_id.strip():
            raise ValueError(f"Buck rule entry {index} in {path} must define a non-empty id")
        if rule_id in rule_ids:
            raise ValueError(f"Duplicate Buck rule id {rule_id!r} found in {path}")
        rule_ids.add(rule_id)
    return rule_ids


def _validate_dialects(document: dict[str, Any], path: Path) -> set[str]:
    """Validate Buck dialect catalog and return the discovered dialect ids."""
    if "rules" in document:
        raise ValueError(f"Buck dialect catalog {path} must not define a top-level 'rules' key")

    raw_dialects = document["dialects"]
    dialect_ids: set[str] = set()
    for index, raw_dialect in enumerate(raw_dialects):
        if not isinstance(raw_dialect, dict):
            raise ValueError(f"Buck dialect entry {index} in {path} must be a mapping")
        dialect_id = raw_dialect.get("id")
        if not isinstance(dialect_id, str) or not dialect_id.strip():
            raise ValueError(f"Buck dialect entry {index} in {path} must define a non-empty id")
        if dialect_id in dialect_ids:
            raise ValueError(f"Duplicate Buck dialect id {dialect_id!r} found in {path}")
        raw_rules = raw_dialect.get("rules", [])
        if not isinstance(raw_rules, list):
            raise ValueError(
                f"Buck dialect entry {dialect_id!r} in {path} must define 'rules' as a list"
            )
        dialect_ids.add(dialect_id)
    return dialect_ids


def _validate_rule_refs(
    dialects_document: dict[str, Any],
    glossary_document: dict[str, Any],
    *,
    rule_ids: set[str],
    dialect_ids: set[str],
    paths: _BuckPaths,
) -> None:
    """Validate cross-file rule and dialect references."""
    for raw_dialect in dialects_document["dialects"]:
        dialect_id = raw_dialect["id"]
        for raw_rule_id in raw_dialect.get("rules", []):
            if not isinstance(raw_rule_id, str) or not raw_rule_id.strip():
                raise ValueError(
                    f"Buck dialect entry {dialect_id!r} in {paths.dialects} "
                    "must contain only non-empty string rule ids"
                )
            if raw_rule_id not in rule_ids:
                raise ValueError(
                    f"Buck dialect entry {dialect_id!r} in {paths.dialects} "
                    f"references unknown rule id {raw_rule_id!r} from {paths.grammar}"
                )

    for index, raw_word in enumerate(glossary_document["words"]):
        if not isinstance(raw_word, dict):
            raise ValueError(f"Buck glossary entry {index} in {paths.glossary} must be a mapping")
        dialect_id = raw_word.get("dialect")
        if not isinstance(dialect_id, str) or not dialect_id.strip():
            raise ValueError(
                f"Buck glossary entry {index} in {paths.glossary} must define a non-empty dialect"
            )
        if dialect_id not in dialect_ids:
            raise ValueError(
                f"Buck glossary entry {index} in {paths.glossary} "
                f"references unknown dialect id {dialect_id!r}"
            )
        if "rule_id" not in raw_word or raw_word["rule_id"] is None:
            continue
        rule_id = raw_word["rule_id"]
        if not isinstance(rule_id, str) or not rule_id.strip():
            raise ValueError(
                f"Buck glossary entry {index} in {paths.glossary} must define a non-empty rule_id"
            )
        if rule_id not in rule_ids:
            raise ValueError(
                f"Buck glossary entry {index} in {paths.glossary} "
                f"references unknown rule id {rule_id!r} from {paths.grammar}"
            )


def _validate_grammar_dialect_refs(
    grammar_document: dict[str, Any],
    *,
    dialect_ids: set[str],
    paths: _BuckPaths,
) -> None:
    """Validate dialect references used inside Buck grammar rules."""
    for raw_rule in grammar_document["rules"]:
        rule_id = raw_rule["id"]
        # Handles YAML null by coercing to []
        dialects = raw_rule.get("affected_dialects") or []
        for dialect_id in dialects:
            if not isinstance(dialect_id, str) or not dialect_id.strip():
                raise ValueError(
                    f"Buck rule {rule_id!r} in {paths.grammar} must contain only "
                    "non-empty strings in affected_dialects"
                )
            if dialect_id not in dialect_ids:
                raise ValueError(
                    f"Buck rule {rule_id!r} in {paths.grammar} references unknown dialect "
                    f"id {dialect_id!r}"
                )
        for variant_index, variant in enumerate(raw_rule.get("variants", []) or []):
            if not isinstance(variant, dict):
                raise ValueError(
                    f"Buck rule {rule_id!r} variant {variant_index} in {paths.grammar} "
                    "must be a mapping"
                )
            for dialect_id in variant.get("dialects", []) or []:
                if not isinstance(dialect_id, str) or not dialect_id.strip():
                    raise ValueError(
                        f"Buck rule {rule_id!r} variant {variant_index} in {paths.grammar} "
                        "must contain only non-empty strings in dialects"
                    )
                if dialect_id not in dialect_ids:
                    raise ValueError(
                        f"Buck rule {rule_id!r} variant {variant_index} in {paths.grammar} "
                        f"references unknown dialect id {dialect_id!r}"
                    )


@lru_cache(maxsize=1)
def _load_buck_data_cached() -> BuckData:
    """Load and validate packaged Buck data once per process.

    Cached via ``@lru_cache(maxsize=1)``: subsequent calls return the
    same object without re-reading files.  Changes to the
    ``PROTEUS_TRUSTED_BUCK_DIR`` environment variable or to the
    underlying data files will **not** be picked up while the process
    is alive.  Tests or callers that need to reload must call
    ``_load_buck_data_cached.cache_clear()`` before invoking this
    function again.  Note that :func:`load_buck_data` returns a
    defensive ``deepcopy``, so mutations of its result never affect
    the cached copy.
    """
    buck_rules_dir = _get_buck_rules_dir()
    try:
        resolved_dir = buck_rules_dir.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Could not find Buck data directory {buck_rules_dir}"
        ) from exc
    if not resolved_dir.is_dir():
        raise FileNotFoundError(f"Buck data path is not a directory: {resolved_dir}")

    paths = _BuckPaths(
        grammar=resolved_dir / "grammar_rules.yaml",
        dialects=resolved_dir / "dialects.yaml",
        glossary=resolved_dir / "glossary.yaml",
    )

    grammar_document = _load_yaml_mapping(paths.grammar, required_list_key="rules")
    dialects_document = _load_yaml_mapping(paths.dialects, required_list_key="dialects")
    glossary_document = _load_yaml_mapping(paths.glossary, required_list_key="words")

    rule_ids = _validate_grammar_rules(grammar_document, paths.grammar)
    dialect_ids = _validate_dialects(dialects_document, paths.dialects)
    _validate_rule_refs(
        dialects_document,
        glossary_document,
        rule_ids=rule_ids,
        dialect_ids=dialect_ids,
        paths=paths,
    )
    _validate_grammar_dialect_refs(
        grammar_document,
        dialect_ids=dialect_ids,
        paths=paths,
    )

    return {
        "grammar_rules": grammar_document,
        "dialects": dialects_document,
        "glossary": glossary_document,
    }


def load_buck_data() -> BuckData:
    """Return a defensive copy of the validated Buck reference data.

    The returned :class:`BuckData` mapping contains three top-level keys:

    * ``"grammar_rules"`` — YAML document with a ``"rules"`` list of rule
      definitions (id, description, affected_dialects, variants, etc.).
    * ``"dialects"`` — YAML document with a ``"dialects"`` list of dialect
      catalog entries (id, name, parent, group, rules).
    * ``"glossary"`` — YAML document with a ``"words"`` list of example
      lexical entries (word, standard_form, dialect, rule_id, etc.).

    Each call returns a ``deepcopy`` of the internally cached data, so
    callers may freely mutate the result without affecting subsequent calls.

    Example::

        data = load_buck_data()
        for rule in data["grammar_rules"]["rules"]:
            print(rule["id"])
    """
    return deepcopy(_load_buck_data_cached())
