"""Validation tests for Buck-normalized Ancient Greek data files."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
BUCK_RULES_DIR = ROOT_DIR / "data" / "languages" / "ancient_greek" / "rules" / "buck"
BUCK_GRAMMAR_RULES_PATH = BUCK_RULES_DIR / "grammar_rules.yaml"
BUCK_DIALECTS_PATH = BUCK_RULES_DIR / "dialects.yaml"
BUCK_GLOSSARY_PATH = BUCK_RULES_DIR / "glossary.yaml"


def _load_yaml(path: Path) -> Any:
    """Read a YAML file from *path* and return the parsed document."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _grammar_rule_ids(grammar_document: dict[str, Any]) -> list[str]:
    """Extract and return a list of rule ``id`` strings from *grammar_document*."""
    return [rule["id"] for rule in grammar_document["rules"]]


def _dialect_ids(dialects_document: dict[str, Any]) -> set[str]:
    """Extract and return a set of dialect ``id`` strings from *dialects_document*."""
    return {dialect["id"] for dialect in dialects_document["dialects"]}


# -- module-scoped fixtures to avoid redundant YAML I/O --


@pytest.fixture(scope="module")
def grammar_document() -> dict[str, Any]:
    """Load and cache the grammar rules YAML document for the module."""
    doc = _load_yaml(BUCK_GRAMMAR_RULES_PATH)
    assert isinstance(doc, dict)
    return doc


@pytest.fixture(scope="module")
def dialects_document() -> dict[str, Any]:
    """Load and cache the dialects YAML document for the module."""
    doc = _load_yaml(BUCK_DIALECTS_PATH)
    assert isinstance(doc, dict)
    return doc


@pytest.fixture(scope="module")
def glossary_document() -> dict[str, Any]:
    """Load and cache the glossary YAML document for the module."""
    doc = _load_yaml(BUCK_GLOSSARY_PATH)
    assert isinstance(doc, dict)
    return doc


def test_buck_directory_contains_expected_three_yaml_files() -> None:
    assert {path.name for path in BUCK_RULES_DIR.glob("*.yaml")} == {
        "dialects.yaml",
        "glossary.yaml",
        "grammar_rules.yaml",
    }


def test_buck_yaml_documents_are_top_level_mappings() -> None:
    for path in (BUCK_GRAMMAR_RULES_PATH, BUCK_DIALECTS_PATH, BUCK_GLOSSARY_PATH):
        document = _load_yaml(path)
        assert isinstance(document, dict), (
            f"{path.name} must contain a top-level mapping"
        )


def test_buck_dialects_file_is_catalog_only(dialects_document: dict[str, Any]) -> None:
    assert "dialects" in dialects_document
    assert "rules" not in dialects_document


def test_buck_rule_ids_are_unique_and_representative_ids_exist(
    grammar_document: dict[str, Any],
) -> None:
    rule_ids = _grammar_rule_ids(grammar_document)
    duplicates = [rule_id for rule_id, count in Counter(rule_ids).items() if count > 1]

    assert not duplicates, f"Duplicate rule IDs found: {duplicates}"
    expected = {"grc_phon_41_4", "grc_phon_60_1", "grc_synt_175", "grc_morph_134_3"}
    actual = set(rule_ids)
    assert expected <= actual, f"Missing representative IDs: {expected - actual}"


def test_buck_dialect_rule_refs_resolve_to_grammar_rules(
    grammar_document: dict[str, Any],
    dialects_document: dict[str, Any],
) -> None:
    rule_ids = set(_grammar_rule_ids(grammar_document))

    missing = sorted(
        {
            rule_id
            for dialect in dialects_document["dialects"]
            for rule_id in dialect.get("rules", [])
            if rule_id not in rule_ids
        }
    )

    assert missing == [], f"Missing rule IDs referenced by dialects: {missing}"


def test_buck_glossary_refs_resolve_to_known_rules_and_dialects(
    grammar_document: dict[str, Any],
    dialects_document: dict[str, Any],
    glossary_document: dict[str, Any],
) -> None:
    rule_ids = set(_grammar_rule_ids(grammar_document))
    dialect_id_set = _dialect_ids(dialects_document)

    missing_rule_ids = sorted(
        {
            entry["rule_id"]
            for entry in glossary_document["words"]
            if "rule_id" in entry
            and isinstance(entry["rule_id"], str)
            and entry["rule_id"] not in rule_ids
        }
    )
    missing_dialect_ids = sorted(
        {
            dialect
            for entry in glossary_document["words"]
            if (dialect := entry.get("dialect"))
            and isinstance(dialect, str)
            and dialect not in dialect_id_set
        }
    )

    assert missing_rule_ids == [], f"Missing rule IDs in glossary: {missing_rule_ids}"
    assert missing_dialect_ids == [], (
        f"Missing dialect IDs in glossary: {missing_dialect_ids}"
    )


def test_buck_grammar_rule_dialect_refs_resolve_to_catalog_entries(
    grammar_document: dict[str, Any],
    dialects_document: dict[str, Any],
) -> None:
    dialect_id_set = _dialect_ids(dialects_document)
    missing = set[str]()

    for rule in grammar_document["rules"]:
        for dialect_id in rule.get("affected_dialects", []) or []:
            if dialect_id not in dialect_id_set:
                missing.add(dialect_id)
        for variant in rule.get("variants", []) or []:
            for dialect_id in variant.get("dialects", []) or []:
                if dialect_id not in dialect_id_set:
                    missing.add(dialect_id)

    assert missing == set(), f"Missing dialect IDs referenced in rules: {missing}"
