"""Validation tests for Buck-normalized Ancient Greek data files."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from jsonschema import Draft202012Validator, FormatChecker


ROOT_DIR = Path(__file__).resolve().parents[1]
BUCK_RULES_DIR = ROOT_DIR / "data" / "languages" / "ancient_greek" / "rules" / "buck"
BUCK_GRAMMAR_RULES_PATH = BUCK_RULES_DIR / "grammar_rules.yaml"
BUCK_DIALECTS_PATH = BUCK_RULES_DIR / "dialects.yaml"
BUCK_GLOSSARY_PATH = BUCK_RULES_DIR / "glossary.yaml"
BUCK_GRAMMAR_SCHEMA_PATH = ROOT_DIR / "data" / "schemas" / "buck_grammar_rules.schema.json"
BUCK_DIALECTS_SCHEMA_PATH = ROOT_DIR / "data" / "schemas" / "buck_dialects.schema.json"
BUCK_GLOSSARY_SCHEMA_PATH = ROOT_DIR / "data" / "schemas" / "buck_glossary.schema.json"


def _load_yaml(path: Path) -> Any:
    """Read a YAML file from *path* and return the parsed document."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_json(path: Path) -> Any:
    """Read a JSON file from *path* and return the parsed document."""
    return json.loads(path.read_text(encoding="utf-8"))


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


@pytest.fixture(scope="module")
def buck_schema_validators() -> dict[Path, Draft202012Validator]:
    """Return JSON Schema validators for Buck YAML data files keyed by path."""
    schema_by_data_path = {
        BUCK_GRAMMAR_RULES_PATH: _load_json(BUCK_GRAMMAR_SCHEMA_PATH),
        BUCK_DIALECTS_PATH: _load_json(BUCK_DIALECTS_SCHEMA_PATH),
        BUCK_GLOSSARY_PATH: _load_json(BUCK_GLOSSARY_SCHEMA_PATH),
    }
    return {
        data_path: Draft202012Validator(schema, format_checker=FormatChecker())
        for data_path, schema in schema_by_data_path.items()
    }


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


@pytest.mark.parametrize(
    "schema_path",
    [BUCK_GRAMMAR_SCHEMA_PATH, BUCK_DIALECTS_SCHEMA_PATH, BUCK_GLOSSARY_SCHEMA_PATH],
)
def test_buck_json_schemas_are_valid_draft_2020_12(schema_path: Path) -> None:
    schema = _load_json(schema_path)

    Draft202012Validator.check_schema(schema)


@pytest.mark.parametrize(
    "data_path",
    [BUCK_GRAMMAR_RULES_PATH, BUCK_DIALECTS_PATH, BUCK_GLOSSARY_PATH],
    ids=lambda path: path.name,
)
def test_buck_yaml_documents_validate_against_shape_schemas(
    buck_schema_validators: dict[Path, Draft202012Validator],
    data_path: Path,
) -> None:
    validator = buck_schema_validators[data_path]
    document = _load_yaml(data_path)

    errors = sorted(validator.iter_errors(document), key=lambda error: error.json_path)

    if errors:
        error_messages = [f"{error.json_path}: {error.message}" for error in errors]
        pytest.fail(
            f"Schema validation errors in {data_path}:\n" + "\n".join(error_messages)
        )


def test_buck_schemas_share_identical_common_defs() -> None:
    """Guard against drift between the duplicated $defs in the Buck schemas."""
    grammar_defs = _load_json(BUCK_GRAMMAR_SCHEMA_PATH)["$defs"]
    dialects_defs = _load_json(BUCK_DIALECTS_SCHEMA_PATH)["$defs"]
    glossary_defs = _load_json(BUCK_GLOSSARY_SCHEMA_PATH)["$defs"]

    shared_defs_by_name = {
        "buckMetadata": [grammar_defs, dialects_defs, glossary_defs],
        "section": [grammar_defs, glossary_defs],
        "stringList": [grammar_defs, dialects_defs],
    }
    for def_name, schema_defs in shared_defs_by_name.items():
        reference = schema_defs[0][def_name]
        for defs in schema_defs[1:]:
            assert defs[def_name] == reference, (
                f"Shared $defs entry {def_name!r} has drifted between the Buck "
                "schema files; keep duplicated definitions identical"
            )


def test_buck_schemas_reject_citation_ready_without_expert_review(
    buck_schema_validators: dict[Path, Draft202012Validator],
) -> None:
    documents = {
        BUCK_GRAMMAR_RULES_PATH: {
            "meta": {
                "status": "provisional",
                "review_status": "not_expert_reviewed",
                "citation_ready": True,
                "source_notes": ["test"],
            },
            "rules": [{"id": "TEST-001"}],
        },
        BUCK_DIALECTS_PATH: {
            "meta": {
                "status": "provisional",
                "review_status": "not_expert_reviewed",
                "citation_ready": True,
                "source_notes": ["test"],
            },
            "dialects": [{"id": "test_dialect", "rules": []}],
        },
        BUCK_GLOSSARY_PATH: {
            "meta": {
                "status": "provisional",
                "review_status": "not_expert_reviewed",
                "citation_ready": True,
                "source_notes": ["test"],
            },
            "words": [{"word": "test", "dialect": "test_dialect"}],
        },
    }

    for data_path, document in documents.items():
        errors = list(buck_schema_validators[data_path].iter_errors(document))

        assert any(
            error.json_path == "$.meta.review_status"
            and error.message == "'expert_reviewed' was expected"
            for error in errors
        ), f"{data_path.name} schema should reject unreviewed citation-ready data"


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


def test_buck_glossary_current_entries_matches_word_count(
    glossary_document: dict[str, Any],
) -> None:
    assert glossary_document["meta"]["current_entries"] == len(
        glossary_document["words"]
    )
