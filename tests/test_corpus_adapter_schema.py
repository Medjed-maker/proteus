"""Tests for corpus source-reference YAML schema."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from jsonschema import Draft202012Validator


ROOT_DIR = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT_DIR / "data" / "schemas" / "corpus_source_reference.schema.json"
DATA_PATH = (
    ROOT_DIR
    / "data"
    / "languages"
    / "ancient_greek"
    / "corpus_sources"
    / "perseus_scaife_sources.yaml"
)


def _validator() -> Draft202012Validator:
    """Load the schema from SCHEMA_PATH and return a Draft202012Validator."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


def _payload() -> dict[str, Any]:
    """Load and return the valid corpus source payload used by tests."""
    payload = yaml.safe_load(DATA_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _first_reference(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the first reference object from the provided test payload."""
    entries = payload["entries"]
    assert isinstance(entries, dict)
    references = next(iter(entries.values()))
    assert isinstance(references, list)
    reference = references[0]
    assert isinstance(reference, dict)
    return reference


def test_valid_corpus_source_yaml_matches_schema() -> None:
    """Validate that the bundled corpus source YAML matches the schema."""
    _validator().validate(_payload())


def test_schema_rejects_missing_required_reference_field() -> None:
    """Validate that omitting a required reference field is rejected."""
    payload = _payload()
    _first_reference(payload).pop("source_id")

    errors = list(_validator().iter_errors(payload))

    assert any("source_id" in error.message for error in errors)


@pytest.mark.parametrize(
    "field_name",
    ["evidence_excerpt", "source_text", "passage_text", "quote"],
)
def test_schema_rejects_source_text_fields(field_name: str) -> None:
    """Validate that prohibited source text fields are rejected."""
    payload = _payload()
    _first_reference(payload)[field_name] = "not allowed"

    errors = list(_validator().iter_errors(payload))

    assert any("Additional properties are not allowed" in error.message for error in errors)


def test_schema_rejects_non_external_url_policy_fields() -> None:
    """Validate that non-external URL policy fields are rejected."""
    payload = _payload()
    _first_reference(payload)["source_text_url"] = "https://example.test/text"

    errors = list(_validator().iter_errors(payload))

    assert any("Additional properties are not allowed" in error.message for error in errors)


def test_schema_rejects_non_http_external_url() -> None:
    """Validate that external_url must use the HTTP or HTTPS scheme."""
    payload = copy.deepcopy(_payload())
    _first_reference(payload)["external_url"] = "ftp://example.test/source"

    errors = list(_validator().iter_errors(payload))

    assert any("does not match" in error.message for error in errors)


def test_schema_rejects_overlong_short_citation() -> None:
    """Validate that short_citation must not exceed 200 characters."""
    payload = copy.deepcopy(_payload())
    _first_reference(payload)["short_citation"] = "x" * 201

    errors = list(_validator().iter_errors(payload))

    assert any("is too long" in error.message for error in errors)
