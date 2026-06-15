"""OpenAPI schema artifact and endpoint coverage tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from api.main import app
from scripts.export_openapi import schema_text

ROOT_DIR = Path(__file__).resolve().parents[1]
OPENAPI_ARTIFACT = ROOT_DIR / "docs" / "api" / "openapi.json"
REGENERATE_OPENAPI_COMMAND = (
    "uv run python scripts/export_openapi.py --output docs/api/openapi.json"
)


def _runtime_openapi() -> dict[str, Any]:
    """Return the current FastAPI OpenAPI schema."""
    return app.openapi()


def test_openapi_schema_is_valid_jsonschema() -> None:
    """The runtime OpenAPI payload should satisfy basic JSON Schema structure."""
    schema = _runtime_openapi()

    assert schema["openapi"] == "3.1.0"
    assert isinstance(schema["paths"], dict)
    assert isinstance(schema["components"]["schemas"], dict)
    for component_schema in schema["components"]["schemas"].values():
        Draft202012Validator.check_schema(component_schema)


def test_openapi_schema_contains_search_endpoint() -> None:
    """The runtime OpenAPI schema should expose the /search endpoint."""
    schema = _runtime_openapi()

    assert "/search" in schema["paths"]


def test_openapi_schema_contains_languages_endpoint() -> None:
    """The runtime OpenAPI schema should expose the /languages endpoint."""
    schema = _runtime_openapi()

    assert "/languages" in schema["paths"]


def test_openapi_schema_contains_version_endpoint() -> None:
    """The runtime OpenAPI schema should expose the /version endpoint."""
    schema = _runtime_openapi()

    assert "/version" in schema["paths"]


def test_openapi_schema_contains_buck_reference_endpoints() -> None:
    """The runtime OpenAPI schema should expose Buck reference endpoints."""
    schema = _runtime_openapi()

    assert {
        "/languages/{language}/buck/rules",
        "/languages/{language}/buck/rules/{rule_id}",
        "/languages/{language}/buck/dialects",
        "/languages/{language}/buck/dialects/{dialect_id}",
        "/languages/{language}/buck/glossary",
    }.issubset(schema["paths"])


def test_openapi_buck_reference_endpoints_document_404_responses() -> None:
    """Buck reference endpoints should document not-found responses."""
    schema = _runtime_openapi()

    for path in (
        "/languages/{language}/buck/rules",
        "/languages/{language}/buck/rules/{rule_id}",
        "/languages/{language}/buck/dialects",
        "/languages/{language}/buck/dialects/{dialect_id}",
        "/languages/{language}/buck/glossary",
    ):
        response = schema["paths"][path]["get"]["responses"]["404"]
        assert response["content"]["application/json"]["schema"]["$ref"] == (
            "#/components/schemas/ErrorResponse"
        )


def test_openapi_buck_reference_endpoints_document_400_responses() -> None:
    """Buck reference endpoints with custom bad requests should document 400s."""
    schema = _runtime_openapi()

    for path in (
        "/languages/{language}/buck/rules",
        "/languages/{language}/buck/glossary",
    ):
        response = schema["paths"][path]["get"]["responses"]["400"]
        assert response["content"]["application/json"]["schema"]["$ref"] == (
            "#/components/schemas/ErrorResponse"
        )


def test_openapi_search_hit_schema_includes_buck_references() -> None:
    """The /search hit schema should expose Buck reference annotations."""
    schema = _runtime_openapi()
    search_hit_schema = schema["components"]["schemas"]["SearchHit"]

    assert "buck_references" in search_hit_schema["properties"]
    assert (
        search_hit_schema["properties"]["buck_references"]["items"]["$ref"]
        == "#/components/schemas/BuckReferenceAnnotation"
    )


def test_openapi_buck_rule_schema_includes_review_fields() -> None:
    """Buck REST item schemas should expose review boundaries."""
    schema = _runtime_openapi()
    rule_schema = schema["components"]["schemas"]["BuckRuleInfo"]
    metadata_schema = schema["components"]["schemas"]["BuckMetadata"]

    assert {"status", "review_status", "citation_ready"}.issubset(
        rule_schema["properties"]
    )
    assert {
        "status",
        "review_status",
        "citation_ready",
        "review_note",
    }.issubset(metadata_schema["properties"])


def test_openapi_buck_glossary_schema_allows_inscription_number_arrays() -> None:
    """Buck glossary entries should expose string or integer-array inscription ids."""
    schema = _runtime_openapi()
    entry_schema = schema["components"]["schemas"]["BuckGlossaryEntryInfo"]
    inscription_schema = entry_schema["properties"]["inscription_no"]

    assert inscription_schema["anyOf"] == [
        {"type": "string"},
        {"items": {"type": "integer"}, "type": "array"},
        {"type": "null"},
    ]


def test_openapi_artifact_matches_app_openapi() -> None:
    """The committed OpenAPI artifact should match the runtime schema."""
    assert OPENAPI_ARTIFACT.exists(), (
        f"{OPENAPI_ARTIFACT} is missing. Regenerate it with:\n"
        f"  {REGENERATE_OPENAPI_COMMAND}"
    )
    expected = schema_text(indent=2)
    actual = OPENAPI_ARTIFACT.read_text(encoding="utf-8")

    assert actual == expected, (
        "docs/api/openapi.json is out of date. Regenerate it with:\n"
        f"  {REGENERATE_OPENAPI_COMMAND}"
    )
