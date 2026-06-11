"""MCP tool schema artifact and contract tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anyio

from scripts.export_mcp_schema import build_schema

ROOT_DIR = Path(__file__).resolve().parents[1]
MCP_SCHEMA_ARTIFACT = ROOT_DIR / "docs" / "mcp" / "tools.json"
REGENERATE_MCP_SCHEMA_COMMAND = (
    "uv run python scripts/export_mcp_schema.py --output docs/mcp/tools.json"
)


def _runtime_mcp_schema() -> dict[str, Any]:
    """Return the current MCP tool schema payload."""
    return anyio.run(build_schema)


def _tool(schema: dict[str, Any], name: str) -> dict[str, Any]:
    """Return the named MCP tool schema."""
    tools = schema["tools"]
    tool = next(
        (item for item in tools if item["name"] == name),
        None,
    )
    assert tool is not None, f"{name} is missing from MCP schema"
    return tool


def _search_tool(schema: dict[str, Any]) -> dict[str, Any]:
    """Return the Ancient Phonology search tool schema."""
    return _tool(schema, "ancient_phonology.search")


def test_mcp_tools_artifact_matches_runtime() -> None:
    """The committed MCP tool schema should match the runtime server state."""
    assert MCP_SCHEMA_ARTIFACT.exists(), (
        f"{MCP_SCHEMA_ARTIFACT} is missing. Regenerate it with:\n"
        f"  {REGENERATE_MCP_SCHEMA_COMMAND}"
    )
    expected = (
        json.dumps(_runtime_mcp_schema(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    )
    actual = MCP_SCHEMA_ARTIFACT.read_text(encoding="utf-8")

    assert actual == expected, (
        "docs/mcp/tools.json is out of date. Regenerate it with:\n"
        f"  {REGENERATE_MCP_SCHEMA_COMMAND}"
    )


def test_mcp_search_tool_input_schema_includes_required_fields() -> None:
    """The MCP ancient_phonology.search input schema should include required fields."""
    schema = _runtime_mcp_schema()
    tool = _search_tool(schema)
    input_schema = tool["inputSchema"]

    assert input_schema["required"] == ["request"]
    request_schema = input_schema["$defs"]["McpSearchInput"]
    assert request_schema["required"] == ["query_form"]
    assert {
        "query_form",
        "source_language",
        "dialect_hint",
        "max_candidates",
        "response_language",
    }.issubset(request_schema["properties"])


def test_mcp_search_tool_output_schema_includes_meta_envelope() -> None:
    """The MCP ancient_phonology.search output schema should include ResponseMeta envelope."""
    schema = _runtime_mcp_schema()
    tool = _search_tool(schema)
    output_schema = tool["outputSchema"]

    assert "meta" in output_schema["properties"]
    assert "meta" in output_schema["required"]
    assert output_schema["properties"]["meta"]["$ref"] == "#/$defs/ResponseMeta"


def test_mcp_search_tool_output_schema_includes_buck_references() -> None:
    """MCP search candidates should expose Buck reference annotations."""
    schema = _runtime_mcp_schema()
    tool = _search_tool(schema)
    output_schema = tool["outputSchema"]

    search_hit_schema = output_schema["$defs"]["SearchHit"]
    assert "buck_references" in search_hit_schema["properties"]
    assert (
        search_hit_schema["properties"]["buck_references"]["items"]["$ref"]
        == "#/$defs/BuckReferenceAnnotation"
    )


def test_mcp_buck_tool_schemas_are_registered() -> None:
    """Buck reference tools should be present in the runtime MCP schema."""
    schema = _runtime_mcp_schema()
    tool_names = {tool["name"] for tool in schema["tools"]}

    assert {
        "ancient_phonology.search_buck_rules",
        "ancient_phonology.get_buck_dialect",
        "ancient_phonology.search_buck_glossary",
    }.issubset(tool_names)


def test_mcp_buck_rule_tool_schema_includes_filters_and_review_fields() -> None:
    """Buck rule schema should expose filters and review metadata."""
    schema = _runtime_mcp_schema()
    tool = _tool(schema, "ancient_phonology.search_buck_rules")
    input_schema = tool["inputSchema"]
    output_schema = tool["outputSchema"]

    assert input_schema["required"] == ["request"]
    request_schema = input_schema["$defs"]["McpBuckRuleSearchInput"]
    assert {
        "rule_id",
        "category",
        "dialect",
        "section",
        "source_language",
        "max_results",
    }.issubset(request_schema["properties"])
    assert "metadata" in output_schema["required"]
    metadata_schema = output_schema["$defs"]["McpBuckMetadata"]
    assert {
        "status",
        "review_status",
        "citation_ready",
        "review_note",
    }.issubset(metadata_schema["properties"])
    rule_schema = output_schema["$defs"]["McpBuckRuleInfo"]
    assert {"status", "review_status", "citation_ready"}.issubset(
        rule_schema["properties"]
    )


def test_mcp_buck_dialect_and_glossary_output_schemas_include_review_fields() -> None:
    """Buck dialect and glossary item schemas should include review boundaries."""
    schema = _runtime_mcp_schema()
    dialect_output_schema = _tool(
        schema,
        "ancient_phonology.get_buck_dialect",
    )["outputSchema"]
    glossary_output_schema = _tool(
        schema,
        "ancient_phonology.search_buck_glossary",
    )["outputSchema"]

    dialect_schema = dialect_output_schema["$defs"]["McpBuckDialectInfo"]
    glossary_schema = glossary_output_schema["$defs"]["McpBuckGlossaryEntryInfo"]
    reference_schema = glossary_output_schema["$defs"]["McpBuckReferenceInfo"]

    assert {"status", "review_status", "citation_ready"}.issubset(
        dialect_schema["properties"]
    )
    assert {"status", "review_status", "citation_ready"}.issubset(
        glossary_schema["properties"]
    )
    assert {"section", "page"}.issubset(reference_schema["properties"])
