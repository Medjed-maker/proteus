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


def _search_tool(schema: dict[str, Any]) -> dict[str, Any]:
    """Return the Ancient Phonology search tool schema."""
    tools = schema["tools"]
    tool = next(
        (item for item in tools if item["name"] == "ancient_phonology.search"),
        None,
    )
    assert tool is not None, "ancient_phonology.search is missing from MCP schema"
    return tool


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
