"""MCP search tool definitions."""

from __future__ import annotations

from typing import Any, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field

from api._models import ResponseMeta, SearchHit


class McpSearchInput(BaseModel):
    """Validated input for the Ancient Phonology MCP search tool."""

    model_config = ConfigDict(frozen=True)

    query_form: str = Field(
        min_length=1, description="Greek word to search for."
    )
    source_language: str = Field(
        default="ancient_greek",
        description="Language profile used for phonological search.",
    )
    dialect_hint: str | None = Field(
        default=None,
        description=(
            "Dialect hint for IPA conversion. ``None`` (the MCP default) means "
            "'defer to the language profile default', which keeps the tool "
            "schema language-agnostic; the REST API uses 'attic' as its "
            "default because Ancient Greek is currently the only profile."
        ),
    )
    max_candidates: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of candidates to return.",
    )
    response_language: Literal["en", "ja"] = Field(
        default="en",
        description="Response prose language.",
    )


class McpSearchOutput(BaseModel):
    """Structured MCP response for Ancient Phonology search."""

    candidates: list[SearchHit] = Field(description="Ranked search candidates.")
    query: str = Field(description="Original query string.")
    query_ipa: str = Field(description="IPA transcription computed for the query.")
    query_mode: Literal["Full-form", "Short-query", "Partial-form"] = Field(
        description="Input classification used by the search engine."
    )
    truncated: bool = Field(
        default=False,
        description="True when candidate annotation was truncated.",
    )
    meta: ResponseMeta = Field(
        description="Version, request, and reproducibility metadata."
    )

    @classmethod
    def output_schema(cls) -> dict[str, object]:
        """Return the JSON schema for this output model."""
        return cls.model_json_schema()


def register_search_tool(app: FastMCP) -> None:
    """Register the Ancient Phonology search tool on ``app``."""

    @app.tool("ancient_phonology.search")
    def ancient_phonology_search(request: McpSearchInput) -> dict[str, Any]:
        """Search Ancient Greek forms using the Proteus phonology engine."""
        # Local import avoids a circular dependency: _search_adapter imports
        # McpSearchInput/McpSearchOutput from this module at load time.
        from mcp_server._search_adapter import _run_search_for_mcp

        output = _run_search_for_mcp(request)
        # Return the JSON-serializable dict so FastMCP forwards the exact
        # payload shape that ``call_tool`` consumers (and our tests) parse.
        return output.model_dump(mode="json")


__all__ = ["McpSearchInput", "McpSearchOutput", "register_search_tool"]
