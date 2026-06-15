"""Shared Buck reference-data presentation helpers.

Pure conversion utilities and review-note constants shared by the REST layer
(``_buck_routes``), the MCP tool layer (``mcp_server.tools.buck``), and the
search-annotation layer (``_buck_annotations``). This module holds no
FastAPI/Pydantic dependency so every Buck-facing surface can reuse a single
source of truth without coupling to a particular API framework.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SUPPORTED_BUCK_LANGUAGE = "ancient_greek"

PROVISIONAL_BUCK_REVIEW_NOTE = (
    "Buck reference data is provisional, not expert-reviewed, and must not be "
    "treated as citation-ready scholarly evidence."
)
CITATION_READY_BUCK_REVIEW_NOTE = (
    "Buck reference data is marked citation-ready; verify the specific context "
    "before scholarly citation."
)


def review_note_for(citation_ready: bool) -> str:
    """Return the review note matching the data's citation-readiness."""
    return (
        CITATION_READY_BUCK_REVIEW_NOTE
        if citation_ready
        else PROVISIONAL_BUCK_REVIEW_NOTE
    )


def inscription_number_as_json(
    inscription_no: str | tuple[int, ...] | None,
) -> str | list[int] | None:
    """Render a Buck inscription number for JSON output (tuple -> list)."""
    if isinstance(inscription_no, tuple):
        return list(inscription_no)
    return inscription_no


def json_mapping(raw_mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively normalize a mapping into JSON-friendly built-ins."""
    return {key: json_value(value) for key, value in raw_mapping.items()}


def json_value(value: Any) -> Any:
    """Normalize one value into JSON-friendly built-ins (tuples -> lists)."""
    if isinstance(value, Mapping):
        return json_mapping(value)
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value
