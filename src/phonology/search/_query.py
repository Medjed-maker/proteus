"""Query mode classification, normalization, and partial-query parsing.

Only helpers that do NOT call ``to_ipa``/``tokenize_ipa`` live here.
``_tokenize_partial_query`` stays in ``phonology.search.__init__`` because
it converts partial-query fragments through ``to_ipa``/``tokenize_ipa``,
which tests monkeypatch at ``search_module``-module level.
"""

from __future__ import annotations

import unicodedata

from .._phones import VOWEL_PHONES
from ._constants import _PARTIAL_QUERY_MARKERS
from ._types import PartialQueryPattern, PartialQueryShape, QueryMode


def _classify_non_partial_query(query: str) -> QueryMode:
    """Classify a query that is already known not to be partial-form."""
    normalized = unicodedata.normalize("NFC", query.strip())
    stripped = normalized.replace(" ", "").replace(".", "").replace("_", "")
    if len(stripped) <= 3:
        return "Short-query"
    return "Full-form"


def classify_query_mode(query: str) -> QueryMode:
    """Classify the input shape for mode-aware search behavior.

    Args:
        query: Raw search query string.

    Returns:
        "Partial-form" if the query contains exactly one wildcard marker,
        "Short-query" if the stripped length is <= 3, "Full-form" otherwise.

    Raises:
        ValueError: If the query is empty or whitespace-only.
    """
    if not query.strip():
        raise ValueError("query must be a non-empty string")
    partial_query = _parse_partial_query(query)
    if partial_query is not None:
        return "Partial-form"
    return _classify_non_partial_query(query)


def _normalize_query_with_pattern(
    query: str, partial_query: PartialQueryPattern | None
) -> str:
    """Normalize a query given its already-parsed partial-query pattern.

    Internal helper used by :func:`normalize_query_for_search` (which parses
    the query itself) and by :func:`search` (which can reuse an existing
    parse result to avoid re-scanning the query string). Keeping the two
    entry points sharing this body ensures the public and internal
    normalization stay in lockstep.
    """
    if partial_query is None:
        return query.strip()
    return f"{partial_query.left_fragment}{partial_query.right_fragment}".strip()


def normalize_query_for_search(query: str) -> str:
    """Remove search markers that should not be passed into IPA conversion.

    Malformed partial queries are tolerated and a cleaned fallback is returned.
    """
    try:
        partial_query = _parse_partial_query(query)
    except ValueError:
        partial_query = None
    return _normalize_query_with_pattern(query, partial_query)


def prepare_query(query: str) -> tuple[QueryMode, str]:
    """Classify and normalize a query in a single pass.

    Combines the work of :func:`classify_query_mode` and
    :func:`normalize_query_for_search` while calling
    ``_parse_partial_query`` only once (instead of twice when the two
    functions are used independently).

    Args:
        query: Raw search query string.

    Returns:
        A ``(query_mode, normalized_query)`` pair.

    Raises:
        ValueError: If the query is empty/whitespace-only or contains
            multiple wildcard markers.
    """
    if not query.strip():
        raise ValueError("query must be a non-empty string")
    partial_query = _parse_partial_query(query)
    if partial_query is not None:
        mode: QueryMode = "Partial-form"
    else:
        mode = _classify_non_partial_query(query)
    normalized = _normalize_query_with_pattern(query, partial_query)
    return mode, normalized


def _parse_partial_query(query: str) -> PartialQueryPattern | None:
    """Parse a query containing a single wildcard marker."""
    normalized = unicodedata.normalize("NFC", query.strip())
    marker_positions = [
        index for index, char in enumerate(normalized) if char in _PARTIAL_QUERY_MARKERS
    ]
    if not marker_positions:
        return None
    if len(marker_positions) > 1:
        raise ValueError("partial-form queries may contain only one wildcard marker")

    marker_index = marker_positions[0]
    left_fragment = normalized[:marker_index].strip()
    right_fragment = normalized[marker_index + 1 :].strip()
    if not left_fragment and not right_fragment:
        raise ValueError("query must be a non-empty string")
    if left_fragment and right_fragment:
        shape: PartialQueryShape = "infix"
    elif left_fragment:
        shape = "prefix"
    else:
        shape = "suffix"
    return PartialQueryPattern(
        shape=shape,
        left_fragment=left_fragment,
        right_fragment=right_fragment,
    )


def _extract_consonant_skeleton(tokens: list[str]) -> list[str]:
    """Drop vowels from a tokenized IPA sequence to form a consonant skeleton."""
    return [token for token in tokens if token not in VOWEL_PHONES]
