"""Entry-id and IPA lookup helpers for the search package.

Core lookup functions that do not depend on custom tokenization live here.
Functions requiring tokenization (``resolve_entry_tokens``,
``build_lexicon_map``) are located in ``_tokenization.py`` or the package
root to support consistent tokenization and monkeypatching in tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

from ..ipa_converter import strip_ignored_ipa_combining_marks
from ._types import (
    LexiconEntry,
    LexiconLookup,
    LexiconLookupValue,
    LexiconRecord,
)

IpaIndex: TypeAlias = dict[str, list[str]]


def _get_required_str(
    entry: LexiconEntry,
    key_or_keys: str | list[str],
    error_message: str,
) -> str:
    """Return a stripped string value for the given key(s) or raise ValueError.

    Args:
        entry: The lexicon entry dict to look up.
        key_or_keys: A single key or list of keys to try in order.
        error_message: The error message to raise if no valid value is found.

    Returns:
        The stripped, non-empty string value.

    Raises:
        ValueError: If no valid string value is found for any of the keys.
    """
    keys = [key_or_keys] if isinstance(key_or_keys, str) else key_or_keys
    for key in keys:
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(error_message)


def _entry_id(entry: LexiconEntry) -> str:
    """Return a stable id for a lexicon entry."""
    return _get_required_str(
        entry,
        ["id", "headword"],
        "Lexicon entries must define a non-empty 'id' or 'headword'",
    )


def _lemma_label(entry: LexiconEntry) -> str:
    """Return the display lemma for a lexicon entry."""
    return _get_required_str(
        entry,
        "headword",
        "Lexicon entries must define a non-empty 'headword'",
    )


def _entry_ipa(entry: LexiconEntry) -> str:
    """Return the IPA field for a lexicon entry."""
    return _get_required_str(
        entry,
        "ipa",
        "Lexicon entries must define a non-empty 'ipa'",
    )


def _normalize_ipa_lookup_key(ipa_text: str) -> str:
    """Return the accent-insensitive key used for exact IPA lookup."""
    return strip_ignored_ipa_combining_marks(ipa_text.strip())


def _build_entry_lookup(lexicon: Sequence[LexiconEntry]) -> dict[str, LexiconEntry]:
    """Build an entry-id lookup without tokenizing IPA strings."""
    result: dict[str, LexiconEntry] = {}
    for entry in lexicon:
        entry_id = _entry_id(entry)
        if entry_id in result:
            raise ValueError(
                f"Duplicate entry ID {entry_id!r} in lexicon; "
                f"existing entry: {result[entry_id]!r}, "
                f"new duplicate entry: {entry!r}"
            )
        result[entry_id] = entry
    return result


def _lookup_entry(record_or_entry: LexiconLookupValue) -> LexiconEntry:
    """Return the underlying lexicon entry from a lookup value."""
    if isinstance(record_or_entry, LexiconRecord):
        return record_or_entry.entry
    return record_or_entry


def build_ipa_index(lexicon_lookup: LexiconLookup) -> IpaIndex:
    """Build a normalized IPA-to-entry-id index from a lexicon lookup.

    ``_entry_ipa`` returns a stripped, non-empty value via ``_get_required_str``.
    This function removes ignored IPA accent marks so exact lookup remains
    stable for accented and unaccented user input.
    """
    ipa_index: IpaIndex = {}
    for entry_id, record_or_entry in lexicon_lookup.items():
        normalized_ipa = _normalize_ipa_lookup_key(
            _entry_ipa(_lookup_entry(record_or_entry))
        )
        ipa_index.setdefault(
            normalized_ipa,
            [],
        ).append(entry_id)
    return ipa_index


def _inject_exact_ipa_matches(
    query_ipa: str,
    candidate_ids: list[str],
    lexicon_lookup: LexiconLookup,
    *,
    ipa_index: IpaIndex | None = None,
    limit: int | None = None,
) -> list[str]:
    """Prepend lexicon entries whose IPA exactly matches the query.

    Ensures that exact phonological matches always reach Stage 2, even when
    seed ranking pushes them past the stage2_limit cutoff, while preserving
    any caller-provided candidate cap.

    Args:
        query_ipa: IPA string to match exactly.
        candidate_ids: Ordered candidate entry IDs selected by earlier stages.
        lexicon_lookup: Mapping from entry ID to lexicon entries or cached
            ``LexiconRecord`` values.
        ipa_index: Optional mapping from normalized IPA strings to ordered
            entry IDs. When omitted, the lookup is scanned for backward
            compatibility with direct callers.
        limit: Optional maximum number of merged candidate IDs to return.

    Returns:
        Candidate IDs with exact IPA matches prepended, capped by ``limit`` when
        provided. Non-positive limits return no candidates.

    Raises:
        ValueError: If a lexicon entry lacks a valid ``ipa`` field.
    """
    normalized_query_ipa = _normalize_ipa_lookup_key(query_ipa)
    candidate_set = set(candidate_ids)
    if ipa_index is None:
        exact_ids = [
            entry_id
            for entry_id, record_or_entry in lexicon_lookup.items()
            if entry_id not in candidate_set
            and _normalize_ipa_lookup_key(_entry_ipa(_lookup_entry(record_or_entry)))
            == normalized_query_ipa
        ]
    else:
        exact_ids = [
            entry_id
            for entry_id in ipa_index.get(normalized_query_ipa, [])
            if entry_id not in candidate_set
        ]
    merged_ids = exact_ids + candidate_ids if exact_ids else candidate_ids
    if limit is None:
        return merged_ids
    if limit <= 0:
        return []
    return merged_ids[:limit]
