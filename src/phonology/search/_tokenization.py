"""Shared tokenization utilities for search and indexing."""

from __future__ import annotations

from collections.abc import Iterable

from ..core.ipa import tokenize_ipa as tokenize_ipa_with_inventory
from ._lookup import (
    _entry_ipa,
    _lookup_entry,
)
from ._types import (
    LexiconLookupValue,
    LexiconRecord,
)


def tokenize_ipa(ipa_text: str) -> list[str]:
    """Backward-compatible one-argument tokenizer seam for generic search tests."""
    return tokenize_ipa_with_inventory(ipa_text, phone_inventory=())


def tokenize_for_inventory(
    ipa_text: str,
    phone_inventory: Iterable[str] | None = None,
) -> list[str]:
    """Tokenize IPA text using a phone inventory when provided.

    Args:
        ipa_text: Compact or space-separated IPA text to tokenize.
        phone_inventory: Iterable of IPA phone strings used for greedy
            longest-match tokenization. When ``None``, tokenization falls back
            to literal character tokens via the backward-compatible search
            shim. This parameter is materialized to a tuple internally; an
            empty tuple triggers the same fallback behavior as ``None``, so
            callers providing an empty iterable will get ``tokenize_ipa``
            behavior.

    Returns:
        A list of IPA token strings.
    """
    if phone_inventory is None:
        return tokenize_ipa(ipa_text)
    # Materialize iterable to tuple to avoid consuming single-use iterables
    phones = tuple(phone_inventory)
    if not phones:
        return tokenize_ipa(ipa_text)
    return tokenize_ipa_with_inventory(ipa_text, phone_inventory=phones)


def resolve_entry_tokens(
    record_or_entry: LexiconLookupValue,
    *,
    phone_inventory: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Return cached IPA tokens when available, tokenizing only as a fallback.

    This helper provides a unified way to access entry tokens across the
    search and scoring modules, ensuring that cached ``LexiconRecord``
    tokens are preferred but custom ``phone_inventory`` can still be applied
    during fallback tokenization.
    """
    if isinstance(record_or_entry, LexiconRecord):
        if record_or_entry.ipa_tokens is not None:
            return record_or_entry.ipa_tokens
        entry = record_or_entry.entry
    else:
        entry = _lookup_entry(record_or_entry)
    return tuple(tokenize_for_inventory(_entry_ipa(entry), phone_inventory))
