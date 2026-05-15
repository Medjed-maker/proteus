"""Data containers and lazy-resolved dependency bundle for the search pipeline.

This module holds the pure ``NamedTuple`` and ``Protocol`` definitions that the
search pipeline passes around, plus the ``_LazySearchDependencies`` helper that
defers expensive index construction until a specific candidate-selection path
actually needs it.

Test-seam policy: ``_LazySearchDependencies`` resolves its builders via late
``from . import ...`` inside each method body so that ``monkeypatch.setattr``
on ``phonology.search._build_lexicon_map_for_inventory`` (used by
``tests/_helpers/fakes.py``) keeps working after this split. Adding a new
builder dependency? Use the same late-import pattern.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import NamedTuple, Protocol

from ._lookup import IpaIndex, _build_entry_lookup, build_ipa_index
from ._types import (
    LexiconEntry,
    LexiconLookup,
    LexiconMap,
    PartialQueryPattern,
    PartialQueryTokens,
    PhoneInventory,
    QueryMode,
    SearchResult,
)


class IpaConverter(Protocol):
    """Callable interface for converting source text into IPA.

    Implementations provide the public converter contract used by search query
    preparation and partial-query fragment conversion. Converters are expected
    to be deterministic for a given text and dialect and to avoid observable
    side effects.

    Args:
        text: Input text to convert.
        dialect: Dialect or conversion mode to apply.

    Returns:
        Converted IPA string.

    Raises:
        Exception: Implementations may raise converter-specific exceptions for
            unsupported dialects or invalid input.
    """

    def __call__(self, text: str, *, dialect: str) -> str: ...


class PreparedQueryIpa(NamedTuple):
    """Query classification, normalization, and IPA data for one search.

    Args:
        query_mode: Search mode selected for the query.
        normalized_query: Normalized search string before IPA conversion.
        partial_query: Parsed partial-query pattern, or ``None`` for
            non-partial searches.
        query_ipa: IPA-normalized query string used by search stages.
        partial_query_tokens: Tokenized partial-query fragments, or ``None``
            when the query is not partial-form.
    """

    query_mode: QueryMode
    normalized_query: str
    partial_query: PartialQueryPattern | None
    query_ipa: str
    partial_query_tokens: PartialQueryTokens | None


class _FallbackLimits(NamedTuple):
    """Effective fallback exploration caps for one search."""

    similarity: int
    unigram: int


class _FinalizationResult(NamedTuple):
    """Finalized search results plus debug-only performance counters."""

    results: list[SearchResult]
    annotated_count: int
    returned_count: int
    truncated: bool = False


class SearchExecutionResult(NamedTuple):
    """Search return value with result metadata for API callers.

    Args:
        results: Ranked search results returned to the caller.
        truncated: Whether the result set was truncated before all candidates
            could be returned.
        query_ipa: IPA-normalized query string used for the search.
        query_mode: Search mode used, for example ``"Full-form"``.
    """

    results: list[SearchResult]
    truncated: bool = False
    query_ipa: str = ""
    query_mode: QueryMode = "Full-form"


class _LazySearchDependencies:
    """Lazily construct search lookups needed by specific candidate paths.

    Builders that tests may monkeypatch on ``phonology.search`` are resolved
    via late ``from . import ...`` inside each method so the patches take
    effect after this split.
    """

    def __init__(
        self,
        *,
        lexicon: Sequence[LexiconEntry],
        prebuilt_lexicon_map: LexiconMap | None,
        prebuilt_ipa_index: IpaIndex | None,
        phone_inventory: PhoneInventory,
        dialect_skeleton_builders: Iterable[Callable[[list[str]], list[str]]]
        | None = None,
    ) -> None:
        self._lexicon = lexicon
        self._lexicon_map = prebuilt_lexicon_map
        self._prebuilt_ipa_index = prebuilt_ipa_index
        self._phone_inventory = phone_inventory
        self._dialect_skeleton_builders = dialect_skeleton_builders
        self._entry_lookup: dict[str, LexiconEntry] | None = None
        self._ipa_index: IpaIndex | None = None

    def entry_lookup(self) -> dict[str, LexiconEntry]:
        """Return an id-to-entry lookup, building it only when needed."""
        if self._entry_lookup is None:
            self._entry_lookup = _build_entry_lookup(self._lexicon)
        return self._entry_lookup

    def lexicon_lookup(self) -> LexiconLookup:
        """Return the cheapest lookup available for scoring or exact matching."""
        if self._lexicon_map is not None:
            return self._lexicon_map
        return self.entry_lookup()

    def tokenized_lexicon_map(self) -> LexiconMap:
        """Return the tokenized lexicon map, building it only when needed.

        Resolves ``_build_lexicon_map_for_inventory`` lazily via the package
        namespace so test monkeypatches on ``phonology.search`` apply here.
        """
        if self._lexicon_map is None:
            from . import _build_lexicon_map_for_inventory

            self._lexicon_map = _build_lexicon_map_for_inventory(
                self._lexicon,
                phone_inventory=self._phone_inventory,
            )
        return self._lexicon_map

    def ipa_index(self) -> IpaIndex:
        """Return the exact IPA index, building it only when needed."""
        if self._prebuilt_ipa_index is not None:
            return self._prebuilt_ipa_index
        if self._ipa_index is None:
            self._ipa_index = build_ipa_index(self.lexicon_lookup())
        return self._ipa_index

    @property
    def phone_inventory(self) -> PhoneInventory:
        """Return the resolved phone inventory shared by this search."""
        return self._phone_inventory

    @property
    def dialect_skeleton_builders(
        self,
    ) -> Iterable[Callable[[list[str]], list[str]]] | None:
        """Return the optional dialect skeleton builders shared by this search."""
        return self._dialect_skeleton_builders
