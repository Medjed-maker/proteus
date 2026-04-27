"""Short-query quality filtering and ranking helpers.

These helpers take a ``lookup_tokens`` callable so the caller can provide
a token resolution strategy. This typically uses ``resolve_entry_tokens``
from ``phonology.search._tokenization``, which ensures consistent results
across the search and scoring modules.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ._constants import _SHORT_QUERY_CONFIDENCE_THRESHOLD
from ._overlap import _is_exact_token_match
from ._types import LexiconLookup, LexiconLookupValue, SearchResult

_LookupTokens = Callable[[LexiconLookupValue], tuple[str, ...]]


def _get_lemma_tokens(
    candidate_id: str | None,
    lexicon_lookup: LexiconLookup,
    lookup_tokens: _LookupTokens,
) -> tuple[str, ...]:
    """Return lemma tokens for a candidate if available, else empty tuple."""
    if candidate_id is not None and candidate_id in lexicon_lookup:
        return lookup_tokens(lexicon_lookup[candidate_id])
    return ()


def _filter_short_query_results(
    query_tokens: Sequence[str],
    results: list[SearchResult],
    lexicon_lookup: LexiconLookup,
    lookup_tokens: _LookupTokens,
) -> list[SearchResult]:
    """Keep only exact, rule-supported, or sufficiently similar short-query hits.

    Exact token matches and explicit rule-supported hits intentionally bypass
    the short-query confidence floor so trustworthy matches are not dropped
    just because the aggregate similarity score is conservative.
    """
    filtered: list[SearchResult] = []
    for result in results:
        lemma_tokens = _get_lemma_tokens(result.entry_id, lexicon_lookup, lookup_tokens)
        if _is_exact_token_match(query_tokens, lemma_tokens) or result.applied_rules:
            filtered.append(result)
            continue
        if result.confidence >= _SHORT_QUERY_CONFIDENCE_THRESHOLD:
            filtered.append(result)
    return filtered


def _rank_short_query_results(
    query_tokens: Sequence[str],
    results: list[SearchResult],
    lexicon_lookup: LexiconLookup,
    lookup_tokens: _LookupTokens,
) -> list[SearchResult]:
    """Order short-query hits so exact matches win before confidence ranking."""

    def _sort_key(result: SearchResult) -> tuple[bool, float, bool, str]:
        lemma_tokens = _get_lemma_tokens(result.entry_id, lexicon_lookup, lookup_tokens)
        return (
            not _is_exact_token_match(query_tokens, lemma_tokens),
            -result.confidence,
            not bool(result.applied_rules),
            result.lemma,
        )

    return sorted(results, key=_sort_key)
