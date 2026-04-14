"""Mode-aware quality filtering and annotation candidate selection.

Contains functions that apply post-scoring quality controls based on
the query mode (short-query, partial-form, or full-form).
"""

from __future__ import annotations

from collections.abc import Sequence

from ._overlap import _is_exact_token_match
from ._partial import _filter_and_rank_partial_results, _match_partial_query
from ._quality import _filter_short_query_results
# NOTE: _resolve_entry_tokens uses import-time binding for tokenize_ipa,
# while __init__._lookup_entry_tokens uses module-level (monkeypatch-sensitive)
# binding. Both produce identical results for LexiconRecord inputs (which carry
# pre-tokenized IPA), but diverge for bare LexiconEntry dicts in test stubs.
# This is intentional: callers in __init__ that pass through the Short-query
# quality ranking use _lookup_entry_tokens explicitly to preserve test patching.
from ._scoring import _resolve_entry_tokens
from ._types import (
    LexiconLookup,
    PartialQueryTokens,
    QueryMode,
    SearchResult,
)


def _apply_mode_quality_filter(
    query_mode: QueryMode,
    query_tokens: Sequence[str],
    partial_query: PartialQueryTokens | None,
    results: list[SearchResult],
    lexicon_lookup: LexiconLookup,
) -> list[SearchResult]:
    """Apply mode-aware post-annotation quality controls."""
    if query_mode == "Short-query":
        return _filter_short_query_results(
            query_tokens, results, lexicon_lookup, _resolve_entry_tokens
        )
    if query_mode == "Partial-form":
        if partial_query is None:
            raise ValueError("partial query metadata is required for Partial-form searches")
        return _filter_and_rank_partial_results(partial_query, results, lexicon_lookup)
    return results


def _select_annotation_candidates(
    query_mode: QueryMode,
    query_tokens: Sequence[str],
    partial_query: PartialQueryTokens | None,
    ranked_scored: list[SearchResult],
    lexicon_lookup: LexiconLookup,
    annotation_limit: int,
) -> list[SearchResult]:
    """Select a bounded, mode-aware annotation window from scored candidates."""
    if len(ranked_scored) <= annotation_limit:
        return list(ranked_scored)
    if query_mode == "Short-query":
        return _rank_short_query_annotation_candidates(
            query_tokens,
            ranked_scored,
            lexicon_lookup,
        )[:annotation_limit]
    if query_mode == "Partial-form":
        if partial_query is None:
            raise ValueError("partial query metadata is required for Partial-form searches")

        def _partial_form_sort_key(
            item: tuple[int, SearchResult]
        ) -> tuple[bool, int, int, float, str, int]:
            index, result = item
            candidate_id = result.entry_id
            lemma_tokens: tuple[str, ...] = ()
            if candidate_id is not None and candidate_id in lexicon_lookup:
                lemma_tokens = _resolve_entry_tokens(lexicon_lookup[candidate_id])
            match_info = _match_partial_query(partial_query, lemma_tokens)
            return (
                not match_info.full_match,  # primary: deprioritize non-full matches
                -match_info.matched_fragments,  # prefer more matched fragments (desc)
                -match_info.overlap_score,  # prefer higher overlap score (desc)
                -result.confidence,  # prefer higher confidence (desc)
                result.lemma,  # tie-break by lemma lexicographically (asc)
                index,  # stable tie-break by original index (asc)
            )

        return [
            result
            for _index, result in sorted(enumerate(ranked_scored), key=_partial_form_sort_key)[
                :annotation_limit
            ]
        ]
    return ranked_scored[:annotation_limit]


def _rank_short_query_annotation_candidates(
    query_tokens: Sequence[str],
    ranked_scored: list[SearchResult],
    lexicon_lookup: LexiconLookup,
) -> list[SearchResult]:
    """Rank short-query candidates before expensive rule annotation."""

    def _short_query_sort_key(item: tuple[int, SearchResult]) -> tuple[bool, float, str, int]:
        index, result = item
        candidate_id = result.entry_id
        lemma_tokens: tuple[str, ...] = ()
        if candidate_id is not None and candidate_id in lexicon_lookup:
            lemma_tokens = _resolve_entry_tokens(lexicon_lookup[candidate_id])
        return (
            not _is_exact_token_match(query_tokens, lemma_tokens),
            -result.confidence,
            result.lemma,
            index,
        )

    return [
        result for _index, result in sorted(enumerate(ranked_scored), key=_short_query_sort_key)
    ]
