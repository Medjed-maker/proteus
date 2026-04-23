"""Mode-aware quality filtering and annotation candidate selection.

Contains functions that apply post-scoring quality controls based on
the query mode (short-query, partial-form, or full-form).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ._constants import _PARTIAL_QUERY_CONFIDENCE_THRESHOLD, _SHORT_QUERY_CONFIDENCE_THRESHOLD
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
    PartialMatchInfo,
    QueryMode,
    SearchResult,
)


@dataclass(frozen=True, slots=True)
class _AnnotationCandidateFeatures:
    """Cheap candidate metadata used to prioritize annotation work."""

    exact_match: bool = False
    meets_confidence: bool = False
    token_count_distance: int = 0
    partial_match: PartialMatchInfo | None = None


_PartialFormSortKey = tuple[bool, int, int, bool, int, float, str, int]
_PartialFormCandidate = tuple[_PartialFormSortKey, SearchResult]
_ShortQueryPrimaryCandidateKey = tuple[bool, bool, int, float, str, int]
_ShortQueryPrimaryCandidate = tuple[_ShortQueryPrimaryCandidateKey, SearchResult]
_ShortQueryExploratoryCandidateKey = tuple[int, float, str, int]
_ShortQueryExploratoryCandidate = tuple[_ShortQueryExploratoryCandidateKey, SearchResult]


def _merge_primary_and_exploratory_candidates(
    primary_candidates: list[SearchResult],
    exploratory_candidates: list[SearchResult],
    *,
    annotation_limit: int,
) -> list[SearchResult]:
    """Return a bounded candidate window within ``annotation_limit``.

    Primary candidates (exact matches, confidence-threshold hits, partial
    overlaps) fill the window first.  Remaining slots are filled from the
    exploratory candidates (token-count-proximity order), giving low-confidence
    rule-only candidates a chance to be annotated without displacing stronger
    primary hits.
    """
    if len(primary_candidates) + len(exploratory_candidates) <= annotation_limit:
        return primary_candidates + exploratory_candidates

    if len(primary_candidates) >= annotation_limit:
        return primary_candidates[:annotation_limit]

    bounded_primary = list(primary_candidates)
    remaining_slots = annotation_limit - len(bounded_primary)
    if remaining_slots <= 0:
        return bounded_primary
    return bounded_primary + exploratory_candidates[:remaining_slots]


def _short_query_candidate_features(
    query_tokens: Sequence[str],
    result: SearchResult,
    lexicon_lookup: LexiconLookup,
) -> _AnnotationCandidateFeatures:
    """Build cheap pre-annotation features for a short-query candidate."""
    candidate_id = result.entry_id
    lemma_tokens: tuple[str, ...] = ()
    if candidate_id is not None and candidate_id in lexicon_lookup:
        lemma_tokens = _resolve_entry_tokens(lexicon_lookup[candidate_id])
    return _AnnotationCandidateFeatures(
        exact_match=_is_exact_token_match(query_tokens, lemma_tokens),
        meets_confidence=result.confidence >= _SHORT_QUERY_CONFIDENCE_THRESHOLD,
        token_count_distance=abs(len(lemma_tokens) - len(query_tokens)),
    )


def _partial_form_candidate_features(
    partial_query: PartialQueryTokens,
    result: SearchResult,
    lexicon_lookup: LexiconLookup,
) -> _AnnotationCandidateFeatures:
    """Build cheap pre-annotation features for a partial-form candidate."""
    candidate_id = result.entry_id
    lemma_tokens: tuple[str, ...] = ()
    if candidate_id is not None and candidate_id in lexicon_lookup:
        lemma_tokens = _resolve_entry_tokens(lexicon_lookup[candidate_id])
    match_info = _match_partial_query(partial_query, lemma_tokens)
    lemma_token_count = len(lemma_tokens)
    context_token_count = len(partial_query.left_tokens) + len(partial_query.right_tokens)
    token_count_distance = abs(lemma_token_count - context_token_count)
    return _AnnotationCandidateFeatures(
        meets_confidence=result.confidence >= _PARTIAL_QUERY_CONFIDENCE_THRESHOLD,
        token_count_distance=token_count_distance,
        partial_match=match_info,
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
    """Select a bounded, mode-aware annotation window from scored candidates.

    **Bounded-window policy**: at most ``annotation_limit`` candidates are
    returned; candidates beyond that cap are never annotated.  This is an
    intentional performance trade-off over the previous multi-batch approach
    (which continued annotation until ``max_results`` were found): annotation
    is O(n × alignment), so unbounded continuation becomes expensive for
    large lexicons.

    Within the window, primary candidates (exact token match, above-threshold
    confidence, or partial overlap) fill the window first; remaining slots are
    filled by exploratory candidates sorted by token-count proximity.  This
    preserves reachability for rule-supported candidates that would otherwise
    rank too low on raw confidence alone.
    """
    if len(ranked_scored) <= annotation_limit:
        if query_mode == "Short-query":
            return _rank_short_query_annotation_candidates(
                query_tokens,
                ranked_scored,
                lexicon_lookup,
                annotation_limit=annotation_limit,
            )
        return list(ranked_scored)
    if query_mode == "Short-query":
        return _rank_short_query_annotation_candidates(
            query_tokens,
            ranked_scored,
            lexicon_lookup,
            annotation_limit=annotation_limit,
        )
    if query_mode == "Partial-form":
        if partial_query is None:
            raise ValueError("partial query metadata is required for Partial-form searches")

        primary_candidates: list[_PartialFormCandidate] = []
        exploratory_candidates: list[_PartialFormCandidate] = []

        for index, result in enumerate(ranked_scored):
            features = _partial_form_candidate_features(partial_query, result, lexicon_lookup)
            match_info = features.partial_match or PartialMatchInfo(False, 0, 0)
            sort_key = (
                not match_info.full_match,
                -match_info.matched_fragments,
                -match_info.overlap_score,
                not features.meets_confidence,
                features.token_count_distance,
                -result.confidence,
                result.lemma,
                index,
            )
            if match_info.overlap_score > 0 or match_info.full_match or features.meets_confidence:
                primary_candidates.append((sort_key, result))
            else:
                exploratory_candidates.append((sort_key, result))

        primary_ranked = [result for _sort_key, result in sorted(primary_candidates, key=lambda item: item[0])]
        exploratory_ranked = [
            result for _sort_key, result in sorted(exploratory_candidates, key=lambda item: item[0])
        ]
        return _merge_primary_and_exploratory_candidates(
            primary_ranked,
            exploratory_ranked,
            annotation_limit=annotation_limit,
        )
    return ranked_scored[:annotation_limit]


def _rank_short_query_annotation_candidates(
    query_tokens: Sequence[str],
    ranked_scored: list[SearchResult],
    lexicon_lookup: LexiconLookup,
    *,
    annotation_limit: int,
) -> list[SearchResult]:
    """Rank short-query candidates before expensive rule annotation.

    Splits candidates into primary (exact token match or above-threshold
    confidence) and exploratory buckets, then merges them into a window of at
    most ``annotation_limit`` entries.  Primary candidates are ordered by
    exact-match priority then descending confidence; exploratory candidates by
    ascending token-count distance.  Remaining slots after primary candidates
    are filled from exploratory, so low-confidence rule-only hits remain
    reachable without displacing stronger candidates.
    """
    primary_candidates: list[_ShortQueryPrimaryCandidate] = []
    exploratory_candidates: list[_ShortQueryExploratoryCandidate] = []

    for index, result in enumerate(ranked_scored):
        features = _short_query_candidate_features(query_tokens, result, lexicon_lookup)
        if features.exact_match or features.meets_confidence:
            primary_candidates.append(
                (
                    (
                        not features.exact_match,
                        not features.meets_confidence,
                        features.token_count_distance,
                        -result.confidence,
                        result.lemma,
                        index,
                    ),
                    result,
                )
            )
            continue
        exploratory_candidates.append(
            (
                (
                    features.token_count_distance,
                    -result.confidence,
                    result.lemma,
                    index,
                ),
                result,
            )
        )

    primary_ranked = [result for _sort_key, result in sorted(primary_candidates, key=lambda item: item[0])]
    exploratory_ranked = [
        result for _sort_key, result in sorted(exploratory_candidates, key=lambda item: item[0])
    ]
    return _merge_primary_and_exploratory_candidates(
        primary_ranked,
        exploratory_ranked,
        annotation_limit=annotation_limit,
    )
