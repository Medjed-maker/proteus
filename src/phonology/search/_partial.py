"""Partial (wildcard) query matching and candidate selection.

Contains functions for matching partial queries against lemma tokens,
selecting seed candidates for partial queries, and filtering/ranking
partial results.

``_select_partial_token_fallback_candidates`` remains in ``__init__``
because it calls ``_match_partial_query`` by bare name, which tests
patch via ``search_module._match_partial_query``.
"""

from __future__ import annotations

from collections.abc import Sequence

from ._constants import (
    _MIN_PARTIAL_STAGE2_CANDIDATES,
    _PARTIAL_QUERY_CONFIDENCE_THRESHOLD,
    _partial_candidate_limit,
)
from ._overlap import (
    _collect_fragment_matches,
    _leading_overlap_length,
    _trailing_overlap_length,
)
from ._scoring import _resolve_entry_tokens
from ._types import (
    LexiconLookup,
    LexiconMap,
    PartialMatchInfo,
    PartialQueryTokens,
    SearchResult,
)


def _match_partial_query(
    partial_query: PartialQueryTokens,
    lemma_tokens: Sequence[str],
) -> PartialMatchInfo:
    """Return fragment-aware match information for a partial query.

    For ``prefix`` and ``suffix`` shapes the match is anchored to the
    lemma edge; the return value reflects the leading/trailing overlap
    of the fragment against that edge.

    For the ``infix`` shape (both left and right fragments present),
    the match must satisfy an **ordering invariant**: the left fragment
    must appear earlier in the lemma token stream than the right
    fragment, non-overlapping (right_start ≥ left_start + left_overlap).
    If both fragments occur in the lemma but only in the wrong order,
    the result is a hard non-match (score zero). If only one fragment
    occurs anywhere in the lemma, that one-fragment match is returned
    as an exploratory partial — the other fragment's absence is not a
    rejection, only a downgrade.
    """
    if partial_query.shape == "prefix":
        overlap = _leading_overlap_length(partial_query.left_tokens, lemma_tokens)
        return PartialMatchInfo(
            full_match=overlap == len(partial_query.left_tokens) and len(partial_query.left_tokens) > 0,
            matched_fragments=int(overlap > 0),
            overlap_score=overlap,
        )
    if partial_query.shape == "suffix":
        overlap = _trailing_overlap_length(partial_query.right_tokens, lemma_tokens)
        return PartialMatchInfo(
            full_match=overlap == len(partial_query.right_tokens) and len(partial_query.right_tokens) > 0,
            matched_fragments=int(overlap > 0),
            overlap_score=overlap,
        )

    # Infix shape. Collect every valid starting offset + overlap length
    # for each fragment in one pass, then pick the best *in-order* pair.
    left_matches = _collect_fragment_matches(partial_query.left_tokens, lemma_tokens)
    right_matches = _collect_fragment_matches(partial_query.right_tokens, lemma_tokens)
    left_len = len(partial_query.left_tokens)
    right_len = len(partial_query.right_tokens)

    best_pair: PartialMatchInfo | None = None
    for left_start, left_overlap in left_matches:
        earliest_right_start = left_start + left_overlap
        for right_start, right_overlap in right_matches:
            if right_start < earliest_right_start:
                continue
            candidate = PartialMatchInfo(
                full_match=(left_overlap == left_len and right_overlap == right_len),
                matched_fragments=2,
                overlap_score=left_overlap + right_overlap,
            )
            if best_pair is None or (
                candidate.full_match,
                candidate.overlap_score,
            ) > (
                best_pair.full_match,
                best_pair.overlap_score,
            ):
                best_pair = candidate

    if best_pair is not None:
        return best_pair

    # No in-order two-fragment pair exists. If both fragments occur
    # anywhere in the lemma, they must be in the wrong order — reject
    # outright per the ordering contract. Otherwise surface the single
    # fragment that did match as an exploratory partial.
    if left_matches and right_matches:
        return PartialMatchInfo(full_match=False, matched_fragments=0, overlap_score=0)
    if left_matches:
        return PartialMatchInfo(
            full_match=False,
            matched_fragments=1,
            overlap_score=max(overlap for _, overlap in left_matches),
        )
    if right_matches:
        return PartialMatchInfo(
            full_match=False,
            matched_fragments=1,
            overlap_score=max(overlap for _, overlap in right_matches),
        )
    return PartialMatchInfo(full_match=False, matched_fragments=0, overlap_score=0)


def _select_partial_seed_candidates(
    partial_query: PartialQueryTokens,
    seed_candidates: Sequence[str],
    lexicon_map: LexiconMap,
    stage2_limit: int,
    *,
    minimum_window: int = _MIN_PARTIAL_STAGE2_CANDIDATES,
) -> list[str]:
    """Preselect partial-query seeds so exact wildcard matches survive Stage 2."""
    exact_match_ids: list[str] = []
    positive_overlap: list[tuple[int, int, int, str]] = []
    zero_overlap_ids: list[str] = []

    for seed_index, candidate_id in enumerate(seed_candidates):
        record = lexicon_map.get(candidate_id)
        if record is None:
            zero_overlap_ids.append(candidate_id)
            continue
        lemma_tokens = record.ipa_tokens
        match_info = _match_partial_query(partial_query, lemma_tokens)
        if match_info.full_match:
            exact_match_ids.append(candidate_id)
            continue
        if match_info.overlap_score > 0:
            positive_overlap.append(
                (
                    match_info.matched_fragments,
                    match_info.overlap_score,
                    seed_index,
                    candidate_id,
                )
            )
            continue
        zero_overlap_ids.append(candidate_id)

    positive_overlap.sort(key=lambda item: (-item[0], -item[1], item[2]))
    window = max(stage2_limit, minimum_window)
    selected = list(exact_match_ids[:window])
    remaining = window - len(selected)
    if remaining > 0:
        selected.extend(
            candidate_id
            for _matched_fragments, _overlap, _index, candidate_id in positive_overlap[:remaining]
        )
    remaining = window - len(selected)
    if remaining > 0:
        selected.extend(zero_overlap_ids[:remaining])
    return selected


def _select_partial_fallback_candidates(
    partial_query: PartialQueryTokens,
    ranked_candidates: Sequence[str],
    lexicon_map: LexiconMap,
    *,
    max_results: int,
    explicit_limit: int | None,
) -> list[str]:
    """Bound partial-form fallback work while preserving late exact matches."""
    stage2_limit = _partial_candidate_limit(max_results)
    if explicit_limit is not None:
        ranked_candidates = ranked_candidates[:explicit_limit]
        stage2_limit = min(stage2_limit, explicit_limit)

    return _select_partial_seed_candidates(
        partial_query,
        ranked_candidates,
        lexicon_map,
        stage2_limit,
        minimum_window=0,
    )


def _filter_and_rank_partial_results(
    partial_query: PartialQueryTokens,
    results: list[SearchResult],
    lexicon_lookup: LexiconLookup,
) -> list[SearchResult]:
    """Apply precision-first filtering and ordering to partial queries."""
    kept: list[tuple[SearchResult, PartialMatchInfo]] = []

    for result in results:
        candidate_id = result.entry_id
        lemma_tokens: tuple[str, ...] = ()
        if candidate_id is not None and candidate_id in lexicon_lookup:
            lemma_tokens = _resolve_entry_tokens(lexicon_lookup[candidate_id])
        match_info = _match_partial_query(partial_query, lemma_tokens)
        rule_supported = bool(result.applied_rules)
        if match_info.full_match:
            kept.append((result, match_info))
            continue
        if match_info.overlap_score <= 0:
            continue
        if rule_supported or result.confidence >= _PARTIAL_QUERY_CONFIDENCE_THRESHOLD:
            kept.append((result, match_info))

    kept.sort(
        key=lambda item: (
            not item[1].full_match,
            -item[1].matched_fragments,
            -item[1].overlap_score,
            not bool(item[0].applied_rules),
            -item[0].confidence,
            item[0].lemma or "",  # safe tie-break: None becomes empty string
        )
    )
    return [result for result, _match_info in kept]
