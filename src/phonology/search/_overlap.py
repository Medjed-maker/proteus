"""Fragment overlap arithmetic and candidate-id merging helpers.

Pure helpers over token sequences and lexicon records. None of these
functions call or reference names that are monkeypatched at
``search_module`` level, so they are safe to live in a submodule.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def _leading_overlap_length(
    fragment_tokens: Sequence[str],
    lemma_tokens: Sequence[str],
) -> int:
    """Return the number of matching tokens from the left edge."""
    overlap = 0
    for fragment_token, lemma_token in zip(fragment_tokens, lemma_tokens, strict=False):
        if fragment_token != lemma_token:
            break
        overlap += 1
    return overlap


def _trailing_overlap_length(
    fragment_tokens: Sequence[str],
    lemma_tokens: Sequence[str],
) -> int:
    """Return the number of matching tokens from the right edge."""
    overlap = 0
    for fragment_token, lemma_token in zip(
        reversed(fragment_tokens), reversed(lemma_tokens), strict=False
    ):
        if fragment_token != lemma_token:
            break
        overlap += 1
    return overlap


def _contiguous_prefix_match_length(
    fragment_tokens: Sequence[str],
    lemma_tokens: Sequence[str],
    start_index: int,
) -> int:
    """Return the contiguous prefix match length from one lemma offset."""
    if start_index < 0:
        return 0
    if start_index >= len(lemma_tokens):
        return 0

    overlap = 0
    max_length = min(len(fragment_tokens), len(lemma_tokens) - start_index)
    while (
        overlap < max_length
        and fragment_tokens[overlap] == lemma_tokens[start_index + overlap]
    ):
        overlap += 1
    return overlap


def _collect_fragment_matches(
    fragment_tokens: Sequence[str],
    lemma_tokens: Sequence[str],
) -> list[tuple[int, int]]:
    """Return every (start_index, overlap_length) pair with overlap > 0.

    Used by the infix branch of :func:`_match_partial_query` to find all
    candidate match offsets for each fragment in a single pass, so the
    downstream pair-matching loop can rely on a compact match list
    instead of re-scanning the whole lemma.
    """
    if not fragment_tokens:
        return []
    matches: list[tuple[int, int]] = []
    for start in range(len(lemma_tokens)):
        overlap = _contiguous_prefix_match_length(fragment_tokens, lemma_tokens, start)
        if overlap > 0:
            matches.append((start, overlap))
    return matches


def _is_exact_token_match(
    query_tokens: Sequence[str],
    lemma_tokens: Sequence[str],
) -> bool:
    """Return whether the two token sequences match exactly."""
    return tuple(query_tokens) == tuple(lemma_tokens)


def _merge_bounded_candidate_ids(
    primary_candidates: Iterable[str],
    supplemental_candidates: Iterable[str] = (),
    *,
    limit: int,
) -> list[str]:
    """Merge candidate streams in priority order without exceeding a hard cap."""
    merged: list[str] = []
    seen: set[str] = set()
    for candidate_id in primary_candidates:
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        merged.append(candidate_id)
        if len(merged) >= limit:
            return merged
    for candidate_id in supplemental_candidates:
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        merged.append(candidate_id)
        if len(merged) >= limit:
            return merged
    return merged
