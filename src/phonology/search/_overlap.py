"""Fragment overlap arithmetic and candidate-id merging helpers.

Pure helpers over token sequences and lexicon records. None of these
functions call or reference names that are monkeypatched at
``search_module`` level, so they are safe to live in a submodule.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from ._lookup import _entry_ipa
from ._types import LexiconRecord


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


def _token_count_proximity_key(
    query_ipa: str,
    query_token_count: int,
    entry_id: str,
    record: LexiconRecord,
) -> tuple[int, bool, str]:
    """Return the fallback ranking key for one tokenized lexicon record.

    Args:
        query_ipa: IPA string of the search query.
        query_token_count: Number of tokens in the query IPA.
        entry_id: Unique identifier for the lexicon entry (used as tiebreaker).
        record: LexiconRecord containing entry metadata and token count.

    Returns:
        A tuple of (int, bool, str) used for sorting:
        - int: Absolute difference between record.token_count and query_token_count
               (lower values rank higher, preferring similar token lengths).
        - bool: Whether the record's IPA differs from query_ipa; True ranks lower,
                so exact IPA matches are preferred.
        - str: entry_id used as final tiebreaker for stable ordering.
    """
    return (
        abs(record.token_count - query_token_count),
        _entry_ipa(record.entry) != query_ipa,
        entry_id,
    )


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
