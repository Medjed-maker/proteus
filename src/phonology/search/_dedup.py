"""Headword deduplication utilities for search results."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from ._types import SearchResult


def _deduplicate_by_headword_common(
    results: Iterable[SearchResult],
) -> list[SearchResult]:
    """Shared core logic for headword deduplication.

    Args:
        results: Iterable of SearchResult objects to deduplicate.

    Returns:
        List of SearchResult objects with duplicates removed, preserving
        the first occurrence of each unique lemma. None lemmas are supported
        and normalized (multiple None lemmas are treated as duplicates).
    """
    seen: set[str | None] = set()
    deduplicated: list[SearchResult] = []
    for result in results:
        if result.lemma in seen:
            continue
        seen.add(result.lemma)
        deduplicated.append(result)
    return deduplicated


def _deduplicate_by_headword(
    results: Sequence[SearchResult], *, check_sorted: bool = True
) -> list[SearchResult]:
    """Keep only the highest-confidence result for each headword.

    Args:
        results: Sequence of SearchResult objects to deduplicate.
        check_sorted: If True (default), validates that results are sorted
            by descending confidence and raises ValueError if not.
            If False, skips the validation (caller must ensure correct ordering
            for meaningful results).

    Returns:
        List of SearchResult objects with duplicates removed.

    Raises:
        ValueError: If check_sorted is True and results are not sorted
            by descending confidence.

    Note:
         For equal-confidence ties, the first entry wins.
    """

    if check_sorted and not all(
        results[i].confidence >= results[i + 1].confidence
        for i in range(len(results) - 1)
    ):
        raise ValueError(
            "_deduplicate_by_headword requires results sorted by descending confidence"
        )
    return _deduplicate_by_headword_common(results)
