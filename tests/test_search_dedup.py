"""Tests for ``_deduplicate_by_headword`` behaviour.

Extracted from ``tests/test_search.py`` during the search package refactor.
"""

from __future__ import annotations

import pytest

from phonology.search import SearchResult, _deduplicate_by_headword


class TestDeduplicateByHeadword:
    """Verify headword-based deduplication of search results."""

    def test_keeps_highest_confidence(self) -> None:
        results = [
            SearchResult(lemma="α", confidence=0.9),
            SearchResult(lemma="β", confidence=0.8),
            SearchResult(lemma="α", confidence=0.7),
        ]
        deduped = _deduplicate_by_headword(results)
        assert [(r.lemma, r.confidence) for r in deduped] == [
            ("α", 0.9),
            ("β", 0.8),
        ]

    def test_rejects_unsorted_later_higher_duplicate_confidence(self) -> None:
        results = [
            SearchResult(lemma="α", confidence=0.7),
            SearchResult(lemma="β", confidence=0.8),
            SearchResult(lemma="α", confidence=0.9),
        ]

        with pytest.raises(ValueError, match="sorted by descending confidence"):
            _deduplicate_by_headword(results)

    def test_can_skip_sorted_check_for_pre_sorted_callers(self) -> None:
        results = [
            SearchResult(lemma="α", confidence=0.7, entry_id="first-alpha"),
            SearchResult(lemma="β", confidence=0.8, entry_id="first-beta"),
            SearchResult(lemma="α", confidence=0.9, entry_id="second-alpha"),
        ]

        deduped = _deduplicate_by_headword(results, check_sorted=False)

        assert [(r.lemma, r.entry_id) for r in deduped] == [
            ("α", "first-alpha"),
            ("β", "first-beta"),
        ]

    def test_boundary_confidence_values_and_equal_ties_are_deterministic(self) -> None:
        results = [
            SearchResult(lemma="α", confidence=1.0, entry_id="first-alpha"),
            SearchResult(lemma="α", confidence=1.0, entry_id="second-alpha"),
            SearchResult(lemma="β", confidence=0.0, entry_id="first-beta"),
            SearchResult(lemma="β", confidence=0.0, entry_id="second-beta"),
        ]

        deduped = _deduplicate_by_headword(results)

        assert [(r.lemma, r.confidence, r.entry_id) for r in deduped] == [
            ("α", 1.0, "first-alpha"),
            ("β", 0.0, "first-beta"),
        ]

    def test_deduplicates_case_sensitive_lemmas_independently(self) -> None:
        results = [
            SearchResult(lemma="Α", confidence=0.9),
            SearchResult(lemma="α", confidence=0.9),
        ]

        deduped = _deduplicate_by_headword(results)

        assert [r.lemma for r in deduped] == ["Α", "α"]

    def test_handles_empty_punctuation_and_unicode_lemmas(self) -> None:
        results = [
            SearchResult(lemma="", confidence=1.0, entry_id="empty-first"),
            SearchResult(lemma="λόγος!", confidence=0.9, entry_id="punct-first"),
            SearchResult(lemma="λόγος!", confidence=0.8, entry_id="punct-second"),
            SearchResult(lemma="", confidence=0.0, entry_id="empty-second"),
        ]

        deduped = _deduplicate_by_headword(results)

        assert [(r.lemma, r.entry_id) for r in deduped] == [
            ("", "empty-first"),
            ("λόγος!", "punct-first"),
        ]

    def test_preserves_order(self) -> None:
        results = [
            SearchResult(lemma="γ", confidence=0.9),
            SearchResult(lemma="α", confidence=0.8),
            SearchResult(lemma="β", confidence=0.7),
        ]
        deduped = _deduplicate_by_headword(results)
        assert [r.lemma for r in deduped] == ["γ", "α", "β"]

    def test_empty_list(self) -> None:
        assert _deduplicate_by_headword([]) == []

    def test_all_same_headword(self) -> None:
        results = [
            SearchResult(lemma="α", confidence=0.9),
            SearchResult(lemma="α", confidence=0.5),
            SearchResult(lemma="α", confidence=0.3),
        ]
        deduped = _deduplicate_by_headword(results)
        assert len(deduped) == 1
        assert deduped[0].confidence == 0.9
