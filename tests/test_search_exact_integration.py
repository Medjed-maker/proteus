"""Integration tests for exact-match boost and deduplication in ``search()``.

Extracted from ``tests/test_search.py`` during the search package refactor.
"""

from __future__ import annotations

import pytest

from phonology import search as search_module
from phonology.distance import load_matrix
from phonology.search import LexiconRecord, SearchResult, search

MATRIX_FILE = "attic_doric.json"


class TestSearchExactMatchIntegration:
    """Integration tests for exact-match boost and deduplication in search()."""

    def test_search_returns_empty_list_for_empty_lexicon(self) -> None:
        matrix = load_matrix(MATRIX_FILE)

        results = search("τεστ", [], matrix, max_results=5, dialect="attic")

        assert results == []

    def test_search_returns_empty_list_when_short_query_has_no_quality_match(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            search_module, "to_ipa", lambda query, dialect="attic": query
        )
        lexicon = [
            {"id": "L1", "headword": "alpha", "ipa": "aaa", "dialect": "attic"},
            {"id": "L2", "headword": "beta", "ipa": "bbb", "dialect": "attic"},
        ]
        matrix = load_matrix(MATRIX_FILE)

        results = search("zz", lexicon, matrix, max_results=5, dialect="attic")

        assert results == []

    def test_search_trims_headword_labels(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            search_module, "to_ipa", lambda query, dialect="attic": query
        )
        lexicon = [
            {"id": "L1", "headword": " alpha ", "ipa": "alpha", "dialect": "attic"},
        ]
        matrix = load_matrix(MATRIX_FILE)

        results = search("alpha", lexicon, matrix, max_results=5, dialect="attic")

        assert [result.lemma for result in results] == ["alpha"]

    def test_search_returns_exact_match_at_top(self) -> None:
        """Verify that a query word appearing in the lexicon is returned as the top result."""
        # Build a lexicon where the target word shares k-mers with many other entries,
        # pushing it past the stage2_limit in seed ranking.
        target = {"id": "TARGET", "headword": "τεστ", "ipa": "test", "dialect": "attic"}
        # Create 30 entries sharing k-mers (consonant skeleton "t s t" has k-mers "t s" and "s t")
        filler = [
            {
                "id": f"F{i:03d}",
                "headword": f"filler{i}",
                "ipa": f"t{'a' * i}st",
                "dialect": "attic",
            }
            for i in range(1, 31)
        ]
        lexicon = filler + [target]  # target last so its ID sorts late

        matrix = load_matrix(MATRIX_FILE)
        results = search("τεστ", lexicon, matrix, max_results=5, dialect="attic")
        assert len(results) > 0, "Expected at least one search result"
        assert results[0].lemma == "τεστ", (
            f"Expected top result lemma 'τεστ', got {results[0].lemma!r}"
        )
        assert results[0].confidence == 1.0, (
            f"Expected top result confidence 1.0, got {results[0].confidence!r}"
        )

    def test_search_deduplicates_homograph_entries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify that duplicate headwords are deduplicated in results."""
        monkeypatch.setattr(
            search_module, "to_ipa", lambda query, dialect="attic": query
        )
        lexicon = [
            {"id": "L1", "headword": "dup", "ipa": "dup", "dialect": "attic"},
            {"id": "L2", "headword": "dup", "ipa": "dup", "dialect": "attic"},
            {"id": "L3", "headword": "other", "ipa": "otʰer", "dialect": "attic"},
        ]
        matrix = load_matrix(MATRIX_FILE)
        results = search("dup", lexicon, matrix, max_results=5, dialect="attic")
        headwords = [r.lemma for r in results]
        assert headwords.count("dup") == 1

    def test_exact_match_injection_preserves_stage2_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exact-match injection must not expand Stage 2 past its hard candidate cap.

        NOTE: This test intentionally mocks private functions (_score_stage,
        _annotate_search_results, filter_stage) and relies on _MIN_STAGE2_CANDIDATES
        to verify exact-match injection behavior. This is intentional to isolate
        the injection logic from full pipeline execution.
        """
        lexicon = [
            {
                "id": f"E{i:02d}",
                "headword": f"dup{i:02d}",
                "ipa": "tat",
                "dialect": "attic",
            }
            for i in range(40)
        ]
        captured: dict[str, list[str]] = {}

        def fake_score_stage(
            *,
            query_ipa: str,
            candidates: list[str],
            lexicon_map: dict[str, LexiconRecord | dict[str, str]],
            matrix: dict[str, dict[str, float]],
        ) -> list[SearchResult]:
            captured["candidate_ids"] = list(candidates)
            return [
                SearchResult(
                    lemma=(
                        lexicon_map[candidate_id].entry["headword"]
                        if isinstance(lexicon_map[candidate_id], LexiconRecord)
                        else lexicon_map[candidate_id]["headword"]
                    ),
                    confidence=1.0,
                    dialect_attribution="lemma dialect: attic",
                    entry_id=candidate_id,
                )
                for candidate_id in candidates
            ]

        monkeypatch.setattr(search_module, "_score_stage", fake_score_stage)
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda **kwargs: kwargs["results"],
        )
        monkeypatch.setattr(
            search_module,
            "filter_stage",
            lambda results, max_results: results[:max_results],
        )
        monkeypatch.setattr(
            search_module, "to_ipa", lambda query, dialect="attic": "tat"
        )

        search("dummy", lexicon, matrix={}, max_results=1, dialect="attic")

        assert len(captured["candidate_ids"]) == search_module._MIN_STAGE2_CANDIDATES
        # Defensive check: ensure candidate_id follows expected "E##" format before parsing
        assert all(
            candidate_id.startswith("E") and candidate_id[1:].isdigit()
            for candidate_id in captured["candidate_ids"]
        )
        assert all(
            lexicon[int(candidate_id.removeprefix("E"))]["ipa"] == "tat"
            for candidate_id in captured["candidate_ids"]
        )

    def test_dedup_does_not_reduce_below_max_results(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dedup before truncation: max_results unique entries are returned."""
        monkeypatch.setattr(
            search_module, "to_ipa", lambda query, dialect="attic": query
        )
        # 2 duplicate headwords + 3 unique = 5 entries.
        # With max_results=3, dedup-before-truncation should yield 3 unique results
        # (not 2 if dedup were applied after truncation).
        lexicon = [
            {"id": "L1", "headword": "alpha", "ipa": "alpha", "dialect": "attic"},
            {"id": "L2", "headword": "alpha", "ipa": "alpha", "dialect": "attic"},
            {"id": "L3", "headword": "beta", "ipa": "alpha", "dialect": "attic"},
            {"id": "L4", "headword": "gamma", "ipa": "alpha", "dialect": "attic"},
            {"id": "L5", "headword": "delta", "ipa": "alpha", "dialect": "attic"},
        ]
        matrix = load_matrix(MATRIX_FILE)
        results = search("alpha", lexicon, matrix, max_results=3, dialect="attic")
        assert len(results) == 3
        headwords = [r.lemma for r in results]
        assert headwords.count("alpha") == 1
