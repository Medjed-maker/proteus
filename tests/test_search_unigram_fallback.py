"""Tests for unigram k=1 fallback behavior in phonology.search."""
# Several tests define monkeypatch stubs whose parameter names must mirror the
# real signatures of the functions they replace (e.g. `_score_stage(query_ipa,
# candidates, lexicon_map, matrix)`). Renaming the unused params to `_prefix`
# would either drift from the real signature or force kwargs-only call sites;
# both are worse than a file-level suppression. Keep this scoped to
# reportUnusedVariable only — other checks stay on.
# pyright: reportUnusedVariable=false

from __future__ import annotations

from collections.abc import Iterable

import pytest

from phonology import search as search_module
from phonology.search import (
    LexiconRecord,
    SearchResult,
    build_kmer_index,
    search,
)

EXPECTED_FALLBACK_CANDIDATE_LIMIT = 2000


@pytest.fixture
def mock_common_search_stages(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Patch shared search stages and return captured score-stage inputs."""
    captured: dict[str, object] = {}

    def fake_to_ipa(query: str, dialect: str = "attic") -> str:
        return {"ποι": "poi", "πα": "pa"}.get(query, "pa")

    def fake_score_stage(
        query_ipa: str,
        candidates: Iterable[str],
        lexicon_map: dict[str, LexiconRecord],
        matrix: object,
    ) -> list[SearchResult]:
        captured["candidate_ids"] = list(candidates)
        return []

    monkeypatch.setattr(search_module, "to_ipa", fake_to_ipa)
    monkeypatch.setattr(search_module, "load_rules", lambda _path: {})
    monkeypatch.setattr(search_module, "_score_stage", fake_score_stage)
    monkeypatch.setattr(
        search_module,
        "_annotate_search_results",
        lambda query_ipa, results, lexicon_map, matrix, language="ancient_greek": results,
    )
    monkeypatch.setattr(search_module, "filter_stage", lambda results, max_results: results)
    return captured


class TestSearchUnigramFallback:
    """Tests for unigram k=1 fallback behavior."""

    def test_search_uses_unigram_fallback_for_single_consonant_query(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A query with 1 consonant should invoke the k=1 fallback path."""
        seed_calls: list[int] = []

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "poi")
        monkeypatch.setattr(search_module, "load_rules", lambda _path: {})
        monkeypatch.setattr(
            search_module,
            "seed_stage",
            lambda query_ipa, index, k=2: seed_calls.append(k) or ([] if k == 2 else ["L1", "L2"]),
        )
        lexicon = [
            {"id": "L1", "headword": "πολύ", "ipa": "poly", "dialect": "attic"},
            {"id": "L2", "headword": "πατρι", "ipa": "patri", "dialect": "attic"},
        ]
        unigram_idx = build_kmer_index(lexicon, k=1)

        search(
            "ποι", lexicon, matrix={}, max_results=2, unigram_index=unigram_idx,
        )

        assert seed_calls == [2, 1]

    def test_short_query_unigram_fallback_can_return_empty_when_seeded_hits_are_still_weak(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Short-query k=1 fallback should not pull in non-unigram tail matches."""
        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "pa")
        monkeypatch.setattr(search_module, "load_rules", lambda _path: {})
        lexicon = [
            {"id": "L1", "headword": "same-consonant", "ipa": "pi", "dialect": "attic"},
            {"id": "L2", "headword": "better-match", "ipa": "ba", "dialect": "attic"},
        ]
        matrix = {"b": {"p": 0.1}, "p": {"b": 0.1}, "i": {"a": 0.9}, "a": {"i": 0.9}}

        results = search("πα", lexicon, matrix=matrix, max_results=2, index={})

        assert results == []

    def test_search_builds_unigram_fallback_when_not_provided(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Direct search() calls should get k=1 fallback without explicit index injection."""
        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "poi")
        monkeypatch.setattr(search_module, "load_rules", lambda _path: {})
        lexicon = [
            {"id": f"L{index:02d}", "headword": f"noise-{index:02d}", "ipa": "kai", "dialect": "attic"}
            for index in range(29)
        ]
        lexicon.append({"id": "L99", "headword": "target", "ipa": "poi", "dialect": "attic"})

        results = search("ποι", lexicon, matrix={}, max_results=1)

        assert [result.lemma for result in results] == ["target"]

    def test_search_accepts_explicit_unigram_fallback_limit(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_common_search_stages: dict[str, object],
    ) -> None:
        """Callers can cap the k=1 fallback after token-count re-ranking."""
        captured = mock_common_search_stages

        def fake_seed_stage(query_ipa: str, index: object, k: int = 2) -> list[str]:
            return [] if k == 2 else ["L1", "L2", "L3", "L4"]

        monkeypatch.setattr(search_module, "seed_stage", fake_seed_stage)

        lexicon = [
            {"id": "L1", "headword": "one", "ipa": "p", "dialect": "attic"},
            {"id": "L2", "headword": "two", "ipa": "pa", "dialect": "attic"},
            {"id": "L3", "headword": "three", "ipa": "poi", "dialect": "attic"},
            {"id": "L4", "headword": "four", "ipa": "poie", "dialect": "attic"},
        ]

        search(
            "ποι",
            lexicon,
            matrix={},
            max_results=1,
            index={},
            unigram_index={},
            unigram_fallback_limit=2,
        )

        assert captured["candidate_ids"] == ["L3", "L2"]

    def test_short_query_unigram_fallback_uses_default_cap(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_common_search_stages: dict[str, object],
    ) -> None:
        """Direct short-query k=1 fallback should not score all unigram hits."""
        captured = mock_common_search_stages

        def fake_seed_stage(query_ipa: str, index: object, k: int = 2) -> list[str]:
            if k == 2:
                return []
            return [f"L{idx:04d}" for idx in range(2500)]

        monkeypatch.setattr(search_module, "seed_stage", fake_seed_stage)

        lexicon = [
            {
                "id": f"L{index:04d}",
                "headword": f"lemma-{index:04d}",
                "ipa": "pa",
                "dialect": "attic",
            }
            for index in range(2500)
        ]

        search("πα", lexicon, matrix={}, max_results=1, index={}, unigram_index={})

        assert len(captured["candidate_ids"]) == EXPECTED_FALLBACK_CANDIDATE_LIMIT

    def test_fullform_unigram_fallback_uses_default_cap(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_common_search_stages: dict[str, object],
    ) -> None:
        """Full-form k=1 fallback should not score all unigram hits by default."""
        captured = mock_common_search_stages

        def fake_seed_stage(query_ipa: str, index: object, k: int = 2) -> list[str]:
            if k == 2:
                return []
            return [f"L{idx:04d}" for idx in range(2500)]

        monkeypatch.setattr(search_module, "seed_stage", fake_seed_stage)

        lexicon = [
            {
                "id": f"L{index:04d}",
                "headword": f"lemma-{index:04d}",
                "ipa": "pa",
                "dialect": "attic",
            }
            for index in range(2500)
        ]

        search("λόγος", lexicon, matrix={}, max_results=1, index={}, unigram_index={})

        assert len(captured["candidate_ids"]) == EXPECTED_FALLBACK_CANDIDATE_LIMIT

    def test_fullform_unigram_fallback_applies_default_cap_when_none(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
        mock_common_search_stages: dict[str, object],
    ) -> None:
        """Full-form k=1 fallback should fall back to the default cap when None is passed."""
        captured = mock_common_search_stages

        def fake_seed_stage(query_ipa: str, index: object, k: int = 2) -> list[str]:
            if k == 2:
                return []
            return [f"L{idx:04d}" for idx in range(2500)]

        monkeypatch.setattr(search_module, "seed_stage", fake_seed_stage)

        lexicon = [
            {
                "id": f"L{index:04d}",
                "headword": f"lemma-{index:04d}",
                "ipa": "pa",
                "dialect": "attic",
            }
            for index in range(2500)
        ]

        caplog.set_level("WARNING", logger="phonology.search")
        search(
            "λόγος",
            lexicon,
            matrix={},
            max_results=1,
            index={},
            unigram_index={},
            unigram_fallback_limit=None,
        )

        assert len(captured["candidate_ids"]) == search_module._DEFAULT_FALLBACK_CANDIDATE_LIMIT
        expected_label = search_module._summarize_query_ipa_for_logs(
            "pa",
            query_token_count=len(search_module.tokenize_ipa("pa")),
            debug_enabled=False,
        )
        assert "unigram_fallback_limit=None for query" in caplog.text
        assert expected_label in caplog.text
        assert "query IPA 'pa'" not in caplog.text
        assert (
            f"applying default cap {search_module._DEFAULT_FALLBACK_CANDIDATE_LIMIT}."
            in caplog.text
        )

    def test_unigram_fallback_limit_uses_token_count_proximity_to_keep_best_match(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A capped k=1 fallback should still keep the closest-length exact match."""
        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "poi")
        monkeypatch.setattr(search_module, "load_rules", lambda _path: {})

        def fake_seed_stage(query_ipa: str, index: object, k: int = 2) -> list[str]:
            return [] if k == 2 else ["L1", "L2"]

        monkeypatch.setattr(search_module, "seed_stage", fake_seed_stage)
        lexicon = [
            {"id": "L1", "headword": "noise", "ipa": "pa", "dialect": "attic"},
            {"id": "L2", "headword": "target", "ipa": "poi", "dialect": "attic"},
        ]

        results = search(
            "ποι",
            lexicon,
            matrix={},
            max_results=1,
            index={},
            unigram_index={},
            unigram_fallback_limit=1,
        )

        assert [result.lemma for result in results] == ["target"]

    def test_unigram_fallback_keeps_exact_match_within_stage2_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unigram fallback should find exact match among candidates within stage2_limit."""
        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "poi")
        monkeypatch.setattr(search_module, "load_rules", lambda _path: {})
        lexicon = [
            {
                "id": f"L{index:02d}",
                "headword": f"noise-{index:02d}",
                "ipa": "pa",
                "dialect": "attic",
            }
            for index in range(20)
        ]
        lexicon.append({"id": "ZZZ", "headword": "target", "ipa": "poi", "dialect": "attic"})

        results = search("ποι", lexicon, matrix={}, max_results=1)

        assert [result.lemma for result in results] == ["target"]

    def test_short_query_unigram_fallback_does_not_keep_non_hit_rule_candidates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Default short-query k=1 fallback should not widen into non-hit rule candidates."""
        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": query)
        monkeypatch.setattr(
            search_module,
            "load_rules",
            lambda _path: {
                "RULE-PQ": {
                    "id": "RULE-PQ",
                    "input": "p",
                    "output": "q",
                    "dialects": ["attic"],
                }
            },
        )
        lexicon = [
            {"id": "L1", "headword": "noise", "ipa": "q x", "dialect": "attic"},
            {"id": "L2", "headword": "rule-hit", "ipa": "p", "dialect": "attic"},
        ]

        results = search(
            "q",
            lexicon,
            matrix={"p": {"q": 0.1}, "q": {"p": 0.1}},
            max_results=2,
            index={},
            unigram_index={"q": ["L1"]},
        )

        assert results == []

    def test_short_query_token_fallback_keeps_rule_supported_candidates_beyond_old_cap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Default short-query token fallback should keep late rule-supported candidates."""
        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": query)
        monkeypatch.setattr(
            search_module,
            "load_rules",
            lambda _path: {
                "RULE-PQ": {
                    "id": "RULE-PQ",
                    "input": "p",
                    "output": "q",
                    "dialects": ["attic"],
                }
            },
        )
        lexicon = [
            {"id": f"L{index:02d}", "headword": f"noise-{index:02d}", "ipa": "t", "dialect": "attic"}
            for index in range(18)
        ]
        lexicon.append({"id": "ZZZ", "headword": "rule-hit", "ipa": "p", "dialect": "attic"})

        results = search(
            "q",
            lexicon,
            matrix={"p": {"q": 0.1}, "q": {"p": 0.1}},
            max_results=2,
            index={},
            unigram_index={},
        )

        assert [result.lemma for result in results] == ["rule-hit"]
        assert results[0].applied_rules == ["RULE-PQ"]
