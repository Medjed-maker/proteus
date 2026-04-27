"""Tests for phonology.search."""
# Several tests define monkeypatch stubs whose parameter names must mirror the
# real signatures of the functions they replace (e.g. `_score_stage(query_ipa,
# candidates, lexicon_map, matrix)`). Renaming the unused params to `_prefix`
# would either drift from the real signature or force kwargs-only call sites;
# both are worse than a file-level suppression. Keep this scoped to
# reportUnusedVariable only — other checks stay on.
# pyright: reportUnusedVariable=false

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pytest

from phonology import search as search_module
from phonology.distance import load_matrix
from phonology.search import (
    LexiconRecord,
    SearchResult,
    build_kmer_index,
    search,
)
# TODO: expose partial-match helpers publicly once that package API is stable.
# These tests exercise _partial._is_partial_match directly, which search() does not expose.
from phonology.search import _partial as partial_module

MATRIX_FILE = "attic_doric.json"

class TestSearch:
    """Verify full three-stage search pipeline including IPA conversion and index reuse."""

    @staticmethod
    def _extract_headword(entry: object) -> str:
        if isinstance(entry, dict):
            return str(entry["headword"])
        return str(entry.entry["headword"])

    @staticmethod
    def _raise_should_not_rebuild_ipa_index(lexicon_lookup: object) -> search_module.IpaIndex:
        raise AssertionError("should not rebuild IPA index")

    @staticmethod
    def _fake_score_stage(
        query_ipa: str,
        candidates: Iterable[str],
        lexicon_map: dict[str, object],
        matrix: object,
        capture: list[str],
        use_headword: bool = True,
    ) -> list[SearchResult]:
        capture[:] = list(candidates)
        return [
            SearchResult(
                lemma=TestSearch._extract_headword(lexicon_map[entry_id]) if use_headword else entry_id,
                confidence=1.0,
                dialect_attribution="lemma dialect: attic",
                entry_id=entry_id,
            )
            for entry_id in capture
        ]

    @staticmethod
    def _make_capturing_score_stage(capture: list[str], use_headword: bool = True):
        """Returns a score-stage mock that captures candidates into the provided list."""

        def fake_score_stage(
            query_ipa: str,
            candidates: Iterable[str],
            lexicon_map: dict[str, object],
            matrix: object,
            **_kwargs: Any,
        ) -> list[SearchResult]:
            return TestSearch._fake_score_stage(
                query_ipa,
                candidates,
                lexicon_map,
                matrix,
                capture=capture,
                use_headword=use_headword,
            )

        return fake_score_stage

    def test_rejects_blank_query(self) -> None:
        with pytest.raises(ValueError, match="query must be a non-empty string"):
            search("   ", [], matrix={})

    @pytest.mark.parametrize("max_results", [0, -1])
    def test_rejects_non_positive_max_results(self, max_results: int) -> None:
        with pytest.raises(ValueError, match="max_results must be a positive integer"):
            search("λόγος", [], matrix={}, max_results=max_results)

    @pytest.mark.parametrize("similarity_fallback_limit", [0, -1])
    def test_rejects_non_positive_similarity_fallback_limit(
        self, similarity_fallback_limit: int
    ) -> None:
        with pytest.raises(
            ValueError,
            match="similarity_fallback_limit must be a positive integer",
        ):
            search(
                "λόγος",
                [],
                matrix={},
                similarity_fallback_limit=similarity_fallback_limit,
            )

    @pytest.mark.parametrize("unigram_fallback_limit", [0, -1])
    def test_rejects_non_positive_unigram_fallback_limit(
        self, unigram_fallback_limit: int
    ) -> None:
        with pytest.raises(
            ValueError,
            match="unigram_fallback_limit must be a positive integer",
        ):
            search(
                "λόγος",
                [],
                matrix={},
                unigram_fallback_limit=unigram_fallback_limit,
            )

    @pytest.mark.parametrize("query", ["*", "-"])
    def test_rejects_wildcard_only_query(self, query: str) -> None:
        with pytest.raises(ValueError, match="query must be a non-empty string"):
            search(query, [], matrix={})

    def test_token_proximity_fallback_does_not_drop_same_length_exact_match_beyond_previous_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Token-count fallback should evaluate same-length exact matches past the old cap."""
        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "aːi")
        monkeypatch.setattr(search_module, "load_rules", lambda _path: {})
        lexicon = [
            {
                "id": f"L{index:02d}",
                "headword": f"noise-{index:02d}",
                "ipa": "ou",
                "dialect": "attic",
            }
            for index in range(30)
        ]
        lexicon.append({"id": "ZZZ", "headword": "target", "ipa": "aːi", "dialect": "attic"})

        results = search("dummy", lexicon, matrix={}, max_results=1)

        assert [result.lemma for result in results] == ["target"]

    def test_token_proximity_fallback_uses_cached_token_counts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Token-count fallback should not re-tokenize lexicon entries while scoring."""
        token_calls: list[str] = []

        def fake_tokenize_ipa(ipa_text: str) -> list[str]:
            token_calls.append(ipa_text)
            return {
                "query-ipa": ["q1", "q2"],
                "aː": ["aː"],
                "i": ["i"],
                "ou": ["o", "u"],
            }.get(ipa_text, [])

        def fake_score_stage(
            query_ipa: str,
            candidates: Iterable[str],
            lexicon_map: dict[str, object],
            matrix: object,
            **_kwargs: Any,
        ) -> list[SearchResult]:
            return [
                SearchResult(
                    lemma="alpha",
                    confidence=1.0,
                    dialect_attribution="lemma dialect: attic",
                )
            ]

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "query-ipa")
        monkeypatch.setattr("phonology.search._tokenization.tokenize_ipa", fake_tokenize_ipa)
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: [])
        monkeypatch.setattr(search_module, "_score_stage", fake_score_stage)
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda query_ipa, results, lexicon_map, matrix, language="ancient_greek", **_kwargs: results,
        )
        monkeypatch.setattr(
            search_module, "filter_stage", lambda results, max_results: results[:max_results]
        )

        lexicon = [
            {"id": "L1", "headword": "alpha", "ipa": "aː", "dialect": "attic"},
            {"id": "L2", "headword": "iota", "ipa": "i", "dialect": "attic"},
            {"id": "L3", "headword": "omicron", "ipa": "ou", "dialect": "attic"},
        ]

        search("query", lexicon, matrix={}, max_results=1, index={}, unigram_index={})

        assert token_calls == ["query-ipa", "aː", "i", "ou"]

    def test_search_uses_prebuilt_lexicon_map_without_retokenizing_entries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Provided lexicon_map should avoid lexicon entry re-tokenization."""
        token_calls: list[str] = []

        def fake_tokenize_ipa(ipa_text: str) -> list[str]:
            token_calls.append(ipa_text)
            return {
                "query-ipa": ["q1", "q2"],
                "aː": ["aː"],
                "i": ["i"],
                "ou": ["o", "u"],
            }.get(ipa_text, [])

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "query-ipa")
        monkeypatch.setattr("phonology.search._tokenization.tokenize_ipa", fake_tokenize_ipa)
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: [])
        monkeypatch.setattr(
            search_module,
            "_score_stage",
            lambda query_ipa, candidates, lexicon_map, matrix: [
                SearchResult(
                    lemma="alpha",
                    confidence=1.0,
                    dialect_attribution="lemma dialect: attic",
                    entry_id="L1",
                )
            ],
        )
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda query_ipa, results, lexicon_map, matrix, language="ancient_greek", **_kwargs: results,
        )
        monkeypatch.setattr(
            search_module, "filter_stage", lambda results, max_results: results[:max_results]
        )

        lexicon = [
            {"id": "L1", "headword": "alpha", "ipa": "aː", "dialect": "attic"},
            {"id": "L2", "headword": "iota", "ipa": "i", "dialect": "attic"},
            {"id": "L3", "headword": "omicron", "ipa": "ou", "dialect": "attic"},
        ]
        prebuilt_lexicon_map = {
            "L1": LexiconRecord(entry=lexicon[0], token_count=1),
            "L2": LexiconRecord(entry=lexicon[1], token_count=1),
            "L3": LexiconRecord(entry=lexicon[2], token_count=2),
        }

        search(
            "query",
            lexicon,
            matrix={},
            max_results=1,
            index={},
            unigram_index={},
            prebuilt_lexicon_map=prebuilt_lexicon_map,
        )

        assert token_calls == ["query-ipa"]

    def test_prebuilt_lexicon_map_reuses_token_fallback_lookup_without_rebuilding(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Token fallback should use a provided lexicon_map without rebuilding it."""
        captured: dict[str, object] = {}

        def fail_build_lexicon_map(lexicon: object) -> search_module.LexiconMap:
            raise AssertionError("should not rebuild lexicon map")

        def fake_rank_by_token_count_proximity(
            query_ipa: str,
            lexicon_map: dict[str, LexiconRecord],
            *,
            max_candidates: int | None = None,
            query_token_count: int | None = None,
            **_kwargs: Any,
        ) -> list[str]:
            captured["rank_lookup_is_prebuilt"] = lexicon_map is prebuilt_lexicon_map
            return ["L1"]

        def fake_score_stage(
            query_ipa: str,
            candidates: Iterable[str],
            lexicon_map: dict[str, object],
            matrix: object,
            **_kwargs: Any,
        ) -> list[SearchResult]:
            captured["score_lookup_is_prebuilt"] = lexicon_map is prebuilt_lexicon_map
            return [
                SearchResult(
                    lemma="alpha",
                    confidence=1.0,
                    dialect_attribution="lemma dialect: attic",
                    entry_id="L1",
                )
            ]

        def fake_annotate_search_results(
            query_ipa: str,
            results: list[SearchResult],
            lexicon_map: dict[str, object],
            matrix: object,
            language: str = "ancient_greek",
            **_kwargs: Any,
        ) -> list[SearchResult]:
            captured["annotate_lookup_is_prebuilt"] = lexicon_map is prebuilt_lexicon_map
            return results

        lexicon = [
            {"id": "L1", "headword": "alpha", "ipa": "aː", "dialect": "attic"},
        ]
        prebuilt_lexicon_map = {
            "L1": LexiconRecord(entry=lexicon[0], token_count=1, ipa_tokens=("aː",)),
        }

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "aː")
        monkeypatch.setattr("phonology.search._tokenization.tokenize_ipa", lambda ipa_text: ["a"])
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: [])
        monkeypatch.setattr(search_module, "build_lexicon_map", fail_build_lexicon_map)
        monkeypatch.setattr(
            search_module,
            "_rank_by_token_count_proximity",
            fake_rank_by_token_count_proximity,
        )
        monkeypatch.setattr(search_module, "_score_stage", fake_score_stage)
        monkeypatch.setattr(search_module, "_annotate_search_results", fake_annotate_search_results)

        search(
            "query",
            lexicon,
            matrix={},
            max_results=1,
            index={},
            prebuilt_lexicon_map=prebuilt_lexicon_map,
        )

        assert captured == {
            "rank_lookup_is_prebuilt": True,
            "score_lookup_is_prebuilt": True,
            "annotate_lookup_is_prebuilt": True,
        }

    def test_fullform_finalization_annotates_only_visible_deduplicated_hits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Full-form finalization should annotate only top-N hits after deduplication."""
        captured: dict[str, object] = {}

        def fake_score_stage(
            query_ipa: str,
            candidates: Iterable[str],
            lexicon_map: dict[str, object],
            matrix: object,
            **_kwargs: Any,
        ) -> list[SearchResult]:
            return [
                SearchResult(lemma="alpha", confidence=0.99, entry_id="L1"),
                SearchResult(lemma="alpha", confidence=0.95, entry_id="L2"),
                SearchResult(lemma="beta", confidence=0.90, entry_id="L3"),
                SearchResult(lemma="gamma", confidence=0.80, entry_id="L4"),
            ]

        def fake_annotate_search_results(
            query_ipa: str,
            results: list[SearchResult],
            lexicon_map: dict[str, object],
            matrix: object,
            language: str = "ancient_greek",
            **_kwargs: Any,
        ) -> list[SearchResult]:
            captured["annotated_entry_ids"] = [result.entry_id for result in results]
            return results

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "pten")
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: ["L1", "L2", "L3", "L4"])
        monkeypatch.setattr(search_module, "_score_stage", fake_score_stage)
        monkeypatch.setattr(search_module, "_annotate_search_results", fake_annotate_search_results)

        lexicon = [
            {"id": "L1", "headword": "alpha", "ipa": "pten", "dialect": "attic"},
            {"id": "L2", "headword": "alpha", "ipa": "pten", "dialect": "attic"},
            {"id": "L3", "headword": "beta", "ipa": "pten", "dialect": "attic"},
            {"id": "L4", "headword": "gamma", "ipa": "pten", "dialect": "attic"},
        ]

        results = search("λόγος", lexicon, matrix={}, max_results=2, index={})

        assert captured["annotated_entry_ids"] == ["L1", "L3"]
        assert [result.entry_id for result in results] == ["L1", "L3"]

    def test_search_does_not_tokenize_full_lexicon_when_k2_seed_hits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Provided k=2 seeds should only need query tokenization before stage 2."""
        token_calls: list[str] = []
        captured: dict[str, object] = {}

        def fake_tokenize_ipa(ipa_text: str) -> list[str]:
            token_calls.append(ipa_text)
            return ["q"]

        def fake_score_stage(
            query_ipa: str,
            candidates: Iterable[str],
            lexicon_map: dict[str, object],
            matrix: object,
            **_kwargs: Any,
        ) -> list[SearchResult]:
            captured["candidate_ids"] = list(candidates)
            captured["lookup_keys"] = list(lexicon_map)
            return []

        def fail_build_lexicon_map(lexicon: object) -> search_module.LexiconMap:
            raise AssertionError("should not build tokenized lexicon map")

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "query-ipa")
        monkeypatch.setattr("phonology.search._tokenization.tokenize_ipa", fake_tokenize_ipa)
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: ["L1"])
        monkeypatch.setattr(search_module, "build_lexicon_map", fail_build_lexicon_map)
        monkeypatch.setattr(search_module, "_score_stage", fake_score_stage)
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda query_ipa, results, lexicon_map, matrix, language="ancient_greek", **_kwargs: results,
        )
        monkeypatch.setattr(search_module, "filter_stage", lambda results, max_results: results)

        lexicon = [
            {"id": "L1", "headword": "alpha", "ipa": "aː", "dialect": "attic"},
            {"id": "L2", "headword": "beta", "ipa": "b", "dialect": "attic"},
        ]

        search("query", lexicon, matrix={}, max_results=1, index={})

        assert token_calls == ["query-ipa"]
        assert captured["candidate_ids"] == ["L1"]
        assert captured["lookup_keys"] == ["L1", "L2"]

    def test_seeded_fullform_applies_initial_stage2_cap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Seeded full-form selection should keep Stage 2 bounded."""
        captured_candidates: list[str] = []
        lexicon = [
            {
                "id": f"L{index:02d}",
                "headword": f"lemma-{index:02d}",
                "ipa": "query-ipa",
                "dialect": "attic",
            }
            for index in range(30)
        ]

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "query-ipa")
        monkeypatch.setattr(
            search_module,
            "seed_stage",
            lambda *_args, **_kwargs: [entry["id"] for entry in lexicon],
        )
        monkeypatch.setattr(
            search_module,
            "_score_stage",
            self._make_capturing_score_stage(captured_candidates),
        )
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda query_ipa, results, lexicon_map, matrix, language="ancient_greek", **_kwargs: results,
        )

        search("query", lexicon, matrix={}, max_results=1)

        assert len(captured_candidates) == search_module._MIN_STAGE2_CANDIDATES

    def test_seeded_fullform_injects_exact_match_beyond_initial_window(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An exact IPA hit outside the initial seed window should survive selection."""
        captured_candidates: list[str] = []
        lexicon = [
            {
                "id": f"N{index:02d}",
                "headword": f"noise-{index:02d}",
                "ipa": "noise-ipa",
                "dialect": "attic",
            }
            for index in range(30)
        ]
        lexicon.append(
            {"id": "TARGET", "headword": "target", "ipa": "query-ipa", "dialect": "attic"}
        )

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "query-ipa")
        monkeypatch.setattr(
            search_module,
            "seed_stage",
            lambda *_args, **_kwargs: [entry["id"] for entry in lexicon],
        )
        monkeypatch.setattr(
            search_module,
            "_score_stage",
            self._make_capturing_score_stage(captured_candidates),
        )
        monkeypatch.setattr(search_module, "_annotate_search_results", lambda **kwargs: kwargs["results"])

        search("query", lexicon, matrix={}, max_results=1)

        assert "TARGET" in captured_candidates
        assert len(captured_candidates) == search_module._MIN_STAGE2_CANDIDATES

    def test_seeded_fullform_adds_length_proximate_candidates_without_exact_match(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Length-proximate full-form seeds should supplement the initial window."""
        captured_candidates: list[str] = []
        lexicon = [
            {
                "id": f"L{index:02d}",
                "headword": f"lemma-{index:02d}",
                "ipa": f"candidate-{index:02d}",
                "dialect": "attic",
            }
            for index in range(30)
        ]
        prebuilt_lexicon_map = {
            entry["id"]: LexiconRecord(
                entry=entry,
                token_count=8 if index < search_module._MIN_STAGE2_CANDIDATES else 3,
                ipa_tokens=("x",) * (8 if index < search_module._MIN_STAGE2_CANDIDATES else 3),
            )
            for index, entry in enumerate(lexicon)
        }

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "query-ipa")
        monkeypatch.setattr("phonology.search._tokenization.tokenize_ipa", lambda ipa_text: ["q", "u", "e"])
        monkeypatch.setattr(
            search_module,
            "seed_stage",
            lambda *_args, **_kwargs: [entry["id"] for entry in lexicon],
        )
        monkeypatch.setattr(
            search_module,
            "_score_stage",
            self._make_capturing_score_stage(captured_candidates),
        )
        monkeypatch.setattr(search_module, "_annotate_search_results", lambda **kwargs: kwargs["results"])

        search(
            "query",
            lexicon,
            matrix={},
            max_results=1,
            prebuilt_lexicon_map=prebuilt_lexicon_map,
        )

        assert captured_candidates[: search_module._MIN_STAGE2_CANDIDATES] == [
            f"L{index:02d}" for index in range(search_module._MIN_STAGE2_CANDIDATES)
        ]
        assert "L25" in captured_candidates
        assert len(captured_candidates) == 30

    def test_seeded_fullform_skips_length_proximate_candidates_when_exact_exists(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exact IPA presence should keep seeded full-form selection at the hard cap."""
        captured_candidates: list[str] = []
        lexicon = [
            {
                "id": f"L{index:02d}",
                "headword": f"lemma-{index:02d}",
                "ipa": "query-ipa" if index == 0 else f"candidate-{index:02d}",
                "dialect": "attic",
            }
            for index in range(30)
        ]
        prebuilt_lexicon_map = {
            entry["id"]: LexiconRecord(
                entry=entry,
                token_count=3,
                ipa_tokens=("q", "u", "e"),
            )
            for entry in lexicon
        }

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "query-ipa")
        monkeypatch.setattr("phonology.search._tokenization.tokenize_ipa", lambda ipa_text: ["q", "u", "e"])
        monkeypatch.setattr(
            search_module,
            "seed_stage",
            lambda *_args, **_kwargs: [entry["id"] for entry in lexicon],
        )
        monkeypatch.setattr(
            search_module,
            "_score_stage",
            self._make_capturing_score_stage(captured_candidates),
        )
        monkeypatch.setattr(search_module, "_annotate_search_results", lambda **kwargs: kwargs["results"])

        search(
            "query",
            lexicon,
            matrix={},
            max_results=1,
            prebuilt_lexicon_map=prebuilt_lexicon_map,
        )

        assert len(captured_candidates) == search_module._MIN_STAGE2_CANDIDATES
        assert "L25" not in captured_candidates

    def test_token_fallback_builds_tokenized_lexicon_map_once_and_reuses_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A lazy tokenized map should be built once and reused through finalization."""
        captured: dict[str, object] = {}
        build_calls = 0

        lexicon = [
            {"id": "L1", "headword": "alpha", "ipa": "aː", "dialect": "attic"},
        ]
        built_lexicon_map = {
            "L1": LexiconRecord(entry=lexicon[0], token_count=1, ipa_tokens=("aː",)),
        }

        def fake_build_lexicon_map(
            lexicon: list[dict[str, str]],
            **_kwargs: Any,
        ) -> dict[str, LexiconRecord]:
            nonlocal build_calls
            build_calls += 1
            return built_lexicon_map

        def fake_rank_by_token_count_proximity(
            query_ipa: str,
            lexicon_map: dict[str, LexiconRecord],
            *,
            max_candidates: int | None = None,
            query_token_count: int | None = None,
            **_kwargs: Any,
        ) -> list[str]:
            captured["rank_lookup_is_built"] = lexicon_map is built_lexicon_map
            return ["L1"]

        def fake_score_stage(
            query_ipa: str,
            candidates: Iterable[str],
            lexicon_map: dict[str, object],
            matrix: object,
            **_kwargs: Any,
        ) -> list[SearchResult]:
            captured["score_lookup_is_built"] = lexicon_map is built_lexicon_map
            return [
                SearchResult(
                    lemma="alpha",
                    confidence=1.0,
                    dialect_attribution="lemma dialect: attic",
                    entry_id="L1",
                )
            ]

        def fake_annotate_search_results(
            query_ipa: str,
            results: list[SearchResult],
            lexicon_map: dict[str, object],
            matrix: object,
            language: str = "ancient_greek",
            **_kwargs: Any,
        ) -> list[SearchResult]:
            captured["annotate_lookup_is_built"] = lexicon_map is built_lexicon_map
            return results

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "aː")
        monkeypatch.setattr("phonology.search._tokenization.tokenize_ipa", lambda ipa_text: ["a"])
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: [])
        monkeypatch.setattr(search_module, "build_lexicon_map", fake_build_lexicon_map)
        monkeypatch.setattr(
            search_module,
            "_rank_by_token_count_proximity",
            fake_rank_by_token_count_proximity,
        )
        monkeypatch.setattr(search_module, "_score_stage", fake_score_stage)
        monkeypatch.setattr(search_module, "_annotate_search_results", fake_annotate_search_results)

        search("query", lexicon, matrix={}, max_results=1, index={})

        assert build_calls == 1
        assert captured == {
            "rank_lookup_is_built": True,
            "score_lookup_is_built": True,
            "annotate_lookup_is_built": True,
        }

    def test_search_uses_only_unigram_hits_for_fullform_unigram_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Full-form k=1 fallback should not append non-unigram lexicon entries."""
        captured: dict[str, object] = {}

        def fake_seed_stage(query_ipa: str, index: object, k: int = 2, **_kwargs: Any) -> list[str]:
            return [] if k == 2 else ["L2", *[f"L{idx:04d}" for idx in range(2500)]]

        def fake_score_stage(
            query_ipa: str,
            candidates: Iterable[str],
            lexicon_map: dict[str, object],
            matrix: object,
            **_kwargs: Any,
        ) -> list[SearchResult]:
            captured["candidate_ids"] = list(candidates)
            return []

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "query-ipa")
        monkeypatch.setattr(search_module, "seed_stage", fake_seed_stage)
        monkeypatch.setattr(search_module, "_score_stage", fake_score_stage)
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda query_ipa, results, lexicon_map, matrix, language="ancient_greek", **_kwargs: results,
        )
        monkeypatch.setattr(search_module, "filter_stage", lambda results, max_results: results)

        lexicon = [
            {"id": "L1", "headword": "alpha", "ipa": "aː", "dialect": "attic"},
            {"id": "L2", "headword": "beta", "ipa": "b", "dialect": "attic"},
        ]

        search("query", lexicon, matrix={}, max_results=1, index={}, unigram_index={})

        assert captured["candidate_ids"] == ["L2"]

    def test_partial_query_token_proximity_fallback_is_bounded_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Partial-form token fallback should use the default exploration cap."""
        captured: dict[str, object] = {}
        partial_match_calls = 0

        def fake_seed_stage(query_ipa: str, index: object, k: int = 2, **_kwargs: Any) -> list[str]:
            return []

        def fake_rank_by_token_count_proximity(
            query_ipa: str,
            lexicon_map: dict[str, LexiconRecord],
            *,
            max_candidates: int | None = None,
            query_token_count: int | None = None,
            **_kwargs: Any,
        ) -> list[str]:
            captured["fallback_max_candidates"] = max_candidates
            captured["fallback_query_token_count"] = query_token_count
            captured["fallback_lexicon_size"] = len(lexicon_map)
            return list(lexicon_map)[:max_candidates]

        original_match_partial_query = search_module._match_partial_query

        def tracking_match_partial_query(
            partial_query: search_module.PartialQueryTokens,
            lemma_tokens: tuple[str, ...],
            **_kwargs: Any,
        ) -> search_module.PartialMatchInfo:
            nonlocal partial_match_calls
            partial_match_calls += 1
            return original_match_partial_query(partial_query, lemma_tokens)

        def fake_score_stage(
            query_ipa: str,
            candidates: Iterable[str],
            lexicon_map: dict[str, object],
            matrix: object,
            **_kwargs: Any,
        ) -> list[SearchResult]:
            captured["candidate_ids"] = list(candidates)
            return []

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": query)
        monkeypatch.setattr(search_module, "seed_stage", fake_seed_stage)
        monkeypatch.setattr(search_module, "_score_stage", fake_score_stage)
        monkeypatch.setattr(search_module, "_rank_by_token_count_proximity", fake_rank_by_token_count_proximity)
        monkeypatch.setattr(search_module, "_match_partial_query", tracking_match_partial_query)
        monkeypatch.setattr(partial_module, "_match_partial_query", tracking_match_partial_query)
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda query_ipa, results, lexicon_map, matrix, language="ancient_greek", **_kwargs: results,
        )

        lexicon = [
            {
                "id": f"N{index:04d}",
                "headword": f"noise-{index:04d}",
                "ipa": "i",
                "dialect": "attic",
            }
            for index in range(1999)
        ]
        lexicon.append({"id": "TARGET", "headword": "target", "ipa": "x a", "dialect": "attic"})
        lexicon.extend(
            {
                "id": f"T{index:04d}",
                "headword": f"tail-{index:04d}",
                "ipa": "i",
                "dialect": "attic",
            }
            for index in range(500)
        )

        search("*a", lexicon, matrix={}, max_results=1, index={}, unigram_index={})

        assert captured["fallback_max_candidates"] == search_module._DEFAULT_FALLBACK_CANDIDATE_LIMIT
        assert captured["fallback_query_token_count"] == 1
        assert captured["fallback_lexicon_size"] == 2500
        assert partial_match_calls == search_module._DEFAULT_FALLBACK_CANDIDATE_LIMIT
        assert len(captured["candidate_ids"]) == search_module._partial_candidate_limit(1)
        assert "TARGET" in captured["candidate_ids"]

    def test_partial_query_token_proximity_fallback_respects_stage2_cap_with_explicit_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit token fallback caps should not widen partial-form stage-2 work."""
        captured: dict[str, object] = {}

        def fake_seed_stage(query_ipa: str, index: object, k: int = 2, **_kwargs: Any) -> list[str]:
            return []

        def fake_score_stage(
            query_ipa: str,
            candidates: Iterable[str],
            lexicon_map: dict[str, object],
            matrix: object,
            **_kwargs: Any,
        ) -> list[SearchResult]:
            captured["candidate_ids"] = list(candidates)
            return []

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": query)
        monkeypatch.setattr(search_module, "seed_stage", fake_seed_stage)
        monkeypatch.setattr(search_module, "_score_stage", fake_score_stage)
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda query_ipa, results, lexicon_map, matrix, language="ancient_greek", **_kwargs: results,
        )

        lexicon = [
            {
                "id": f"N{index:02d}",
                "headword": f"noise-{index:02d}",
                "ipa": f"{chr(ord('b') + index)} c",
                "dialect": "attic",
            }
            for index in range(30)
        ]
        lexicon.append({"id": "TARGET", "headword": "target", "ipa": "a x x x x c", "dialect": "attic"})

        search(
            "a*c",
            lexicon,
            matrix={},
            max_results=1,
            index={},
            unigram_index={},
            similarity_fallback_limit=31,
        )

        assert len(captured["candidate_ids"]) == search_module._partial_candidate_limit(1)
        assert "TARGET" in captured["candidate_ids"]

    def test_partial_query_unigram_fallback_is_bounded_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Partial-form k=1 fallback should rank unigram hits with the default cap."""
        captured: dict[str, object] = {}
        unigram_hits = [f"N{index:02d}" for index in range(30)] + ["TARGET"]

        def fake_seed_stage(query_ipa: str, index: object, k: int = 2, **_kwargs: Any) -> list[str]:
            return [] if k == 2 else unigram_hits

        def fake_rank_by_token_count_proximity(
            query_ipa: str,
            lexicon_map: dict[str, LexiconRecord],
            *,
            max_candidates: int | None = None,
            query_token_count: int | None = None,
            **_kwargs: Any,
        ) -> list[str]:
            captured["fallback_max_candidates"] = max_candidates
            captured["fallback_query_token_count"] = query_token_count
            captured["fallback_lexicon_keys"] = list(lexicon_map)
            return list(lexicon_map)[:max_candidates]

        def fake_score_stage(
            query_ipa: str,
            candidates: Iterable[str],
            lexicon_map: dict[str, object],
            matrix: object,
            **_kwargs: Any,
        ) -> list[SearchResult]:
            captured["candidate_ids"] = list(candidates)
            return []

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": query)
        monkeypatch.setattr(search_module, "seed_stage", fake_seed_stage)
        monkeypatch.setattr(search_module, "_score_stage", fake_score_stage)
        monkeypatch.setattr(search_module, "_rank_by_token_count_proximity", fake_rank_by_token_count_proximity)
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda query_ipa, results, lexicon_map, matrix, language="ancient_greek", **_kwargs: results,
        )

        lexicon = [
            {
                "id": f"N{index:02d}",
                "headword": f"noise-{index:02d}",
                "ipa": f"{chr(ord('b') + index)} c",
                "dialect": "attic",
            }
            for index in range(30)
        ]
        lexicon.append({"id": "TARGET", "headword": "target", "ipa": "a x x x x c", "dialect": "attic"})

        search("a*c", lexicon, matrix={}, max_results=1, index={})

        assert captured["fallback_max_candidates"] == search_module._DEFAULT_FALLBACK_CANDIDATE_LIMIT
        assert captured["fallback_query_token_count"] == 2
        assert captured["fallback_lexicon_keys"] == unigram_hits
        assert len(captured["candidate_ids"]) == search_module._partial_candidate_limit(1)
        assert "TARGET" in captured["candidate_ids"]

    def test_partial_query_unigram_fallback_respects_stage2_cap_with_explicit_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit unigram fallback caps should not widen partial-form stage-2 work."""
        captured: dict[str, object] = {}
        unigram_hits = [f"N{index:02d}" for index in range(30)] + ["TARGET"]

        def fake_seed_stage(query_ipa: str, index: object, k: int = 2, **_kwargs: Any) -> list[str]:
            return [] if k == 2 else unigram_hits

        def fake_score_stage(
            query_ipa: str,
            candidates: Iterable[str],
            lexicon_map: dict[str, object],
            matrix: object,
            **_kwargs: Any,
        ) -> list[SearchResult]:
            captured["candidate_ids"] = list(candidates)
            return []

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": query)
        monkeypatch.setattr(search_module, "seed_stage", fake_seed_stage)
        monkeypatch.setattr(search_module, "_score_stage", fake_score_stage)
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda query_ipa, results, lexicon_map, matrix, language="ancient_greek", **_kwargs: results,
        )

        lexicon = [
            {
                "id": f"N{index:02d}",
                "headword": f"noise-{index:02d}",
                "ipa": f"{chr(ord('b') + index)} c",
                "dialect": "attic",
            }
            for index in range(30)
        ]
        lexicon.append({"id": "TARGET", "headword": "target", "ipa": "a x x x x c", "dialect": "attic"})

        search(
            "a*c",
            lexicon,
            matrix={},
            max_results=1,
            index={},
            unigram_fallback_limit=31,
        )

        assert len(captured["candidate_ids"]) == search_module._partial_candidate_limit(1)
        assert "TARGET" in captured["candidate_ids"]

    def test_unigram_fallback_logs_missing_tokenized_candidates(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Missing tokenized unigram candidates should warn and continue."""
        captured: dict[str, object] = {}

        def fake_seed_stage(query_ipa: str, index: object, k: int = 2, **_kwargs: Any) -> list[str]:
            return [] if k == 2 else ["L2", "missing", "L1", "missing"]

        def fake_build_lexicon_map(
            lexicon: list[dict[str, str]],
            **_kwargs: Any,
        ) -> dict[str, LexiconRecord]:
            return {
                "L1": LexiconRecord(entry=lexicon[0], token_count=1, ipa_tokens=("p",)),
                "L2": LexiconRecord(entry=lexicon[1], token_count=2, ipa_tokens=("p", "a")),
            }

        def fake_score_stage(
            query_ipa: str,
            candidates: Iterable[str],
            lexicon_map: dict[str, object],
            matrix: object,
            **_kwargs: Any,
        ) -> list[SearchResult]:
            captured["candidate_ids"] = list(candidates)
            captured["lookup_keys"] = list(lexicon_map)
            return []

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "pa")
        monkeypatch.setattr(search_module, "seed_stage", fake_seed_stage)
        monkeypatch.setattr(search_module, "build_lexicon_map", fake_build_lexicon_map)
        monkeypatch.setattr(search_module, "_score_stage", fake_score_stage)
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda query_ipa, results, lexicon_map, matrix, language="ancient_greek", **_kwargs: results,
        )
        monkeypatch.setattr(search_module, "filter_stage", lambda results, max_results: results)

        lexicon = [
            {"id": "L1", "headword": "alpha", "ipa": "p", "dialect": "attic"},
            {"id": "L2", "headword": "beta", "ipa": "pa", "dialect": "attic"},
        ]

        with caplog.at_level("WARNING"):
            search(
                "query",
                lexicon,
                matrix={},
                max_results=1,
                index={},
                unigram_index={},
                unigram_fallback_limit=2,
            )

        assert captured["candidate_ids"] == ["L2", "L1"]
        assert captured["lookup_keys"] == ["L1", "L2"]
        expected_label = search_module._summarize_query_ipa_for_logs(
            "pa",
            query_token_count=len(search_module.tokenize_ipa("pa")),
            debug_enabled=False,
        )
        assert expected_label in caplog.text
        assert "query IPA 'pa'" not in caplog.text
        assert "missing" in caplog.text

    def test_search_debug_logs_redacted_query_label_for_performance_metrics(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Performance debug logs should use a redacted query label only."""
        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "p t e n")
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: ["L1"])
        monkeypatch.setattr(
            search_module,
            "_score_stage",
            lambda query_ipa, candidates, lexicon_map, matrix: [
                SearchResult(
                    lemma="alpha",
                    confidence=1.0,
                    dialect_attribution="lemma dialect: attic",
                    entry_id="L1",
                )
            ],
        )
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda query_ipa, results, lexicon_map, matrix, language="ancient_greek", **_kwargs: list(results),
        )

        lexicon = [{"id": "L1", "headword": "alpha", "ipa": "p t e n", "dialect": "attic"}]

        caplog.set_level("DEBUG", logger="phonology.search")
        search("λόγος", lexicon, matrix={}, max_results=1, index={})

        debug_messages = "\n".join(
            record.getMessage() for record in caplog.records if record.levelname == "DEBUG"
        )

        assert "candidate selection completed" in debug_messages
        assert "scoring completed" in debug_messages
        assert "annotation/final filtering completed" in debug_messages
        assert "λόγος" not in debug_messages
        assert "p t e n" not in debug_messages
        assert "sha256=" in debug_messages

    def test_search_passes_debug_disabled_flag_when_debug_logging_is_off(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Normal search path should skip hash generation when DEBUG logging is off."""
        captured: list[bool] = []

        def fake_summarize_query_ipa_for_logs(
            query_ipa: str,
            *,
            query_token_count: int,
            debug_enabled: bool = True,
        ) -> str:
            captured.append(debug_enabled)
            return "tokens=2 chars=2 sha256=<debug-disabled>"

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "pa")
        monkeypatch.setattr(
            search_module,
            "_summarize_query_ipa_for_logs",
            fake_summarize_query_ipa_for_logs,
        )

        lexicon = [{"id": "L1", "headword": "alpha", "ipa": "pa", "dialect": "attic"}]

        caplog.set_level("INFO", logger="phonology.search")
        search("λόγος", lexicon, matrix={}, max_results=1, index={})

        assert captured == [False]

    def test_search_default_paths_never_use_uncapped_token_proximity_sort(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every default token-proximity path should pass an explicit candidate cap."""
        rank_calls = 0

        def fake_rank_by_token_count_proximity(
            query_ipa: str,
            lexicon_map: dict[str, LexiconRecord],
            *,
            max_candidates: int | None = None,
            query_token_count: int | None = None,
            **_kwargs: Any,
        ) -> list[str]:
            nonlocal rank_calls
            rank_calls += 1
            assert max_candidates is not None
            return ["L1"]

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "aː")
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: [])
        monkeypatch.setattr(
            search_module,
            "_rank_by_token_count_proximity",
            fake_rank_by_token_count_proximity,
        )
        monkeypatch.setattr(
            search_module,
            "_score_stage",
            lambda query_ipa, candidates, lexicon_map, matrix: [
                SearchResult(
                    lemma="alpha",
                    confidence=1.0,
                    dialect_attribution="lemma dialect: attic",
                    entry_id="L1",
                )
            ],
        )
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda query_ipa, results, lexicon_map, matrix, language="ancient_greek", **_kwargs: list(results),
        )

        lexicon = [{"id": "L1", "headword": "alpha", "ipa": "aː", "dialect": "attic"}]

        search("query", lexicon, matrix={}, max_results=1, index={}, unigram_index={})

        assert rank_calls > 0

    def test_search_skips_building_unigram_index_for_pure_vowel_query(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pure-vowel queries should jump straight to token-proximity fallback."""
        build_k_calls: list[int] = []
        captured: dict[str, object] = {}
        prebuilt_lexicon_map = {
            "L1": LexiconRecord(
                entry={"id": "L1", "headword": "alpha", "ipa": "aː", "dialect": "attic"},
                token_count=1,
            )
        }

        def fake_build_kmer_index(
            lexicon: object, k: int = 2
        ) -> dict[str, list[str]]:
            build_k_calls.append(k)
            return {}

        def fake_rank_by_token_count_proximity(
            query_ipa: str,
            lexicon_map: dict[str, LexiconRecord],
            *,
            max_candidates: int | None = None,
            query_token_count: int | None = None,
            **_kwargs: Any,
        ) -> list[str]:
            captured["query_token_count"] = query_token_count
            captured["lookup_keys"] = list(lexicon_map)
            return ["L1"]

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "aː")
        monkeypatch.setattr(search_module, "build_kmer_index", fake_build_kmer_index)
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: [])
        monkeypatch.setattr(search_module, "_rank_by_token_count_proximity", fake_rank_by_token_count_proximity)
        monkeypatch.setattr(
            search_module,
            "extend_stage",
            lambda query_ipa, candidates, lexicon_map, matrix, language="ancient_greek": [],
        )
        monkeypatch.setattr(search_module, "filter_stage", lambda results, max_results: results)

        lexicon = [
            {"id": "L1", "headword": "alpha", "ipa": "aː", "dialect": "attic"},
        ]

        search(
            "query",
            lexicon,
            matrix={},
            max_results=1,
            index={},
            prebuilt_lexicon_map=prebuilt_lexicon_map,
        )

        assert build_k_calls == []
        assert captured["query_token_count"] == 1
        assert captured["lookup_keys"] == ["L1"]

    def test_search_uses_to_ipa_and_respects_max_results(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}

        def fake_to_ipa(query: str, dialect: str = "attic") -> str:
            captured["query"] = query
            captured["dialect"] = dialect
            return "pten"

        monkeypatch.setattr(search_module, "to_ipa", fake_to_ipa)
        monkeypatch.setattr(search_module, "load_rules", lambda _path: {})
        lexicon = [
            {"id": "L1", "headword": "πτην", "ipa": "pten", "dialect": "attic"},
            {"id": "L2", "headword": "πτω", "ipa": "pto", "dialect": "attic"},
            {"id": "L3", "headword": "κτην", "ipa": "kten", "dialect": "doric"},
        ]

        results = search("πτην", lexicon, matrix={}, max_results=1)

        assert captured == {"query": "πτην", "dialect": "attic"}
        assert [result.lemma for result in results] == ["πτην"]

    def test_search_accepts_explicit_dialect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def fake_to_ipa(query: str, dialect: str = "attic") -> str:
            captured["query"] = query
            captured["dialect"] = dialect
            return "pten"

        monkeypatch.setattr(search_module, "to_ipa", fake_to_ipa)
        monkeypatch.setattr(search_module, "load_rules", lambda _path: {})
        lexicon = [
            {"id": "L1", "headword": "πτην", "ipa": "pten", "dialect": "attic"},
        ]

        results = search("πτην", lexicon, matrix={}, dialect="ionic", max_results=1)

        assert captured == {"query": "πτην", "dialect": "ionic"}
        assert [result.lemma for result in results] == ["πτην"]

    def test_search_reuses_prebuilt_index(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def raise_should_not_rebuild_index(
            lexicon: list[dict[str, str]], k: int = 2
        ) -> search_module.KmerIndex:
            raise AssertionError("should not rebuild index")

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "pten")
        monkeypatch.setattr(search_module, "load_rules", lambda _path: {})
        monkeypatch.setattr(
            search_module,
            "build_kmer_index",
            raise_should_not_rebuild_index,
        )
        lexicon = [
            {"id": "L1", "headword": "πτην", "ipa": "pten", "dialect": "attic"},
            {"id": "L2", "headword": "πτω", "ipa": "pto", "dialect": "attic"},
        ]
        index = {"p t": ["L1", "L2"], "t n": ["L1"]}

        results = search("πτην", lexicon, matrix={}, max_results=1, index=index)

        assert [result.lemma for result in results] == ["πτην"]

    def test_search_with_prebuilt_hit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A matching prebuilt IPA index key should be used without rebuilding."""
        captured_candidates: list[str] = []

        def fail_build_lexicon_map(lexicon: object) -> search_module.LexiconMap:
            raise AssertionError("should not build tokenized lexicon map")

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "pten")
        monkeypatch.setattr(search_module, "load_rules", lambda _path: {})
        monkeypatch.setattr(search_module, "build_ipa_index", self._raise_should_not_rebuild_ipa_index)
        monkeypatch.setattr(search_module, "build_lexicon_map", fail_build_lexicon_map)
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: ["L2"])

        monkeypatch.setattr(
            search_module,
            "_score_stage",
            self._make_capturing_score_stage(captured_candidates, use_headword=False),
        )
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda **kwargs: kwargs["results"],
        )
        lexicon = [
            {"id": "L1", "headword": "πτην", "ipa": "pten", "dialect": "attic"},
            {"id": "L2", "headword": "πτω", "ipa": "pto", "dialect": "attic"},
        ]

        results = search(
            "πτην",
            lexicon,
            matrix={},
            max_results=1,
            prebuilt_ipa_index={"pten": ["L1"]},
        )

        assert captured_candidates == ["L1", "L2"]
        assert [result.lemma for result in results] == ["L1"]

    def test_search_with_empty_prebuilt_ipa_index(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An empty prebuilt IPA index should not inject an exact IPA match."""
        captured_candidates: list[str] = []

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "pten")
        monkeypatch.setattr(search_module, "load_rules", lambda _path: {})
        monkeypatch.setattr(search_module, "build_ipa_index", self._raise_should_not_rebuild_ipa_index)
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: ["L2"])

        monkeypatch.setattr(
            search_module,
            "_score_stage",
            self._make_capturing_score_stage(captured_candidates),
        )
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda **kwargs: kwargs["results"],
        )
        lexicon = [
            {"id": "L1", "headword": "πτην", "ipa": "pten", "dialect": "attic"},
            {"id": "L2", "headword": "πτω", "ipa": "pto", "dialect": "attic"},
        ]

        results = search("πτην", lexicon, matrix={}, max_results=1, prebuilt_ipa_index={})

        assert captured_candidates == ["L2"]
        assert results[0].lemma == "πτω"

    def test_search_with_missing_ipa_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A missing query key in prebuilt IPA index should not inject exact matches."""
        captured_candidates: list[str] = []

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "pten")
        monkeypatch.setattr(search_module, "load_rules", lambda _path: {})
        monkeypatch.setattr(search_module, "build_ipa_index", self._raise_should_not_rebuild_ipa_index)
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: ["L2"])

        monkeypatch.setattr(
            search_module,
            "_score_stage",
            self._make_capturing_score_stage(captured_candidates),
        )
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda **kwargs: kwargs["results"],
        )
        lexicon = [
            {"id": "L1", "headword": "πτην", "ipa": "pten", "dialect": "attic"},
            {"id": "L2", "headword": "πτω", "ipa": "pto", "dialect": "attic"},
        ]

        results = search(
            "πτην",
            lexicon,
            matrix={},
            max_results=1,
            prebuilt_ipa_index={"other": ["L1"]},
        )

        assert captured_candidates == ["L2"]
        assert results[0].lemma == "πτω"

    def test_koine_seed_index_keeps_attic_target_within_stage2_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            search_module,
            "to_ipa",
            lambda query, dialect="attic": "aflas" if dialect == "koine" else "apʰlas",
        )
        lexicon = [
            {
                "id": f"D{index:02d}",
                "headword": f"distractor-{index:02d}",
                "ipa": "nals",
                "dialect": "attic",
            }
            for index in range(60)
        ]
        lexicon.append(
            {
                "id": "TARGET",
                "headword": "target",
                "ipa": "apʰlas",
                "dialect": "attic",
            }
        )

        results = search(
            "ignored",
            lexicon,
            matrix=load_matrix(MATRIX_FILE),
            max_results=5,
            dialect="koine",
            index=build_kmer_index(lexicon),
        )

        assert len(results) > 0
        assert len(results) <= 5
        assert results[0].lemma == "target"

    def test_short_koine_query_keeps_attic_source_as_top_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Short koine query should match attic χ→kʰ; distractor shares consonant."""
        monkeypatch.setattr(
            search_module,
            "to_ipa",
            lambda query, dialect="attic": "xa" if dialect == "koine" else "kʰa",
        )
        lexicon = [
            {"id": "TARGET", "headword": "target", "ipa": "kʰa", "dialect": "attic"},
            {"id": "DISTRACTOR", "headword": "alpha", "ipa": "kʰe", "dialect": "attic"},
        ]

        results = search(
            "ignored",
            lexicon,
            matrix=load_matrix(MATRIX_FILE),
            max_results=2,
            dialect="koine",
        )

        assert results[0].lemma == "target"
        assert [result.lemma for result in results] == ["target", "alpha"]

    @pytest.mark.parametrize(
        ("language_kwarg", "expected_language"),
        [
            ({"language": "test_lang"}, "test_lang"),
            ({}, "ancient_greek"),
        ],
        ids=["explicit_language", "default_language"],
    )
    def test_search_forwards_language_to_extend_stage(
        self,
        monkeypatch: pytest.MonkeyPatch,
        language_kwarg: dict[str, str],
        expected_language: str,
    ) -> None:
        captured: dict[str, object] = {}

        def fake_annotate_results(
            query_ipa: str,
            results: list[SearchResult],
            lexicon_map: dict[str, dict[str, object]],
            matrix: object,
            language: str = "ancient_greek",
            **_kwargs: Any,
        ) -> list[SearchResult]:
            captured["language"] = language
            return results

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "pten")
        monkeypatch.setattr(
            search_module,
            "_score_stage",
            lambda query_ipa, candidates, lexicon_map, matrix: [
                SearchResult(
                    lemma="πτην",
                    confidence=1.0,
                    dialect_attribution="lemma dialect: attic",
                    entry_id="L1",
                )
            ],
        )
        monkeypatch.setattr(search_module, "_annotate_search_results", fake_annotate_results)
        monkeypatch.setattr(
            search_module, "filter_stage", lambda results, max_results: results[:max_results]
        )
        lexicon = [
            {"id": "L1", "headword": "πτην", "ipa": "pten", "dialect": "attic"},
        ]

        search("πτην", lexicon, matrix={}, max_results=1, **language_kwarg)

        assert captured["language"] == expected_language

    def test_rejects_both_prepared_query_and_query_ipa(self) -> None:
        """search() must raise when both prepared_query and query_ipa are provided."""
        prepared = search_module.prepare_query_ipa("λόγος", dialect="attic")
        lexicon: list[dict[str, object]] = [
            {"id": "L1", "headword": "λόγος", "ipa": "loɡos", "dialect": "attic"},
        ]

        with pytest.raises(ValueError, match="Pass either prepared_query or query_ipa"):
            search(
                "λόγος",
                lexicon,
                matrix={},
                max_results=1,
                prepared_query=prepared,
                query_ipa="loɡos",
            )
