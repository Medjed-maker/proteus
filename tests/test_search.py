"""Tests for proteus.phonology.search."""

from __future__ import annotations

from collections.abc import Generator

import pytest
import yaml

from proteus.phonology import search as search_module
from proteus.phonology.explainer import RuleApplication
from proteus.phonology.search import (
    SearchResult,
    build_kmer_index,
    extend_stage,
    filter_stage,
    search,
    seed_stage,
)


@pytest.fixture(autouse=True)
def clear_rule_cache() -> Generator[None, None, None]:
    """Reset cached rule registry state between tests."""
    search_module._get_rules_registry.cache_clear()

    yield

    search_module._get_rules_registry.cache_clear()


@pytest.fixture
def sample_search_results() -> list[SearchResult]:
    """Return a small unsorted result list for ranking tests."""
    return [
        SearchResult(lemma="γ", confidence=0.5, dialect_attribution="lemma dialect: attic"),
        SearchResult(lemma="α", confidence=0.9, dialect_attribution="lemma dialect: attic"),
        SearchResult(lemma="β", confidence=0.9, dialect_attribution="lemma dialect: attic"),
    ]


@pytest.fixture
def sample_lexicon() -> list[dict[str, str]]:
    """Return a compact lexicon fixture with deterministic ids and IPA."""
    return [
        {"id": "L1", "headword": "πτην", "ipa": "pten", "dialect": "attic"},
        {"id": "L2", "headword": "πτω", "ipa": "pto", "dialect": "attic"},
        {"id": "L3", "headword": "κτην", "ipa": "kten", "dialect": "doric"},
    ]


class TestFilterStage:
    """Verify final filtering sorts by confidence and rejects invalid limits."""

    def test_sorts_by_confidence_desc_then_lemma(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        filtered = filter_stage(sample_search_results, max_results=2)

        assert [(result.lemma, result.confidence) for result in filtered] == [
            ("α", 0.9),
            ("β", 0.9),
        ]

    @pytest.mark.parametrize("max_results", [0, -1])
    def test_rejects_non_positive_max_results(self, max_results: int) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            filter_stage([], max_results=max_results)


class TestBuildKmerIndex:
    """Verify consonant-skeleton k-mer index construction and parameter validation."""

    def test_builds_consonant_skeleton_kmer_index(
        self, sample_lexicon: list[dict[str, str]]
    ) -> None:
        index = build_kmer_index(sample_lexicon, k=2)

        assert index["p t"] == ["L1", "L2"]
        assert index["t n"] == ["L1", "L3"]
        assert "p n" not in index

    @pytest.mark.parametrize("k", [0, -1])
    def test_rejects_non_positive_k(self, k: int) -> None:
        with pytest.raises(ValueError, match="build_kmer_index.*k"):
            build_kmer_index([], k=k)


class TestSeedStage:
    """Verify stage-1 seed ranking by shared consonant-skeleton k-mers."""

    def test_ranks_candidates_by_shared_seed_count(
        self, sample_lexicon: list[dict[str, str]]
    ) -> None:
        index = build_kmer_index(sample_lexicon, k=2)

        candidates = seed_stage("pten", index, k=2)

        assert candidates == ["L1", "L2", "L3"]

    def test_returns_empty_list_when_query_has_no_seedable_consonant_skeleton(self) -> None:
        assert seed_stage("aː", {"p t": ["L1"]}, k=2) == []


class TestExtendStage:
    """Verify stage-2 Smith-Waterman extension, rule detection, and dialect attribution."""

    def test_get_rules_registry_uses_requested_language(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}

        def fake_load_rules(language: str) -> dict[str, dict[str, object]]:
            captured["language"] = language
            return {"RULE": {"id": "RULE"}}

        monkeypatch.setattr(search_module, "load_rules", fake_load_rules)

        assert search_module._get_rules_registry("test_language") == {
            "RULE": {"id": "RULE"}
        }
        assert captured == {"language": "test_language"}

    def test_get_rules_registry_raises_descriptive_error_for_invalid_language(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raise_value_error(language: str) -> None:
            """Raise ValueError to simulate missing rules for the given language."""
            raise ValueError(f"missing {language}")

        monkeypatch.setattr(
            search_module,
            "load_rules",
            _raise_value_error,
        )

        with pytest.raises(
            ValueError,
            match="_get_rules_registry failed to load rules for language 'missing_language'",
        ):
            search_module._get_rules_registry("missing_language")

    def test_get_rules_registry_wraps_yaml_parse_errors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raise_yaml_error(language: str) -> None:
            """Raise YAMLError to simulate a parse failure for the given language."""
            raise yaml.YAMLError(f"bad yaml for {language}")

        monkeypatch.setattr(
            search_module,
            "load_rules",
            _raise_yaml_error,
        )

        with pytest.raises(
            ValueError,
            match="_get_rules_registry failed to load rules for language 'broken_language'",
        ):
            search_module._get_rules_registry("broken_language")

    def test_apply_rule_markers_does_not_mutate_input_markers(self) -> None:
        baseline_markers = [".", ".", "."]
        applications = [
            RuleApplication(
                rule_id="RULE-1",
                rule_name="Rule 1",
                input_phoneme="b",
                output_phoneme="d",
                dialects=["attic"],
                position=1,
            )
        ]

        updated_markers = search_module._apply_rule_markers(
            baseline_markers,
            aligned_query=["a", "d", "c"],
            aligned_lemma=["a", "b", "c"],
            applications=applications,
        )

        assert baseline_markers == [".", ".", "."]
        assert updated_markers == [".", ":", "."]

    def test_collect_application_dialects_preserves_first_seen_order(self) -> None:
        applications = [
            RuleApplication(
                rule_id="RULE-1",
                rule_name="Rule 1",
                input_phoneme="a",
                output_phoneme="b",
                dialects=["ionic", "attic"],
                position=0,
            ),
            RuleApplication(
                rule_id="RULE-2",
                rule_name="Rule 2",
                input_phoneme="b",
                output_phoneme="c",
                dialects=["attic", "doric"],
                position=1,
            ),
        ]

        assert search_module._collect_application_dialects(applications) == [
            "ionic",
            "attic",
            "doric",
        ]

    def test_exact_match_returns_confidence_one_and_three_line_visualization(self) -> None:
        lexicon_map = {
            "L1": {"headword": "λόγος", "ipa": "lóɡos", "dialect": "attic"}
        }

        results = extend_stage("lóɡos", ["L1"], lexicon_map, matrix={})

        assert len(results) == 1
        result = results[0]
        assert result.lemma == "λόγος"
        assert result.ipa == "lóɡos"
        assert result.confidence == pytest.approx(1.0)
        assert result.applied_rules == []
        assert result.rule_applications == []
        assert result.dialect_attribution == "lemma dialect: attic"
        lines = result.alignment_visualization.splitlines()
        assert len(lines) == 3
        assert lines[0].startswith("query:")
        assert lines[1].startswith("       ")
        assert lines[2].startswith("lemma:")

    def test_extend_stage_uses_requested_language_for_rules_registry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}

        def fake_get_rules_registry(language: str = "ancient_greek") -> dict[str, dict[str, object]]:
            captured["language"] = language
            return {}

        monkeypatch.setattr(search_module, "_get_rules_registry", fake_get_rules_registry)

        extend_stage(
            "aː",
            ["L1"],
            {"L1": {"headword": "γᾱ", "ipa": "aː", "dialect": "attic"}},
            matrix={},
            language="test_language",
        )

        assert captured == {"language": "test_language"}

    def test_detects_single_token_rule_and_reports_dialect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            search_module,
            "load_rules",
            lambda _path: {
                "VSH-TEST": {
                    "id": "VSH-TEST",
                    "input": "aː",
                    "output": "ɛː",
                    "dialects": ["ionic"],
                }
            },
        )
        lexicon_map = {
            "L1": {"headword": "γᾱ", "ipa": "aː", "dialect": "attic"}
        }

        results = extend_stage("ɛː", ["L1"], lexicon_map, matrix={"aː": {"ɛː": 0.3}})

        assert results[0].applied_rules == ["VSH-TEST"]
        assert [application.position for application in results[0].rule_applications] == [0]
        assert (
            results[0].dialect_attribution
            == "lemma dialect: attic; query-compatible dialects: ionic"
        )
        assert ":" in results[0].alignment_visualization.splitlines()[1]

    def test_prefers_contextual_rule_from_shared_explainer_logic(self) -> None:
        lexicon_map = {
            "L1": {"headword": "χώρα", "ipa": "rɛː", "dialect": "attic"}
        }

        results = extend_stage("raː", ["L1"], lexicon_map, matrix={"ɛː": {"aː": 0.1}})

        assert results[0].applied_rules == ["VSH-010"]
        assert [application.position for application in results[0].rule_applications] == [1]
        assert (
            results[0].dialect_attribution
            == "lemma dialect: attic; query-compatible dialects: attic"
        )
        assert ":" in results[0].alignment_visualization.splitlines()[1]

    def test_detects_multi_token_rule(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            search_module,
            "load_rules",
            lambda _path: {
                "CCH-TEST": {
                    "id": "CCH-TEST",
                    "input": "ss",
                    "output": "tt",
                    "dialects": ["attic"],
                }
            },
        )
        lexicon_map = {
            "L1": {"headword": "θάλασσα", "ipa": "ss", "dialect": "ionic"}
        }

        results = extend_stage("tt", ["L1"], lexicon_map, matrix={"s": {"t": 0.4}})

        assert results[0].applied_rules == ["CCH-TEST"]
        assert [application.position for application in results[0].rule_applications] == [0]
        assert (
            results[0].dialect_attribution
            == "lemma dialect: ionic; query-compatible dialects: attic"
        )

    def test_detects_deletion_rule_inside_alignment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            search_module,
            "load_rules",
            lambda _path: {
                "CCH-DEL": {
                    "id": "CCH-DEL",
                    "input": "s",
                    "output": "",
                    "dialects": ["ionic"],
                }
            },
        )
        lexicon_map = {
            "L1": {"headword": "γένεσος", "ipa": "asa", "dialect": "attic"}
        }

        results = extend_stage("aa", ["L1"], lexicon_map, matrix={})

        assert results[0].applied_rules == ["CCH-DEL"]
        assert [application.position for application in results[0].rule_applications] == [1]
        assert (
            results[0].dialect_attribution
            == "lemma dialect: attic; query-compatible dialects: ionic"
        )

    def test_skips_stale_candidates_missing_from_lexicon_map(self) -> None:
        lexicon_map = {
            "L1": {"headword": "λόγος", "ipa": "lóɡos", "dialect": "attic"}
        }

        results = extend_stage("lóɡos", ["L1", "missing"], lexicon_map, matrix={})

        assert [result.lemma for result in results] == ["λόγος"]

    def test_ignores_non_list_rule_dialects(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            search_module,
            "load_rules",
            lambda _path: {
                "VSH-TEST": {
                    "id": "VSH-TEST",
                    "input": "aː",
                    "output": "ɛː",
                    "dialects": "attic",
                }
            },
        )
        lexicon_map = {
            "L1": {"headword": "γᾱ", "ipa": "aː", "dialect": "attic"}
        }

        results = extend_stage("ɛː", ["L1"], lexicon_map, matrix={"aː": {"ɛː": 0.3}})

        assert results[0].applied_rules == ["VSH-TEST"]
        assert results[0].dialect_attribution == "lemma dialect: attic"

    @pytest.mark.parametrize("dialect", [None, "", "   "])
    def test_uses_unknown_when_entry_dialect_is_missing_or_blank(
        self, monkeypatch: pytest.MonkeyPatch, dialect: object
    ) -> None:
        monkeypatch.setattr(search_module, "load_rules", lambda _path: {})
        lexicon_map = {"L1": {"headword": "γᾱ", "ipa": "aː", "dialect": dialect}}

        results = extend_stage("aː", ["L1"], lexicon_map, matrix={})

        assert results[0].dialect_attribution == "lemma dialect: unknown"

    def test_near_zero_score_prefers_diagonal_traceback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(search_module, "_GAP_PENALTY", 1e-12)
        monkeypatch.setattr(search_module, "_substitution_score", lambda *_args: 0.0)

        score, aligned_query, aligned_lemma = search_module._smith_waterman_alignment(
            ["q"],
            ["l"],
            matrix={},
        )

        assert score == pytest.approx(1e-12)
        assert aligned_query == ["q"]
        assert aligned_lemma == ["l"]


class TestSearch:
    """Verify full three-stage search pipeline including IPA conversion and index reuse."""

    def test_rejects_blank_query(self) -> None:
        with pytest.raises(ValueError, match="query must be a non-empty string"):
            search("   ", [], matrix={})

    def test_falls_back_to_full_lexicon_when_query_has_no_seedable_skeleton(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "aː")
        monkeypatch.setattr(search_module, "load_rules", lambda _path: {})
        lexicon = [
            {"id": "L1", "headword": "alpha", "ipa": "aː", "dialect": "attic"},
            {"id": "L2", "headword": "iota", "ipa": "i", "dialect": "attic"},
            {"id": "L3", "headword": "upsilon", "ipa": "y", "dialect": "attic"},
        ]

        results = search("ἄ", lexicon, matrix={}, max_results=2)

        assert [result.lemma for result in results] == ["alpha", "iota"]

    def test_full_lexicon_fallback_passes_all_candidates_to_extend_stage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}

        def fake_extend_stage(
            query_ipa: str,
            candidates: object,
            lexicon_map: dict[str, dict[str, object]],
            matrix: dict[str, dict[str, float]],
            language: str = "ancient_greek",
        ) -> list[SearchResult]:
            captured["query_ipa"] = query_ipa
            captured["candidate_type"] = type(candidates)
            captured["candidate_ids"] = list(candidates)
            return [
                SearchResult(
                    lemma=lexicon_map[candidate_id]["headword"],
                    confidence=1.0,
                    dialect_attribution="lemma dialect: attic",
                )
                for candidate_id in captured["candidate_ids"]
            ]

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "aː")
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: [])
        monkeypatch.setattr(search_module, "extend_stage", fake_extend_stage)
        monkeypatch.setattr(search_module, "filter_stage", lambda results, max_results: results[:max_results])
        lexicon = [
            {
                "id": f"L{index}",
                "headword": f"lemma-{index}",
                "ipa": "aː",
                "dialect": "attic",
            }
            for index in range(30)
        ]

        results = search("ἄ", lexicon, matrix={}, max_results=1)

        assert captured["query_ipa"] == "aː"
        assert captured["candidate_type"].__name__ == "dict_keys"
        assert captured["candidate_ids"] == [f"L{index}" for index in range(30)]
        assert [result.lemma for result in results] == ["lemma-0"]

    def test_full_lexicon_fallback_can_find_exact_match_beyond_previous_cap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "aː")
        monkeypatch.setattr(search_module, "load_rules", lambda _path: {})
        lexicon = [
            {
                "id": f"L{index}",
                "headword": f"lemma-{index}",
                "ipa": "i",
                "dialect": "attic",
            }
            for index in range(29)
        ]
        lexicon.append(
            {"id": "L29", "headword": "target", "ipa": "aː", "dialect": "attic"}
        )

        results = search("ἄ", lexicon, matrix={}, max_results=1)

        assert [result.lemma for result in results] == ["target"]

    def test_falls_back_to_full_lexicon_when_seed_lookup_finds_no_hits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "mn")
        monkeypatch.setattr(search_module, "load_rules", lambda _path: {})
        lexicon = [
            {"id": "L1", "headword": "πτω", "ipa": "pto", "dialect": "attic"},
            {"id": "L2", "headword": "κτω", "ipa": "kto", "dialect": "attic"},
        ]

        results = search("μν", lexicon, matrix={}, max_results=2)

        assert len(results) == 2
        assert {result.lemma for result in results} == {"πτω", "κτω"}

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

        def fake_extend_stage(
            query_ipa: str,
            candidates: object,
            lexicon_map: dict[str, dict[str, object]],
            matrix: object,
            language: str = "ancient_greek",
        ) -> list[SearchResult]:
            captured["language"] = language
            return [
                SearchResult(
                    lemma="πτην",
                    confidence=1.0,
                    dialect_attribution="lemma dialect: attic",
                )
            ]

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "pten")
        monkeypatch.setattr(search_module, "extend_stage", fake_extend_stage)
        monkeypatch.setattr(
            search_module, "filter_stage", lambda results, max_results: results[:max_results]
        )
        lexicon = [
            {"id": "L1", "headword": "πτην", "ipa": "pten", "dialect": "attic"},
        ]

        search("πτην", lexicon, matrix={}, max_results=1, **language_kwarg)

        assert captured["language"] == expected_language
