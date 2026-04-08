"""Tests for phonology.search."""

from __future__ import annotations

from collections.abc import Generator, Iterable

import pytest
import yaml

from phonology import search as search_module
from phonology.distance import load_matrix
from phonology.explainer import RuleApplication
from phonology.ipa_converter import to_ipa, tokenize_ipa
from phonology.search import (
    LexiconRecord,
    SearchResult,
    _deduplicate_by_headword,
    _inject_exact_ipa_matches,
    build_kmer_index,
    build_lexicon_map,
    extend_stage,
    filter_stage,
    search,
    seed_stage,
)

MATRIX_FILE = "attic_doric.json"


@pytest.fixture(autouse=True)
def clear_rule_cache() -> Generator[None, None, None]:
    """Reset cached rule registry state between tests."""
    search_module._get_rules_registry.cache_clear()
    search_module._get_tokenized_rules.cache_clear()

    yield

    search_module._get_rules_registry.cache_clear()
    search_module._get_tokenized_rules.cache_clear()


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


class TestBuildLexiconMap:
    """Verify lexicon map construction and duplicate detection."""

    def test_returns_empty_dict_for_empty_lexicon(self) -> None:
        assert build_lexicon_map([]) == {}

    def test_builds_map_with_token_counts(self) -> None:
        lexicon = [
            {"id": "L1", "headword": "λόγος", "ipa": "lóɡos", "dialect": "attic"},
        ]
        result = build_lexicon_map(lexicon)
        assert "L1" in result
        assert result["L1"].entry == lexicon[0]
        assert result["L1"].token_count == 5

    def test_raises_on_duplicate_entry_id(self) -> None:
        lexicon = [
            {"id": "L1", "headword": "αλφα", "ipa": "alfa", "dialect": "attic"},
            {"id": "L1", "headword": "βητα", "ipa": "beta", "dialect": "attic"},
        ]
        with pytest.raises(ValueError, match="Duplicate entry ID"):
            build_lexicon_map(lexicon)

    @pytest.mark.parametrize("ipa_text", ["", "   "])
    def test_raises_on_empty_or_whitespace_only_ipa(self, ipa_text: str) -> None:
        lexicon = [
            {"id": "L1", "headword": "test", "ipa": ipa_text, "dialect": "attic"},
        ]

        with pytest.raises(ValueError, match="non-empty 'ipa'"):
            build_lexicon_map(lexicon)

    @pytest.mark.parametrize("ipa_text", ["!?", "ã"])
    def test_builds_map_for_non_empty_special_character_ipa(self, ipa_text: str) -> None:
        lexicon = [
            {"id": "L1", "headword": "test", "ipa": ipa_text, "dialect": "attic"},
        ]

        result = build_lexicon_map(lexicon)

        assert "L1" in result
        assert result["L1"].entry == lexicon[0]
        assert result["L1"].token_count == len(tokenize_ipa(ipa_text))


class TestBuildKmerIndex:
    """Verify consonant-skeleton k-mer index construction and parameter validation."""

    def test_builds_consonant_skeleton_kmer_index(
        self, sample_lexicon: list[dict[str, str]]
    ) -> None:
        index = build_kmer_index(sample_lexicon, k=2)

        assert index["p t"] == ["L1", "L2"]
        assert index["t n"] == ["L1", "L3"]
        assert "p n" not in index

    def test_adds_koine_compatible_kmers_without_duplicate_entry_ids(self) -> None:
        index = build_kmer_index(
            [{"id": "L1", "headword": "target", "ipa": "apʰlas", "dialect": "attic"}],
            k=2,
        )

        assert index["pʰ l"] == ["L1"]
        assert index["f l"] == ["L1"]
        assert index["l s"] == ["L1"]

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

    def test_k1_finds_candidates_for_single_consonant_query(self) -> None:
        """k=1 unigram index should find entries sharing a single consonant."""
        lexicon = [
            {"id": "L1", "headword": "πτω", "ipa": "pto", "dialect": "attic"},
            {"id": "L2", "headword": "κτω", "ipa": "kto", "dialect": "attic"},
            {"id": "L3", "headword": "ποι", "ipa": "poi", "dialect": "attic"},
        ]
        unigram_idx = build_kmer_index(lexicon, k=1)

        # Query IPA "poieɔː" has consonant skeleton ['p'], so k=1 produces ["p"].
        candidates = seed_stage("poieɔː", unigram_idx, k=1)

        # L1 (pto → skeleton p,t) and L3 (poi → skeleton p) contain "p"
        assert "L1" in candidates
        assert "L2" not in candidates
        assert "L3" in candidates

    def test_k1_returns_empty_for_pure_vowel_query(self) -> None:
        """k=1 should still return empty for a query with zero consonants."""
        unigram_idx: dict[str, list[str]] = {"p": ["L1"]}
        assert seed_stage("aː", unigram_idx, k=1) == []


class TestExtendStage:
    """Verify stage-2 extension behavior, including an integration-style packaged-data case.

    This class includes a test that depends on packaged resources
    ``attic_doric.json`` and rule ``MPH-013`` to exercise the runtime suffix
    matching path end-to-end.
    """

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

    def test_is_observed_application_matches_obs_prefix(self) -> None:
        # Canonical case: uppercase OBS- prefix with suffix
        assert search_module._is_observed_application(
            RuleApplication(
                rule_id="OBS-SUB",
                rule_name="Observed substitution",
                input_phoneme="a",
                output_phoneme="x",
                position=0,
            )
        )
        # Prefix only (still starts with OBS-)
        assert search_module._is_observed_application(
            RuleApplication(
                rule_id="OBS-",
                rule_name="Observed",
                input_phoneme="a",
                output_phoneme="x",
                position=0,
            )
        )
        # Non-observed rule
        assert not search_module._is_observed_application(
            RuleApplication(
                rule_id="RULE-1",
                rule_name="Rule 1",
                input_phoneme="a",
                output_phoneme="x",
                position=0,
            )
        )
        # Edge case: no dash after OBS
        assert not search_module._is_observed_application(
            RuleApplication(
                rule_id="OBS",
                rule_name="Observed",
                input_phoneme="a",
                output_phoneme="x",
                position=0,
            )
        )
        # Edge case: lowercase prefix
        assert not search_module._is_observed_application(
            RuleApplication(
                rule_id="obs-sub",
                rule_name="Observed",
                input_phoneme="a",
                output_phoneme="x",
                position=0,
            )
        )
        # Edge case: empty string
        assert not search_module._is_observed_application(
            RuleApplication(
                rule_id="",
                rule_name="Empty",
                input_phoneme="a",
                output_phoneme="x",
                position=0,
            )
        )

    def test_apply_rule_markers_ignores_observed_steps(self) -> None:
        updated_markers = search_module._apply_rule_markers(
            ["."],
            aligned_query=["x"],
            aligned_lemma=["a"],
            applications=[
                RuleApplication(
                    rule_id="OBS-SUB",
                    rule_name="Observed substitution",
                    input_phoneme="a",
                    output_phoneme="x",
                    position=0,
                )
            ],
        )

        assert updated_markers == ["."]

    def test_exact_match_returns_confidence_one_and_three_line_visualization(self) -> None:
        lexicon_map = {
            "L1": LexiconRecord(entry={"headword": "λόγος", "ipa": "lóɡos", "dialect": "attic"}, token_count=5)
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
            {"L1": LexiconRecord(entry={"headword": "γᾱ", "ipa": "aː", "dialect": "attic"}, token_count=1)},
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
            "L1": LexiconRecord(entry={"headword": "γᾱ", "ipa": "aː", "dialect": "attic"}, token_count=1)
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
            "L1": LexiconRecord(entry={"headword": "χώρα", "ipa": "rɛː", "dialect": "attic"}, token_count=2)
        }

        results = extend_stage("raː", ["L1"], lexicon_map, matrix={"ɛː": {"aː": 0.1}})

        assert results[0].applied_rules == ["VSH-010"]
        assert [application.position for application in results[0].rule_applications] == [1]
        assert (
            results[0].dialect_attribution
            == "lemma dialect: attic; query-compatible dialects: attic"
        )
        assert ":" in results[0].alignment_visualization.splitlines()[1]

    def test_extend_stage_uses_packaged_morphophonemic_rule_for_runtime_suffix_match(self) -> None:
        lexicon_map = {
            "L1": LexiconRecord(entry={
                "headword": "βασιλεύς",
                "ipa": to_ipa("βασιλεύς"),
                "dialect": "attic",
            }, token_count=7)
        }

        results = extend_stage(
            to_ipa("βασιλέος"),
            ["L1"],
            lexicon_map,
            matrix=load_matrix(MATRIX_FILE),
        )

        assert len(results) == 1
        assert results[0].applied_rules == ["MPH-013"]
        assert [application.rule_id for application in results[0].rule_applications] == ["MPH-013"]
        assert results[0].dialect_attribution == (
            "lemma dialect: attic; query-compatible dialects: attic, ionic"
        )
        assert ":" in results[0].alignment_visualization.splitlines()[1]

    def test_extend_stage_uses_packaged_runtime_velar_assimilation_rule(self) -> None:
        lexicon_map = {
            "L1": LexiconRecord(entry={
                "headword": "ἄνκυρα",
                "ipa": to_ipa("ἄνκυρα"),
                "dialect": "attic",
            }, token_count=6)
        }

        results = extend_stage(
            to_ipa("ἄγκυρα"),
            ["L1"],
            lexicon_map,
            matrix=load_matrix(MATRIX_FILE),
        )

        assert len(results) == 1
        assert "CCH-015" in results[0].applied_rules
        assert results[0].dialect_attribution == (
            f"lemma dialect: {lexicon_map['L1'].entry['dialect']}; "
            "query-compatible dialects: attic, ionic, doric, koine"
        )
        assert any(
            application.rule_id == "CCH-015"
            and application.input_phoneme == "n"
            and application.output_phoneme == "ɡ"
            for application in results[0].rule_applications
        )

    @pytest.mark.parametrize(
        ("query_word", "expected_rule_ids"),
        [
            ("λόγος", {"CCH-009"}),
            ("ἀδελφός", {"CCH-010", "CCH-011"}),
            ("φῶς", {"CCH-011"}),
            ("θεός", {"CCH-012"}),
            ("χείρ", {"CCH-013"}),
        ],
    )
    def test_search_supports_koine_query_side_spirantization_rules(
        self,
        query_word: str,
        expected_rule_ids: set[str],
    ) -> None:
        lexicon = [
            {
                "id": "L1",
                "headword": query_word,
                "ipa": to_ipa(query_word, dialect="attic"),
                "dialect": "attic",
            }
        ]

        results = search(
            query_word,
            lexicon=lexicon,
            matrix=load_matrix(MATRIX_FILE),
            max_results=1,
            dialect="koine",
        )

        assert len(results) == 1
        assert expected_rule_ids <= set(results[0].applied_rules)
        assert "query-compatible dialects: koine" in results[0].dialect_attribution

    @pytest.mark.parametrize(
        ("query_tokens", "lemma_tokens", "expected_query", "expected_lemma"),
        [
            (["f", "ɔː", "s"], ["pʰ", "ɔː", "s"], ["f", "ɔː", "s"], ["pʰ", "ɔː", "s"]),
            (["θ", "e", "o", "s"], ["tʰ", "e", "o", "s"], ["θ", "e", "o", "s"], ["tʰ", "e", "o", "s"]),
        ],
    )
    def test_smith_waterman_alignment_retains_edge_substitutions(
        self,
        query_tokens: list[str],
        lemma_tokens: list[str],
        expected_query: list[str],
        expected_lemma: list[str],
    ) -> None:
        _score, aligned_query, aligned_lemma = search_module._smith_waterman_alignment(
            query_tokens,
            lemma_tokens,
            load_matrix(MATRIX_FILE),
        )

        assert aligned_query == expected_query
        assert aligned_lemma == expected_lemma

    @pytest.mark.parametrize(
        ("query_tokens", "lemma_tokens", "expected_query", "expected_lemma"),
        [
            (
                ["a", "p", "t", "k"],
                ["a", "s", "x", "p", "t", "k"],
                ["a", None, None, "p", "t", "k"],
                ["a", "s", "x", "p", "t", "k"],
            ),
            (
                ["p", "t", "k", "a"],
                ["p", "t", "k", "s", "x", "a"],
                ["p", "t", "k", None, None, "a"],
                ["p", "t", "k", "s", "x", "a"],
            ),
        ],
    )
    def test_smith_waterman_alignment_preserves_shared_edge_matches(
        self,
        query_tokens: list[str],
        lemma_tokens: list[str],
        expected_query: list[str | None],
        expected_lemma: list[str | None],
    ) -> None:
        _score, aligned_query, aligned_lemma = search_module._smith_waterman_alignment(
            query_tokens,
            lemma_tokens,
            load_matrix(MATRIX_FILE),
        )

        assert aligned_query == expected_query
        assert aligned_lemma == expected_lemma

    @pytest.mark.parametrize(
        ("query_tokens", "lemma_tokens", "expected_applications"),
        [
            (
                ["a", "p", "t", "k"],
                ["a", "s", "x", "p", "t", "k"],
                [("OBS-DEL", "s", "", 1), ("OBS-DEL", "x", "", 2)],
            ),
            (
                ["p", "t", "k", "a"],
                ["p", "t", "k", "s", "x", "a"],
                [("OBS-DEL", "s", "", 3), ("OBS-DEL", "x", "", 4)],
            ),
        ],
    )
    def test_explain_does_not_invent_observed_changes_for_shared_edge_matches(
        self,
        query_tokens: list[str],
        lemma_tokens: list[str],
        expected_applications: list[tuple[str, str, str, int]],
    ) -> None:
        _score, aligned_query, aligned_lemma = search_module._smith_waterman_alignment(
            query_tokens,
            lemma_tokens,
            load_matrix(MATRIX_FILE),
        )

        applications = search_module.explain(
            query_tokens=query_tokens,
            lemma_tokens=lemma_tokens,
            alignment=search_module.Alignment(
                aligned_query=tuple(aligned_query),
                aligned_lemma=tuple(aligned_lemma),
            ),
            rules=[],
        )

        assert [
            (
                application.rule_id,
                application.input_phoneme,
                application.output_phoneme,
                application.position,
            )
            for application in applications
        ] == expected_applications

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
            "L1": LexiconRecord(entry={"headword": "θάλασσα", "ipa": "ss", "dialect": "ionic"}, token_count=2)
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
            "L1": LexiconRecord(entry={"headword": "γένεσος", "ipa": "asa", "dialect": "attic"}, token_count=3)
        }

        results = extend_stage("aa", ["L1"], lexicon_map, matrix={})

        assert results[0].applied_rules == ["CCH-DEL"]
        assert [application.position for application in results[0].rule_applications] == [1]
        assert (
            results[0].dialect_attribution
            == "lemma dialect: attic; query-compatible dialects: ionic"
        )

    def test_observed_steps_remain_visible_but_not_catalogued(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(search_module, "load_rules", lambda _path: {})
        lexicon_map = {
            "L1": LexiconRecord(entry={"headword": "χ", "ipa": "a", "dialect": "attic"}, token_count=1)
        }

        results = extend_stage("x", ["L1"], lexicon_map, matrix={"a": {"x": 0.5}})

        assert results[0].applied_rules == []
        assert [application.rule_id for application in results[0].rule_applications] == ["OBS-SUB"]
        assert "." in results[0].alignment_visualization.splitlines()[1]
        assert ":" not in results[0].alignment_visualization.splitlines()[1]

    def test_catalogued_rules_survive_alongside_observed_steps(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            search_module,
            "load_rules",
            lambda _path: {
                "RULE-BX": {
                    "id": "RULE-BX",
                    "input": "b",
                    "output": "x",
                    "dialects": ["attic"],
                }
            },
        )
        monkeypatch.setattr(
            search_module,
            "_smith_waterman_alignment",
            # Returns alignment simulating: position 0 = deletion (None vs "a"),
            # position 1 = substitution ("x" vs "b")
            lambda query_tokens, lemma_tokens, matrix: (
                1.0,
                [None, "x"],
                ["a", "b"],
            ),
        )
        lexicon_map = {
            "L1": LexiconRecord(entry={"headword": "χ", "ipa": "ab", "dialect": "ionic"}, token_count=2)
        }

        results = extend_stage("x", ["L1"], lexicon_map, matrix={"a": {"x": 0.5}, "b": {"x": 0.4}})

        assert results[0].applied_rules == ["RULE-BX"]
        assert [application.rule_id for application in results[0].rule_applications] == [
            "OBS-DEL",
            "RULE-BX",
        ]
        marker_line = results[0].alignment_visualization.splitlines()[1]
        assert marker_line.endswith(" :")
        assert marker_line.count(":") == 1

    def test_skips_stale_candidates_missing_from_lexicon_map(self) -> None:
        lexicon_map = {
            "L1": LexiconRecord(entry={"headword": "λόγος", "ipa": "lóɡos", "dialect": "attic"}, token_count=5)
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
            "L1": LexiconRecord(entry={"headword": "γᾱ", "ipa": "aː", "dialect": "attic"}, token_count=1)
        }

        results = extend_stage("ɛː", ["L1"], lexicon_map, matrix={"aː": {"ɛː": 0.3}})

        assert results[0].applied_rules == ["VSH-TEST"]
        assert results[0].dialect_attribution == "lemma dialect: attic"

    @pytest.mark.parametrize("dialect", [None, "", "   "])
    def test_uses_unknown_when_entry_dialect_is_missing_or_blank(
        self, monkeypatch: pytest.MonkeyPatch, dialect: object
    ) -> None:
        monkeypatch.setattr(search_module, "load_rules", lambda _path: {})
        lexicon_map = {"L1": LexiconRecord(entry={"headword": "γᾱ", "ipa": "aː", "dialect": dialect}, token_count=1)}

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

    def test_uses_token_proximity_when_query_has_no_seedable_skeleton(
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

    def test_k2_seed_candidates_respect_stage2_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only the primary k=2 seed path should cap candidates at stage2_limit."""
        captured: dict[str, object] = {}

        def get_headword(record: object) -> str:
            if isinstance(record, LexiconRecord):
                return str(record.entry["headword"])
            assert isinstance(record, dict)
            return str(record["headword"])

        def fake_score_stage(
            query_ipa: str,
            candidates: Iterable[str],
            lexicon_map: dict[str, object],
            matrix: dict[str, dict[str, float]],
        ) -> list[SearchResult]:
            captured["candidate_ids"] = list(candidates)
            return [
                SearchResult(
                    lemma=get_headword(lexicon_map[cid]),
                    confidence=1.0,
                    dialect_attribution="lemma dialect: attic",
                    entry_id=str(cid),
                )
                for cid in captured["candidate_ids"]
            ]

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "aː")
        seed_calls = iter(
            [
                [f"L{index:02d}" for index in range(30)],
            ]
        )
        monkeypatch.setattr(
            search_module,
            "seed_stage",
            lambda *_args, **_kwargs: next(seed_calls),
        )
        monkeypatch.setattr(search_module, "_score_stage", fake_score_stage)
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda query_ipa, results, lexicon_map, matrix, language="ancient_greek": results,
        )
        monkeypatch.setattr(
            search_module, "filter_stage", lambda results, max_results: results[:max_results]
        )
        lexicon = [
            {
                "id": f"L{index:02d}",
                "headword": f"lemma-{index}",
                "ipa": f"aː{chr(ord('a') + index)}",
                "dialect": "attic",
            }
            for index in range(30)
        ]

        # max_results=1 → stage2_limit = max(25, 1*10) = 25, capping 30 entries.
        # No entry has IPA matching query "aː", so exact-match injection adds nothing.
        search("ἄ", lexicon, matrix={}, max_results=1)

        assert len(captured["candidate_ids"]) == 25

    def test_token_proximity_finds_exact_match_by_length_similarity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Token-count proximity fallback should prefer entries with similar IPA length."""
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

        # "aː" (1 token) should match "aː" (1 token) over "i" (1 token) by
        # exact phonological similarity via Smith-Waterman. The fallback is
        # capped, so exact IPA matches must still sort into the evaluated set.
        assert [result.lemma for result in results] == ["target"]

    def test_token_proximity_fallback_does_not_drop_exact_match_after_sorting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exact matches must survive fallback when target has the closest token count."""
        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "aː")
        monkeypatch.setattr(search_module, "load_rules", lambda _path: {})
        # 20 noise entries with 3-token IPA, 1 target with 1-token IPA matching the query.
        # Token-count proximity puts the target (diff=0) ahead of noise (diff=2).
        lexicon = [
            {
                "id": f"L{index:02d}",
                "headword": f"noise-{index:02d}",
                "ipa": "poi",
                "dialect": "attic",
            }
            for index in range(20)
        ]
        lexicon.append({"id": "L99", "headword": "target", "ipa": "aː", "dialect": "attic"})

        results = search("ἄ", lexicon, matrix={}, max_results=1)

        assert [result.lemma for result in results] == ["target"]

    def test_token_proximity_fallback_when_no_kmer_matches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When neither k=2 nor k=1 seeds match, token-count proximity ranks candidates."""
        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "mn")
        monkeypatch.setattr(search_module, "load_rules", lambda _path: {})
        lexicon = [
            {"id": "L1", "headword": "πτω", "ipa": "pto", "dialect": "attic"},
            {"id": "L2", "headword": "κτω", "ipa": "kto", "dialect": "attic"},
        ]

        results = search("μν", lexicon, matrix={}, max_results=2)

        assert len(results) == 2
        assert {result.lemma for result in results} == {"πτω", "κτω"}

    def test_token_proximity_fallback_is_uncapped_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The token-count fallback should pass the full ranked list by default."""
        captured: dict[str, object] = {}

        def fake_score_stage(
            query_ipa: str,
            candidates: Iterable[str],
            lexicon_map: dict[str, object],
            matrix: object,
        ) -> list[SearchResult]:
            captured["candidate_ids"] = list(candidates)
            return []

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "aː")
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: [])
        monkeypatch.setattr(search_module, "_score_stage", fake_score_stage)
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda query_ipa, results, lexicon_map, matrix, language="ancient_greek": results,
        )
        monkeypatch.setattr(search_module, "filter_stage", lambda results, max_results: results)

        lexicon = [
            {
                "id": f"L{index:02d}",
                "headword": f"lemma-{index:02d}",
                "ipa": "i",
                "dialect": "attic",
            }
            for index in range(30)
        ]

        # Default fallback should stay uncapped even when stage2_limit is 25.
        search("ἄ", lexicon, matrix={}, max_results=1, index={}, unigram_index={})

        assert len(captured["candidate_ids"]) == 30

    def test_search_accepts_explicit_similarity_fallback_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Callers can tighten the token-count fallback candidate cap."""
        captured: dict[str, object] = {}

        def fake_score_stage(
            query_ipa: str,
            candidates: Iterable[str],
            lexicon_map: dict[str, object],
            matrix: object,
        ) -> list[SearchResult]:
            captured["candidate_ids"] = list(candidates)
            return []

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "aː")
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: [])
        monkeypatch.setattr(search_module, "_score_stage", fake_score_stage)
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda query_ipa, results, lexicon_map, matrix, language="ancient_greek": results,
        )
        monkeypatch.setattr(search_module, "filter_stage", lambda results, max_results: results)

        lexicon = [
            {
                "id": f"L{index:02d}",
                "headword": f"lemma-{index:02d}",
                "ipa": "i",
                "dialect": "attic",
            }
            for index in range(10)
        ]

        search(
            "ἄ",
            lexicon,
            matrix={},
            max_results=1,
            index={},
            unigram_index={},
            similarity_fallback_limit=3,
        )

        assert len(captured["candidate_ids"]) == 3

    def test_token_proximity_fallback_keeps_closer_same_length_target_beyond_old_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A later same-length target should survive default uncapped fallback."""
        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "a")
        monkeypatch.setattr(search_module, "load_rules", lambda _path: {})
        lexicon = [
            {
                "id": f"L{index:02d}",
                "headword": f"noise-{index:02d}",
                "ipa": "i",
                "dialect": "attic",
            }
            for index in range(30)
        ]
        lexicon.append({"id": "ZZZ", "headword": "target", "ipa": "e", "dialect": "attic"})
        matrix = {"e": {"a": 0.1}, "i": {"a": 0.9}, "a": {"e": 0.1, "i": 0.9}}

        results = search("dummy", lexicon, matrix=matrix, max_results=1, index={}, unigram_index={})

        assert [result.lemma for result in results] == ["target"]

    def test_search_uses_unigram_fallback_for_single_consonant_query(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A query with 1 consonant uses k=1 to find consonant-matching candidates."""
        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "poi")
        monkeypatch.setattr(search_module, "load_rules", lambda _path: {})
        lexicon = [
            {"id": "L1", "headword": "πολύ", "ipa": "poly", "dialect": "attic"},
            {"id": "L2", "headword": "πατρι", "ipa": "patri", "dialect": "attic"},
        ]
        unigram_idx = build_kmer_index(lexicon, k=1)

        results = search(
            "ποι", lexicon, matrix={}, max_results=2, unigram_index=unigram_idx,
        )

        assert results[0].lemma == "πολύ"
        assert len(results) == 2

    def test_unigram_fallback_preserves_full_lexicon_coverage_for_stage2(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Single-consonant fallback must not drop better stage-2 matches."""
        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "pa")
        monkeypatch.setattr(search_module, "load_rules", lambda _path: {})
        lexicon = [
            {"id": "L1", "headword": "same-consonant", "ipa": "pi", "dialect": "attic"},
            {"id": "L2", "headword": "better-match", "ipa": "ba", "dialect": "attic"},
        ]
        matrix = {"b": {"p": 0.1}, "p": {"b": 0.1}, "i": {"a": 0.9}, "a": {"i": 0.9}}

        results = search("dummy", lexicon, matrix=matrix, max_results=2, index={})

        assert [result.lemma for result in results] == ["better-match", "same-consonant"]

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
        lexicon.append({"id": "L99", "headword": "target", "ipa": "poly", "dialect": "attic"})


        results = search("ποι", lexicon, matrix={}, max_results=1)

        assert [result.lemma for result in results] == ["target"]

    def test_search_accepts_explicit_unigram_fallback_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Callers can cap the k=1 fallback after token-count re-ranking."""
        captured: dict[str, object] = {}

        def fake_seed_stage(query_ipa: str, index: object, k: int = 2) -> list[str]:
            return [] if k == 2 else ["L1", "L2", "L3", "L4"]

        def fake_score_stage(
            query_ipa: str,
            candidates: Iterable[str],
            lexicon_map: dict[str, object],
            matrix: object,
        ) -> list[SearchResult]:
            captured["candidate_ids"] = list(candidates)
            return []

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "poi")
        monkeypatch.setattr(search_module, "seed_stage", fake_seed_stage)
        monkeypatch.setattr(search_module, "_score_stage", fake_score_stage)
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda query_ipa, results, lexicon_map, matrix, language="ancient_greek": results,
        )
        monkeypatch.setattr(search_module, "filter_stage", lambda results, max_results: results)

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

    def test_unigram_fallback_passes_all_candidates_to_stage2(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unigram fallback should prepend hits without dropping later candidates."""
        captured: dict[str, object] = {}

        def fake_score_stage(
            query_ipa: str,
            candidates: Iterable[str],
            lexicon_map: dict[str, object],
            matrix: object,
        ) -> list[SearchResult]:
            captured["candidate_ids"] = list(candidates)
            return []

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "poi")
        seed_calls: list[int] = []

        def fake_seed_stage(query_ipa: str, index: object, k: int = 2) -> list[str]:
            seed_calls.append(k)
            if k == 2:
                return []
            return ["L05", "L02", "L29"]

        monkeypatch.setattr(search_module, "seed_stage", fake_seed_stage)
        monkeypatch.setattr(search_module, "_score_stage", fake_score_stage)
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda query_ipa, results, lexicon_map, matrix, language="ancient_greek": results,
        )
        monkeypatch.setattr(search_module, "filter_stage", lambda results, max_results: results)

        lexicon = [
            {
                "id": f"L{index:02d}",
                "headword": f"noise-{index:02d}",
                "ipa": "pa",
                "dialect": "attic",
            }
            for index in range(30)
        ]
        all_lexicon_ids = [f"L{index:02d}" for index in range(30)]
        unigram_hits = ["L05", "L02", "L29"]

        # max_results=1 still yields full lexicon coverage on the k=1 path.
        search("ποι", lexicon, matrix={}, max_results=1, index={}, unigram_index={})

        assert seed_calls == [2, 1]
        assert len(captured["candidate_ids"]) == 30
        assert captured["candidate_ids"][:3] == unigram_hits
        assert captured["candidate_ids"][3:] == [
            entry_id for entry_id in all_lexicon_ids if entry_id not in set(unigram_hits)
        ]

    def test_search_annotates_only_ranked_results(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Search should defer explanation work until after top-N ranking."""
        captured: dict[str, object] = {}

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "poi")
        monkeypatch.setattr(
            search_module,
            "seed_stage",
            lambda *_args, **_kwargs: [f"L{index:02d}" for index in range(30)],
        )
        monkeypatch.setattr(
            search_module,
            "_score_stage",
            lambda query_ipa, candidates, lexicon_map, matrix: [
                SearchResult(
                    lemma=f"lemma-{index:02d}",
                    confidence=1.0 - (index * 0.01),
                    dialect_attribution="lemma dialect: attic",
                    entry_id=f"L{index:02d}",
                )
                for index, _candidate_id in enumerate(candidates)
            ],
        )

        def fake_annotate_results(
            query_ipa: str,
            results: list[SearchResult],
            lexicon_map: dict[str, object],
            matrix: object,
            language: str = "ancient_greek",
        ) -> list[SearchResult]:
            captured["annotated_count"] = len(results)
            return results

        monkeypatch.setattr(search_module, "_annotate_search_results", fake_annotate_results)

        lexicon = [
            {"id": f"L{index:02d}", "headword": f"lemma-{index:02d}", "ipa": "poi", "dialect": "attic"}
            for index in range(30)
        ]

        search("ποι", lexicon, matrix={}, max_results=5)

        assert captured["annotated_count"] == 5

    def test_extend_stage_tokenizes_rules_once_for_all_candidates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Full extend_stage should reuse one tokenized-rules batch for all candidates."""
        tokenize_calls: list[int] = []

        monkeypatch.setattr(search_module, "_get_rules_registry", lambda language="ancient_greek": {})
        monkeypatch.setattr(
            search_module,
            "tokenize_rules_for_matching",
            lambda rules: tokenize_calls.append(len(rules)) or [],
        )
        lexicon_map = {
            "L1": LexiconRecord(entry={"id": "L1", "headword": "alpha", "ipa": "pa", "dialect": "attic"}, token_count=2, ipa_tokens=("p", "a")),
            "L2": LexiconRecord(entry={"id": "L2", "headword": "beta", "ipa": "pi", "dialect": "attic"}, token_count=2, ipa_tokens=("p", "i")),
            "L3": LexiconRecord(entry={"id": "L3", "headword": "gamma", "ipa": "po", "dialect": "attic"}, token_count=2, ipa_tokens=("p", "o")),
        }

        extend_stage("poi", ["L1", "L2", "L3"], lexicon_map, matrix={})

        assert tokenize_calls == [0]

    def test_single_consonant_query_keeps_alignment_and_rule_applications(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Single-consonant fallback should still annotate the final top hit."""
        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "poi")
        monkeypatch.setattr(search_module, "load_rules", lambda _path: {})
        lexicon = [
            {"id": "L1", "headword": "πολύ", "ipa": "poly", "dialect": "attic"},
            {"id": "L2", "headword": "πατρι", "ipa": "patri", "dialect": "attic"},
        ]

        results = search("ποι", lexicon, matrix={}, max_results=1)

        assert results[0].alignment_visualization
        assert len(results[0].rule_applications) > 0

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
        ) -> list[SearchResult]:
            return [
                SearchResult(
                    lemma="alpha",
                    confidence=1.0,
                    dialect_attribution="lemma dialect: attic",
                )
            ]

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "query-ipa")
        monkeypatch.setattr(search_module, "tokenize_ipa", fake_tokenize_ipa)
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: [])
        monkeypatch.setattr(search_module, "_score_stage", fake_score_stage)
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda query_ipa, results, lexicon_map, matrix, language="ancient_greek": results,
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
        monkeypatch.setattr(search_module, "tokenize_ipa", fake_tokenize_ipa)
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
            lambda query_ipa, results, lexicon_map, matrix, language="ancient_greek": results,
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
        ) -> list[SearchResult]:
            captured["candidate_ids"] = list(candidates)
            captured["lookup_keys"] = list(lexicon_map)
            return []

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "query-ipa")
        monkeypatch.setattr(search_module, "tokenize_ipa", fake_tokenize_ipa)
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: ["L1"])
        monkeypatch.setattr(search_module, "_score_stage", fake_score_stage)
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda query_ipa, results, lexicon_map, matrix, language="ancient_greek": results,
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

    def test_search_does_not_tokenize_full_lexicon_when_unigram_fallback_hits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """k=1 fallback should reuse lightweight lookup when token counts are unnecessary."""
        token_calls: list[str] = []
        captured: dict[str, object] = {}

        def fake_tokenize_ipa(ipa_text: str) -> list[str]:
            token_calls.append(ipa_text)
            return ["q"]

        def fake_seed_stage(query_ipa: str, index: object, k: int = 2) -> list[str]:
            return [] if k == 2 else ["L2"]

        def fake_score_stage(
            query_ipa: str,
            candidates: Iterable[str],
            lexicon_map: dict[str, object],
            matrix: object,
        ) -> list[SearchResult]:
            captured["candidate_ids"] = list(candidates)
            captured["lookup_keys"] = list(lexicon_map)
            return []

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "query-ipa")
        monkeypatch.setattr(search_module, "tokenize_ipa", fake_tokenize_ipa)
        monkeypatch.setattr(search_module, "seed_stage", fake_seed_stage)
        monkeypatch.setattr(search_module, "_score_stage", fake_score_stage)
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda query_ipa, results, lexicon_map, matrix, language="ancient_greek": results,
        )
        monkeypatch.setattr(search_module, "filter_stage", lambda results, max_results: results)

        lexicon = [
            {"id": "L1", "headword": "alpha", "ipa": "aː", "dialect": "attic"},
            {"id": "L2", "headword": "beta", "ipa": "b", "dialect": "attic"},
        ]

        search("query", lexicon, matrix={}, max_results=1, index={}, unigram_index={})

        assert token_calls == ["query-ipa"]
        assert captured["candidate_ids"] == ["L2", "L1"]
        assert captured["lookup_keys"] == ["L1", "L2"]

    def test_extend_stage_reuses_cached_alignment_during_annotation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """extend_stage should not recompute alignment during annotation."""
        alignment_calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

        def fake_alignment(
            query_tokens: list[str],
            lemma_tokens: list[str],
            matrix: object,
        ) -> tuple[float, list[str | None], list[str | None]]:
            alignment_calls.append((tuple(query_tokens), tuple(lemma_tokens)))
            return 2.0, list(query_tokens), list(lemma_tokens)

        monkeypatch.setattr(search_module, "_smith_waterman_alignment", fake_alignment)
        monkeypatch.setattr(search_module, "_get_rules_registry", lambda language="ancient_greek": {})
        monkeypatch.setattr(search_module, "tokenize_rules_for_matching", lambda rules: [])

        lexicon_map = {
            "L1": LexiconRecord(
                entry={"headword": "alpha", "ipa": "pa", "dialect": "attic"},
                token_count=2,
                ipa_tokens=("p", "a"),
            ),
            "L2": LexiconRecord(
                entry={"headword": "beta", "ipa": "pi", "dialect": "attic"},
                token_count=2,
                ipa_tokens=("p", "i"),
            ),
        }

        extend_stage("pa", ["L1", "L2"], lexicon_map, matrix={})

        assert alignment_calls == [(("p", "a"), ("p", "a")), (("p", "a"), ("p", "i"))]

    def test_unigram_fallback_logs_missing_tokenized_candidates(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Missing tokenized unigram candidates should warn and continue."""
        captured: dict[str, object] = {}

        def fake_seed_stage(query_ipa: str, index: object, k: int = 2) -> list[str]:
            return [] if k == 2 else ["L2", "missing", "L1", "missing"]

        def fake_build_lexicon_map(
            lexicon: list[dict[str, str]],
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
            lambda query_ipa, results, lexicon_map, matrix, language="ancient_greek": results,
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
        assert "query IPA 'pa'" in caplog.text
        assert "missing" in caplog.text

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


class TestInjectExactIpaMatches:
    """Verify exact IPA match injection into candidate lists."""

    def test_prepends_exact_match_not_in_candidates(self) -> None:
        lookup = {
            "L1": {"id": "L1", "headword": "α", "ipa": "aaa"},
            "L2": {"id": "L2", "headword": "β", "ipa": "bbb"},
            "L3": {"id": "L3", "headword": "γ", "ipa": "aaa"},
        }
        candidates = ["L2"]
        result = _inject_exact_ipa_matches("aaa", candidates, lookup)
        # L1 and L3 match IPA "aaa" and should be prepended
        assert result[0] in ("L1", "L3")
        assert result[-1] == "L2"
        assert set(result) == {"L1", "L2", "L3"}

    def test_respects_limit_after_prepending_exact_matches(self) -> None:
        lookup = {
            "L1": {"id": "L1", "headword": "α", "ipa": "aaa"},
            "L2": {"id": "L2", "headword": "β", "ipa": "bbb"},
            "L3": {"id": "L3", "headword": "γ", "ipa": "aaa"},
            "L4": {"id": "L4", "headword": "δ", "ipa": "aaa"},
        }
        candidates = ["L2", "L1"]
        result = _inject_exact_ipa_matches("aaa", candidates, lookup, limit=3)
        assert len(result) == 3
        assert result[:2] == ["L3", "L4"]
        assert result[2] == "L2"

    def test_returns_unchanged_when_no_match(self) -> None:
        lookup = {
            "L1": {"id": "L1", "headword": "α", "ipa": "aaa"},
        }
        candidates = ["L1"]
        result = _inject_exact_ipa_matches("zzz", candidates, lookup)
        assert result == ["L1"]

    def test_no_duplicate_when_already_in_candidates(self) -> None:
        lookup = {
            "L1": {"id": "L1", "headword": "α", "ipa": "aaa"},
            "L2": {"id": "L2", "headword": "β", "ipa": "bbb"},
        }
        candidates = ["L1", "L2"]
        result = _inject_exact_ipa_matches("aaa", candidates, lookup)
        # L1 already in candidates, should not be duplicated
        assert result == ["L1", "L2"]

    def test_works_with_lexicon_records(self) -> None:
        lookup = {
            "L1": LexiconRecord(
                entry={"id": "L1", "headword": "α", "ipa": "aaa"},
                token_count=3,
                ipa_tokens=("a", "a", "a"),
            ),
        }
        candidates: list[str] = []
        result = _inject_exact_ipa_matches("aaa", candidates, lookup)
        assert result == ["L1"]


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


class TestSearchExactMatchIntegration:
    """Integration tests for exact-match boost and deduplication in search()."""

    def test_search_returns_exact_match_at_top(self) -> None:
        """Verify that a query word appearing in the lexicon is returned as the top result."""
        # Build a lexicon where the target word shares k-mers with many other entries,
        # pushing it past the stage2_limit in seed ranking.
        target = {"id": "TARGET", "headword": "τεστ", "ipa": "test", "dialect": "attic"}
        # Create 30 entries sharing k-mers (consonant skeleton "t s t" has k-mers "t s" and "s t")
        filler = [
            {"id": f"F{i:03d}", "headword": f"filler{i}", "ipa": f"t{'a' * i}st", "dialect": "attic"}
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

    def test_search_deduplicates_homograph_entries(self) -> None:
        """Verify that duplicate headwords are deduplicated in results."""
        lexicon = [
            {"id": "L1", "headword": "dup", "ipa": "dup", "dialect": "attic"},
            {"id": "L2", "headword": "dup", "ipa": "dup", "dialect": "attic"},
            {"id": "L3", "headword": "other", "ipa": "otʰer", "dialect": "attic"},
        ]
        matrix = load_matrix(MATRIX_FILE)
        results = search("dup", lexicon, matrix, max_results=5, dialect="attic")
        headwords = [r.lemma for r in results]
        assert headwords.count("dup") == 1

    def test_exact_match_injection_preserves_stage2_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Exact-match injection must not expand Stage 2 past its hard candidate cap."""
        lexicon = [
            {"id": f"E{i:02d}", "headword": f"dup{i:02d}", "ipa": "tat", "dialect": "attic"}
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
        monkeypatch.setattr(search_module, "_annotate_search_results", lambda **kwargs: kwargs["results"])
        monkeypatch.setattr(search_module, "filter_stage", lambda results, max_results: results[:max_results])
        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "tat")

        search("dummy", lexicon, matrix={}, max_results=1, dialect="attic")

        assert len(captured["candidate_ids"]) == 25
        assert all(
            lexicon[int(candidate_id.removeprefix("E"))]["ipa"] == "tat"
            for candidate_id in captured["candidate_ids"]
        )

    def test_dedup_does_not_reduce_below_max_results(self) -> None:
        """Dedup before truncation: max_results unique entries are returned."""
        # 2 duplicate headwords + 3 unique = 5 entries.
        # With max_results=3, dedup-before-truncation should yield 3 unique results
        # (not 2 if dedup were applied after truncation).
        lexicon = [
            {"id": "L1", "headword": "alpha", "ipa": "alp", "dialect": "attic"},
            {"id": "L2", "headword": "alpha", "ipa": "alp", "dialect": "attic"},
            {"id": "L3", "headword": "beta", "ipa": "bet", "dialect": "attic"},
            {"id": "L4", "headword": "gamma", "ipa": "ɡam", "dialect": "attic"},
            {"id": "L5", "headword": "delta", "ipa": "del", "dialect": "attic"},
        ]
        matrix = load_matrix(MATRIX_FILE)
        results = search("alp", lexicon, matrix, max_results=3, dialect="attic")
        assert len(results) == 3
        headwords = [r.lemma for r in results]
        assert headwords.count("alpha") == 1
