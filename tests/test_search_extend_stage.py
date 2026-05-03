"""Tests for the stage-2 extend pipeline and low-level helpers used by it.

Contains ``TestExtendStage`` — rule registry loading, rule-marker logic,
Smith-Waterman alignment edge behavior, observed-step handling, packaged
rule integration, and dialect attribution. This class previously lived
alongside ``TestSearch`` in ``tests/test_search.py``.
"""

from __future__ import annotations

import pytest
import yaml

from phonology import search as search_module
from phonology.distance import load_matrix
from phonology.explainer import RuleApplication
from phonology.ipa_converter import to_ipa
from phonology.search import (
    LexiconRecord,
    extend_stage,
    search,
)
from phonology.search import _scoring as scoring_module

MATRIX_FILE = "attic_doric.json"


class TestExtendStage:
    """Verify stage-2 extension behavior, including an integration-style packaged-data case.

    This class includes a test that depends on packaged resources
    ``attic_doric.json`` and rule ``MPH-013`` to exercise the runtime suffix
    matching path end-to-end.
    """

    def test_get_rules_registry_resolves_language_id_to_profile_rules_dir(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pathlib import Path

        fake_rules_dir = Path("/fake/rules")
        captured: list[object] = []

        def fake_get_profile(lang: str) -> object:
            class _Profile:
                rules_dir = fake_rules_dir

            return _Profile()

        def fake_load_rules(source: object) -> dict[str, dict[str, object]]:
            captured.append(source)
            return {"RULE": {"id": "RULE"}}

        monkeypatch.setattr(search_module, "get_language_profile", fake_get_profile)
        monkeypatch.setattr(search_module, "load_rules", fake_load_rules)

        assert search_module.get_rules_registry("test_language") == {
            "RULE": {"id": "RULE"}
        }
        assert captured == [fake_rules_dir]

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
            match="get_rules_registry failed to load rules for language 'missing_language'",
        ):
            search_module.get_rules_registry("missing_language")

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
            match="get_rules_registry failed to load rules for language 'broken_language'",
        ):
            search_module.get_rules_registry("broken_language")

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

    def test_build_alignment_markers_rejects_length_mismatch(self) -> None:
        with pytest.raises(ValueError):
            search_module._build_alignment_markers(["a", "b"], ["a"])

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

    def test_exact_match_returns_confidence_one_and_three_line_visualization(
        self,
    ) -> None:
        lexicon_map = {
            "L1": LexiconRecord(
                entry={"headword": "λόγος", "ipa": "lóɡos", "dialect": "attic"},
                token_count=5,
            )
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

        def fake_get_rules_registry(
            language: str = "ancient_greek",
        ) -> dict[str, dict[str, object]]:
            captured["language"] = language
            return {}

        monkeypatch.setattr(
            search_module, "get_rules_registry", fake_get_rules_registry
        )

        extend_stage(
            "aː",
            ["L1"],
            {
                "L1": LexiconRecord(
                    entry={"headword": "γᾱ", "ipa": "aː", "dialect": "attic"},
                    token_count=1,
                )
            },
            matrix={},
            language="test_language",
        )

        assert captured == {"language": "test_language"}

    def test_detects_single_token_rule_and_reports_dialect(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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
            "L1": LexiconRecord(
                entry={"headword": "γᾱ", "ipa": "aː", "dialect": "attic"}, token_count=1
            )
        }

        results = extend_stage("ɛː", ["L1"], lexicon_map, matrix={"aː": {"ɛː": 0.3}})

        assert results[0].applied_rules == ["VSH-TEST"]
        assert [
            application.position for application in results[0].rule_applications
        ] == [0]
        assert (
            results[0].dialect_attribution
            == "lemma dialect: attic; query-compatible dialects: ionic"
        )
        assert ":" in results[0].alignment_visualization.splitlines()[1]

    def test_prefers_contextual_rule_from_shared_explainer_logic(self) -> None:
        lexicon_map = {
            "L1": LexiconRecord(
                entry={"headword": "χώρα", "ipa": "rɛː", "dialect": "attic"},
                token_count=2,
            )
        }

        results = extend_stage("raː", ["L1"], lexicon_map, matrix={"ɛː": {"aː": 0.1}})

        assert results[0].applied_rules == ["VSH-010"]
        assert [
            application.position for application in results[0].rule_applications
        ] == [1]
        assert (
            results[0].dialect_attribution
            == "lemma dialect: attic; query-compatible dialects: attic"
        )
        assert ":" in results[0].alignment_visualization.splitlines()[1]

    def test_extend_stage_uses_packaged_morphophonemic_rule_for_runtime_suffix_match(
        self,
    ) -> None:
        """Integration test verifying MPH-013 morphophonemic rule produces βασιλεύς→βασιλέος suffix transformation."""
        lexicon_map = {
            "L1": LexiconRecord(
                entry={
                    "headword": "βασιλεύς",
                    "ipa": to_ipa("βασιλεύς"),
                    "dialect": "attic",
                },
                token_count=7,
            )
        }

        results = extend_stage(
            to_ipa("βασιλέος"),
            ["L1"],
            lexicon_map,
            matrix=load_matrix(MATRIX_FILE),
        )

        assert len(results) == 1
        assert results[0].applied_rules == ["MPH-013"]
        assert [
            application.rule_id for application in results[0].rule_applications
        ] == ["MPH-013"]
        assert results[0].dialect_attribution == (
            "lemma dialect: attic; query-compatible dialects: attic, ionic"
        )
        assert ":" in results[0].alignment_visualization.splitlines()[1]

    def test_extend_stage_uses_packaged_neuter_ion_final_nu_absence_rule(self) -> None:
        lexicon_map = {
            "L1": LexiconRecord(
                entry={
                    "headword": "παιδίον",
                    "ipa": to_ipa("παιδίον"),
                    "dialect": "attic",
                },
                token_count=6,
            )
        }

        results = extend_stage(
            to_ipa("παιδίο"),
            ["L1"],
            lexicon_map,
            matrix=load_matrix(MATRIX_FILE),
        )

        assert len(results) == 1
        assert results[0].applied_rules == ["MPH-015"]
        assert [
            application.rule_id for application in results[0].rule_applications
        ] == ["MPH-015"]
        assert results[0].rule_applications[0].input_phoneme == "ion"
        assert results[0].rule_applications[0].output_phoneme == "io"
        assert results[0].dialect_attribution == (
            "lemma dialect: attic; query-compatible dialects: attic, ionic, koine"
        )
        assert ":" in results[0].alignment_visualization.splitlines()[1]

    def test_extend_stage_uses_packaged_neuter_eion_final_nu_absence_rule(self) -> None:
        lexicon_map = {
            "L1": LexiconRecord(
                entry={
                    "headword": "μνημεῖον",
                    "ipa": to_ipa("μνημεῖον"),
                    "dialect": "attic",
                },
                token_count=7,
            )
        }

        results = extend_stage(
            to_ipa("μνημεῖο"),
            ["L1"],
            lexicon_map,
            matrix=load_matrix(MATRIX_FILE),
        )

        assert len(results) == 1
        assert results[0].applied_rules == ["MPH-016"]
        assert [
            application.rule_id for application in results[0].rule_applications
        ] == ["MPH-016"]
        assert results[0].rule_applications[0].input_phoneme == "eːon"
        assert results[0].rule_applications[0].output_phoneme == "eːo"
        assert results[0].dialect_attribution == (
            "lemma dialect: attic; query-compatible dialects: attic, ionic, koine"
        )
        assert ":" in results[0].alignment_visualization.splitlines()[1]

    def test_extend_stage_uses_packaged_neuter_on_final_nu_absence_rule(self) -> None:
        lexicon_map = {
            "L1": LexiconRecord(
                entry={
                    "headword": "τέκνον",
                    "ipa": to_ipa("τέκνον"),
                    "dialect": "attic",
                    "gender": "neuter",
                },
                token_count=6,
            )
        }

        results = extend_stage(
            to_ipa("τέκνο"),
            ["L1"],
            lexicon_map,
            matrix=load_matrix(MATRIX_FILE),
        )

        assert len(results) == 1
        assert results[0].applied_rules == ["MPH-017"]
        assert [
            application.rule_id for application in results[0].rule_applications
        ] == ["MPH-017"]
        assert results[0].rule_applications[0].input_phoneme == "on"
        assert results[0].rule_applications[0].output_phoneme == "o"
        assert results[0].dialect_attribution == (
            "lemma dialect: attic; query-compatible dialects: attic, ionic, koine"
        )
        assert ":" in results[0].alignment_visualization.splitlines()[1]

    def test_extend_stage_does_not_use_neuter_on_rule_for_common_gender(self) -> None:
        lexicon_map = {
            "L1": LexiconRecord(
                entry={
                    "headword": "ἄλλον",
                    "ipa": to_ipa("ἄλλον"),
                    "dialect": "attic",
                    "gender": "common",
                },
                token_count=5,
            )
        }

        results = extend_stage(
            to_ipa("ἄλλο"),
            ["L1"],
            lexicon_map,
            matrix=load_matrix(MATRIX_FILE),
        )

        assert len(results) == 1
        assert results[0].applied_rules == []
        assert [
            application.rule_id for application in results[0].rule_applications
        ] == ["OBS-DEL"]
        assert results[0].dialect_attribution == "lemma dialect: attic"
        marker_line = results[0].alignment_visualization.splitlines()[1]
        assert ":" not in marker_line

    def test_extend_stage_uses_packaged_runtime_velar_assimilation_rule(self) -> None:
        lexicon_map = {
            "L1": LexiconRecord(
                entry={
                    "headword": "ἄνκυρα",
                    "ipa": to_ipa("ἄνκυρα"),
                    "dialect": "attic",
                },
                token_count=6,
            )
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
            (
                ["θ", "e", "o", "s"],
                ["tʰ", "e", "o", "s"],
                ["θ", "e", "o", "s"],
                ["tʰ", "e", "o", "s"],
            ),
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
            "L1": LexiconRecord(
                entry={"headword": "θάλασσα", "ipa": "ss", "dialect": "ionic"},
                token_count=2,
            )
        }

        results = extend_stage("tt", ["L1"], lexicon_map, matrix={"s": {"t": 0.4}})

        assert results[0].applied_rules == ["CCH-TEST"]
        assert [
            application.position for application in results[0].rule_applications
        ] == [0]
        assert (
            results[0].dialect_attribution
            == "lemma dialect: ionic; query-compatible dialects: attic"
        )

    def test_detects_deletion_rule_inside_alignment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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
            "L1": LexiconRecord(
                entry={"headword": "γένεσος", "ipa": "asa", "dialect": "attic"},
                token_count=3,
            )
        }

        results = extend_stage("aa", ["L1"], lexicon_map, matrix={})

        assert results[0].applied_rules == ["CCH-DEL"]
        assert [
            application.position for application in results[0].rule_applications
        ] == [1]
        assert (
            results[0].dialect_attribution
            == "lemma dialect: attic; query-compatible dialects: ionic"
        )

    def test_observed_steps_remain_visible_but_not_catalogued(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(search_module, "load_rules", lambda _path: {})
        lexicon_map = {
            "L1": LexiconRecord(
                entry={"headword": "χ", "ipa": "a", "dialect": "attic"}, token_count=1
            )
        }

        results = extend_stage("x", ["L1"], lexicon_map, matrix={"a": {"x": 0.5}})

        assert results[0].applied_rules == []
        assert [
            application.rule_id for application in results[0].rule_applications
        ] == ["OBS-SUB"]
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
            "L1": LexiconRecord(
                entry={"headword": "χ", "ipa": "ab", "dialect": "ionic"}, token_count=2
            )
        }

        results = extend_stage(
            "x", ["L1"], lexicon_map, matrix={"a": {"x": 0.5}, "b": {"x": 0.4}}
        )

        assert results[0].applied_rules == ["RULE-BX"]
        assert [
            application.rule_id for application in results[0].rule_applications
        ] == [
            "OBS-DEL",
            "RULE-BX",
        ]
        marker_line = results[0].alignment_visualization.splitlines()[1]
        assert marker_line.endswith(" :")
        assert marker_line.count(":") == 1

    def test_skips_stale_candidates_missing_from_lexicon_map(self) -> None:
        lexicon_map = {
            "L1": LexiconRecord(
                entry={"headword": "λόγος", "ipa": "lóɡos", "dialect": "attic"},
                token_count=5,
            )
        }

        results = extend_stage("lóɡos", ["L1", "missing"], lexicon_map, matrix={})

        assert [result.lemma for result in results] == ["λόγος"]

    def test_ignores_non_list_rule_dialects(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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
            "L1": LexiconRecord(
                entry={"headword": "γᾱ", "ipa": "aː", "dialect": "attic"}, token_count=1
            )
        }

        results = extend_stage("ɛː", ["L1"], lexicon_map, matrix={"aː": {"ɛː": 0.3}})

        assert results[0].applied_rules == ["VSH-TEST"]
        assert results[0].dialect_attribution == "lemma dialect: attic"

    @pytest.mark.parametrize("dialect", [None, "", "   "])
    def test_uses_unknown_when_entry_dialect_is_missing_or_blank(
        self, monkeypatch: pytest.MonkeyPatch, dialect: object
    ) -> None:
        monkeypatch.setattr(search_module, "load_rules", lambda _path: {})
        lexicon_map = {
            "L1": LexiconRecord(
                entry={"headword": "γᾱ", "ipa": "aː", "dialect": dialect}, token_count=1
            )
        }

        results = extend_stage("aː", ["L1"], lexicon_map, matrix={})

        assert results[0].dialect_attribution == "lemma dialect: unknown"

    def test_near_zero_score_prefers_diagonal_traceback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(scoring_module, "_GAP_PENALTY", 1e-12)
        monkeypatch.setattr(scoring_module, "_substitution_score", lambda *_args: 0.0)

        score, aligned_query, aligned_lemma = scoring_module._smith_waterman_alignment(
            ["q"],
            ["l"],
            matrix={},
        )

        assert score == pytest.approx(1e-12)
        assert aligned_query == ["q"]
        assert aligned_lemma == ["l"]
