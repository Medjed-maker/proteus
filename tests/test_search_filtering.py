"""Direct tests for mode-aware search filtering helpers."""

from __future__ import annotations

import pytest

from phonology.search._filtering import (
    _apply_mode_quality_filter,
    _rank_short_query_annotation_candidates,
    _select_annotation_candidates,
)
from phonology.search._types import LexiconRecord, PartialQueryTokens, SearchResult


def _record(entry_id: str, ipa_tokens: tuple[str, ...]) -> LexiconRecord:
    return LexiconRecord(
        {
            "id": entry_id,
            "headword": entry_id,
            "ipa": " ".join(ipa_tokens),
            "dialect": "attic",
        },
        token_count=len(ipa_tokens),
        ipa_tokens=ipa_tokens,
    )


def _result(entry_id: str, lemma: str, confidence: float, **kwargs: object) -> SearchResult:
    return SearchResult(
        lemma=lemma,
        confidence=confidence,
        entry_id=entry_id,
        **kwargs,
    )


class TestApplyModeQualityFilter:
    def test_short_query_keeps_exact_rule_supported_and_confident_results(self) -> None:
        lexicon_lookup = {
            "exact": _record("exact", ("a",)),
            "rule": _record("rule", ("b",)),
            "confident": _record("confident", ("c",)),
            "weak": _record("weak", ("d",)),
        }
        results = [
            _result("weak", "weak", 0.10),
            _result("exact", "exact", 0.10),
            _result("rule", "rule", 0.10, applied_rules=["TEST"]),
            _result("confident", "confident", 0.70),
        ]

        filtered = _apply_mode_quality_filter(
            "Short-query",
            ("a",),
            None,
            results,
            lexicon_lookup,
        )

        assert [result.entry_id for result in filtered] == ["exact", "rule", "confident"]

    def test_partial_form_filters_and_ranks_by_fragment_match_quality(self) -> None:
        partial_query = PartialQueryTokens("prefix", ("a", "b"), ())
        lexicon_lookup = {
            "weak_full": _record("weak_full", ("a", "b", "x")),
            "rule_overlap": _record("rule_overlap", ("a", "x")),
            "weak_overlap": _record("weak_overlap", ("a", "z")),
            "no_overlap": _record("no_overlap", ("z", "a")),
        }
        results = [
            _result("no_overlap", "no-overlap", 0.99),
            _result("weak_overlap", "weak-overlap", 0.10),
            _result("rule_overlap", "rule-overlap", 0.10, applied_rules=["TEST"]),
            _result("weak_full", "weak-full", 0.10),
        ]

        filtered = _apply_mode_quality_filter(
            "Partial-form",
            ("a", "b"),
            partial_query,
            results,
            lexicon_lookup,
        )

        assert [result.entry_id for result in filtered] == ["weak_full", "rule_overlap"]

    def test_full_form_returns_input_results_unchanged(self) -> None:
        results = [_result("one", "one", 0.1), _result("two", "two", 0.2)]

        filtered = _apply_mode_quality_filter("Full-form", ("a",), None, results, {})

        assert filtered is results

    def test_partial_form_requires_partial_query_metadata(self) -> None:
        with pytest.raises(ValueError, match="partial query metadata is required"):
            _apply_mode_quality_filter("Partial-form", ("a",), None, [], {})


class TestSelectAnnotationCandidates:
    def test_under_limit_full_form_returns_copy_in_ranked_order(self) -> None:
        results = [_result("one", "one", 0.1), _result("two", "two", 0.2)]

        selected = _select_annotation_candidates(
            "Full-form",
            ("a",),
            None,
            results,
            {},
            annotation_limit=5,
        )

        assert selected == results
        assert selected is not results

    def test_under_limit_short_query_still_uses_short_query_ranking(self) -> None:
        lexicon_lookup = {
            "far": _record("far", ("x", "y", "z")),
            "exact": _record("exact", ("a",)),
        }
        results = [
            _result("far", "far", 0.99),
            _result("exact", "exact", 0.10),
        ]

        selected = _select_annotation_candidates(
            "Short-query",
            ("a",),
            None,
            results,
            lexicon_lookup,
            annotation_limit=5,
        )

        assert [result.entry_id for result in selected] == ["exact", "far"]

    def test_short_query_ranking_prioritizes_exact_then_confident_then_exploratory(self) -> None:
        lexicon_lookup = {
            "beta": _record("beta", ("a",)),
            "alpha": _record("alpha", ("a",)),
            "confident": _record("confident", ("a", "x")),
            "exploratory": _record("exploratory", ("x",)),
        }
        results = [
            _result("exploratory", "exploratory", 0.10),
            _result("confident", "confident", 0.90),
            _result("beta", "beta", 0.80),
            _result("alpha", "alpha", 0.80),
        ]

        selected = _rank_short_query_annotation_candidates(
            ("a",),
            results,
            lexicon_lookup,
            annotation_limit=4,
        )

        assert [result.entry_id for result in selected] == [
            "alpha",
            "beta",
            "confident",
            "exploratory",
        ]

    def test_short_query_limit_uses_primary_window_before_exploratory_candidates(self) -> None:
        lexicon_lookup = {
            "exact": _record("exact", ("a",)),
            "confident": _record("confident", ("x", "y")),
            "near": _record("near", ("x",)),
            "far": _record("far", ("x", "y", "z")),
        }
        results = [
            _result("far", "far", 0.40),
            _result("near", "near", 0.30),
            _result("confident", "confident", 0.80),
            _result("exact", "exact", 0.10),
        ]

        selected = _select_annotation_candidates(
            "Short-query",
            ("a",),
            None,
            results,
            lexicon_lookup,
            annotation_limit=3,
        )

        assert [result.entry_id for result in selected] == ["exact", "confident", "near"]

    def test_partial_form_ranking_keeps_primary_matches_before_exploratory(self) -> None:
        partial_query = PartialQueryTokens("infix", ("a",), ("c",))
        lexicon_lookup = {
            "full": _record("full", ("a", "b", "c")),
            "left": _record("left", ("a", "x", "x")),
            "confident_zero": _record("confident_zero", ("x", "y")),
            "exploratory_zero": _record("exploratory_zero", ("y", "z", "q")),
        }
        results = [
            _result("exploratory_zero", "exploratory-zero", 0.10),
            _result("confident_zero", "confident-zero", 0.95),
            _result("left", "left", 0.20, applied_rules=["TEST"]),
            _result("full", "full", 0.10),
        ]

        selected = _select_annotation_candidates(
            "Partial-form",
            ("a", "c"),
            partial_query,
            results,
            lexicon_lookup,
            annotation_limit=3,
        )

        assert [result.entry_id for result in selected] == ["full", "left", "confident_zero"]

    def test_partial_form_requires_metadata_when_selection_is_capped(self) -> None:
        with pytest.raises(ValueError, match="partial query metadata is required"):
            _select_annotation_candidates(
                "Partial-form",
                ("a",),
                None,
                [_result("one", "one", 0.1), _result("two", "two", 0.2)],
                {},
                annotation_limit=1,
            )

    def test_full_form_over_limit_truncates_without_mode_specific_ranking(self) -> None:
        results = [
            _result("one", "one", 0.1),
            _result("two", "two", 0.2),
            _result("three", "three", 0.3),
        ]

        selected = _select_annotation_candidates(
            "Full-form",
            ("a",),
            None,
            results,
            {},
            annotation_limit=2,
        )

        assert [result.entry_id for result in selected] == ["one", "two"]
