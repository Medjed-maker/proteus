"""Tests for phonology.log_odds — NW alignment, count aggregation, log-odds."""

from __future__ import annotations

import math
from collections import Counter

import pytest

from phonology.explainer import Alignment
from phonology.log_odds import (
    CountTables,
    NWParams,
    accumulate_counts,
    build_matrix_document,
    compute_log_odds,
    needleman_wunsch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_alignment(q: tuple, lemma: tuple) -> Alignment:
    return Alignment(aligned_query=q, aligned_lemma=lemma)


def _make_counts(
    pair_counts: dict,
    phone_totals: dict,
    *,
    indel_counts: dict | None = None,
) -> CountTables:
    pt = Counter(phone_totals)
    pair_total = sum(pair_counts.values())
    phone_total = sum(pt.values())
    return CountTables(
        pair_counts=pair_counts,
        phone_totals=pt,
        indel_counts=Counter(indel_counts or {}),
        pair_total=pair_total,
        phone_total=phone_total,
    )


# ---------------------------------------------------------------------------
# TestNeedlemanWunsch
# ---------------------------------------------------------------------------


class TestNeedlemanWunsch:
    def test_both_empty(self):
        result = needleman_wunsch([], [])
        assert result == Alignment(aligned_query=(), aligned_lemma=())

    def test_seq1_empty(self):
        result = needleman_wunsch([], ["a", "b"])
        assert result.aligned_query == (None, None)
        assert result.aligned_lemma == ("a", "b")

    def test_seq2_empty(self):
        result = needleman_wunsch(["a", "b"], [])
        assert result.aligned_query == ("a", "b")
        assert result.aligned_lemma == (None, None)

    def test_identical_sequences(self):
        result = needleman_wunsch(["a", "b", "c"], ["a", "b", "c"])
        assert result.aligned_query == ("a", "b", "c")
        assert result.aligned_lemma == ("a", "b", "c")

    def test_single_mismatch_at_end(self):
        result = needleman_wunsch(["a", "b", "c"], ["a", "b", "d"])
        assert result.aligned_query == ("a", "b", "c")
        assert result.aligned_lemma == ("a", "b", "d")

    def test_gap_in_seq2(self):
        # seq1 = ["a","b","c"], seq2 = ["a","c"] → b gets a gap in seq2
        result = needleman_wunsch(["a", "b", "c"], ["a", "c"])
        assert result.aligned_query == ("a", "b", "c")
        assert result.aligned_lemma == ("a", None, "c")

    def test_gap_in_seq1(self):
        # seq1 = ["a","c"], seq2 = ["a","b","c"] → b gets a gap in seq1
        result = needleman_wunsch(["a", "c"], ["a", "b", "c"])
        assert result.aligned_query == ("a", None, "c")
        assert result.aligned_lemma == ("a", "b", "c")

    def test_single_phone_match(self):
        result = needleman_wunsch(["x"], ["x"])
        assert result.aligned_query == ("x",)
        assert result.aligned_lemma == ("x",)

    def test_single_phone_mismatch(self):
        result = needleman_wunsch(["x"], ["y"])
        assert result.aligned_query == ("x",)
        assert result.aligned_lemma == ("y",)

    def test_tie_break_prefers_diag(self):
        # When diag cost == gap cost, DIAG wins.
        # params: match=1.0, mismatch=1.0, gap=1.0 → all moves cost the same.
        # For ["a"] vs ["b"], DIAG (mismatch, cost=1) vs UP (cost=1) vs LEFT (cost=1).
        # Tie-break: DIAG > UP > LEFT.
        params = NWParams(match=1.0, mismatch=1.0, gap=1.0)
        result = needleman_wunsch(["a"], ["b"], params)
        assert result.aligned_query == ("a",)
        assert result.aligned_lemma == ("b",)

    def test_alignment_length(self):
        # Result length equals max(len(seq1), len(seq2)) at minimum.
        # More precisely: every position in both sequences is accounted for.
        seq1 = ["a", "b", "c", "d"]
        seq2 = ["x", "y"]
        result = needleman_wunsch(seq1, seq2)
        assert len(result.aligned_query) == len(result.aligned_lemma)
        non_none_q = [x for x in result.aligned_query if x is not None]
        non_none_l = [x for x in result.aligned_lemma if x is not None]
        assert non_none_q == list(seq1)
        assert non_none_l == list(seq2)

    def test_custom_gap_cost(self):
        # With a very high gap cost, alignment should prefer mismatches over gaps.
        params = NWParams(match=0.0, mismatch=0.5, gap=100.0)
        result = needleman_wunsch(["a", "b"], ["c", "d"], params)
        # Two mismatches (cost 1.0) < any gap arrangement (cost >= 100.0)
        assert result.aligned_query == ("a", "b")
        assert result.aligned_lemma == ("c", "d")

    def test_deterministic(self):
        # Same input always gives same output.
        seq1 = ["p", "a", "t", "e", "r"]
        seq2 = ["p", "a", "t", "r"]
        r1 = needleman_wunsch(seq1, seq2)
        r2 = needleman_wunsch(seq1, seq2)
        assert r1 == r2


# ---------------------------------------------------------------------------
# TestAccumulateCounts
# ---------------------------------------------------------------------------


class TestAccumulateCounts:
    def test_empty_iterator(self):
        result = accumulate_counts([])
        assert result.pair_counts == {}
        assert result.phone_totals == Counter()
        assert result.indel_counts == Counter()
        assert result.pair_total == 0
        assert result.phone_total == 0

    def test_single_match_column(self):
        alignment = _make_alignment(("a",), ("b",))
        result = accumulate_counts([alignment])
        assert result.pair_counts == {("a", "b"): 1}
        assert result.phone_totals["a"] == 1
        assert result.phone_totals["b"] == 1
        assert result.pair_total == 1
        assert result.phone_total == 2
        assert result.indel_counts == Counter()

    def test_self_pair_increments_totals_twice(self):
        alignment = _make_alignment(("a",), ("a",))
        result = accumulate_counts([alignment])
        assert result.pair_counts == {("a", "a"): 1}
        assert result.phone_totals["a"] == 2
        assert result.pair_total == 1
        assert result.phone_total == 2

    def test_symmetrised_pair_key(self):
        # (b, a) and (a, b) should map to the same canonical key.
        a1 = _make_alignment(("a",), ("b",))
        a2 = _make_alignment(("b",), ("a",))
        result = accumulate_counts([a1, a2])
        assert ("a", "b") in result.pair_counts
        assert ("b", "a") not in result.pair_counts
        assert result.pair_counts[("a", "b")] == 2

    def test_gap_in_query_increments_indels(self):
        alignment = _make_alignment((None,), ("z",))
        result = accumulate_counts([alignment])
        assert result.pair_total == 0
        assert result.indel_counts["z"] == 1

    def test_gap_in_lemma_increments_indels(self):
        alignment = _make_alignment(("z",), (None,))
        result = accumulate_counts([alignment])
        assert result.pair_total == 0
        assert result.indel_counts["z"] == 1

    def test_multiple_alignments(self):
        # 4 columns: (a,a), (a,b), (b,a), (b,b)
        alignments = [
            _make_alignment(("a",), ("a",)),
            _make_alignment(("a",), ("b",)),
            _make_alignment(("b",), ("a",)),
            _make_alignment(("b",), ("b",)),
        ]
        result = accumulate_counts(alignments)
        assert result.pair_counts[("a", "a")] == 1
        assert result.pair_counts[("a", "b")] == 2
        assert result.pair_counts[("b", "b")] == 1
        assert result.pair_total == 4
        assert result.phone_totals["a"] == 4
        assert result.phone_totals["b"] == 4
        assert result.phone_total == 8


# ---------------------------------------------------------------------------
# TestComputeLogOdds
# ---------------------------------------------------------------------------


class TestComputeLogOdds:
    def test_empty_counts_returns_empty(self):
        counts = CountTables(
            pair_counts={},
            phone_totals=Counter(),
            indel_counts=Counter(),
            pair_total=0,
            phone_total=0,
        )
        result = compute_log_odds(counts)
        assert result == {}

    def test_symmetric_output(self):
        # Build counts from 4 aligned columns.
        alignments = [
            _make_alignment(("a",), ("a",)),
            _make_alignment(("a",), ("b",)),
            _make_alignment(("b",), ("a",)),
            _make_alignment(("b",), ("b",)),
        ]
        counts = accumulate_counts(alignments)
        scores = compute_log_odds(counts)
        assert scores["a"]["b"] == pytest.approx(scores["b"]["a"])

    def test_known_case_laplace(self):
        # 4 columns: (a,a)×1, (a,b)×2, (b,b)×1.
        # pair_counts = {("a","a"):1, ("a","b"):2, ("b","b"):1}
        # phone_totals = {"a":4, "b":4}, pair_total=4, phone_total=8
        # n=2, n_pairs=3, smoothed_total=7
        # p_a = p_b = 0.5
        # S(a,a): q=2/7, expected=0.25 → log2(8/7)
        # S(a,b): q=3/7, expected=0.5 → log2(6/7)
        # S(b,b): q=2/7, expected=0.25 → log2(8/7)
        alignments = [
            _make_alignment(("a",), ("a",)),
            _make_alignment(("a",), ("b",)),
            _make_alignment(("b",), ("a",)),
            _make_alignment(("b",), ("b",)),
        ]
        counts = accumulate_counts(alignments)
        scores = compute_log_odds(counts, smoothing="laplace")
        assert scores["a"]["a"] == pytest.approx(math.log2(8 / 7))
        assert scores["b"]["b"] == pytest.approx(math.log2(8 / 7))
        assert scores["a"]["b"] == pytest.approx(math.log2(6 / 7))

    def test_laplace_zero_count_is_finite(self):
        # (a,b) count is zero; with Laplace, S(a,b) should be a finite negative.
        counts = _make_counts(
            pair_counts={("a", "a"): 5, ("b", "b"): 3},
            phone_totals={"a": 10, "b": 6},
        )
        scores = compute_log_odds(counts, smoothing="laplace")
        assert math.isfinite(scores["a"]["b"])
        assert scores["a"]["b"] < 0

    def test_floor_smoothing_zero_count_equals_floor(self):
        counts = _make_counts(
            pair_counts={("a", "a"): 5, ("b", "b"): 3},
            phone_totals={"a": 10, "b": 6},
        )
        scores = compute_log_odds(counts, smoothing="floor", floor=-7.5)
        assert scores["a"]["b"] == pytest.approx(-7.5)

    def test_min_count_filters_cells(self):
        counts = _make_counts(
            pair_counts={("a", "a"): 10, ("a", "b"): 1, ("b", "b"): 10},
            phone_totals={"a": 21, "b": 21},
        )
        scores = compute_log_odds(counts, min_count=5)
        assert "b" not in scores["a"]
        assert "a" not in scores["b"]
        assert "a" in scores["a"]
        assert "b" in scores["b"]

    def test_self_score_positive(self):
        # Self-pairs should score positive when same-phone matches dominate
        # over cross-phone pairings.  Use equal background frequencies
        # (p_a = p_b = 0.5) with 3 same-phone pairs vs 2 cross-phone pairs.
        # S(a,a): smoothed q_aa = 4/11 > expected 0.25 → log2 > 0.
        alignments = [
            _make_alignment(("a",), ("a",)),
            _make_alignment(("a",), ("a",)),
            _make_alignment(("a",), ("a",)),
            _make_alignment(("a",), ("b",)),
            _make_alignment(("b",), ("a",)),
            _make_alignment(("b",), ("b",)),
            _make_alignment(("b",), ("b",)),
            _make_alignment(("b",), ("b",)),
        ]
        counts = accumulate_counts(alignments)
        scores = compute_log_odds(counts)
        assert scores["a"]["a"] > 0
        assert scores["b"]["b"] > 0

    def test_lidstone_with_alpha_zero_hits_floor_for_unseen(self):
        counts = _make_counts(
            pair_counts={("a", "a"): 5, ("b", "b"): 3},
            phone_totals={"a": 10, "b": 6},
        )
        scores = compute_log_odds(
            counts, smoothing="lidstone", lidstone_alpha=0.0, floor=-9.0
        )
        assert scores["a"]["b"] == pytest.approx(-9.0)

    def test_alphabet_order_is_unicode_sorted(self):
        counts = _make_counts(
            pair_counts={("b", "z"): 1, ("a", "z"): 1},
            phone_totals={"a": 1, "b": 1, "z": 2},
        )
        scores = compute_log_odds(counts)
        assert list(scores.keys()) == sorted(scores.keys())


# ---------------------------------------------------------------------------
# TestBuildMatrixDocument
# ---------------------------------------------------------------------------


class TestBuildMatrixDocument:
    def _make_simple_doc(
        self,
        *,
        source_path: str = "data/training/test.json",
        smoothing: str = "laplace",
        smoothing_params: dict | None = None,
        nw_params: NWParams | None = None,
        source_pair_count: int = 1,
        alignments_used: int = 1,
    ) -> dict:
        counts = accumulate_counts([_make_alignment(("a",), ("b",))])
        scores = compute_log_odds(counts)
        return build_matrix_document(
            counts,
            scores,
            source_path=source_path,
            smoothing=smoothing,
            smoothing_params=smoothing_params or {},
            nw_params=nw_params or NWParams(),
            source_pair_count=source_pair_count,
            alignments_used=alignments_used,
        )

    def test_top_level_keys(self):
        doc = self._make_simple_doc()
        assert "_meta" in doc
        assert "scores" in doc

    def test_meta_required_fields(self):
        doc = self._make_simple_doc()
        meta = doc["_meta"]
        for field in (
            "description",
            "version",
            "generated_at",
            "method",
            "range",
            "source",
            "source_pair_count",
            "alignments_used",
            "alphabet",
            "smoothing",
            "nw_params",
            "indel_counts",
            "totals",
        ):
            assert field in meta, f"Missing _meta field: {field}"

    def test_method_metadata_describes_unordered_pair_expectation(self):
        doc = self._make_simple_doc()
        method = doc["_meta"]["method"]
        assert "expected_ij" in method
        assert "p_i^2" in method
        assert "i == j" in method
        assert "2 * p_i * p_j" in method
        assert "i != j" in method

    def test_generated_at_format(self):
        import re

        doc = self._make_simple_doc()
        ts = doc["_meta"]["generated_at"]
        assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}\+00:00", ts)

    def test_alphabet_is_sorted(self):
        counts = accumulate_counts(
            [
                _make_alignment(("z",), ("a",)),
                _make_alignment(("m",), ("a",)),
            ]
        )
        scores = compute_log_odds(counts)
        doc = build_matrix_document(
            counts,
            scores,
            source_path="x",
            smoothing="laplace",
            smoothing_params={},
            nw_params=NWParams(),
            source_pair_count=2,
            alignments_used=2,
        )
        alphabet = doc["_meta"]["alphabet"]
        assert alphabet == sorted(alphabet)

    def test_scores_matches_computed(self):
        counts = accumulate_counts([_make_alignment(("a",), ("b",))])
        scores = compute_log_odds(counts)
        doc = build_matrix_document(
            counts,
            scores,
            source_path="x",
            smoothing="laplace",
            smoothing_params={},
            nw_params=NWParams(),
            source_pair_count=1,
            alignments_used=1,
        )
        assert doc["scores"] is scores

    def test_nw_params_recorded(self):
        params = NWParams(match=0.5, mismatch=2.0, gap=3.0)
        doc = self._make_simple_doc(nw_params=params)
        assert doc["_meta"]["nw_params"]["match"] == 0.5
        assert doc["_meta"]["nw_params"]["mismatch"] == 2.0
        assert doc["_meta"]["nw_params"]["gap"] == 3.0
