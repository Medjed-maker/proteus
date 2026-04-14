"""Tests for ``_match_partial_query`` infix ordering invariants.

Extracted from ``tests/test_search.py`` during the search package refactor to
keep individual test files focused and under the maintainability budget.
"""

from __future__ import annotations

from phonology import search as search_module
from phonology.search import PartialQueryTokens
from phonology.search._overlap import _contiguous_prefix_match_length


class TestPartialQueryInfixOrdering:
    """Pin the ordering invariant of ``_match_partial_query`` infix matching.

    These tests lock in the contract that when both the left and right
    fragments are provided (infix shape), the match must find them in
    lemma order (left before right, non-overlapping). A wrong-order
    occurrence must produce a hard rejection rather than a partial match,
    and an overlapping placement must also be rejected.
    """

    @staticmethod
    def _infix_tokens(left: tuple[str, ...], right: tuple[str, ...]) -> PartialQueryTokens:
        return PartialQueryTokens(
            shape="infix",
            left_tokens=left,
            right_tokens=right,
        )

    def test_rejects_when_only_wrong_order_pair_exists(self) -> None:
        """Case 1: lemma=[a,b,c,d,e], left=(d,), right=(b,) — wrong order only."""
        partial = self._infix_tokens(left=("d",), right=("b",))
        info = search_module._match_partial_query(partial, ["a", "b", "c", "d", "e"])
        # Both fragments occur, but only in reversed order. Hard reject per contract.
        assert info.matched_fragments == 0
        assert info.overlap_score == 0
        assert info.full_match is False

    def test_matches_when_pair_is_in_order(self) -> None:
        """Case 2: lemma=[a,b,c,d,e], left=(b,), right=(d,) — valid in-order pair."""
        partial = self._infix_tokens(left=("b",), right=("d",))
        info = search_module._match_partial_query(partial, ["a", "b", "c", "d", "e"])
        assert info.matched_fragments == 2
        assert info.full_match is True
        # Left overlap 1 + right overlap 1 = 2.
        assert info.overlap_score == 2

    def test_matches_repeated_fragment_when_two_distinct_occurrences_exist(self) -> None:
        """Case 3: lemma=[a,b,c,b,d], left=(b,), right=(b,) — pick two distinct b tokens."""
        partial = self._infix_tokens(left=("b",), right=("b",))
        info = search_module._match_partial_query(partial, ["a", "b", "c", "b", "d"])
        assert info.matched_fragments == 2
        assert info.full_match is True
        assert info.overlap_score == 2

    def test_rejects_when_pair_would_require_overlap(self) -> None:
        """Case 4: lemma=[a,b,c], left=(b,c), right=(c,) — no non-overlapping pair."""
        partial = self._infix_tokens(left=("b", "c"), right=("c",))
        info = search_module._match_partial_query(partial, ["a", "b", "c"])
        # left_fragment = [b, c] matches at start=1 with overlap 2, ending at index 3.
        # The only right_fragment match is at start=2 (overlap 1), which lies
        # *inside* the left match — so the earliest allowed right start (3)
        # exceeds all right_matches. Both fragments occur, wrong placement → reject.
        assert info.matched_fragments == 0
        assert info.overlap_score == 0
        assert info.full_match is False

    def test_negative_start_index_has_no_contiguous_prefix_match(self) -> None:
        assert _contiguous_prefix_match_length(["b"], ["a", "b"], -1) == 0
