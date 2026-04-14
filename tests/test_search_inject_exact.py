"""Tests for ``_inject_exact_ipa_matches`` candidate injection.

Extracted from ``tests/test_search.py`` during the search package refactor.
"""

from __future__ import annotations

import pytest

from phonology.search import LexiconRecord
from phonology.search._lookup import build_ipa_index, _inject_exact_ipa_matches


@pytest.fixture
def basic_lookup() -> dict[str, dict[str, str]]:
    """Return a basic lookup dict used by multiple tests."""
    return {
        "L1": {"id": "L1", "headword": "α", "ipa": "aaa"},
        "L2": {"id": "L2", "headword": "β", "ipa": "bbb"},
    }


class TestInjectExactIpaMatches:
    """Verify exact IPA match injection into candidate lists."""

    def test_does_not_mutate_input_candidates(self) -> None:
        lookup = {
            "L1": {"id": "L1", "headword": "α", "ipa": "aaa"},
        }
        candidates = ["L2"]
        original = list(candidates)
        result = _inject_exact_ipa_matches("aaa", candidates, lookup)
        assert candidates == original
        assert result is not candidates
        # Exact match L1 is prepended before original candidates
        assert result == ["L1", "L2"]

    def test_handles_special_characters_in_ipa(self) -> None:
        lookup = {
            "L1": {"id": "L1", "headword": "α", "ipa": "a.b+c"},
            "L2": {"id": "L2", "headword": "β", "ipa": "bbb"},
        }
        candidates = ["L2"]
        result = _inject_exact_ipa_matches("a.b+c", candidates, lookup)
        assert result == ["L1", "L2"]

    def test_normalizes_query_ipa_whitespace(self) -> None:
        lookup = {
            "L1": {"id": "L1", "headword": "α", "ipa": "aaa"},
            "L2": {"id": "L2", "headword": "β", "ipa": "bbb"},
        }

        result = _inject_exact_ipa_matches(" aaa ", ["L2"], lookup)

        assert result == ["L1", "L2"]

    def test_normalizes_lookup_ipa_whitespace_without_index(self) -> None:
        lookup = {
            "L1": {"id": "L1", "headword": "α", "ipa": " aaa "},
            "L2": {"id": "L2", "headword": "β", "ipa": "bbb"},
        }

        result = _inject_exact_ipa_matches("aaa", ["L2"], lookup)

        assert result == ["L1", "L2"]

    def test_uses_ipa_index_for_exact_matches(self) -> None:
        lookup = {
            "L1": {"id": "L1", "headword": "α", "ipa": " aaa "},
            "L2": {"id": "L2", "headword": "β", "ipa": "bbb"},
        }

        result = _inject_exact_ipa_matches(
            "aaa",
            ["L2"],
            lookup,
            ipa_index=build_ipa_index(lookup),
        )

        assert result == ["L1", "L2"]

    def test_matches_unaccented_query_to_accented_lookup_without_index(self) -> None:
        lookup = {
            "L1": {"id": "L1", "headword": "νῦν", "ipa": "nýn"},
            "L2": {"id": "L2", "headword": "β", "ipa": "bbb"},
        }

        result = _inject_exact_ipa_matches("nyn", ["L2"], lookup)

        assert result == ["L1", "L2"]

    def test_matches_unaccented_query_to_accented_lookup_with_index(self) -> None:
        lookup = {
            "L1": {"id": "L1", "headword": "νῦν", "ipa": "nýn"},
            "L2": {"id": "L2", "headword": "β", "ipa": "bbb"},
        }

        result = _inject_exact_ipa_matches(
            "nyn",
            ["L2"],
            lookup,
            ipa_index=build_ipa_index(lookup),
        )

        assert result == ["L1", "L2"]

    def test_matches_accented_query_to_unaccented_lookup(self) -> None:
        lookup = {
            "L1": {"id": "L1", "headword": "νυν", "ipa": "nyn"},
            "L2": {"id": "L2", "headword": "β", "ipa": "bbb"},
        }

        result = _inject_exact_ipa_matches("nýn", ["L2"], lookup)

        assert result == ["L1", "L2"]

    def test_normalizes_whitespace_and_accents_for_indexed_lookup(self) -> None:
        lookup = {
            "L1": {"id": "L1", "headword": "νῦν", "ipa": " nýn "},
            "L2": {"id": "L2", "headword": "β", "ipa": "bbb"},
        }

        result = _inject_exact_ipa_matches(
            " nyn ",
            ["L2"],
            lookup,
            ipa_index=build_ipa_index(lookup),
        )

        assert result == ["L1", "L2"]

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
        """L1 is already in candidates so only other IPA matches (L3,L4) should be prepended."""
        lookup = {
            "L1": {"id": "L1", "headword": "α", "ipa": "aaa"},
            "L2": {"id": "L2", "headword": "β", "ipa": "bbb"},
            "L3": {"id": "L3", "headword": "γ", "ipa": "aaa"},
            "L4": {"id": "L4", "headword": "δ", "ipa": "aaa"},
        }
        candidates = ["L2", "L1"]
        result = _inject_exact_ipa_matches("aaa", candidates, lookup, limit=3)
        assert len(result) == 3
        assert set(result[:2]) == {"L3", "L4"}
        assert result[2] == "L2"

    def test_returns_candidates_when_lookup_is_empty(
        self, basic_lookup: dict[str, dict[str, str]]
    ) -> None:
        result = _inject_exact_ipa_matches("aaa", ["L1"], {})

        assert result == ["L1"]

    def test_limit_zero_returns_empty_list(self) -> None:
        lookup = {
            "L1": LexiconRecord(
                entry={"id": "L1", "headword": "α", "ipa": "aaa"},
                token_count=3,
                ipa_tokens=("a", "a", "a"),
            ),
        }

        result = _inject_exact_ipa_matches("aaa", ["L1"], lookup, limit=0)

        assert result == []

    def test_truncates_when_exact_matches_exceed_limit(self) -> None:
        """Results are returned in insertion order (dict iteration order, L1, L2, L3)."""
        lookup = {
            "L1": LexiconRecord(
                entry={"id": "L1", "headword": "α", "ipa": "aaa"},
                token_count=3,
                ipa_tokens=("a", "a", "a"),
            ),
            "L2": LexiconRecord(
                entry={"id": "L2", "headword": "β", "ipa": "aaa"},
                token_count=3,
                ipa_tokens=("a", "a", "a"),
            ),
            "L3": LexiconRecord(
                entry={"id": "L3", "headword": "γ", "ipa": "aaa"},
                token_count=3,
                ipa_tokens=("a", "a", "a"),
            ),
        }

        result = _inject_exact_ipa_matches("aaa", [], lookup, limit=2)

        assert len(result) == 2
        # Results follow insertion order: L1, L2 (L3 truncated)
        assert result == ["L1", "L2"]

    def test_returns_unchanged_when_no_match(
        self, basic_lookup: dict[str, dict[str, str]]
    ) -> None:
        candidates = ["L1"]
        result = _inject_exact_ipa_matches("zzz", candidates, basic_lookup)
        assert result == ["L1"]

    def test_no_duplicate_when_already_in_candidates(
        self, basic_lookup: dict[str, dict[str, str]]
    ) -> None:
        candidates = ["L1", "L2"]
        result = _inject_exact_ipa_matches("aaa", candidates, basic_lookup)
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
