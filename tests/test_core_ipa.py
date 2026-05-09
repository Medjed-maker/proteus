"""Tests for phonology.core.ipa tokenize_ipa equivalence and fallback behavior."""

from __future__ import annotations

import pytest

from phonology.core.ipa import tokenize_ipa
from phonology.search._tokenization import tokenize_ipa as tokenize_ipa_shim


class TestTokenizeIpa:
    """Verify that the core tokenize_ipa and the search shim behave consistently."""

    def test_empty_inventory_splits_ascii_literals(self) -> None:
        """Empty phone_inventory tokenizes as individual character literals."""
        assert tokenize_ipa("pa", phone_inventory=()) == ["p", "a"]

    def test_empty_inventory_splits_space_separated(self) -> None:
        """Space-separated IPA is split on whitespace with empty inventory."""
        assert tokenize_ipa("p a", phone_inventory=()) == ["p", "a"]

    def test_empty_string_returns_empty_list(self) -> None:
        """Empty IPA input tokenizes to no phones."""
        assert tokenize_ipa("", phone_inventory=()) == []
        assert tokenize_ipa_shim("") == tokenize_ipa("", phone_inventory=())

    def test_whitespace_only_returns_empty_list(self) -> None:
        """Whitespace-only IPA input tokenizes to no phones."""
        assert tokenize_ipa("   ", phone_inventory=()) == []
        assert tokenize_ipa_shim("   ") == tokenize_ipa("   ", phone_inventory=())

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("pá", ["p", "a"]),  # acute (precomposed; NFD → U+0301)
            ("pà", ["p", "a"]),  # grave (precomposed; NFD → U+0300)
            ("pâ", ["p", "a"]),  # circumflex (precomposed; NFD → U+0302)
            ("p̂", ["p"]),  # bare combining Latin circumflex U+0302
            ("a͂", ["a"]),  # Greek perispomeni U+0342
        ],
    )
    def test_accent_marks_are_stripped_before_tokenizing(
        self, text: str, expected: list[str]
    ) -> None:
        """Acute, grave, circumflex (U+0302) and perispomeni (U+0342) drop out."""
        result = tokenize_ipa(text, phone_inventory=())
        assert result == expected

    def test_non_accent_combining_marks_are_preserved(self) -> None:
        """Combining marks that are not accent marks survive tokenization."""
        # U+0303 combining tilde (nasalization) is not an accent.
        result = tokenize_ipa("a\u0303", phone_inventory=())
        assert result == ["ã"]

    def test_inventory_enables_digraph_matching(self) -> None:
        """When phone_inventory includes 'kʰ', it is returned as a single token."""
        result = tokenize_ipa("kʰ a", phone_inventory=("kʰ", "k", "a"))
        assert result == ["kʰ", "a"]

    def test_empty_inventory_splits_digraph_into_chars(self) -> None:
        """Without an inventory, 'kʰ' splits into individual characters."""
        assert tokenize_ipa("kʰ", phone_inventory=()) == ["k", "ʰ"]

    @pytest.mark.parametrize("text", ["pa", "kʰ a", "pá"])
    def test_shim_matches_empty_inventory_behavior(self, text: str) -> None:
        """The search shim (no phone_inventory arg) should match empty-inventory core."""
        assert tokenize_ipa_shim(text) == tokenize_ipa(text, phone_inventory=())

    def test_shim_splits_space_separated_aspirated_stop(self) -> None:
        """The search shim should split an aspirated stop without inventory help."""
        assert tokenize_ipa_shim("kʰ a") == ["k", "ʰ", "a"]
