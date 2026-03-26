"""Tests for the Beta Code to Unicode converter."""

from __future__ import annotations

import unicodedata

import pytest

from phonology.betacode import beta_to_unicode


class TestBetaToUnicode:
    """Core Beta Code → Unicode conversions."""

    @pytest.mark.parametrize(
        "beta, expected",
        [
            ("a)/nqrwpos", "ἄνθρωπος"),
            ("lo/gos", "λόγος"),
            ("yuxh/", "ψυχή"),
            ("po/lis", "πόλις"),
            ("qeo/s", "θεός"),
            ("ba/dhn", "βάδην"),
        ],
    )
    def test_known_conversions(self, beta: str, expected: str) -> None:
        assert beta_to_unicode(beta) == expected

    def test_final_sigma(self) -> None:
        result = beta_to_unicode("lo/gos")
        assert result.endswith("ς")
        assert "σ" not in result  # no medial sigma at end

    def test_medial_sigma_preserved(self) -> None:
        # σ in the middle of a word stays as σ
        result = beta_to_unicode("sw=ma")
        assert result[0] == "σ"  # initial sigma stays σ (not ς)

    def test_uppercase_with_smooth_breathing(self) -> None:
        result = beta_to_unicode("*)aqh/nh")
        assert result.startswith("Ἀ")

    def test_uppercase_with_rough_breathing(self) -> None:
        result = beta_to_unicode("*(hraklh=s")
        # Should start with Ἡ (capital eta with rough breathing)
        assert result.startswith("Ἡ")

    def test_circumflex(self) -> None:
        result = beta_to_unicode("sw=ma")
        # ω with circumflex → ῶ
        assert "ῶ" in result

    def test_iota_subscript(self) -> None:
        result = beta_to_unicode("tw=|")
        # Should contain omega with circumflex and iota subscript
        assert "ῷ" in result

    def test_diaeresis(self) -> None:
        # a+i/ applies diaeresis to alpha and acute to iota.
        result = beta_to_unicode("a+i/")
        expected = unicodedata.normalize("NFC", "α̈ί")
        assert unicodedata.normalize("NFC", result) == expected

    def test_grave_accent(self) -> None:
        result = beta_to_unicode("a\\")
        assert "ὰ" in result

    def test_empty_string(self) -> None:
        assert beta_to_unicode("") == ""

    def test_passthrough_punctuation(self) -> None:
        result = beta_to_unicode("a, b")
        assert "," in result

    @pytest.mark.parametrize("beta", ["*", "*)", "*(", "*()/"])
    def test_invalid_uppercase_marker_raises(self, beta: str) -> None:
        with pytest.raises(ValueError, match=r"index 0.*diacritics"):
            beta_to_unicode(beta)

    def test_numeric_only_input_is_preserved(self) -> None:
        assert beta_to_unicode("12345") == "12345"

    def test_existing_unicode_input_is_preserved(self) -> None:
        assert beta_to_unicode("αβγ") == "αβγ"

    def test_mixed_unicode_and_betacode_input_is_converted_selectively(self) -> None:
        assert beta_to_unicode("μῖξ lo/gos") == "μῖξ λόγος"

    def test_multiple_words_final_sigma(self) -> None:
        result = beta_to_unicode("lo/gos kai\\ mu=qos")
        # Both word-final sigma should be ς
        words = result.split()
        assert words[0].endswith("ς")
        assert words[2].endswith("ς")

    @pytest.mark.parametrize(
        "beta, expected_punctuation",
        [
            (f"lo/gos{chr(0x037E)}", ";"),  # Greek question mark
            (f"lo/gos{chr(0x0387)}", "·"),  # Greek ano teleia (middle dot)
        ],
    )
    def test_final_sigma_before_greek_punctuation(
        self, beta: str, expected_punctuation: str
    ) -> None:
        result = beta_to_unicode(beta)
        normalized_result = unicodedata.normalize("NFC", result)
        assert normalized_result[-1] == expected_punctuation
        assert normalized_result[-2] == "ς"
