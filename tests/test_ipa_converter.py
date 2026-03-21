"""Tests for proteus.phonology.ipa_converter."""

import unicodedata

import pytest

from proteus.phonology.distance import word_distance
from proteus.phonology.ipa_converter import (
    greek_to_ipa,
    strip_diacritics,
    to_ipa,
    tokenize_ipa,
)


class TestStripDiacritics:
    def test_removes_polytonic_marks(self) -> None:
        assert strip_diacritics("Ἅγιος") == "Αγιος"

    def test_removes_monotonic_accents(self) -> None:
        assert strip_diacritics("άέήίόύώ") == "αεηιουω"

    def test_plain_text_is_unchanged(self) -> None:
        assert strip_diacritics("αβγ") == "αβγ"

    def test_empty_string_returns_empty_string(self) -> None:
        assert strip_diacritics("") == ""


class TestGreekToIpa:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("βγδ", "bɡd"),
            ("ΑΙ", "ai"),
            ("οὐρανός", "oːranos"),
            ("εὖ", "eu"),
            ("ηυ", "ɛːy"),
        ],
    )
    def test_maps_basic_letters_and_diphthongs(
        self, text: str, expected: str
    ) -> None:
        assert "".join(greek_to_ipa(text)) == expected

    def test_handles_mixed_case_after_stripping(self) -> None:
        assert "".join(greek_to_ipa("ΔηΜΟσ")) == "dɛːmos"

    def test_handles_nfd_input(self) -> None:
        text = unicodedata.normalize("NFD", "Αἰών")

        assert "".join(greek_to_ipa(text)) == "aiɔːn"

    def test_empty_string_returns_empty_list(self) -> None:
        assert greek_to_ipa("") == []

    def test_unknown_and_non_greek_characters_are_skipped(self) -> None:
        assert "".join(greek_to_ipa("λόγος,!?")) == "loɡos"

    def test_mixed_alphanumeric_input_keeps_only_known_greek_letters(self) -> None:
        assert "".join(greek_to_ipa("α1β?")) == "ab"

    def test_unknown_characters_emit_debug_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level("DEBUG", logger="proteus.phonology.ipa_converter")

        assert "".join(greek_to_ipa("α1?")) == "a"

        assert "Skipping unknown Greek character '1'" in caplog.text
        assert "Skipping unknown Greek character '?'" in caplog.text

    def test_diaeresis_prevents_false_diphthong(self) -> None:
        assert "".join(greek_to_ipa("ἀϋτή")) == "aytɛː"

    def test_rough_breathing_on_second_diphthong_element_keeps_diphthong(self) -> None:
        assert "".join(greek_to_ipa("αὑτός")) == "autos"
        assert "".join(greek_to_ipa("εὑρίσκω")) == "euriskɔː"
        assert "".join(greek_to_ipa("ηὑρίσκω")) == "ɛːyriskɔː"

    def test_iota_subscript_is_expanded_instead_of_dropped(self) -> None:
        assert "".join(greek_to_ipa("τῇ")) == "tɛːi"

    def test_diphthong_boundary_is_silently_consumed(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Diphthong boundary marker is consumed without a spurious debug log."""
        caplog.set_level("DEBUG", logger="proteus.phonology.ipa_converter")

        result = greek_to_ipa("ἀϋτή")

        assert result == ["a", "y", "t", "ɛː"]
        assert "|" not in caplog.text


class TestToIpa:
    def test_attic_conversion_succeeds(self) -> None:
        assert to_ipa("Δημοσθένης", dialect="attic") == "dɛːmostʰenɛːs"

    def test_rough_breathing_is_preserved_as_h(self) -> None:
        assert to_ipa("ἁλος", dialect="attic") == "halos"

    def test_rough_breathing_distinguishes_otherwise_identical_words(self) -> None:
        assert to_ipa("ἁλος", dialect="attic") == "halos"
        assert to_ipa("ἀλος", dialect="attic") == "alos"

    def test_rough_breathed_diphthongs_do_not_gain_extra_h_phone(self) -> None:
        assert to_ipa("αὑτός", dialect="attic") == "autos"
        assert to_ipa("εὑρίσκω", dialect="attic") == "euriskɔː"
        assert to_ipa("ηὑρίσκω", dialect="attic") == "ɛːyriskɔː"

    def test_compact_output_can_be_compared_against_stressed_ipa(self) -> None:
        assert word_distance(to_ipa("λόγος"), "lóɡos", {}) == pytest.approx(0.0)

    def test_rough_breathed_diphthongs_compare_without_extra_phone_penalty(self) -> None:
        assert word_distance(to_ipa("αὑτός"), "autos", {}) == pytest.approx(0.0)

    def test_non_attic_dialect_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError, match="ionic"):
            to_ipa("λόγος", dialect="ionic")

    def test_invalid_dialect_fails_before_conversion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fail_if_called(_: str) -> list[str]:
            raise AssertionError("greek_to_ipa should not be called")

        monkeypatch.setattr("proteus.phonology.ipa_converter.greek_to_ipa", fail_if_called)

        with pytest.raises(NotImplementedError, match="ionic"):
            to_ipa("λόγος", dialect="ionic")


class TestTokenizeIpa:
    def test_known_h_phone_is_not_treated_as_unknown_literal(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level("DEBUG", logger="proteus.phonology.ipa_converter")

        assert tokenize_ipa("halos") == ["h", "a", "l", "o", "s"]
        assert "Treating unknown IPA token" not in caplog.text

    def test_tokenizes_long_diphthong_as_single_token(self) -> None:
        assert tokenize_ipa("ɛːy") == ["ɛːy"]

    def test_tokenizes_long_upsilon_as_single_token(self) -> None:
        assert tokenize_ipa("pyː") == ["p", "yː"]

    def test_tokenizes_psykhe_without_stray_length_marker(self) -> None:
        assert tokenize_ipa("psyːkʰɛ́ː") == ["ps", "yː", "kʰ", "ɛː"]

    def test_unknown_tokens_emit_debug_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level("DEBUG", logger="proteus.phonology.ipa_converter")

        assert tokenize_ipa("a!") == ["a", "!"]

        assert "Treating unknown IPA token '!'" in caplog.text
        assert "at index 1" in caplog.text
