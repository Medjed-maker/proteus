"""Tests for phonology.ipa_converter."""

import unicodedata

import pytest

from phonology.distance import word_distance
from phonology.ipa_converter import (
    apply_koine_consonant_shifts,
    greek_to_ipa,
    strip_diacritics,
    strip_ignored_ipa_combining_marks,
    to_ipa,
    tokenize_ipa,
    get_known_phones,
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


class TestStripIgnoredIpaCombiningMarks:
    def test_public_strip_ignored_ipa_combining_marks(self) -> None:
        assert strip_ignored_ipa_combining_marks("a\u0301b\u0300c\u0342") == "abc"

    def test_empty_string(self) -> None:
        assert strip_ignored_ipa_combining_marks("") == ""

    def test_no_combining_marks(self) -> None:
        assert strip_ignored_ipa_combining_marks("abc") == "abc"

    def test_consecutive_combining_marks(self) -> None:
        assert strip_ignored_ipa_combining_marks("a\u0301\u0300\u0342") == "a"


class TestGreekToIpa:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("βγδ", "bɡd"),
            ("ΑΙ", "ai"),
            ("οὐρανός", "oːranós"),
            ("εὖ", "éu"),
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

        assert "".join(greek_to_ipa(text)) == "aiɔ́ːn"

    def test_empty_string_returns_empty_list(self) -> None:
        assert greek_to_ipa("") == []

    def test_unknown_and_non_greek_characters_are_skipped(self) -> None:
        assert "".join(greek_to_ipa("λόγος,!?")) == "lóɡos"

    def test_mixed_alphanumeric_input_keeps_only_known_greek_letters(self) -> None:
        assert "".join(greek_to_ipa("α1β?")) == "ab"

    def test_unknown_characters_emit_debug_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level("DEBUG", logger="phonology.ipa_converter")

        assert "".join(greek_to_ipa("α1?")) == "a"

        assert "Skipping unknown Greek character '1'" in caplog.text
        assert "Skipping unknown Greek character '?'" in caplog.text

    def test_literal_h_is_not_treated_as_a_rough_breathing_marker(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level("DEBUG", logger="phonology.ipa_converter")

        assert "".join(greek_to_ipa("αhβ")) == "ab"

        assert "Skipping unknown Greek character 'h'" in caplog.text

    def test_diaeresis_prevents_false_diphthong(self) -> None:
        assert "".join(greek_to_ipa("ἀϋτή")) == "aytɛ́ː"

    def test_rough_breathing_on_second_diphthong_element_keeps_diphthong(self) -> None:
        assert "".join(greek_to_ipa("αὑτός")) == "autós"
        assert "".join(greek_to_ipa("εὑρίσκω")) == "eurískɔː"
        assert "".join(greek_to_ipa("ηὑρίσκω")) == "ɛːyrískɔː"

    def test_iota_subscript_is_expanded_instead_of_dropped(self) -> None:
        assert "".join(greek_to_ipa("τῇ")) == "tɛ́ːi"

    @pytest.mark.parametrize("text", ["ᾱ", unicodedata.normalize("NFD", "ᾱ")])
    def test_macron_alpha_maps_to_long_alpha(self, text: str) -> None:
        assert greek_to_ipa(text) == ["aː"]

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("ἅγιος", "háɡios"),
            ("ἄγω", "áɡɔː"),
            ("αΐ", "aí"),
            ("τοῦτό", "tóːtó"),
            ("ᾧ", "hɔ́ːi"),
            ("ᾅ", "hái"),
        ],
    )
    def test_accent_edge_cases(self, text: str, expected: str) -> None:
        """Accent marks are correctly placed across edge cases."""
        assert "".join(greek_to_ipa(text)) == expected

    def test_diphthong_boundary_is_silently_consumed(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Diphthong boundary marker is consumed without a spurious debug log."""
        caplog.set_level("DEBUG", logger="phonology.ipa_converter")

        result = greek_to_ipa("ἀϋτή")

        assert result == ["a", "y", "t", "ɛ́ː"]
        assert "|" not in caplog.text

    def test_nfc_nfd_equivalence_with_complex_diacritics(self) -> None:
        nfc_text = "ᾅ"
        nfd_text = unicodedata.normalize("NFD", nfc_text)
        assert greek_to_ipa(nfc_text) == greek_to_ipa(nfd_text)
        assert greek_to_ipa(nfc_text) == ["h", unicodedata.normalize("NFC", "a\u0301i")]

    def test_only_diacritics_returns_empty_list(self) -> None:
        assert greek_to_ipa("\u0301\u0314") == []


class TestToIpa:
    def test_attic_conversion_succeeds(self) -> None:
        assert to_ipa("Δημοσθένης", dialect="attic") == "dɛːmostʰénɛːs"

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("λόγος", "lóɣos"),
            ("οδός", "oðós"),
            ("φῶς", "fɔ́ːs"),
            ("θεός", "θeós"),
            ("χείρ", "xéːr"),
        ],
    )
    def test_koine_conversion_applies_supported_consonant_shifts(
        self, text: str, expected: str
    ) -> None:
        assert to_ipa(text, dialect="koine") == expected

    def test_rough_breathing_is_preserved_as_h(self) -> None:
        assert to_ipa("ἁλος", dialect="attic") == "halos"

    def test_rough_breathing_distinguishes_otherwise_identical_words(self) -> None:
        assert to_ipa("ἁλος", dialect="attic") == "halos"
        assert to_ipa("ἀλος", dialect="attic") == "alos"

    def test_rough_breathed_diphthongs_do_not_gain_extra_h_phone(self) -> None:
        assert to_ipa("αὑτός", dialect="attic") == "autós"
        assert to_ipa("εὑρίσκω", dialect="attic") == "eurískɔː"
        assert to_ipa("ηὑρίσκω", dialect="attic") == "ɛːyrískɔː"

    def test_compact_output_can_be_compared_against_stressed_ipa(self) -> None:
        assert word_distance(to_ipa("λόγος"), "lóɡos", {}) == pytest.approx(0.0)

    def test_rough_breathed_diphthongs_compare_without_extra_phone_penalty(self) -> None:
        assert word_distance(to_ipa("αὑτός"), "autos", {}) == pytest.approx(0.0)

    def test_digamma_is_converted_to_w(self) -> None:
        assert to_ipa("ϝοῖκος") == "wóikos"

    def test_unsupported_dialect_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError, match="ionic"):
            to_ipa("λόγος", dialect="ionic")

    def test_invalid_dialect_fails_before_conversion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fail_if_called(_: str) -> list[str]:
            raise AssertionError("greek_to_ipa should not be called")

        monkeypatch.setattr("phonology.ipa_converter.greek_to_ipa", fail_if_called)

        with pytest.raises(NotImplementedError, match="ionic"):
            to_ipa("λόγος", dialect="ionic")


class TestApplyKoineConsonantShifts:
    def test_preserves_accent_for_direct_shift(self) -> None:
        assert apply_koine_consonant_shifts(["kʰ\u0301"]) == ["x\u0301"]

    def test_preserves_accent_for_intervocalic_shift(self) -> None:
        assert apply_koine_consonant_shifts(["a", "d\u0301", "a"]) == ["a", "ð\u0301", "a"]


class TestTokenizeIpa:
    def test_circumflex_alpha_maps_to_long_alpha_token(self) -> None:
        ipa = to_ipa("δᾶμος")

        assert tokenize_ipa(ipa) == ["d", "aː", "m", "o", "s"]

    def test_ignored_accent_marks_are_removed_and_recomposed_to_nfc(self) -> None:
        normalized = strip_ignored_ipa_combining_marks("éu")

        assert normalized == "eu"
        assert unicodedata.is_normalized("NFC", normalized)

    def test_accented_input_is_normalized_once_before_tokenization(self) -> None:
        assert tokenize_ipa("éu") == tokenize_ipa("eu")

    def test_known_h_phone_is_not_treated_as_unknown_literal(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level("DEBUG", logger="phonology.ipa_converter")

        assert tokenize_ipa("halos") == ["h", "a", "l", "o", "s"]
        assert "Treating unknown IPA token" not in caplog.text

    def test_tokenizes_long_diphthong_as_single_token(self) -> None:
        assert tokenize_ipa("ɛːy") == ["ɛːy"]

    def test_tokenizes_long_upsilon_as_single_token(self) -> None:
        assert tokenize_ipa("pyː") == ["p", "yː"]

    def test_tokenizes_psykhe_without_stray_length_marker(self) -> None:
        assert tokenize_ipa("psyːkʰɛ́ː") == ["ps", "yː", "kʰ", "ɛː"]

    def test_preserves_non_accent_combining_marks_during_tokenization(self) -> None:
        assert tokenize_ipa("n̩") == ["n̩"]
        assert tokenize_ipa("ã") == ["ã"]

    def test_attaches_combining_marks_to_preceding_known_phone(self) -> None:
        assert tokenize_ipa("ɛː̃") == ["ɛː̃"]

    def test_attaches_combining_marks_to_preceding_literal_token(self) -> None:
        assert tokenize_ipa("!̃") == ["!̃"]

    def test_unknown_tokens_emit_debug_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level("DEBUG", logger="phonology.core.ipa")

        assert tokenize_ipa("a!") == ["a", "!"]

        assert "Treating unknown IPA token '!'" in caplog.text
        assert "at index 1" in caplog.text


class TestGetKnownPhones:
    def test_get_known_phones_consistency(self) -> None:
        phones: list[str] = get_known_phones()
        assert phones, "get_known_phones() returned an empty list"
        for phone in phones:
            assert tokenize_ipa(phone) == [phone]

    def test_get_known_phones_includes_supported_koine_outputs(self) -> None:
        phones = set(get_known_phones())

        assert {"ɣ", "ð", "f", "θ", "x"} <= phones
