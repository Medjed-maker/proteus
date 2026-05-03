"""Tests for Greek to Latin transliteration."""

from __future__ import annotations

import logging

import pytest

from phonology.transliterate import transliterate


class TestTransliterate:
    """Transliteration must match existing sample data conventions."""

    @pytest.mark.parametrize(
        "greek, expected",
        [
            ("ἄνθρωπος", "anthrōpos"),
            ("λόγος", "logos"),
            ("ψυχή", "psykhē"),
            ("πόλις", "polis"),
            ("θεός", "theos"),
        ],
    )
    def test_matches_existing_samples(self, greek: str, expected: str) -> None:
        assert transliterate(greek) == expected

    def test_rough_breathing_prefix_h(self) -> None:
        # ἡ (eta with rough breathing) → hē
        assert transliterate("ἡμέρα") == "hēmera"

    def test_smooth_breathing_no_h(self) -> None:
        # ἀ (alpha with smooth breathing) → a (no h prefix)
        result = transliterate("ἄνθρωπος")
        assert result.startswith("a")
        assert not result.startswith("ha")

    def test_diphthongs(self) -> None:
        assert transliterate("αἴτιος") == "aitios"
        assert transliterate("οὐρανός") == "ouranos"
        assert transliterate("εὐχή") == "eukhē"

    def test_rough_breathed_diphthong_preserves_initial_h(self) -> None:
        assert transliterate("αὑτός") == "hautos"
        assert transliterate("εὑρίσκω") == "heuriskō"

    def test_diaeresis_prevents_false_diphthong(self) -> None:
        assert transliterate("ἀϋτή") == "aytē"
        assert transliterate("αὐτή") == "autē"
        assert transliterate("εϋ") == "ey"
        assert transliterate("ευ") == "eu"

    def test_rough_rho_uses_rh(self) -> None:
        assert transliterate("ῥόδος") == "rhodos"

    def test_final_sigma(self) -> None:
        assert transliterate("λόγος") == "logos"

    def test_chi_as_kh(self) -> None:
        assert transliterate("χρόνος") == "khronos"

    def test_phi_as_ph(self) -> None:
        assert transliterate("φίλος") == "philos"

    @pytest.mark.parametrize(
        "greek, expected",
        [
            ("ἄγγελος", "angelos"),
            ("ἐγκώμιον", "enkōmion"),
            ("ἄγχη", "ankhē"),
            ("ἄγξω", "anxō"),
        ],
    )
    def test_nasal_assimilation_before_velars(self, greek: str, expected: str) -> None:
        assert transliterate(greek) == expected

    def test_gamma_before_non_velar_remains_g(self) -> None:
        assert transliterate("ἀγορά") == "agora"

    def test_empty_string(self) -> None:
        assert transliterate("") == ""

    def test_non_greek_characters(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="phonology.transliterate"):
            assert transliterate("hello") == "hello"
            assert transliterate("123") == "123"

        assert len(caplog.records) == 0
        caplog.clear()

        with caplog.at_level(logging.WARNING, logger="phonology.transliterate"):
            assert transliterate("λόγ🙂ος") == "log🙂os"

        assert len(caplog.records) == 1
        assert "Unsupported non-ASCII character" in caplog.records[0].message

    def test_multiple_words(self) -> None:
        assert transliterate("ὁ λόγος") == "ho logos"

    def test_uppercase_greek(self) -> None:
        assert transliterate("ΛΟΓΟΣ") == "logos"

    def test_eta_as_long_e(self) -> None:
        result = transliterate("ψυχή")
        assert result.endswith("ē")

    def test_omega_as_long_o(self) -> None:
        assert transliterate("σῶμα") == "sōma"

    def test_digamma(self) -> None:
        assert transliterate("ϝοῖκος") == "woikos"

    @pytest.mark.parametrize(
        "greek, expected",
        [
            ("τῇ", "tēi"),
            ("ᾠδή", "ōidē"),
            ("ᾅδω", "haidō"),
        ],
    )
    def test_iota_subscript_is_expanded_to_explicit_i(
        self, greek: str, expected: str
    ) -> None:
        assert transliterate(greek) == expected
