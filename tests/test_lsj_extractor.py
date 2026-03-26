"""Tests for the LSJ XML extraction module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from phonology.betacode import beta_to_unicode
from phonology.lsj_extractor import (
    build_lexicon_document,
    extract_entry,
    validate_document,
)

if TYPE_CHECKING:
    from lxml.etree import _Element


def _make_entry_xml(
    entry_id: str = "n100",
    key: str = "lo/gos",
    entry_type: str = "main",
    orth: str = "lo/gos",
    gen: str = "o(",
    pos: str | None = None,
    tr: str = "word, speech",
    dialect: str | None = None,
) -> _Element:
    """Build a minimal <entryFree> element for testing."""
    from lxml import etree

    parts = [f'<entryFree id="{entry_id}" key="{key}" type="{entry_type}">']
    if orth:
        parts.append(f'<orth extent="full" lang="greek">{orth}</orth>')
    if gen:
        parts.append(f'<gen lang="greek">{gen}</gen>')
    if pos:
        parts.append(f"<pos>{pos}</pos>")
    if dialect:
        parts.append(f'<gramGrp><gram type="dialect">{dialect}</gram></gramGrp>')
    parts.append('<sense id="s1" n="A" level="1">')
    if tr:
        parts.append(f"<tr>{tr}</tr>")
    parts.append("</sense>")
    parts.append("</entryFree>")
    return etree.fromstring("".join(parts))


# --------------------------------------------------------------------------
# extract_entry tests
# --------------------------------------------------------------------------


class TestExtractEntry:
    """Test single entry extraction."""

    @pytest.fixture(autouse=True)
    def _skip_without_lxml(self) -> None:
        pytest.importorskip("lxml")

    def test_basic_noun_extraction(self) -> None:
        elem = _make_entry_xml(
            entry_id="n100", orth="lo/gos", gen="o(", tr="word, speech"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["id"] == "LSJ-000100"
        assert result["headword"] == "λόγος"
        assert result["pos"] == "noun"
        assert result["gender"] == "masculine"
        assert result["gloss"] == "word, speech"
        assert result["dialect"] == "attic"
        assert "ipa" in result
        assert isinstance(result["ipa"], str)
        assert result["ipa"].strip() != ""
        assert "transliteration" in result
        assert isinstance(result["transliteration"], str)
        assert result["transliteration"].strip() != ""

    def test_explicit_pos_adjective(self) -> None:
        elem = _make_entry_xml(
            entry_id="n200",
            orth="kalo/s",
            gen="",
            pos="Adj.",
            tr="beautiful",
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["pos"] == "adjective"
        # Gender should default to "common" for adjectives without gender
        assert result["gender"] == "common"

    def test_explicit_pos_adverb(self) -> None:
        elem = _make_entry_xml(
            entry_id="n300",
            orth="ba/dhn",
            gen="",
            pos="Adv.",
            tr="step by step",
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["pos"] == "adverb"
        assert "gender" not in result  # adverbs don't need gender

    def test_skips_gloss_type(self) -> None:
        elem = _make_entry_xml(entry_type="gloss")
        result = extract_entry(elem)
        assert result is None

    def test_skips_no_gloss(self) -> None:
        elem = _make_entry_xml(tr="")
        result = extract_entry(elem)
        assert result is None

    def test_skips_no_id(self) -> None:
        from lxml import etree

        elem = etree.fromstring(
            '<entryFree key="test" type="main">'
            '<orth extent="full" lang="greek">test</orth>'
            '<sense id="s1" n="A" level="1"><tr>test</tr></sense>'
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is None

    def test_feminine_gender(self) -> None:
        elem = _make_entry_xml(
            entry_id="n400", orth="yuxh/", gen="h(", tr="soul"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["gender"] == "feminine"

    def test_neuter_gender(self) -> None:
        elem = _make_entry_xml(
            entry_id="n500", orth="sw=ma", gen="to/", tr="body"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["gender"] == "neuter"

    @pytest.mark.parametrize(
        ("gen_value", "expected_gender"),
        [("ὁ", "masculine"), ("ἡ", "feminine"), ("τό", "neuter"), ("τὸ", "neuter")],
    )
    def test_unicode_article_gender_fallback(
        self, gen_value: str, expected_gender: str
    ) -> None:
        elem = _make_entry_xml(
            entry_id="n550", orth="lo/gos", gen=gen_value, tr="word"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["gender"] == expected_gender

    def test_dialect_extraction(self) -> None:
        elem = _make_entry_xml(
            entry_id="n600", orth="lo/gos", gen="o(", tr="word", dialect="Dor."
        )
        result = extract_entry(elem)
        assert result is None

    def test_attic_entries_pass_dialect_through_to_ipa(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        elem = _make_entry_xml(
            entry_id="n605", orth="lo/gos", gen="o(", tr="word"
        )
        captured: dict[str, str] = {}

        def fake_to_ipa(greek_text: str, dialect: str = "attic") -> str:
            captured["greek_text"] = greek_text
            captured["dialect"] = dialect
            return "mock-ipa"

        monkeypatch.setattr("phonology.lsj_extractor.to_ipa", fake_to_ipa)

        result = extract_entry(elem)

        assert result is not None
        assert result["ipa"] == "mock-ipa"
        assert captured == {"greek_text": "λόγος", "dialect": "attic"}

    @pytest.mark.parametrize(
        ("dialect_label", "expected_dialect"),
        [("Dor.", "doric"), ("Ion.", "ionic"), ("Ep.", "ionic")],
    )
    def test_non_attic_entries_are_skipped_with_info_log(
        self,
        dialect_label: str,
        expected_dialect: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        elem = _make_entry_xml(
            entry_id="n608", orth="lo/gos", gen="o(", tr="word", dialect=dialect_label
        )
        caplog.set_level("INFO", logger="phonology.lsj_extractor")

        result = extract_entry(elem)

        assert result is None
        assert (
            f"Skipping non-Attic entry LSJ-000608 (λόγος): dialect={expected_dialect}"
            in caplog.text
        )

    def test_rough_breathed_diphthong_transliteration(self) -> None:
        elem = _make_entry_xml(
            entry_id="n610", orth="au(to/s", gen="", pos="Pron.", tr="self"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["headword"] == "αὑτός"
        assert result["transliteration"] == "hautos"

    def test_diaeresis_breaks_diphthong_transliteration(self) -> None:
        elem = _make_entry_xml(
            entry_id="n615", orth="a)u+th/", gen="", pos="Pron.", tr="herself"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["headword"] == "ἀϋτή"
        assert result["transliteration"] == "aytē"
        assert result["ipa"].strip() != ""

    def test_rough_rho_transliteration(self) -> None:
        elem = _make_entry_xml(
            entry_id="n620", orth="r(o/dos", gen="o(", tr="rose"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["headword"] == "ῥόδος"
        assert result["transliteration"] == "rhodos"

    def test_iota_subscript_transliteration_is_preserved(self) -> None:
        elem = _make_entry_xml(
            entry_id="n625", orth="w)|dh/", gen="h(", tr="song"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["headword"] == "ᾠδή"
        assert result["transliteration"] == "ōidē"

    def test_skips_single_letter_headword(self) -> None:
        elem = _make_entry_xml(
            entry_id="n630", key="a", orth="a", gen="", pos="Part.", tr="alpha"
        )
        assert extract_entry(elem) is None

    def test_id_formatting(self) -> None:
        elem = _make_entry_xml(entry_id="n5")
        result = extract_entry(elem)
        assert result is not None
        assert result["id"] == "LSJ-000005"

    def test_large_id_formatting(self) -> None:
        elem = _make_entry_xml(entry_id="n116502")
        result = extract_entry(elem)
        assert result is not None
        assert result["id"] == "LSJ-116502"

    def test_skips_expected_ipa_conversion_errors_with_info_log(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        elem = _make_entry_xml(entry_id="n700", orth="lo/gos", gen="o(", tr="word")

        def fail_to_ipa(_: str, dialect: str = "attic") -> str:
            raise ValueError("bad conversion")

        monkeypatch.setattr("phonology.lsj_extractor.to_ipa", fail_to_ipa)
        caplog.set_level("INFO", logger="phonology.lsj_extractor")

        result = extract_entry(elem)

        assert result is None
        assert "IPA conversion failed for LSJ-000700 (λόγος): ValueError: bad conversion" in caplog.text

    @pytest.mark.parametrize("error", [TypeError("bad type"), AttributeError("missing attr")])
    def test_unexpected_ipa_conversion_errors_are_reraised(
        self,
        monkeypatch: pytest.MonkeyPatch,
        error: Exception,
    ) -> None:
        elem = _make_entry_xml(entry_id="n710", orth="lo/gos", gen="o(", tr="word")

        def fail_to_ipa(_: str, dialect: str = "attic") -> str:
            raise error

        monkeypatch.setattr("phonology.lsj_extractor.to_ipa", fail_to_ipa)

        with pytest.raises(type(error), match=str(error)):
            extract_entry(elem)


# --------------------------------------------------------------------------
# Document building tests
# --------------------------------------------------------------------------


class TestBuildDocument:
    """Test document assembly and validation."""

    def _sample_entries(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "LSJ-000001",
                "headword": "λόγος",
                "transliteration": "logos",
                "ipa": "lóɡos",
                "pos": "noun",
                "gender": "masculine",
                "gloss": "word, speech",
                "dialect": "attic",
            },
            {
                "id": "LSJ-000002",
                "headword": "βάδην",
                "transliteration": "badēn",
                "ipa": "bádɛːn",
                "pos": "adverb",
                "gloss": "step by step",
                "dialect": "attic",
            },
        ]

    def test_build_document_structure(self) -> None:
        entries = self._sample_entries()
        doc = build_lexicon_document(entries)
        assert doc["schema_version"] == "2.0.0"
        assert "_meta" in doc
        assert doc["_meta"]["source"].startswith("LSJ")
        assert doc["_meta"]["dialect"] == "attic"
        assert "filtered to Attic entries" in doc["_meta"]["description"]
        assert doc["_meta"]["note"].endswith("output dialect is attic")
        assert doc["_meta"]["license"] == "CC-BY-SA 4.0"
        assert doc["lemmas"] == entries

    def test_build_document_uses_single_entry_dialect_in_metadata_text(self) -> None:
        entries = [
            {
                "id": "LSJ-000003",
                "headword": "θάλασσα",
                "transliteration": "thalassa",
                "ipa": "tʰálassa",
                "pos": "noun",
                "gender": "feminine",
                "gloss": "sea",
                "dialect": "ionic",
            }
        ]

        doc = build_lexicon_document(entries)

        assert doc["_meta"]["dialect"] == "ionic"
        assert "filtered to Ionic entries" in doc["_meta"]["description"]
        assert doc["_meta"]["note"].endswith("output dialect is ionic")

    def test_build_document_rejects_mixed_dialects(self) -> None:
        entries = self._sample_entries()
        entries.append(
            {
                "id": "LSJ-000003",
                "headword": "θάλασσα",
                "transliteration": "thalassa",
                "ipa": "tʰálassa",
                "pos": "noun",
                "gender": "feminine",
                "gloss": "sea",
                "dialect": "ionic",
            }
        )

        with pytest.raises(ValueError, match="single output dialect"):
            build_lexicon_document(entries)

    def test_validate_valid_document(self) -> None:
        entries = self._sample_entries()
        doc = build_lexicon_document(entries)
        # Should not raise
        validate_document(doc)

    def test_validate_rejects_missing_fields(self) -> None:
        doc = build_lexicon_document(
            [{"id": "LSJ-000001", "headword": "test"}]  # missing required fields
        )
        with pytest.raises(ValueError, match="Schema validation failed"):
            validate_document(doc)

    def test_validate_rejects_gender_for_noun(self) -> None:
        doc = build_lexicon_document(
            [
                {
                    "id": "LSJ-000001",
                    "headword": "λόγος",
                    "transliteration": "logos",
                    "ipa": "logos",
                    "pos": "noun",
                    # missing gender — required for nouns
                    "gloss": "word",
                    "dialect": "attic",
                }
            ]
        )
        with pytest.raises(ValueError, match="Schema validation failed"):
            validate_document(doc)

    def test_validate_rejects_invalid_pos(self) -> None:
        doc = build_lexicon_document(
            [
                {
                    "id": "LSJ-000001",
                    "headword": "λόγος",
                    "transliteration": "logos",
                    "ipa": "logos",
                    "pos": "unknown_pos",
                    "gender": "masculine",
                    "gloss": "word",
                    "dialect": "attic",
                }
            ]
        )
        with pytest.raises(ValueError, match="Schema validation failed"):
            validate_document(doc)

    def test_validate_rejects_invalid_dialect(self) -> None:
        doc = build_lexicon_document(
            [
                {
                    "id": "LSJ-000001",
                    "headword": "λόγος",
                    "transliteration": "logos",
                    "ipa": "logos",
                    "pos": "noun",
                    "gender": "masculine",
                    "gloss": "word",
                    "dialect": "invalid_dialect",
                }
            ]
        )
        with pytest.raises(ValueError, match="Schema validation failed"):
            validate_document(doc)

    def test_validate_accepts_empty_lemmas(self) -> None:
        doc = build_lexicon_document([])
        assert doc["_meta"]["dialect"] == "attic"
        # Schema permits an empty lemmas array; the CLI main() enforces
        # at least one entry at a higher level.
        validate_document(doc)


# --------------------------------------------------------------------------
# Beta Code integration
# --------------------------------------------------------------------------


class TestBetaCodeIntegration:
    """Ensure Beta Code → Unicode → IPA pipeline works end-to-end."""

    @pytest.mark.parametrize(
        "beta, expected_headword, expected_ipa",
        [
            ("lo/gos", "λόγος", "lóɡos"),
            ("a)/nqrwpos", "ἄνθρωπος", "ántʰrɔːpos"),
            ("yuxh/", "ψυχή", "psykʰɛ́ː"),
            pytest.param("", "", "", id="empty-string"),
            pytest.param("?!", "?!", "", id="punctuation"),
            pytest.param("*", None, None, id="invalid-uppercase-marker"),
        ],
    )
    def test_beta_to_headword_to_ipa(
        self,
        beta: str,
        expected_headword: str | None,
        expected_ipa: str | None,
    ) -> None:
        from phonology.ipa_converter import to_ipa

        if expected_headword is None:
            with pytest.raises(ValueError, match=r"Uppercase marker '\*' at index 0"):
                beta_to_unicode(beta)
            return

        headword = beta_to_unicode(beta)
        assert headword == expected_headword
        ipa = to_ipa(headword)
        assert ipa == expected_ipa
