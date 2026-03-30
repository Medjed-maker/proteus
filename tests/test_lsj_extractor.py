"""Tests for the LSJ XML extraction module."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest
etree = pytest.importorskip("lxml.etree")

pytestmark = pytest.mark.usefixtures("reset_pos_overrides_cache")

from phonology.betacode import beta_to_unicode
import phonology.lsj_extractor as lsj_extractor_module
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


class TestLoadPosOverrides:
    """Test POS override loading behavior."""

    def test_missing_pos_overrides_file_returns_empty_overrides(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(lsj_extractor_module, "_pos_overrides", None)
        monkeypatch.setattr(lsj_extractor_module, "resolve_repo_data_dir", lambda _name: tmp_path)

        caplog.set_level("ERROR", logger="phonology.lsj_extractor")
        result = lsj_extractor_module._load_pos_overrides()

        assert result == {
            "common_gender_keys": frozenset(),
            "numeral_keys": frozenset(),
        }
        assert lsj_extractor_module._pos_overrides == result
        assert "POS overrides file not found" in caplog.text

    def test_invalid_pos_overrides_yaml_returns_empty_overrides(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(lsj_extractor_module, "_pos_overrides", None)
        monkeypatch.setattr(lsj_extractor_module, "resolve_repo_data_dir", lambda _name: tmp_path)
        (tmp_path / "pos_overrides.yaml").write_text(
            "common_gender_keys: [unterminated\n",
            encoding="utf-8",
        )

        caplog.set_level("ERROR", logger="phonology.lsj_extractor")
        result = lsj_extractor_module._load_pos_overrides()

        assert result == {
            "common_gender_keys": frozenset(),
            "numeral_keys": frozenset(),
        }
        assert lsj_extractor_module._pos_overrides == result
        assert "Failed to parse POS overrides YAML" in caplog.text

    def test_invalid_utf8_pos_overrides_returns_empty_overrides(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(lsj_extractor_module, "_pos_overrides", None)
        monkeypatch.setattr(lsj_extractor_module, "resolve_repo_data_dir", lambda _name: tmp_path)
        (tmp_path / "pos_overrides.yaml").write_bytes(b"\xff\xfe")

        caplog.set_level("ERROR", logger="phonology.lsj_extractor")
        result = lsj_extractor_module._load_pos_overrides()

        assert result == {
            "common_gender_keys": frozenset(),
            "numeral_keys": frozenset(),
        }
        assert lsj_extractor_module._pos_overrides == result
        assert "Failed to read POS overrides YAML" in caplog.text

    def test_missing_lexicon_data_dir_returns_empty_overrides(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(lsj_extractor_module, "_pos_overrides", None)
        monkeypatch.setattr(
            lsj_extractor_module,
            "resolve_repo_data_dir",
            Mock(side_effect=FileNotFoundError("missing lexicon dir")),
        )

        caplog.set_level("ERROR", logger="phonology.lsj_extractor")
        result = lsj_extractor_module._load_pos_overrides()

        assert result == {
            "common_gender_keys": frozenset(),
            "numeral_keys": frozenset(),
        }
        assert lsj_extractor_module._pos_overrides == result
        assert "lexicon data dir missing" in caplog.text

    def test_missing_lexicon_data_dir_raises_in_cli_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(lsj_extractor_module, "_pos_overrides", None)
        monkeypatch.setattr(
            lsj_extractor_module,
            "resolve_repo_data_dir",
            Mock(side_effect=FileNotFoundError("missing lexicon dir")),
        )

        with pytest.raises(FileNotFoundError, match="missing lexicon dir"):
            lsj_extractor_module._load_pos_overrides(cli_mode=True)

    def test_oserror_reading_pos_overrides_returns_empty_overrides(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(lsj_extractor_module, "_pos_overrides", None)
        monkeypatch.setattr(lsj_extractor_module, "resolve_repo_data_dir", lambda _name: tmp_path)
        (tmp_path / "pos_overrides.yaml").write_text("common_gender_keys: []\n", encoding="utf-8")

        def _raise_oserror(self: Path, **_kwargs: object) -> str:
            raise OSError("read failed")

        monkeypatch.setattr(Path, "read_text", _raise_oserror)
        caplog.set_level("ERROR", logger="phonology.lsj_extractor")

        result = lsj_extractor_module._load_pos_overrides()

        assert result == {
            "common_gender_keys": frozenset(),
            "numeral_keys": frozenset(),
        }
        assert lsj_extractor_module._pos_overrides == result
        assert "Failed to read POS overrides YAML" in caplog.text


class TestExtractEntry:
    """Test single entry extraction."""

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
            entry_id="n610",
            key="au(to/s",
            orth="au(to/s",
            gen="",
            pos="Pron.",
            tr="self",
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["headword"] == "αὑτός"
        assert result["transliteration"] == "hautos"

    def test_diaeresis_breaks_diphthong_transliteration(self) -> None:
        elem = _make_entry_xml(
            entry_id="n615",
            key="a)u+th/",
            orth="a)u+th/",
            gen="",
            pos="Pron.",
            tr="herself",
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["headword"] == "ἀϋτή"
        assert result["transliteration"] == "aytē"
        assert result["ipa"].strip() != ""

    def test_rough_rho_transliteration(self) -> None:
        elem = _make_entry_xml(
            entry_id="n620", key="r(o/dos", orth="r(o/dos", gen="o(", tr="rose"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["headword"] == "ῥόδος"
        assert result["transliteration"] == "rhodos"

    def test_iota_subscript_transliteration_is_preserved(self) -> None:
        elem = _make_entry_xml(
            entry_id="n625", key="w)|dh/", orth="w)|dh/", gen="h(", tr="song"
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

    # -- Gloss cleaning ---------------------------------------------------

    def test_gloss_strips_trailing_comma_from_tr_elements(self) -> None:
        """<tr> elements with trailing commas should produce clean gloss."""
        elem = etree.fromstring(
            '<entryFree id="n1" key="lo/gos" type="main">'
            '<orth extent="full" lang="greek">lo/gos</orth>'
            '<gen lang="greek">o(</gen>'
            '<sense id="s1" n="A" level="1">'
            "<tr>word,</tr><tr>speech</tr>"
            "</sense>"
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["gloss"] == "word, speech"
        assert ",," not in result["gloss"]

    # -- POS and gender heuristics ---------------------------------------

    def test_verb_indicator_requires_verb_like_headword(self) -> None:
        """Sense-level mood markup should not turn adverbs into verbs."""

        elem = etree.fromstring(
            '<entryFree id="n25000" key="w(sanei/" type="main">'
            '<orth extent="full" lang="greek">w(sanei/</orth>'
            '<sense id="s1" n="A" level="1"><tr>as if, as it were</tr>'
            '<mood>part.</mood></sense>'
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["pos"] == "adverb"
        assert "gender" not in result

    def test_capitalized_verb_like_key_stays_verb(self) -> None:
        """Uppercase Beta Code alone should not reclassify a verb as a noun."""

        elem = etree.fromstring(
            '<entryFree id="n25001" key="*dhmosqeni/zw" type="main">'
            '<orth extent="full" lang="greek">*dhmosqeni/zw</orth>'
            '<sense id="s1" n="A" level="1"><tr>imitate Demosthenes</tr>'
            '<mood>inf.</mood></sense>'
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["pos"] == "verb"
        assert "gender" not in result

    def test_homonym_numbered_verb_key_stays_verb(self) -> None:
        """LSJ homonym numbering should not block verb inference."""

        elem = etree.fromstring(
            '<entryFree id="n25010" key="a)ke/w2" type="main">'
            '<orth extent="full" lang="greek">a)ke/w</orth>'
            '<sense id="s1" n="A" level="1"><tr>to be silent</tr>'
            '<tns>pres.</tns></sense>'
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["headword"] == "ἀκέω"
        assert result["pos"] == "verb"
        assert "gender" not in result

    def test_homonym_numbered_adverb_key_stays_adverb(self) -> None:
        """LSJ homonym numbering should not block adverb inference."""

        elem = etree.fromstring(
            '<entryFree id="n25011" key="o(/pws2" type="main">'
            '<orth extent="full" lang="greek">o(/pws dh/</orth>'
            '<sense id="s1" n="A" level="1"><tr>how</tr></sense>'
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["headword"] == "ὅπως"
        assert result["pos"] == "adverb"
        assert "gender" not in result

    def test_itype_pair_infers_adjective(self) -> None:
        """Direct iType pairs like ``a, on`` should infer adjectives."""

        elem = etree.fromstring(
            '<entryFree id="n25002" key="*e(llhniko/s" type="main">'
            '<orth extent="full" lang="greek">*e(llhniko/s</orth>'
            '<itype lang="greek">a</itype>'
            '<itype lang="greek">on</itype>'
            '<sense id="s1" n="A" level="1"><tr>Hellenic, Greek</tr></sense>'
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["pos"] == "adjective"
        assert result["gender"] == "common"

    def test_two_termination_itype_infers_adjective_from_accented_key(self) -> None:
        """Accented ``-o/s`` keys with ``itype=on`` should infer adjectives."""

        elem = etree.fromstring(
            '<entryFree id="n25027" key="kalo/s" type="main">'
            '<orth extent="full" lang="greek">kalo/s</orth>'
            '<itype lang="greek">on</itype>'
            '<sense id="s1" n="A" level="1"><tr>beautiful</tr></sense>'
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["headword"] == "καλός"
        assert result["pos"] == "adjective"
        assert result["gender"] == "common"

    def test_inline_pronoun_label_is_extracted_from_entry_intro(self) -> None:
        """Pronoun entries without direct ``<pos>`` tags should still be kept."""

        elem = etree.fromstring(
            '<entryFree id="n25012" key="au)to/s" type="main">'
            '<orth extent="full" lang="greek">au)to/s</orth>'
            "(Cret. <orth extent=\"full\" lang=\"greek\">a)vto/s</orth>, al.), "
            "<foreign lang=\"greek\">au)th/, au)to/</foreign> reflexive Pron., "
            '<sense id="s1" n="A" level="1"><tr>self</tr></sense>'
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["headword"] == "αὐτός"
        assert result["pos"] == "pronoun"
        assert result["gender"] == "common"

    def test_pronoun_label_after_leading_citation_is_still_detected(self) -> None:
        """Sense-intro POS labels can appear after an opening citation block."""

        elem = etree.fromstring(
            '<entryFree id="n25018" key="au)to/s" type="main">'
            '<orth extent="full" lang="greek">au)to/s</orth> '
            '<sense id="s1" n="A" level="1">'
            '<cit><quote lang="greek">au)to/n</quote></cit>, reflexive Pron., '
            '<tr>self</tr>'
            "</sense>"
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["headword"] == "αὐτός"
        assert result["pos"] == "pronoun"
        assert result["gender"] == "common"

    def test_inline_conjunction_wins_over_later_adverb_subusage(self) -> None:
        """Heading-level conjunction labels should outrank later adverb sub-uses."""

        elem = etree.fromstring(
            '<entryFree id="n25013" key="kai/1" type="main">'
            '<orth extent="full" lang="greek">kai/</orth>, Conj., copulative, '
            'joining words and sentences, '
            '<sense id="s1" n="A" level="1">'
            "<tr>and</tr>; also <pos>Adv.</pos>, <tr>even</tr>"
            "</sense>"
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["headword"] == "καί"
        assert result["pos"] == "conjunction"
        assert "gender" not in result

    def test_inline_interjection_label_is_extracted_from_entry_intro(self) -> None:
        """Interjections labeled only in the heading should be preserved."""

        elem = etree.fromstring(
            '<entryFree id="n25014" key="i)ou/" type="main">'
            '<orth extent="full" lang="greek">i)ou/</orth> or '
            '<orth extent="full" lang="greek">i)ou=</orth>, Interj., '
            '<sense id="s1" n="A" level="1"><tr>ah!</tr></sense>'
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["headword"] == "ἰού"
        assert result["pos"] == "interjection"
        assert "gender" not in result

    def test_attic_article_sense_outranks_non_attic_pronoun_sense(self) -> None:
        """Attic-tagged article senses should win over earlier non-Attic pronoun senses."""

        elem = etree.fromstring(
            '<entryFree id="n25015" key="o(1" type="main">'
            '<orth extent="full" lang="greek">o(</orth>, '
            '<orth extent="full" lang="greek">h(</orth>, '
            '<gen lang="greek">to/</gen>, '
            '<sense id="s1" n="A" level="1">demonstr. Pronoun.</sense>'
            '<sense id="s2" n="B" level="1">in <gramGrp><gram type="dialect">Att.</gram></gramGrp>, '
            'definite or prepositive Article.</sense>'
            '<sense id="s3" n="C" level="1"><tr>the</tr></sense>'
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["headword"] == "ὁ"
        assert result["pos"] == "article"
        assert result["gender"] == "common"

    def test_known_numeral_key_infers_numeral_without_explicit_pos(self) -> None:
        """Common cardinal numerals should not be dropped when LSJ omits a POS tag."""

        elem = etree.fromstring(
            '<entryFree id="n25017" key="de/ka" type="main">'
            '<orth extent="full" lang="greek">de/ka</orth>'
            '<sense id="s1" n="A" level="1"><tr>ten</tr></sense>'
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["headword"] == "δέκα"
        assert result["pos"] == "numeral"
        assert result["gender"] == "common"

    def test_sense_level_non_attic_variants_do_not_flip_entry_dialect(self) -> None:
        """Inline variant dialect notes inside senses should not drop Attic entries."""

        elem = etree.fromstring(
            '<entryFree id="n25016" key="e)gw/" type="main">'
            '<orth extent="full" lang="greek">e)gw/</orth>, <title>I</title>: Pron. '
            'of the first person: '
            '<sense id="s1" n="A" level="1">'
            '<gramGrp><gram type="dialect">Ep.</gram></gramGrp> mostly '
            '<orth extent="full" lang="greek">e)gw/n</orth> before vowels; '
            '<gramGrp><gram type="dialect">Dor.</gram></gramGrp> '
            '<orth extent="full" lang="greek">e)gw/nga</orth>; '
            '<gramGrp><gram type="dialect">Boeot.</gram></gramGrp> '
            '<orth extent="full" lang="greek">i(w/nga</orth>; '
            '<tr>I at least, for my part</tr>'
            "</sense>"
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["headword"] == "ἐγώ"
        assert result["pos"] == "pronoun"
        assert result["dialect"] == "attic"
        assert result["gender"] == "common"

    def test_multiple_non_attic_heading_labels_skip_entry(self) -> None:
        """Entries with only non-Attic heading labels should be excluded."""

        elem = etree.fromstring(
            '<entryFree id="n25021" key="a)gapa/zw" type="main">'
            '<orth extent="full" lang="greek">a)gapa/zw</orth>, '
            '<gramGrp><gram type="dialect">Ep.</gram></gramGrp> and Lyr. form of '
            '<foreign lang="greek">a)gapa/w</foreign>; '
            '<gramGrp><gram type="dialect">Dor.</gram></gramGrp>'
            '<sense id="s1" n="A" level="1"><tns>impf.</tns><tr>love</tr></sense>'
            "</entryFree>"
        )
        assert extract_entry(elem) is None

    def test_heading_variant_dialects_after_pos_keep_attic_entry(self) -> None:
        """Dialect notes after a heading POS label should be treated as variants."""

        elem = etree.fromstring(
            '<entryFree id="n25026" key="e)gw/" type="main">'
            '<orth extent="full" lang="greek">e)gw/</orth>, <title>I</title>: '
            'Pron. of the first person:—'
            '<gramGrp><gram type="dialect">Ep.</gram></gramGrp> mostly '
            '<orth extent="full" lang="greek">e)gw/n</orth> before vowels; '
            '<gramGrp><gram type="dialect">Boeot.</gram></gramGrp> '
            '<orth extent="full" lang="greek">i(w/n</orth>:—'
            '<sense id="s1" n="A" level="1"><tr>I at least, for my part</tr></sense>'
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["pos"] == "pronoun"
        assert result["dialect"] == "attic"
        assert result["gender"] == "common"

    def test_heading_entry_level_non_attic_dialect_after_morphology_skips_entry(self) -> None:
        """Heading dialect labels without a following variant form should filter the entry."""

        elem = etree.fromstring(
            '<entryFree id="n25028" key="a)mfagapa/zw" type="main">'
            '<orth extent="full" lang="greek">a)mfagapa/zw</orth>, '
            '<tns>impf.</tns> <foreign lang="greek">a)mfaga/pazon</foreign>, '
            '<tns>pres.</tns> <mood>part.</mood> '
            '<foreign lang="greek">-omenos</foreign>; by later '
            '<gramGrp><gram type="dialect">Ep.</gram></gramGrp> only in '
            '<tns>pres.</tns>, <tns>impf.</tns> '
            '<sense id="s1" n="A" level="1"><tr>embrace with love</tr></sense>'
            "</entryFree>"
        )

        assert extract_entry(elem) is None

    def test_non_top_level_art_usage_does_not_reclassify_common_noun(self) -> None:
        """Later level-3 usage notes like ``with the Art.`` must not become POS labels."""

        elem = etree.fromstring(
            '<entryFree id="n25019" key="a)/nqrwpos" type="main">'
            '<orth extent="full" lang="greek">a)/nqrwpos</orth>, '
            '<gen lang="greek">h(</gen>, '
            '<gramGrp><gram type="dialect">Att.</gram></gramGrp> crasis '
            '<foreign lang="greek">a(/nqrwpos</foreign>, '
            '<gramGrp><gram type="dialect">Ion.</gram></gramGrp> '
            '<foreign lang="greek">w(/nqrwpos</foreign>:'
            '<sense id="s1" n="A" level="1"><tr>man</tr></sense>'
            '<sense id="s2" n="2" level="3">uses it both with and without the Art. '
            'to denote <tr>man generically</tr></sense>'
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["headword"] == "ἄνθρωπος"
        assert result["pos"] == "noun"
        assert result["gender"] == "common"

    def test_bibliography_title_art_does_not_reclassify_noun(self) -> None:
        """Bibliography titles such as ``Art.`` must not become article labels."""

        elem = etree.fromstring(
            '<entryFree id="n25022" key="a)ski/on" type="main">'
            '<orth extent="full" lang="greek">a)ski/on</orth>, '
            '<gen lang="greek">to/</gen>, '
            '<bibl><author>Plu.</author><title>Art.</title><biblScope>12</biblScope></bibl>: '
            '<sense id="s1" n="A" level="1"><tr>empty threats</tr></sense>'
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["pos"] == "noun"
        assert result["gender"] == "neuter"

    def test_bibliography_title_pron_does_not_reclassify_adjective(self) -> None:
        """Bibliography titles such as ``Pron.`` must not become pronoun labels."""

        elem = etree.fromstring(
            '<entryFree id="n25023" key="e)pi/memptos" type="main">'
            '<orth extent="suff" lang="greek">e)pi/mempt-os</orth>'
            '<itype lang="greek">on</itype>, '
            '<bibl><author>A.D.</author><title>Pron.</title><biblScope>86.2</biblScope></bibl>, '
            'al. <sense id="s1" n="2" level="3">. <tr>blaming</tr></sense>'
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["pos"] == "adjective"
        assert result["gender"] == "common"

    def test_prep_postponed_note_does_not_reclassify_verb(self) -> None:
        """Usage notes like ``Prep. postponed`` must not become prepositions."""

        elem = etree.fromstring(
            '<entryFree id="n25024" key="a)po/llumi" type="main">'
            '<orth extent="full" lang="greek">a)po/llu_mi</orth>, '
            '<tns>impf.</tns> <sense id="s1" n="A" level="1">'
            'freq. in tmesi; Prep. postponed in some constructions. '
            '<tr>destroy utterly</tr>'
            "</sense>"
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["pos"] == "verb"
        assert "gender" not in result

    def test_participle_of_intro_with_singular_gender_stays_noun(self) -> None:
        """Lexicalized nouns like ἄρχων should not be forced to participles."""

        elem = etree.fromstring(
            '<entryFree id="n25025" key="a)/rxwn" type="main">'
            '<orth extent="full" lang="greek">a)/rxwn</orth>, '
            '<itype lang="greek">ontos</itype>, <gen lang="greek">o(</gen>, '
            '(<mood>part.</mood> of <foreign lang="greek">a)/rxw</foreign>) '
            '<sense id="s1" n="A" level="1"><tr>ruler</tr></sense>'
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["pos"] == "noun"
        assert result["gender"] == "masculine"

    def test_participle_of_intro_is_not_misread_as_particle(self) -> None:
        """`part. of` mood notes should classify participial lemmas as participles."""

        elem = etree.fromstring(
            '<entryFree id="n25020" key="o)/nta" type="main">'
            '<orth extent="full" lang="greek">o)/nta</orth>, '
            '<gen lang="greek">ta/</gen>, neut. pl. '
            '<mood>part.</mood> of <foreign lang="greek">ei)mi/</foreign> '
            '<sense id="s1" n="A" level="1">'
            '<tr>the things which actually exist</tr>'
            "</sense>"
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["headword"] == "ὄντα"
        assert result["pos"] == "participle"
        assert result["gender"] == "common"

    def test_explicit_adverb_pos_is_not_overridden_by_itypes(self) -> None:
        """Explicit adverb tags should win over adjective-like iTypes."""

        elem = etree.fromstring(
            '<entryFree id="n25004" key="*persiko/s" type="main">'
            '<orth extent="full" lang="greek">*persiko/s</orth>'
            '<itype lang="greek">h/</itype>'
            '<itype lang="greek">o/n</itype>'
            '<pos>Adv.</pos>'
            '<sense id="s1" n="A" level="1"><tr>Persianly</tr></sense>'
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["pos"] == "adverb"
        assert "gender" not in result

    def test_falls_back_to_key_before_suffixed_orth(self) -> None:
        """Suffixed orthography should normalize to the canonical lemma key."""

        elem = etree.fromstring(
            '<entryFree id="n25005" key="yuxh/" type="main">'
            '<orth extent="suff" lang="greek">yu_x-h/</orth>'
            '<gen lang="greek">h(</gen>'
            '<sense id="s1" n="A" level="1"><tr>soul</tr></sense>'
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["headword"] == "ψυχή"
        assert result["pos"] == "noun"

    def test_strips_homonym_suffix_from_key_fallback(self) -> None:
        """LSJ key numbering should not leak into canonical headwords."""

        elem = etree.fromstring(
            '<entryFree id="n25006" key="de/smh1" type="main">'
            '<orth extent="suff" lang="greek">de/sm-h</orth>'
            '<gen lang="greek">h(</gen>'
            '<sense id="s1" n="A" level="1"><tr>bundle</tr></sense>'
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["headword"] == "δέσμη"
        assert result["pos"] == "noun"

    def test_prefers_key_over_full_suffix_only_orth_for_sis_nouns(self) -> None:
        """Full orth suffixes should not replace the canonical headword key."""

        elem = etree.fromstring(
            '<entryFree id="n25007" key="a)na/klisis" type="main">'
            '<orth extent="suff" lang="greek">a)na/-kli^sis</orth>'
            '<orth extent="full" lang="greek">ews</orth>'
            '<gen lang="greek">h(</gen>'
            '<sense id="s1" n="A" level="1"><tr>reclining</tr></sense>'
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["headword"] == "ἀνάκλισις"
        assert result["pos"] == "noun"
        assert result["gender"] == "feminine"

    def test_prefers_key_over_full_suffix_only_orth_for_ma_nouns(self) -> None:
        """Full orth case endings should not replace neuter lemma keys."""

        elem = etree.fromstring(
            '<entryFree id="n25008" key="a)potei/xisma" type="main">'
            '<orth extent="suff" lang="greek">a)potei/x-isma</orth>'
            '<orth extent="full" lang="greek">atos</orth>'
            '<gen lang="greek">to/</gen>'
            '<sense id="s1" n="A" level="1"><tr>lines of blockade</tr></sense>'
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["headword"] == "ἀποτείχισμα"
        assert result["pos"] == "noun"
        assert result["gender"] == "neuter"

    def test_itypes_on_later_variant_do_not_reclassify_headword(self) -> None:
        """Variant-form iTypes should not turn a verb headword into an adjective."""

        elem = etree.fromstring(
            '<entryFree id="n25009" key="qumiati/zw" type="main">'
            '<orth extent="suff" lang="greek">qu_mi-a_ti/zw</orth>'
            '<orth extent="suff" lang="greek">qu_mi-a_tiko/s</orth>'
            '<itype lang="greek">h/</itype>'
            '<itype lang="greek">o/n</itype>'
            '<sense id="s1" n="A" level="1"><tr>good for burning as incense</tr></sense>'
            "</entryFree>"
        )
        assert extract_entry(elem) is None

    def test_anthropos_is_common_gender(self) -> None:
        """LSJ marks ἄνθρωπος with ``h(``, but the schema should expose common gender."""
        elem = _make_entry_xml(
            entry_id="n25003",
            key="a)/nqrwpos",
            orth="a)/nqrwpos",
            gen="h(",
            pos=None,
            tr="man",
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["pos"] == "noun"
        assert result["gender"] == "common"

    def test_attic_inline_conjunction_outranks_earlier_general_particle_label(self) -> None:
        """Attic-specific POS labels should only upgrade the matching candidate."""

        elem = etree.fromstring(
            '<entryFree id="n25029" key="kai/" type="main">'
            '<orth extent="full" lang="greek">kai/</orth>'
            '<sense id="s1" n="A" level="1">Part., in prose mostly Att. Conj.<tr>and</tr></sense>'
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["headword"] == "καί"
        assert result["pos"] == "conjunction"
        assert "gender" not in result

    def test_skips_entry_with_invalid_beta_code_headword(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Malformed Beta Code in headword is caught and entry skipped."""

        elem = etree.fromstring(
            '<entryFree id="n9000" key="*" type="main">'
            '<orth extent="full" lang="greek">*</orth>'
            '<sense id="s1" n="A" level="1"><tr>test</tr></sense>'
            "</entryFree>"
        )
        caplog.set_level("INFO", logger="phonology.lsj_extractor")
        result = extract_entry(elem)
        assert result is None
        assert "Beta Code conversion failed" in caplog.text


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
