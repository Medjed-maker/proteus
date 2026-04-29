"""Tests for the LSJ XML extraction module."""

from __future__ import annotations

import json
from pathlib import Path
import sys
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


def _make_heading_context(
    **overrides: Any,
) -> lsj_extractor_module._HeadingContext:
    """Build a minimal heading context for dialect predicate tests."""
    defaults: dict[str, Any] = {
        "following_greek_form_count": 0,
        "has_following_form": False,
        "has_following_surface_form": False,
        "has_following_gen_marker": False,
        "has_following_itype_marker": False,
        "has_non_dialect_gramgrp_before_gen": False,
        "has_preceding_form": False,
        "has_preceding_itype_marker": False,
        "has_prior_dialect_label": False,
        "has_following_dialect_label": False,
        "has_following_attic_label": False,
    }
    defaults.update(overrides)
    return lsj_extractor_module._HeadingContext(**defaults)


def _make_dialect_decision_context(
    **overrides: Any,
) -> lsj_extractor_module._DialectDecisionContext:
    """Build a minimal dialect decision context for predicate tests."""
    defaults: dict[str, Any] = {
        "mapped_dialect": "doric",
        "heading": _make_heading_context(),
        "variant_context": False,
        "entry_level_constraint": False,
        "prior_headword_context": False,
        "following_headword_context": False,
        "has_distinct_following_surface_form": False,
        "has_extra_following_forms": False,
    }
    defaults.update(overrides)
    return lsj_extractor_module._DialectDecisionContext(**defaults)


# --------------------------------------------------------------------------
# extract_entry tests
# --------------------------------------------------------------------------


class TestDialectVariantPredicates:
    """Test dialect variant-only predicate helpers directly."""

    def test_is_attic_without_prior(self) -> None:
        context = _make_dialect_decision_context(mapped_dialect="attic")

        assert lsj_extractor_module._is_attic_without_prior(context) is True

    def test_is_single_dialect_surface_variant_with_headword_context(self) -> None:
        context = _make_dialect_decision_context(
            heading=_make_heading_context(has_following_surface_form=True),
            prior_headword_context=True,
        )

        assert (
            lsj_extractor_module._is_single_dialect_surface_variant(context) is True
        )

    def test_has_dialect_variant_chain(self) -> None:
        context = _make_dialect_decision_context(
            heading=_make_heading_context(
                has_following_dialect_label=True,
                has_following_form=True,
                has_following_surface_form=True,
            )
        )

        assert lsj_extractor_module._has_dialect_variant_chain(context) is True

    def test_has_nominal_morphology_continuation(self) -> None:
        context = _make_dialect_decision_context(
            heading=_make_heading_context(
                has_following_gen_marker=True,
                has_following_itype_marker=True,
            )
        )

        assert (
            lsj_extractor_module._has_nominal_morphology_continuation(context)
            is True
        )

    def test_has_distinct_nominal_surface_variant(self) -> None:
        context = _make_dialect_decision_context(
            heading=_make_heading_context(has_following_gen_marker=True),
            has_distinct_following_surface_form=True,
        )

        assert (
            lsj_extractor_module._has_distinct_nominal_surface_variant(context)
            is True
        )

    def test_qualifies_by_context_with_inherited_variant_chain(self) -> None:
        context = _make_dialect_decision_context()

        assert (
            lsj_extractor_module._qualifies_by_context(
                context,
                has_dialect_variant_chain=True,
                has_distinct_nominal_surface_variant=False,
                is_single_dialect_surface_variant=False,
            )
            is True
        )

    @pytest.mark.parametrize(
        ("heading", "prior_headword_context"),
        [
            (_make_heading_context(has_following_surface_form=True), False),
            (_make_heading_context(has_preceding_form=True), True),
        ],
    )
    def test_qualifies_by_nearby_variant_note(
        self,
        heading: lsj_extractor_module._HeadingContext,
        prior_headword_context: bool,
    ) -> None:
        context = _make_dialect_decision_context(
            heading=heading,
            variant_context=True,
            prior_headword_context=prior_headword_context,
        )

        assert lsj_extractor_module._qualifies_by_nearby_variant_note(context) is True

    @pytest.mark.parametrize(
        ("has_nominal_morphology_continuation", "prior_headword_context"),
        [(True, False), (False, True)],
    )
    def test_qualifies_by_gen_marker(
        self,
        has_nominal_morphology_continuation: bool,
        prior_headword_context: bool,
    ) -> None:
        context = _make_dialect_decision_context(
            heading=_make_heading_context(has_following_gen_marker=True),
            prior_headword_context=prior_headword_context,
        )

        assert (
            lsj_extractor_module._qualifies_by_gen_marker(
                context,
                has_nominal_morphology_continuation=has_nominal_morphology_continuation,
            )
            is True
        )

    def test_is_variant_only_respects_entry_level_constraint(self) -> None:
        context = _make_dialect_decision_context(
            heading=_make_heading_context(has_following_form=True),
            entry_level_constraint=True,
        )

        assert (
            lsj_extractor_module._is_variant_only(
                context,
                is_attic_without_prior=False,
                qualifies_by_context=True,
                qualifies_by_gen_marker=False,
                qualifies_by_nearby_variant_note=False,
            )
            is False
        )


class TestLeadingDialectLabels:
    """Test heading dialect label extraction directly from XML fixtures."""

    def test_attic_primary_heading(self) -> None:
        elem = etree.fromstring(
            '<entryFree id="n25100" key="a)ru/w" type="main">'
            '<orth extent="full" lang="greek">a)ru/w</orth>, '
            '<gramGrp><gram type="dialect">Att.</gram></gramGrp> '
            '<orth extent="full" lang="greek">a)ru/tw</orth>, '
            '<sense id="s1" n="A" level="1"><tr>draw</tr></sense>'
            "</entryFree>"
        )

        assert lsj_extractor_module._leading_dialect_labels(elem) == ["attic"]

    def test_non_attic_primary_heading(self) -> None:
        elem = etree.fromstring(
            '<entryFree id="n25101" key="dw=ron" type="main">'
            '<orth extent="full" lang="greek">dw=ron</orth>, '
            '<gramGrp><gram type="dialect">Dor.</gram></gramGrp> '
            '<gen lang="greek">to/</gen>, '
            '<sense id="s1" n="A" level="1"><tr>gift</tr></sense>'
            "</entryFree>"
        )

        assert lsj_extractor_module._leading_dialect_labels(elem) == ["doric"]

    def test_single_variant_form_is_variant_only(self) -> None:
        elem = etree.fromstring(
            '<entryFree id="n25102" key="a)bohti/" type="main">'
            '<orth extent="full" lang="greek">a)bohti/</orth>, '
            '<gramGrp><gram type="dialect">Dor.</gram></gramGrp> '
            '<orth extent="suff" lang="greek">a)bohq-a_ti/</orth>, '
            '<pos>Adv.</pos> '
            '<sense id="s1" n="A" level="1"><tr>without summons</tr></sense>'
            "</entryFree>"
        )

        assert lsj_extractor_module._leading_dialect_labels(elem) == []

    def test_variant_chain_is_variant_only(self) -> None:
        elem = etree.fromstring(
            '<entryFree id="n25103" key="a)/ella" type="main">'
            '<orth extent="full" lang="greek">a)/ella</orth>, '
            '<gramGrp><gram type="dialect">Ep.</gram></gramGrp> '
            '<orth extent="full" lang="greek">a)e/llh</orth>, '
            '<gramGrp><gram type="dialect">Aeol.</gram></gramGrp> '
            '<orth extent="full" lang="greek">au)/ella</orth>, '
            '<gen lang="greek">h(</gen>, '
            '<sense id="s1" n="A" level="1"><tr>stormy wind</tr></sense>'
            "</entryFree>"
        )

        assert lsj_extractor_module._leading_dialect_labels(elem) == []

    def test_nominal_morphology_continuation_is_variant_only(self) -> None:
        elem = etree.fromstring(
            '<entryFree id="n25104" key="a)na/stasis" type="main">'
            '<orth extent="full" lang="greek">a)na/-sta^sis</orth>, '
            '<itype lang="greek">ews</itype>, '
            '<gramGrp><gram type="dialect">Ion.</gram></gramGrp> '
            '<itype lang="greek">ios</itype>, '
            '<gen lang="greek">h(</gen>, '
            '<sense id="s1" n="A" level="1"><tr>raising up</tr></sense>'
            "</entryFree>"
        )

        assert lsj_extractor_module._leading_dialect_labels(elem) == []

    def test_entry_level_constraint_keeps_dialect_label(self) -> None:
        elem = etree.fromstring(
            '<entryFree id="n25105" key="a)mfagapa/zw" type="main">'
            '<orth extent="full" lang="greek">a)mfagapa/zw</orth>, '
            '<tns>impf.</tns> <foreign lang="greek">a)mfaga/pazon</foreign>, '
            '<gramGrp><gram type="dialect">Ep.</gram></gramGrp> only in '
            '<tns>pres.</tns>, '
            '<sense id="s1" n="A" level="1"><tr>embrace</tr></sense>'
            "</entryFree>"
        )

        assert lsj_extractor_module._leading_dialect_labels(elem) == ["ionic"]

    def test_preserves_label_order_and_ignores_unrecognized_dialects(self) -> None:
        elem = etree.fromstring(
            '<entryFree id="n25106" key="dw=ron" type="main">'
            '<orth extent="full" lang="greek">dw=ron</orth>, '
            '<gramGrp><gram type="dialect">Dor.</gram></gramGrp> '
            '<gramGrp><gram type="dialect">Boeot.</gram></gramGrp> '
            '<gramGrp><gram type="dialect">Ion.</gram></gramGrp> '
            '<sense id="s1" n="A" level="1"><tr>gift</tr></sense>'
            "</entryFree>"
        )

        assert lsj_extractor_module._leading_dialect_labels(elem) == [
            "doric",
            "ionic",
        ]

    @pytest.mark.parametrize(
        ("dialect_label", "tail", "following_markup", "expected"),
        [
            ("Dor.", " mostly ", '<orth lang="greek">dw/rion</orth><pos>Noun</pos>', []),
            ("Ion.", " form of ", '<orth lang="greek">dw/rion</orth>', ["ionic"]),
            ("Att.", " ", '<orth lang="greek">dw/rion</orth>', ["attic"]),
        ],
    )
    def test_table_driven_dialect_variant_classification(
        self,
        dialect_label: str,
        tail: str,
        following_markup: str,
        expected: list[str],
    ) -> None:
        elem = etree.fromstring(
            '<entryFree id="n25107" key="dw=ron" type="main">'
            '<orth extent="full" lang="greek">dw=ron</orth>, '
            f'<gramGrp><gram type="dialect">{dialect_label}</gram></gramGrp>{tail}'
            f"{following_markup}"
            '<sense id="s1" n="A" level="1"><tr>gift</tr></sense>'
            "</entryFree>"
        )

        assert lsj_extractor_module._leading_dialect_labels(elem) == expected


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

    def test_malformed_pos_override_lists_ignore_non_strings_with_warning(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(lsj_extractor_module, "_pos_overrides", None)
        monkeypatch.setattr(lsj_extractor_module, "resolve_repo_data_dir", lambda _name: tmp_path)
        (tmp_path / "pos_overrides.yaml").write_text(
            "common_gender_keys: [a)/nqrwpos, 123]\n"
            "numeral_keys: [de/ka, false]\n",
            encoding="utf-8",
        )
        caplog.set_level("WARNING", logger="phonology.lsj_extractor")

        result = lsj_extractor_module._load_pos_overrides()

        assert result == {
            "common_gender_keys": frozenset({"a)/nqrwpos"}),
            "numeral_keys": frozenset({"de/ka"}),
        }
        assert "Ignoring non-string values in POS overrides list" in caplog.text


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

    def test_skips_non_numeric_id(self) -> None:
        elem = _make_entry_xml(entry_id="not-numeric", orth="lo/gos", gen="o(", tr="word")

        assert extract_entry(elem) is None

    def test_skips_when_transliteration_is_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        elem = _make_entry_xml(entry_id="n101", orth="lo/gos", gen="o(", tr="word")
        monkeypatch.setattr(lsj_extractor_module, "transliterate", lambda _headword: "")

        assert extract_entry(elem) is None

    def test_skips_when_ipa_conversion_returns_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        elem = _make_entry_xml(entry_id="n102", orth="lo/gos", gen="o(", tr="word")
        monkeypatch.setattr(lsj_extractor_module, "to_ipa", lambda *_args, **_kwargs: "")

        assert extract_entry(elem) is None

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

    def test_find_texts_filters_direct_children_by_tag_and_attributes(self) -> None:
        elem = etree.fromstring(
            '<entryFree id="n1">'
            '<orth lang="greek">lo/gos</orth>'
            '<orth lang="latin">logos</orth>'
            '<sense><orth lang="greek">nested</orth></sense>'
            "</entryFree>"
        )

        assert lsj_extractor_module._find_texts(elem, "orth", lang="greek") == ["lo/gos"]

    def test_extract_gloss_falls_back_to_direct_tr_and_truncates_long_text(self) -> None:
        long_gloss = "x" * 250
        elem = etree.fromstring(
            f'<entryFree id="n1" key="lo/gos" type="main"><tr>{long_gloss}</tr></entryFree>'
        )

        gloss = lsj_extractor_module._extract_gloss(elem)

        assert len(gloss) == 200
        assert gloss.endswith("...")

    # -- POS and gender heuristics ---------------------------------------

    def test_pos_rule_explicit_pos_beats_adjective_itype(self) -> None:
        elem = etree.fromstring(
            '<entryFree id="n25200" key="*persiko/s" type="main">'
            '<orth extent="full" lang="greek">*persiko/s</orth>'
            '<itype lang="greek">h/</itype>'
            '<itype lang="greek">o/n</itype>'
            '<pos>Adv.</pos>'
            '<sense id="s1" n="A" level="1"><tr>Persianly</tr></sense>'
            "</entryFree>"
        )

        assert lsj_extractor_module._extract_pos(elem) == "adverb"

    def test_pos_rule_inline_prose_beats_known_numeral_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            lsj_extractor_module,
            "_load_pos_overrides",
            lambda: {
                "common_gender_keys": frozenset(),
                "numeral_keys": frozenset({"de/ka"}),
            },
        )
        elem = etree.fromstring(
            '<entryFree id="n25201" key="de/ka" type="main">'
            '<orth extent="full" lang="greek">de/ka</orth>, Conj., '
            '<sense id="s1" n="A" level="1"><tr>ten</tr></sense>'
            "</entryFree>"
        )

        assert lsj_extractor_module._extract_pos(elem) == "conjunction"

    def test_pos_rule_gender_based_noun_beats_adverb_ending(self) -> None:
        elem = etree.fromstring(
            '<entryFree id="n25202" key="o(/pws" type="main">'
            '<orth extent="full" lang="greek">o(/pws</orth>'
            '<gen lang="greek">o(</gen>'
            '<sense id="s1" n="A" level="1"><tr>way</tr></sense>'
            "</entryFree>"
        )

        assert lsj_extractor_module._extract_pos(elem) == "noun"

    def test_pos_rule_adverb_ending_beats_adjective_itype(self) -> None:
        elem = etree.fromstring(
            '<entryFree id="n25203" key="o(/pws" type="main">'
            '<orth extent="full" lang="greek">o(/pws</orth>'
            '<itype lang="greek">a</itype>'
            '<itype lang="greek">on</itype>'
            '<sense id="s1" n="A" level="1"><tr>how</tr></sense>'
            "</entryFree>"
        )

        assert lsj_extractor_module._extract_pos(elem) == "adverb"

    def test_pos_rule_adjective_itype_beats_verb_indicator(self) -> None:
        elem = etree.fromstring(
            '<entryFree id="n25204" key="a)gapa/w" type="main">'
            '<orth extent="full" lang="greek">a)gapa/w</orth>'
            '<itype lang="greek">a</itype>'
            '<itype lang="greek">on</itype>'
            '<sense id="s1" n="A" level="1"><tr>beloved</tr>'
            '<mood>inf.</mood></sense>'
            "</entryFree>"
        )

        assert lsj_extractor_module._extract_pos(elem) == "adjective"

    def test_pos_rule_verb_indicator_beats_post_gloss_fallback(self) -> None:
        elem = etree.fromstring(
            '<entryFree id="n25205" key="a)ke/w" type="main">'
            '<orth extent="full" lang="greek">a)ke/w</orth>'
            '<sense id="s1" n="A" level="1">'
            '<mood>inf.</mood><tr>to be silent</tr>: Pron. of another form'
            "</sense>"
            "</entryFree>"
        )

        assert lsj_extractor_module._extract_pos(elem) == "verb"

    def test_pos_rule_post_gloss_fallback_beats_final_participle(self) -> None:
        elem = etree.fromstring(
            '<entryFree id="n25206" key="o)/nta" type="main">'
            '<orth extent="full" lang="greek">o)/nta</orth>, '
            '<mood>part.</mood> of <foreign lang="greek">ei)mi/</foreign> '
            '<sense id="s1" n="A" level="1">'
            '<tr>beings</tr>: Pron. of another form'
            "</sense>"
            "</entryFree>"
        )

        assert lsj_extractor_module._extract_pos(elem) == "pronoun"

    def test_pos_rule_undetermined_returns_none(self) -> None:
        elem = etree.fromstring(
            '<entryFree id="n25207" key="a)ran" type="main">'
            '<orth extent="full" lang="greek">a)ran</orth>'
            '<sense id="s1" n="A" level="1"><tr>unclear</tr></sense>'
            "</entryFree>"
        )

        assert lsj_extractor_module._extract_pos(elem) is None

    @pytest.mark.parametrize(
        ("pos_text", "expected"),
        [
            ("Article", "article"),
            ("prep", "preposition"),
            ("Pron", "pronoun"),
            ("unknown", None),
        ],
    )
    def test_explicit_pos_inference_accepts_periodless_labels(
        self,
        pos_text: str,
        expected: str | None,
    ) -> None:
        elem = etree.fromstring(
            f'<entryFree id="n25208" key="o(" type="main"><pos>{pos_text}</pos></entryFree>'
        )

        assert lsj_extractor_module._infer_explicit_pos(elem) == expected

    @pytest.mark.parametrize(
        ("xml", "expected"),
        [
            (
                '<entryFree id="n25209" key="o)/nta" type="main">'
                '<orth extent="full" lang="greek">o)/nta</orth>, '
                '<mood>part.</mood> of <foreign lang="greek">ei)mi/</foreign>'
                "</entryFree>",
                "participle",
            ),
            (
                '<entryFree id="n25210" key="o)/nta" type="main">'
                '<orth extent="full" lang="greek">o)/nta</orth>, '
                '<mood>part.</mood> of <foreign lang="greek">ei)mi/</foreign>'
                '<gen lang="greek">ta/</gen>'
                "</entryFree>",
                "participle",
            ),
        ],
    )
    def test_participle_fallback_inference_paths(self, xml: str, expected: str) -> None:
        elem = etree.fromstring(xml)

        assert lsj_extractor_module._extract_pos(elem) == expected

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

    def test_heading_variant_parenthetical_doric_note_keeps_attic_pronoun(self) -> None:
        """Parenthetical variant notes like ``(so in Dor.)`` must not flip Attic entries."""

        elem = etree.fromstring(
            '<entryFree id="n25030" key="e)gw/" type="main">'
            '<orth extent="full" lang="greek">e)gw/</orth>, <title>I</title>: '
            'Pron. of the first person:—'
            '<gramGrp><gram type="dialect">Ep.</gram></gramGrp> mostly '
            '<orth extent="full" lang="greek">e)gw/n</orth> before vowels (so in '
            '<gramGrp><gram type="dialect">Dor.</gram></gramGrp> before consonants), '
            'rarely in Trag.; '
            '<gramGrp><gram type="dialect">Boeot.</gram></gramGrp> '
            '<orth extent="full" lang="greek">i(w/n</orth>:— strengthd. '
            '<orth extent="full" lang="greek">e)/gwge</orth>, '
            '<sense id="s1" n="A" level="1">'
            '<tr>I at least, for my part</tr> (more freq. in '
            '<gramGrp><gram type="dialect">Att.</gram></gramGrp> than in Hom.)'
            "</sense>"
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["headword"] == "ἐγώ"
        assert result["pos"] == "pronoun"
        assert result["dialect"] == "attic"
        assert result["gender"] == "common"

    def test_heading_variant_dialect_after_title_pos_text_keeps_attic_entry(self) -> None:
        """Title-wrapped heading prose should count as prior context for variant notes."""

        elem = etree.fromstring(
            '<entryFree id="n25045" key="e)gw/" type="main">'
            '<orth extent="full" lang="greek">e)gw/</orth>'
            '<title>Pron. of the first person</title>'
            '<gramGrp><gram type="dialect">Ep.</gram></gramGrp> '
            '<orth extent="full" lang="greek">e)gw/n</orth>, '
            '<sense id="s1" n="A" level="1"><tr>I</tr></sense>'
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["headword"] == "ἐγώ"
        assert result["pos"] == "pronoun"
        assert result["dialect"] == "attic"
        assert result["gender"] == "common"

    def test_heading_plain_gloss_title_does_not_hide_non_attic_entry(self) -> None:
        """A bare English title before a dialect label must not imply variant context."""

        elem = etree.fromstring(
            '<entryFree id="n25056" key="dw=ron" type="main">'
            '<orth extent="full" lang="greek">dw=ron</orth>, '
            '<title>gift</title>, '
            '<gramGrp><gram type="dialect">Dor.</gram></gramGrp> '
            '<gen lang="greek">to/</gen>, '
            '<sense id="s1" n="A" level="1"><tr>gift</tr></sense>'
            "</entryFree>"
        )
        assert extract_entry(elem) is None

    def test_heading_plain_gloss_tail_does_not_hide_non_attic_entry(self) -> None:
        """Bare tail prose before a dialect label must not imply variant context."""

        elem = etree.fromstring(
            '<entryFree id="n25057" key="dw=ron" type="main">'
            '<orth extent="full" lang="greek">dw=ron</orth>, gift, '
            '<gramGrp><gram type="dialect">Dor.</gram></gramGrp> '
            '<gen lang="greek">to/</gen>, '
            '<sense id="s1" n="A" level="1"><tr>gift</tr></sense>'
            "</entryFree>"
        )
        assert extract_entry(elem) is None

    def test_heading_single_doric_variant_form_keeps_attic_adverb(self) -> None:
        """A lone dialect spelling variant should not filter an Attic headword."""

        elem = etree.fromstring(
            '<entryFree id="n25047" key="a)bohti/" type="main">'
            '<orth extent="full" lang="greek">a)bohti/</orth>, '
            '<gramGrp><gram type="dialect">Dor.</gram></gramGrp> '
            '<orth extent="suff" lang="greek">a)bohq-a_ti/</orth>, '
            '<pos>Adv.</pos> (<etym lang="greek">boa/w</etym>) '
            '<sense id="s1" n="A" level="1"><tr>without summons</tr></sense>'
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["headword"] == "ἀβοητί"
        assert result["pos"] == "adverb"
        assert result["dialect"] == "attic"
        assert "gender" not in result

    def test_heading_single_non_attic_surface_form_without_context_skips_entry(self) -> None:
        """A bare dialect label plus one surface form still marks a non-Attic entry."""

        elem = etree.fromstring(
            '<entryFree id="n25054" key="dw=ron" type="main">'
            '<orth extent="full" lang="greek">dw=ron</orth>, '
            '<gramGrp><gram type="dialect">Dor.</gram></gramGrp> '
            '<orth extent="full" lang="greek">dw=ron</orth>, '
            '<gen lang="greek">to/</gen>, '
            '<sense id="s1" n="A" level="1"><tr>gift</tr></sense>'
            "</entryFree>"
        )
        assert extract_entry(elem) is None

    def test_heading_multiple_dialect_variant_forms_keep_attic_noun(self) -> None:
        """Multi-dialect variant chains should not drop Attic headwords."""

        elem = etree.fromstring(
            '<entryFree id="n25029" key="a)/ella" type="main">'
            '<orth extent="full" lang="greek">a)/ella</orth>, '
            '<gramGrp><gram type="dialect">Ep.</gram></gramGrp> '
            '<orth extent="full" lang="greek">a)e/llh</orth>, '
            '<itype lang="greek">hs</itype>, '
            '<gramGrp><gram type="dialect">Aeol.</gram></gramGrp> '
            '<orth extent="full" lang="greek">au)/ella</orth>, '
            '<gen lang="greek">h(</gen>, '
            '<sense id="s1" n="A" level="1"><tr>stormy wind, whirlwind</tr></sense>'
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["headword"] == "ἄελλα"
        assert result["pos"] == "noun"
        assert result["dialect"] == "attic"
        assert result["gender"] == "feminine"

    def test_heading_single_form_of_dialect_label_stays_non_attic(self) -> None:
        """A lone heading dialect label before ``form of`` must still filter the entry."""

        elem = etree.fromstring(
            '<entryFree id="n25034" key="a)gapa/zw" type="main">'
            '<orth extent="full" lang="greek">a)gapa/zw</orth>, '
            '<gramGrp><gram type="dialect">Ep.</gram></gramGrp> form of '
            '<foreign lang="greek">a)gapa/w</foreign>'
            '<sense id="s1" n="A" level="1"><tns>impf.</tns><tr>love</tr></sense>'
            "</entryFree>"
        )
        assert extract_entry(elem) is None

    def test_heading_attic_variant_inside_non_attic_entry_stays_filtered(self) -> None:
        """An Attic variant inside a Doric heading must not flip the entry to Attic."""

        elem = etree.fromstring(
            '<entryFree id="n25039" key="dw=ron" type="main">'
            '<orth extent="full" lang="greek">dw=ron</orth>, '
            '<gramGrp><gram type="dialect">Dor.</gram></gramGrp> '
            '<orth extent="full" lang="greek">dw=ron</orth>, '
            '<gramGrp><gram type="dialect">Att.</gram></gramGrp> '
            '<orth extent="full" lang="greek">dw=ron</orth>, '
            '<gen lang="greek">to/</gen>, '
            '<sense id="s1" n="A" level="1"><tr>gift</tr></sense>'
            "</entryFree>"
        )
        assert extract_entry(elem) is None

    def test_heading_laconian_spelling_after_itypes_keeps_attic_adjective(self) -> None:
        """Non-Attic spellings after iTypes must not drop Attic headwords."""

        elem = etree.fromstring(
            '<entryFree id="n25036" key="a)gaqo/s" type="main">'
            '<orth extent="full" lang="greek">a)ga^qo/s</orth>'
            '<pron extent="full" lang="greek">[a^g]</pron>, '
            '<itype lang="greek">h/</itype>, '
            '<itype lang="greek">o/n</itype>, '
            '<gramGrp><gram type="dialect">Lacon.</gram></gramGrp> '
            '<orth extent="full" lang="greek">a)gaso/s</orth> '
            '<bibl><author>Ar.</author><title>Lys.</title><biblScope>1301</biblScope></bibl>, '
            '<orth extent="full" lang="greek">a)zaqo/s</orth>:—'
            '<sense id="s1" n="A" level="1"><tr>good</tr></sense>'
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["headword"] == "ἀγαθός"
        assert result["pos"] == "adjective"
        assert result["dialect"] == "attic"
        assert result["gender"] == "common"

    def test_heading_dialect_with_mostly_but_no_alt_form_stays_non_attic(self) -> None:
        """Variant cue words alone must not discard an entry-level dialect label."""

        elem = etree.fromstring(
            '<entryFree id="n25041" key="dw=ron" type="main">'
            '<orth extent="full" lang="greek">dw=ron</orth>, '
            '<gramGrp><gram type="dialect">Dor.</gram></gramGrp> mostly in lyric poetry, '
            '<gen lang="greek">to/</gen>, '
            '<sense id="s1" n="A" level="1"><tr>gift</tr></sense>'
            "</entryFree>"
        )
        assert extract_entry(elem) is None

    def test_heading_ionic_variant_chain_keeps_attic_noun(self) -> None:
        """Variant chains with multiple Greek forms must not flip Attic nouns."""

        elem = etree.fromstring(
            '<entryFree id="n25037" key="a)gaqoergi/a" type="main">'
            '<orth extent="suff" lang="greek">a)gaqo-ergi/a</orth>, '
            '<gramGrp><gram type="dialect">Ion.</gram></gramGrp> '
            '<foreign lang="greek">-ih,</foreign> '
            '<gramGrp><gram type="var">contr.</gram></gramGrp> '
            '<orth extent="suff" lang="greek">a)gaqo-ourgi/a</orth>, '
            '<gen lang="greek">h(</gen>, '
            '<sense id="s1" n="A" level="1"><tr>good deed, service</tr></sense>'
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["headword"] == "ἀγαθοεργία"
        assert result["pos"] == "noun"
        assert result["dialect"] == "attic"
        assert result["gender"] == "feminine"

    def test_heading_attic_label_with_following_form_still_keeps_entry(self) -> None:
        """Attic labels should not be discarded as mere variant notes."""

        elem = etree.fromstring(
            '<entryFree id="n25038" key="a)ru/w1" type="main">'
            '<orth extent="full" lang="greek">a)ru/w</orth> '
            '<pron extent="full" lang="greek">[a^]</pron>, '
            '<gramGrp><gram type="dialect">Att.</gram></gramGrp> '
            '<orth extent="full" lang="greek">a)ru/tw</orth> '
            '<pron extent="full" lang="greek">[u^]</pron>; '
            '<gramGrp><gram type="dialect">Aeol.</gram></gramGrp> '
            '<mood>part.</mood> '
            '<sense id="s1" n="A" level="1"><tr>draw</tr></sense>'
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["headword"] == "ἀρύω"
        assert result["pos"] == "verb"
        assert result["dialect"] == "attic"

    def test_heading_variant_dialect_after_explicit_pos_tag_keeps_attic_entry(self) -> None:
        """Tagged heading POS labels should count as prior prose for variant dialect notes."""

        elem = etree.fromstring(
            '<entryFree id="n25032" key="e)gw/" type="main">'
            '<orth extent="full" lang="greek">e)gw/</orth>, '
            '<pos>Pron.</pos>'
            '<gramGrp><gram type="dialect">Ep.</gram></gramGrp> '
            '<orth extent="full" lang="greek">e)gw/n</orth>, '
            '<sense id="s1" n="A" level="1"><tr>I</tr></sense>'
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["headword"] == "ἐγώ"
        assert result["pos"] == "pronoun"
        assert result["dialect"] == "attic"
        assert result["gender"] == "common"

    def test_strengthd_variant_context_with_non_dialect_gramgrp_keeps_attic_entry(self) -> None:
        """A standalone ``strengthd.`` cue should keep variant dialect labels from flipping the entry."""

        elem = etree.fromstring(
            '<entryFree id="n25040" key="fo/os" type="main">'
            '<orth extent="full" lang="greek">fo/os</orth>, <title>light</title>: '
            '<gramGrp><gram type="dialect">Dor.</gram></gramGrp> strengthd. '
            '<gramGrp><gram type="var">sync.</gram></gramGrp> '
            '<orth extent="full" lang="greek">fo/oss</orth>, '
            '<gen lang="greek">to/</gen>, '
            '<sense id="s1" n="A" level="1"><tr>light</tr></sense>'
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["headword"] == "φόος"
        assert result["pos"] == "noun"
        assert result["dialect"] == "attic"
        assert result["gender"] == "neuter"

    def test_heading_dialect_followed_only_by_gen_still_counts_as_variant(self) -> None:
        """A heading `gen` marker can be the only signal for a dialectal variant form."""

        elem = etree.fromstring(
            '<entryFree id="n25044" key="a)lw/phc" type="main">'
            '<orth extent="full" lang="greek">a)lw/phc</orth>, older form, '
            '<gramGrp><gram type="dialect">Dor.</gram></gramGrp> '
            '<gen lang="greek">h(</gen>, '
            '<sense id="s1" n="A" level="1"><tr>fox</tr></sense>'
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["headword"] == "ἀλώπηξ"
        assert result["pos"] == "noun"
        assert result["dialect"] == "attic"
        assert result["gender"] == "feminine"

    def test_heading_var_label_before_gen_keeps_non_attic_dialect(self) -> None:
        """A non-dialect `gramGrp` before `gen` must not erase an entry-level dialect."""

        elem = etree.fromstring(
            '<entryFree id="n25052" key="dw=ron" type="main">'
            '<orth extent="full" lang="greek">dw=ron</orth>, <title>gift</title>, '
            '<gramGrp><gram type="dialect">Dor.</gram></gramGrp> '
            '<gramGrp><gram type="var">contr.</gram></gramGrp> '
            '<gen lang="greek">to/</gen>, '
            '<sense id="s1" n="A" level="1"><tr>gift</tr></sense>'
            "</entryFree>"
        )
        assert extract_entry(elem) is None

    def test_heading_pos_then_var_label_before_gen_keeps_non_attic_dialect(self) -> None:
        """Tagged POS before a dialect note must not reclassify var-plus-gen headings as Attic."""

        elem = etree.fromstring(
            '<entryFree id="n25053" key="dw=ron" type="main">'
            '<orth extent="full" lang="greek">dw=ron</orth>, <pos>Subst.</pos> '
            '<gramGrp><gram type="dialect">Dor.</gram></gramGrp> '
            '<gramGrp><gram type="var">contr.</gram></gramGrp> '
            '<gen lang="greek">to/</gen>, '
            '<sense id="s1" n="A" level="1"><tr>gift</tr></sense>'
            "</entryFree>"
        )
        assert extract_entry(elem) is None

    def test_heading_surface_form_plus_gen_without_prose_keeps_attic_noun(self) -> None:
        """A lone dialect form plus `gen` should still keep the Attic headword."""

        elem = etree.fromstring(
            '<entryFree id="n25046" key="a)ggei=on" type="main">'
            '<orth extent="full" lang="greek">a)ggei=on</orth>, '
            '<gramGrp><gram type="dialect">Ion.</gram></gramGrp> '
            '<orth extent="suff" lang="greek">a)ggeio-h/ion</orth>, '
            '<gen lang="greek">to/</gen>, '
            '<sense id="s1" n="A" level="1"><tr>vessel</tr></sense>'
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["headword"] == "ἀγγεῖον"
        assert result["pos"] == "noun"
        assert result["dialect"] == "attic"
        assert result["gender"] == "neuter"

    def test_heading_multiple_dialect_cited_forms_keep_attic_verb(self) -> None:
        """Cited dialect form chains should still preserve the Attic headword."""

        elem = etree.fromstring(
            '<entryFree id="n25048" key="a)dike/w" type="main">'
            '<orth extent="full" lang="greek">a)di^ke/w</orth>, '
            '<gramGrp><gram type="dialect">Aeol.</gram></gramGrp> '
            '<orth extent="suff" lang="greek">a)di-h/w</orth> '
            '<bibl><author>Sapph.</author><biblScope>1.20</biblScope></bibl>, '
            '<gramGrp><gram type="dialect">Dor.</gram></gramGrp> '
            '<orth extent="suff" lang="greek">a)di-i/w</orth> '
            '<bibl><title>Tab.Heracl.</title><biblScope>1.138</biblScope></bibl>, '
            '<sense id="s1" n="A" level="1"><tr>do wrong</tr><tns>impf.</tns></sense>'
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["headword"] == "ἀδικέω"
        assert result["pos"] == "verb"
        assert result["dialect"] == "attic"

    def test_heading_nominal_morphology_continuation_keeps_attic_noun(self) -> None:
        """Dialect-specific iType and gender continuations should stay variant-only."""

        elem = etree.fromstring(
            '<entryFree id="n25049" key="a)na/stasis" type="main">'
            '<orth extent="full" lang="greek">a)na/-sta^sis</orth>, '
            '<itype lang="greek">ews</itype>, '
            '<gramGrp><gram type="dialect">Ion.</gram></gramGrp> '
            '<itype lang="greek">ios</itype>, '
            '<gen lang="greek">h(</gen>, '
            '<sense id="s1" n="A" level="1"><tr>raising up</tr></sense>'
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["headword"] == "ἀνάστασις"
        assert result["pos"] == "noun"
        assert result["dialect"] == "attic"
        assert result["gender"] == "feminine"

    def test_heading_ionic_morphology_continuation_keeps_attic_goatherd(self) -> None:
        """A dialect label bracketed by iTypes and gen should not filter the entry."""

        elem = etree.fromstring(
            '<entryFree id="n25050" key="ai)gonomeu/s" type="main">'
            '<orth extent="suff" lang="greek">ai)go-nomeu/s</orth>, '
            '<itype lang="greek">e/ws</itype>, '
            '<gramGrp><gram type="dialect">Ion.</gram></gramGrp> '
            '<itype lang="greek">h=os</itype>, '
            '<gen lang="greek">o(</gen>, '
            '<sense id="s1" n="A" level="1"><tr>goat-herd</tr></sense>'
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["headword"] == "αἰγονομεύς"
        assert result["pos"] == "noun"
        assert result["dialect"] == "attic"
        assert result["gender"] == "masculine"

    def test_heading_multiple_dialect_spelling_variants_keep_attic_noun(self) -> None:
        """Sequential dialect spellings should remain heading variants, not entry labels."""

        elem = etree.fromstring(
            '<entryFree id="n25051" key="a)ni/a" type="main">'
            '<orth extent="full" lang="greek">a)ni/a</orth>, '
            '<gramGrp><gram type="dialect">Ion.</gram></gramGrp> '
            '<orth extent="full" lang="greek">a)ni/h</orth>, '
            '<gramGrp><gram type="dialect">Aeol.</gram></gramGrp> '
            '<orth extent="full" lang="greek">o)ni/a</orth>, '
            '<gen lang="greek">h(</gen>, '
            '<sense id="s1" n="A" level="1"><tr>grief, sorrow</tr></sense>'
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["headword"] == "ἀνία"
        assert result["pos"] == "noun"
        assert result["dialect"] == "attic"
        assert result["gender"] == "feminine"

    def test_pos_from_trailing_sense_prose_handles_second_person_plural_pronoun(self) -> None:
        """POS notes after a gloss inside a sense should still classify pronouns."""

        elem = etree.fromstring(
            '<entryFree id="n25031" key="u(mei=s" type="main">'
            '<orth extent="full" lang="greek">u(mei=s</orth>, '
            '<sense id="s1" n="A" level="1">'
            '<tr>ye</tr>: Pron. of the second pers. pl.:—'
            '<gramGrp><gram type="dialect">Ep.</gram></gramGrp> nom. '
            '<orth extent="full" lang="greek">u)/mmes</orth>;'
            "</sense>"
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["headword"] == "ὑμεῖς"
        assert result["pos"] == "pronoun"
        assert result["dialect"] == "attic"
        assert result["gender"] == "common"

    def test_tagged_pos_after_gloss_keeps_second_person_plural_pronoun(self) -> None:
        """Tagged POS labels after a gloss should still classify the entry."""

        elem = etree.fromstring(
            '<entryFree id="n25033" key="u(mei=s" type="main">'
            '<orth extent="full" lang="greek">u(mei=s</orth>, '
            '<sense id="s1" n="A" level="1">'
            '<tr>ye</tr>: <pos>Pron.</pos> of the second pers. pl.:—'
            '<gramGrp><gram type="dialect">Ep.</gram></gramGrp> nom. '
            '<orth extent="full" lang="greek">u)/mmes</orth>;'
            "</sense>"
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["headword"] == "ὑμεῖς"
        assert result["pos"] == "pronoun"
        assert result["dialect"] == "attic"
        assert result["gender"] == "common"

    def test_secondary_post_gloss_pos_note_does_not_override_adjective(self) -> None:
        """Secondary post-gloss POS notes must not replace the main headword POS."""

        elem = etree.fromstring(
            '<entryFree id="n25035" key="toi=os" type="main">'
            '<orth extent="full" lang="greek">toi=os</orth>'
            '<itype lang="greek">a</itype>'
            '<itype lang="greek">on</itype>'
            '<sense id="s1" n="A" level="1">'
            '<tr>such</tr>; also <pos>Pron.</pos>, <tr>such a one</tr>'
            "</sense>"
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["headword"] == "τοῖος"
        assert result["pos"] == "adjective"
        assert result["gender"] == "common"

    def test_post_gloss_pronoun_note_does_not_override_itype_adjective(self) -> None:
        """Fallback post-gloss POS notes must not beat stronger adjective signals."""

        elem = etree.fromstring(
            '<entryFree id="n25042" key="toi=os" type="main">'
            '<orth extent="full" lang="greek">toi=os</orth>'
            '<itype lang="greek">a</itype>'
            '<itype lang="greek">on</itype>'
            '<sense id="s1" n="A" level="1">'
            '<tr>such</tr>: Pron. of the correlative form in late usage'
            "</sense>"
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["headword"] == "τοῖος"
        assert result["pos"] == "adjective"
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

    def test_participle_spelled_out_in_mood_intro_classifies_participle(self) -> None:
        """Spelled-out ``Participle of`` mood notes should still classify participles."""

        elem = etree.fromstring(
            '<entryFree id="n25055" key="o)/nta2" type="main">'
            '<orth extent="full" lang="greek">o)/nta</orth>, '
            '<gen lang="greek">ta/</gen>, neut. pl. '
            '<mood>Participle</mood> of <foreign lang="greek">ei)mi/</foreign> '
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

    def test_mood_participle_without_of_classifies_participle(self) -> None:
        """<mood>Participle</mood> without 'of' should still classify as participle."""

        elem = etree.fromstring(
            '<entryFree id="n25060" key="pa/qwn" type="main">'
            '<orth extent="full" lang="greek">pa/qwn</orth>, '
            '<gen lang="greek">o(</gen>, '
            '<mood>Participle</mood> pres. act. '
            '<sense id="s1" n="A" level="1"><tr>suffering</tr></sense>'
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["pos"] == "participle"
        assert result["gender"] == "masculine"

    def test_mood_partic_abbrev_without_of_classifies_participle(self) -> None:
        """<mood>Partic.</mood> without 'of' should still classify as participle."""

        elem = etree.fromstring(
            '<entryFree id="n25061" key="pa/qwn2" type="main">'
            '<orth extent="full" lang="greek">pa/qwn</orth>, '
            '<gen lang="greek">o(</gen>, '
            '<mood>Partic.</mood> pres. act. '
            '<sense id="s1" n="A" level="1"><tr>suffering</tr></sense>'
            "</entryFree>"
        )
        result = extract_entry(elem)
        assert result is not None
        assert result["pos"] == "participle"
        assert result["gender"] == "masculine"

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


def _sample_document_entries() -> list[dict[str, Any]]:
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


class TestBuildDocument:
    """Test document assembly and validation."""

    def _sample_entries(self) -> list[dict[str, Any]]:
        return _sample_document_entries()

    def test_build_document_structure(self) -> None:
        entries = self._sample_entries()
        doc = build_lexicon_document(entries)
        assert doc["schema_version"] == "2.0.0"
        assert "_meta" in doc
        assert doc["_meta"]["source"].startswith("LSJ")
        assert doc["_meta"]["dialect"] == "attic"
        assert "filtered to Attic entries" in doc["_meta"]["description"]
        assert doc["_meta"]["note"].endswith("output dialect is attic")
        assert doc["_meta"]["license"] == "CC-BY-SA-4.0"
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


class TestXmlIterationAndCli:
    """Test XML file discovery, extraction orchestration, and CLI behavior."""

    def _write_xml_file(self, xml_dir: Path, name: str, entries: list[str]) -> Path:
        xml_dir.mkdir(parents=True, exist_ok=True)
        xml_path = xml_dir / name
        xml_path.write_text("<root>" + "".join(entries) + "</root>", encoding="utf-8")
        return xml_path

    def _entry_xml(self, entry_id: str, key: str = "lo/gos") -> str:
        return etree.tostring(
            _make_entry_xml(entry_id=entry_id, key=key, orth=key),
            encoding="unicode",
        )

    def test_find_xml_files_sorts_by_eng_suffix_number(self, tmp_path: Path) -> None:
        xml_dir = tmp_path / "xml"
        xml_dir.mkdir()
        for name in [
            "grc.lsj.perseus-eng10.xml",
            "grc.lsj.perseus-eng.xml",
            "grc.lsj.perseus-eng2.xml",
            "ignored.xml",
        ]:
            (xml_dir / name).write_text("<root/>", encoding="utf-8")

        files = lsj_extractor_module.find_xml_files(xml_dir)

        assert [path.name for path in files] == [
            "grc.lsj.perseus-eng.xml",
            "grc.lsj.perseus-eng2.xml",
            "grc.lsj.perseus-eng10.xml",
        ]

    def test_find_xml_files_reports_missing_inputs(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No LSJ XML files found"):
            lsj_extractor_module.find_xml_files(tmp_path)

    def test_iter_xml_entries_yields_valid_entries(self, tmp_path: Path) -> None:
        xml_path = self._write_xml_file(
            tmp_path,
            "grc.lsj.perseus-eng.xml",
            [self._entry_xml("n1")],
        )

        entries = list(lsj_extractor_module.iter_xml_entries(xml_path))

        assert [entry["id"] for entry in entries] == ["LSJ-000001"]

    def test_extract_all_deduplicates_ids_and_honors_limit(self, tmp_path: Path) -> None:
        xml_dir = tmp_path / "xml"
        self._write_xml_file(
            xml_dir,
            "grc.lsj.perseus-eng.xml",
            [self._entry_xml("n1"), self._entry_xml("n1"), self._entry_xml("n2", "a)/nqrwpos")],
        )

        entries = list(lsj_extractor_module.extract_all(xml_dir, limit=2))

        assert [entry["id"] for entry in entries] == ["LSJ-000001", "LSJ-000002"]

    def test_main_returns_one_when_no_entries_are_extracted(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(lsj_extractor_module, "extract_all", lambda *_args, **_kwargs: iter(()))
        caplog.set_level("ERROR", logger="phonology.lsj_extractor")

        assert lsj_extractor_module.main(xml_dir=tmp_path, output_path=tmp_path / "out.json") == 1
        assert "No entries extracted" in caplog.text

    def test_main_returns_one_when_validation_fails(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(
            lsj_extractor_module,
            "extract_all",
            lambda *_args, **_kwargs: iter([_sample_document_entries()[0]]),
        )
        monkeypatch.setattr(
            lsj_extractor_module,
            "validate_document",
            Mock(side_effect=ValueError("bad schema")),
        )
        caplog.set_level("ERROR", logger="phonology.lsj_extractor")

        assert lsj_extractor_module.main(xml_dir=tmp_path, output_path=tmp_path / "out.json") == 1
        assert "Validation failed: bad schema" in caplog.text

    def test_main_dry_run_prints_counts_without_writing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        output_path = tmp_path / "out.json"
        monkeypatch.setattr(
            lsj_extractor_module,
            "extract_all",
            lambda *_args, **_kwargs: iter(_sample_document_entries()),
        )
        monkeypatch.setattr(lsj_extractor_module, "validate_document", lambda *_args, **_kwargs: None)

        assert (
            lsj_extractor_module.main(
                xml_dir=tmp_path,
                output_path=output_path,
                dry_run=True,
            )
            == 0
        )

        captured = capsys.readouterr()
        assert "Dry run: 2 entries would be written" in captured.out
        assert "  adverb: 1" in captured.out
        assert "  noun: 1" in captured.out
        assert not output_path.exists()

    def test_main_writes_valid_document(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        output_path = tmp_path / "out.json"
        monkeypatch.setattr(
            lsj_extractor_module,
            "extract_all",
            lambda *_args, **_kwargs: iter([_sample_document_entries()[0]]),
        )
        monkeypatch.setattr(lsj_extractor_module, "validate_document", lambda *_args, **_kwargs: None)

        assert lsj_extractor_module.main(xml_dir=tmp_path, output_path=output_path) == 0

        assert json.loads(output_path.read_text(encoding="utf-8"))["lemmas"][0]["id"] == "LSJ-000001"
        assert "Lexicon written: 1 entries" in capsys.readouterr().out

    def test_main_requires_xml_dir(self) -> None:
        with pytest.raises(ValueError, match="xml_dir is required"):
            lsj_extractor_module.main(xml_dir=None, output_path=Path("out.json"))

    def test_main_propagates_output_write_errors(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            lsj_extractor_module,
            "extract_all",
            lambda *_args, **_kwargs: iter([_sample_document_entries()[0]]),
        )
        monkeypatch.setattr(lsj_extractor_module, "validate_document", lambda *_args, **_kwargs: None)

        with pytest.raises(FileNotFoundError):
            lsj_extractor_module.main(
                xml_dir=tmp_path,
                output_path=tmp_path / "missing" / "out.json",
            )

    def test_run_cli_forwards_arguments_and_preloads_overrides(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, object] = {}
        xml_dir = tmp_path / "xml"
        output_path = tmp_path / "out.json"

        def fake_load_pos_overrides(*, cli_mode: bool = False) -> dict[str, frozenset[str]]:
            captured["cli_mode"] = cli_mode
            return lsj_extractor_module._empty_pos_overrides()

        def fake_main(**kwargs: object) -> int:
            captured.update(kwargs)
            return 0

        monkeypatch.setattr(lsj_extractor_module, "_load_pos_overrides", fake_load_pos_overrides)
        monkeypatch.setattr(lsj_extractor_module, "main", fake_main)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "lsj_extractor.py",
                "--xml-dir",
                str(xml_dir),
                "--output",
                str(output_path),
                "--limit",
                "3",
                "--dry-run",
                "--verbose",
            ],
        )

        assert lsj_extractor_module.run_cli() == 0
        assert captured == {
            "cli_mode": True,
            "xml_dir": xml_dir,
            "output_path": output_path,
            "limit": 3,
            "dry_run": True,
        }

    def test_run_cli_returns_one_for_preload_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(
            lsj_extractor_module,
            "_load_pos_overrides",
            Mock(side_effect=RuntimeError("bad overrides")),
        )
        monkeypatch.setattr(
            lsj_extractor_module,
            "main",
            Mock(side_effect=AssertionError("main should not run")),
        )
        monkeypatch.setattr(sys, "argv", ["lsj_extractor.py", "--xml-dir", "xml"])
        caplog.set_level("ERROR", logger="phonology.lsj_extractor")

        assert lsj_extractor_module.run_cli() == 1
        assert "Extraction failed: bad overrides" in caplog.text


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


class TestNestedGenElement:
    """Verify that <gen> nested inside <foreign> is found by the deep fallback."""

    def test_nested_gen_inside_foreign_extracts_as_noun(self) -> None:
        """βίος-like entry: <gen> is inside <foreign>, not a direct child."""
        xml = etree.fromstring(
            '<entryFree id="n19972" key="bi/os1" type="main">'
            '<orth extent="full" lang="greek">bi/os</orth>'
            '<foreign lang="greek">i^], <gen lang="greek">o(</gen></foreign>'
            '<sense id="s1" n="A" level="1"><tr>life</tr></sense>'
            "</entryFree>"
        )
        result = extract_entry(xml)
        assert result is not None
        assert result["pos"] == "noun"
        assert result["gender"] == "masculine"

    def test_direct_child_gen_still_preferred(self) -> None:
        """Normal entry: direct-child <gen> is found without needing deep search."""
        xml = _make_entry_xml(gen="o(", tr="word")
        result = extract_entry(xml)
        assert result is not None
        assert result["gender"] == "masculine"

    def test_find_gen_text_deep_returns_empty_when_no_gen(self) -> None:
        """Deep search returns empty string when no <gen> exists anywhere."""
        from phonology.lsj_extractor import _find_gen_text

        xml = etree.fromstring(
            '<entryFree id="n100" key="test" type="main">'
            '<orth extent="full" lang="greek">test</orth>'
            "</entryFree>"
        )
        assert _find_gen_text(xml) == ""

    def test_deep_search_does_not_cross_sense_boundary(self) -> None:
        """<gen> inside <sense> should not be found by the deep fallback."""
        from phonology.lsj_extractor import _find_text_deep

        xml = etree.fromstring(
            '<entryFree id="n100" key="test" type="main">'
            '<orth extent="full" lang="greek">test</orth>'
            '<sense id="s1" n="A" level="1">'
            '<gen lang="greek">h(</gen>'
            "<tr>some word</tr>"
            "</sense>"
            "</entryFree>"
        )
        # The <gen> is inside <sense>, so deep search should NOT find it
        assert _find_text_deep(xml, "gen", lang="greek") == ""

    def test_deep_search_ignores_citation_subtrees(self) -> None:
        """<gen> inside heading citations should not be treated as headword gender."""
        from phonology.lsj_extractor import _find_gen_text

        xml = etree.fromstring(
            '<entryFree id="n101" key="a)gaqo/s" type="main">'
            '<orth extent="full" lang="greek">a)gaqo/s</orth>'
            '<cit><quote><foreign lang="greek">ti <gen lang="greek">h(</gen></foreign></quote></cit>'
            '<sense id="s1" n="A" level="1"><tr>good</tr></sense>'
            "</entryFree>"
        )
        assert _find_gen_text(xml) == ""
        assert extract_entry(xml) is None
