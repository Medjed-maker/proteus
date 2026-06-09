"""Tests for the Buck reference service."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from types import MappingProxyType
import unicodedata

import pytest

from phonology._trusted_paths import TRUSTED_DIR_OVERRIDES_OPT_IN_ENV_VAR
from phonology.languages.ancient_greek import buck as buck_module
from phonology.languages.ancient_greek import buck_service as buck_service_module
from phonology.languages.ancient_greek.buck import load_buck_data
from phonology.languages.ancient_greek.buck_service import (
    BuckReferenceIndex,
    build_buck_reference_index,
    canonicalize_buck_section,
)


def _write_buck_fixture(
    buck_dir: Path,
    *,
    grammar_rules: str,
    dialects: str,
    glossary: str,
) -> None:
    buck_dir.mkdir(parents=True)
    (buck_dir / "grammar_rules.yaml").write_text(grammar_rules, encoding="utf-8")
    (buck_dir / "dialects.yaml").write_text(dialects, encoding="utf-8")
    (buck_dir / "glossary.yaml").write_text(glossary, encoding="utf-8")


@pytest.fixture(autouse=True)
def reset_buck_loader_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TRUSTED_DIR_OVERRIDES_OPT_IN_ENV_VAR, "1")
    monkeypatch.delenv("PROTEUS_TRUSTED_BUCK_DIR", raising=False)
    buck_module.clear_buck_data_cache()


@pytest.fixture
def fixture_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> BuckReferenceIndex:
    buck_dir = tmp_path / "buck"
    _write_buck_fixture(
        buck_dir,
        grammar_rules=(
            "meta:\n"
            "  status: provisional\n"
            "  review_status: not_expert_reviewed\n"
            "  citation_ready: false\n"
            "rules:\n"
            "  - id: R1\n"
            "    buck_section: 41.4\n"
            "    category: vowels\n"
            "    description: Attic vowel change.\n"
            "    transformation:\n"
            "      from: a\n"
            "      to: e\n"
            "    affected_dialects: [attic]\n"
            "    variants:\n"
            "      - form: leos\n"
            "        dialects: [attic]\n"
            "    notes: First note.\n"
            "  - id: R2\n"
            "    buck_section: '10'\n"
            "    category: consonants\n"
            "    description: Doric consonant rule.\n"
            "    affected_dialects: [doric]\n"
            "  - id: R3\n"
            "    buck_section: '20.1'\n"
            "    category: vowels\n"
            "    description: Inherited group rule.\n"
            "  - id: R4\n"
            "    buck_section: '20.2'\n"
            "    description: Parent-only inherited rule.\n"
        ),
        dialects=(
            "meta:\n"
            "  status: provisional\n"
            "  review_status: not_expert_reviewed\n"
            "  citation_ready: false\n"
            "dialects:\n"
            "  - id: parent\n"
            "    name: Parent\n"
            "    kind: group\n"
            "    group: Test\n"
            "    rules: [R3, R4]\n"
            "  - id: child\n"
            "    name: Child\n"
            "    kind: dialect\n"
            "    group: Test\n"
            "    parent: parent\n"
            "    rules: [R1, R3]\n"
            "  - id: attic\n"
            "    name: Attic\n"
            "    kind: dialect\n"
            "    group: Test\n"
            "    rules: [R1]\n"
            "  - id: doric\n"
            "    name: Doric\n"
            "    kind: dialect\n"
            "    group: Test\n"
            "    rules: [R2]\n"
        ),
        glossary=(
            "meta:\n"
            "  status: provisional\n"
            "  review_status: not_expert_reviewed\n"
            "  citation_ready: false\n"
            "words:\n"
            "  - word: λεώς\n"
            "    standard_form: λαός\n"
            "    dialect: attic\n"
            "    rule_id: R1\n"
            "    definition: people\n"
            "    inscription_no: IG I 1\n"
            "    buck_ref:\n"
            "      section: 41.4\n"
            "      page: 130\n"
            "    notes: Example note.\n"
            "  - word: Δωρικός\n"
            "    standard_form: Δωρικός\n"
            "    dialect: doric\n"
            "    rule_id: R2\n"
            "    definition: Doric\n"
        ),
    )
    monkeypatch.setenv("PROTEUS_TRUSTED_BUCK_DIR", str(buck_dir))
    return build_buck_reference_index()


def test_canonicalize_buck_section_accepts_numeric_and_string_values() -> None:
    assert canonicalize_buck_section("41.4") == "41.4"
    assert canonicalize_buck_section(41.4) == "41.4"
    assert canonicalize_buck_section(10) == "10"
    assert canonicalize_buck_section(" 10 ") == "10"


def test_rule_lookup_section_lookup_and_category_filter(
    fixture_index: BuckReferenceIndex,
) -> None:
    rule = fixture_index.get_rule("R1")

    assert rule is not None
    assert rule.id == "R1"
    assert rule.buck_section == "41.4"
    assert rule.category == "vowels"
    assert rule.transformation == MappingProxyType({"from": "a", "to": "e"})
    assert rule.affected_dialects == ("attic",)
    assert rule.review_status == "not_expert_reviewed"
    assert rule.citation_ready is False
    assert [item.id for item in fixture_index.get_rules_by_section("41.4")] == ["R1"]
    assert [item.id for item in fixture_index.get_rules_by_section(41.4)] == ["R1"]
    assert [item.id for item in fixture_index.list_rules(category="vowels")] == [
        "R1",
        "R3",
    ]


def test_rule_filter_by_dialect_uses_rule_affected_dialects(
    fixture_index: BuckReferenceIndex,
) -> None:
    assert [rule.id for rule in fixture_index.list_rules(dialect="attic")] == ["R1"]
    assert [rule.id for rule in fixture_index.list_rules(dialect="doric")] == ["R2"]


def test_dialect_lookup_and_inherited_rules_dedupe_first_seen(
    fixture_index: BuckReferenceIndex,
) -> None:
    dialect = fixture_index.get_dialect("child")

    assert dialect is not None
    assert dialect.id == "child"
    assert dialect.parent == "parent"
    assert dialect.rules == ("R1", "R3")
    assert dialect.review_status == "not_expert_reviewed"
    assert [item.id for item in fixture_index.list_dialects(kind="dialect")] == [
        "child",
        "attic",
        "doric",
    ]
    assert [
        rule.id for rule in fixture_index.get_dialect_rules("child")
    ] == ["R1", "R3", "R4"]
    assert [
        rule.id
        for rule in fixture_index.get_dialect_rules("child", include_inherited=False)
    ] == ["R1", "R3"]


def test_glossary_lookup_filters_and_uses_nfc_strict_matching(
    fixture_index: BuckReferenceIndex,
) -> None:
    decomposed_standard_form = unicodedata.normalize("NFD", "λαός")

    by_word = fixture_index.find_glossary_by_word("λεώς")
    by_standard_form = fixture_index.find_glossary_by_standard_form(
        decomposed_standard_form
    )

    assert [entry.word for entry in by_word] == ["λεώς"]
    assert [entry.word for entry in by_standard_form] == ["λεώς"]
    assert fixture_index.find_glossary_by_standard_form("λαος") == ()
    assert [entry.word for entry in fixture_index.list_glossary_entries(dialect="attic")] == [
        "λεώς",
    ]
    assert [entry.word for entry in fixture_index.list_glossary_entries(rule_id="R2")] == [
        "Δωρικός",
    ]
    assert by_word[0].buck_ref is not None
    assert by_word[0].buck_ref.section == "41.4"
    assert by_word[0].buck_ref.page == 130
    assert by_word[0].review_status == "not_expert_reviewed"
    assert by_word[0].citation_ready is False


def test_packaged_buck_data_counts_and_metadata_are_exposed() -> None:
    index = build_buck_reference_index()

    assert len(index.rules) >= 99
    assert len(index.dialects) >= 22
    assert len(index.glossary_entries) >= 15
    assert index.metadata.status == "provisional"
    assert index.metadata.review_status == "not_expert_reviewed"
    assert index.metadata.citation_ready is False
    assert index.get_rule("grc_phon_41_4") is not None


def test_service_treats_non_boolean_citation_ready_as_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_load_buck_data() -> dict[str, object]:
        return {
            "grammar_rules": {
                "meta": {
                    "status": "provisional",
                    "review_status": "not_expert_reviewed",
                    "citation_ready": "false",
                },
                "rules": [{"id": "R1"}],
            },
            "dialects": {"dialects": []},
            "glossary": {"words": []},
        }

    monkeypatch.setattr(buck_service_module, "load_buck_data", fake_load_buck_data)

    index = build_buck_reference_index()

    assert index.metadata.citation_ready is False
    assert index.get_rule("R1").citation_ready is False


def test_index_is_not_affected_by_mutating_loaded_source_copy(
    fixture_index: BuckReferenceIndex,
) -> None:
    source_copy = load_buck_data()
    source_copy["grammar_rules"]["rules"][0]["id"] = "MUTATED"
    source_copy["dialects"]["dialects"][0]["rules"].append("R1")
    source_copy["glossary"]["words"][0]["word"] = "mutated"

    assert fixture_index.get_rule("R1") is not None
    assert fixture_index.get_rule("MUTATED") is None
    assert [entry.word for entry in fixture_index.find_glossary_by_word("λεώς")] == [
        "λεώς"
    ]


def test_index_models_do_not_expose_mutable_yaml_structures(
    fixture_index: BuckReferenceIndex,
) -> None:
    rule = fixture_index.get_rule("R1")
    dialect = fixture_index.get_dialect("child")
    glossary_entry = fixture_index.find_glossary_by_word("λεώς")[0]

    assert rule is not None
    assert dialect is not None
    assert isinstance(rule.transformation, MappingProxyType)
    assert rule.variants[0] == MappingProxyType(
        {"form": "leos", "dialects": ("attic",)}
    )
    with pytest.raises(TypeError):
        rule.transformation["from"] = "mutated"
    with pytest.raises(FrozenInstanceError):
        rule.category = "mutated"
    with pytest.raises(FrozenInstanceError):
        dialect.name = "mutated"
    with pytest.raises(FrozenInstanceError):
        glossary_entry.word = "mutated"
