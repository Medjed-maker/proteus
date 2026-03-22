"""Tests for proteus.phonology.explainer."""

from pathlib import Path

import pytest

from proteus.phonology import explainer as explainer_module
from proteus.phonology.explainer import (
    Alignment,
    Explanation,
    RuleApplication,
    explain,
    explain_alignment,
    load_rules,
    to_prose,
)


def _rule(
    *,
    rule_id: str,
    input_phoneme: str,
    output_phoneme: str,
    context: str | None = "all environments",
    name_ja: str = "テスト規則",
) -> dict[str, object]:
    """テスト用の音韻規則辞書を生成する。"""
    return {
        "id": rule_id,
        "name_ja": name_ja,
        "input": input_phoneme,
        "output": output_phoneme,
        "context": context,
        "dialects": ["attic"],
    }


def test_to_prose_returns_canonical_prose_without_mutating_explanation() -> None:
    """to_prose returns prose while leaving Explanation.prose unchanged for callers."""
    explanation = Explanation(
        source="λόγος",
        target="λογος",
        source_ipa="loɡos",
        target_ipa="loɡos",
        distance=0.5,
        steps=[
            RuleApplication(
                rule_id="rule-1",
                rule_name="Accent Loss",
                from_phone="o",
                to_phone="o",
                position=1,
                dialects=["attic"],
                weight=0.5,
            )
        ],
    )

    result = to_prose(explanation)

    assert "Accent Loss" in result
    assert "λόγος /loɡos/ aligns to λογος /loɡos/" in result
    assert "weight 0.5" in result
    assert explanation.prose == ""


def test_rule_application_accepts_legacy_alias_fields() -> None:
    application = RuleApplication(
        rule_id="LEGACY-001",
        rule_name="Legacy Name",
        from_phone="a",
        to_phone="b",
        position=3,
    )

    assert application.description == "Legacy Name"
    assert application.input_phoneme == "a"
    assert application.output_phoneme == "b"
    assert application.rule_name == "Legacy Name"
    assert application.from_phone == "a"
    assert application.to_phone == "b"


def test_rule_application_prefers_explicit_rule_name_over_description_parsing() -> None:
    application = RuleApplication(
        rule_id="EXPLICIT-001",
        description="Parsed Name: /a/ → /b/",
        rule_name="Explicit Name",
        position=0,
    )

    assert application.description == "Parsed Name: /a/ → /b/"
    assert application.rule_name == "Explicit Name"


def test_rule_application_defaults_description_and_rule_name_together() -> None:
    application = RuleApplication(
        rule_id="DEFAULT-001",
        description=None,
        rule_name=None,
        position=0,
    )

    assert application.description == ""
    assert application.rule_name == ""


def test_explain_detects_single_token_substitution_with_japanese_description() -> None:
    rules = [
        _rule(
            rule_id="VSH-001",
            input_phoneme="aː",
            output_phoneme="ɛː",
            name_ja="長母音 ā > ē 推移",
        )
    ]

    applications = explain(
        query_ipa=["k", "ɛː", "s"],
        lemma_ipa=["k", "aː", "s"],
        alignment=Alignment(
            aligned_query=("k", "ɛː", "s"),
            aligned_lemma=("k", "aː", "s"),
        ),
        rules=rules,
    )

    assert len(applications) == 1
    application = applications[0]
    assert application.rule_id == "VSH-001"
    assert application.position == 1
    assert application.input_phoneme == "aː"
    assert application.output_phoneme == "ɛː"
    assert "長母音 ā > ē 推移" in application.description
    assert "/aː/ → /ɛː/" in application.description


def test_explain_prefers_longest_multi_token_rule() -> None:
    rules = [
        _rule(
            rule_id="CCH-LONG",
            input_phoneme="ss",
            output_phoneme="tt",
            name_ja="長い規則",
        ),
        _rule(
            rule_id="CCH-SHORT",
            input_phoneme="s",
            output_phoneme="t",
            name_ja="短い規則",
        ),
    ]

    applications = explain(
        query_ipa=["t", "t"],
        lemma_ipa=["s", "s"],
        alignment=Alignment(
            aligned_query=("t", "t"),
            aligned_lemma=("s", "s"),
        ),
        rules=rules,
    )

    assert [application.rule_id for application in applications] == ["CCH-LONG"]
    assert applications[0].position == 0
    assert applications[0].input_phoneme == "ss"
    assert applications[0].output_phoneme == "tt"


def test_explain_detects_deletion_and_uses_lemma_position() -> None:
    rules = [
        _rule(
            rule_id="CCH-DEL",
            input_phoneme="s",
            output_phoneme="",
            context="V_V",
            name_ja="母音間 s 脱落",
        )
    ]

    applications = explain(
        query_ipa=["a", "a"],
        lemma_ipa=["a", "s", "a"],
        alignment=Alignment(
            aligned_query=("a", None, "a"),
            aligned_lemma=("a", "s", "a"),
        ),
        rules=rules,
    )

    assert len(applications) == 1
    assert applications[0].rule_id == "CCH-DEL"
    assert applications[0].position == 1
    assert applications[0].output_phoneme == ""


def test_explain_skips_empty_output_rule_without_query_gap() -> None:
    rules = [
        _rule(
            rule_id="CCH-DEL",
            input_phoneme="s",
            output_phoneme="",
            context="all environments",
            name_ja="削除規則",
        )
    ]

    applications = explain(
        query_ipa=["a"],
        lemma_ipa=["s"],
        alignment=Alignment(
            aligned_query=("a",),
            aligned_lemma=("s",),
        ),
        rules=rules,
    )

    assert applications == []


def test_explain_ignores_empty_input_rule_without_insertion_flag() -> None:
    rules = [
        _rule(
            rule_id="INS-NOFLAG",
            input_phoneme="",
            output_phoneme="a",
            context="all environments",
            name_ja="未許可挿入",
        )
    ]

    applications = explain(
        query_ipa=["a"],
        lemma_ipa=[],
        alignment=Alignment(
            aligned_query=("a",),
            aligned_lemma=(None,),
        ),
        rules=rules,
    )

    assert applications == []


def test_explain_matches_empty_input_rule_only_when_explicitly_allowed() -> None:
    rule = _rule(
        rule_id="INS-ALLOWED",
        input_phoneme="",
        output_phoneme="a",
        context="all environments",
        name_ja="許可済み挿入",
    )
    rule["is_insertion"] = True

    applications = explain(
        query_ipa=["a"],
        lemma_ipa=[],
        alignment=Alignment(
            aligned_query=("a",),
            aligned_lemma=(None,),
        ),
        rules=[rule],
    )

    assert [application.rule_id for application in applications] == ["INS-ALLOWED"]


def test_explain_uses_context_to_choose_between_competing_rules() -> None:
    rules = [
        _rule(
            rule_id="RET-AFTER",
            input_phoneme="ɛː",
            output_phoneme="aː",
            context="after e, i, or r",
            name_ja="後続規則",
        ),
        _rule(
            rule_id="RET-ELSEWHERE",
            input_phoneme="ɛː",
            output_phoneme="aː",
            context="all environments except after e, i, or r",
            name_ja="それ以外規則",
        ),
    ]

    after_result = explain(
        query_ipa=["e", "aː"],
        lemma_ipa=["e", "ɛː"],
        alignment=Alignment(
            aligned_query=("e", "aː"),
            aligned_lemma=("e", "ɛː"),
        ),
        rules=rules,
    )
    elsewhere_result = explain(
        query_ipa=["o", "aː"],
        lemma_ipa=["o", "ɛː"],
        alignment=Alignment(
            aligned_query=("o", "aː"),
            aligned_lemma=("o", "ɛː"),
        ),
        rules=rules,
    )

    assert [application.rule_id for application in after_result] == ["RET-AFTER"]
    assert [application.rule_id for application in elsewhere_result] == ["RET-ELSEWHERE"]


def test_explain_prefers_specific_context_over_generic_rule() -> None:
    rules = [
        _rule(
            rule_id="RET-GENERIC",
            input_phoneme="ɛː",
            output_phoneme="aː",
            context="all environments",
            name_ja="汎用規則",
        ),
        _rule(
            rule_id="RET-AFTER",
            input_phoneme="ɛː",
            output_phoneme="aː",
            context="after e, i, or r",
            name_ja="条件付き規則",
        ),
    ]

    applications = explain(
        query_ipa=["r", "aː"],
        lemma_ipa=["r", "ɛː"],
        alignment=Alignment(
            aligned_query=("r", "aː"),
            aligned_lemma=("r", "ɛː"),
        ),
        rules=rules,
    )

    assert [application.rule_id for application in applications] == ["RET-AFTER"]


@pytest.mark.parametrize(
    ("rule", "query_ipa", "lemma_ipa", "alignment", "expected_rule_id"),
    [
        (
            _rule(
                rule_id="CTX-VV",
                input_phoneme="s",
                output_phoneme="",
                context="V_V",
                name_ja="V_V 規則",
            ),
            ["a", "a"],
            ["a", "s", "a"],
            Alignment(
                aligned_query=("a", None, "a"),
                aligned_lemma=("a", "s", "a"),
            ),
            "CTX-VV",
        ),
        (
            _rule(
                rule_id="CTX-NC",
                input_phoneme="a",
                output_phoneme="aː",
                context="_NC",
                name_ja="_NC 規則",
            ),
            ["aː", "n", "t"],
            ["a", "n", "t"],
            Alignment(
                aligned_query=("aː", "n", "t"),
                aligned_lemma=("a", "n", "t"),
            ),
            "CTX-NC",
        ),
        (
            _rule(
                rule_id="CTX-LOOKAHEAD",
                input_phoneme="pʰ",
                output_phoneme="p",
                context="_...pʰ",
                name_ja="先読み規則",
            ),
            ["p", "e", "pʰ"],
            ["pʰ", "e", "pʰ"],
            Alignment(
                aligned_query=("p", "e", "pʰ"),
                aligned_lemma=("pʰ", "e", "pʰ"),
            ),
            "CTX-LOOKAHEAD",
        ),
        (
            _rule(
                rule_id="CTX-SET",
                input_phoneme="k",
                output_phoneme="t",
                context="_{e,i}",
                name_ja="集合規則",
            ),
            ["t", "e"],
            ["k", "e"],
            Alignment(
                aligned_query=("t", "e"),
                aligned_lemma=("k", "e"),
            ),
            "CTX-SET",
        ),
    ],
)
def test_explain_supports_primary_context_notation(
    rule: dict[str, object],
    query_ipa: list[str],
    lemma_ipa: list[str],
    alignment: Alignment,
    expected_rule_id: str,
) -> None:
    applications = explain(
        query_ipa=query_ipa,
        lemma_ipa=lemma_ipa,
        alignment=alignment,
        rules=[rule],
    )

    assert [application.rule_id for application in applications] == [expected_rule_id]


def test_explain_supports_nc_context_with_query_side_fallback_after_lemma_end() -> None:
    applications = explain(
        query_ipa=["aː", "n", "t"],
        lemma_ipa=["a"],
        alignment=Alignment(
            aligned_query=("aː", "n", "t"),
            aligned_lemma=("a", None, None),
        ),
        rules=[
            _rule(
                rule_id="CTX-NC-FALLBACK",
                input_phoneme="a",
                output_phoneme="aː",
                context="_NC",
                name_ja="_NC fallback 規則",
            )
        ],
    )

    assert [application.rule_id for application in applications] == ["CTX-NC-FALLBACK"]


def test_explain_skips_unmatched_difference_blocks() -> None:
    applications = explain(
        query_ipa=["x"],
        lemma_ipa=["a"],
        alignment=Alignment(
            aligned_query=("x",),
            aligned_lemma=("a",),
        ),
        rules=[_rule(rule_id="NOPE", input_phoneme="b", output_phoneme="c")],
    )

    assert applications == []


def test_explain_alignment_wraps_rule_ids_into_explanation_steps() -> None:
    explanation = explain_alignment(
        source_ipa="aː",
        target_ipa="ɛː",
        rule_ids=["VSH-001"],
        distance=0.25,
        all_rules={
            "VSH-001": {
                "id": "VSH-001",
                "name_ja": "長母音 ā > ē 推移",
                "input": "aː",
                "output": "ɛː",
                "context": "all environments",
            }
        },
    )

    assert explanation.source == "aː"
    assert explanation.target == "ɛː"
    assert explanation.distance == 0.25
    assert len(explanation.steps) == 1
    assert explanation.steps[0].rule_id == "VSH-001"
    # explain_alignment and RuleApplication use -1 as the sentinel for an unspecified position.
    assert explanation.steps[0].position == -1
    assert explanation.steps[0].input_phoneme == "aː"
    assert explanation.steps[0].output_phoneme == "ɛː"


def test_explain_alignment_normalizes_dialects_with_deduplication() -> None:
    explanation = explain_alignment(
        source_ipa="aː",
        target_ipa="ɛː",
        rule_ids=["VSH-001"],
        all_rules={
            "VSH-001": {
                "id": "VSH-001",
                "name_ja": "長母音 ā > ē 推移",
                "input": "aː",
                "output": "ɛː",
                "dialects": ["attic", "ionic", "attic", 1, None, "ionic"],
            }
        },
    )

    assert explanation.steps[0].dialects == ["attic", "ionic"]


def test_explain_alignment_defaults_distance_to_zero_for_backward_compatibility() -> None:
    explanation = explain_alignment(
        source_ipa="aː",
        target_ipa="ɛː",
        rule_ids=[],
        all_rules={},
    )

    assert explanation.distance == 0.0


def test_load_rules_reads_yaml_rules_from_temp_rules_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """load_rules reads rule ids from a controlled rules directory."""
    rules_base = tmp_path / "rules"
    rules_dir = rules_base / "ancient_greek"
    rules_dir.mkdir(parents=True)
    monkeypatch.setattr(explainer_module, "_RULES_BASE_DIR_OVERRIDE", rules_base)

    (rules_dir / "consonants.yaml").write_text(
        "rules:\n"
        "  - id: TEST-CCH-001\n"
        "    name: consonant shift\n",
        encoding="utf-8",
    )
    (rules_dir / "vowels.yaml").write_text(
        "rules:\n"
        "  - id: TEST-VSH-001\n"
        "    name: vowel shift\n",
        encoding="utf-8",
    )

    rules = load_rules(rules_dir)

    assert "TEST-CCH-001" in rules
    assert "TEST-VSH-001" in rules


def test_load_rules_accepts_bare_relative_directory_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rules_base = tmp_path / "rules"
    rules_dir = rules_base / "ancient_greek"
    elsewhere_dir = tmp_path / "elsewhere"
    rules_dir.mkdir(parents=True)
    elsewhere_dir.mkdir()
    monkeypatch.setattr(explainer_module, "_RULES_BASE_DIR_OVERRIDE", rules_base)
    monkeypatch.chdir(elsewhere_dir)

    (rules_dir / "rules.yaml").write_text(
        "rules:\n"
        "  - id: TEST-001\n"
        "    name: relative lookup\n",
        encoding="utf-8",
    )

    rules = load_rules("ancient_greek")

    assert "TEST-001" in rules


def test_load_rules_accepts_legacy_repo_style_relative_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rules_base = tmp_path / "rules"
    rules_dir = rules_base / "ancient_greek"
    elsewhere_dir = tmp_path / "elsewhere"
    rules_dir.mkdir(parents=True)
    elsewhere_dir.mkdir()
    monkeypatch.setattr(explainer_module, "_RULES_BASE_DIR_OVERRIDE", rules_base)
    monkeypatch.chdir(elsewhere_dir)

    (rules_dir / "rules.yaml").write_text(
        "rules:\n"
        "  - id: TEST-LEGACY-001\n"
        "    name: legacy relative lookup\n",
        encoding="utf-8",
    )

    rules = load_rules("data/rules/ancient_greek")

    assert "TEST-LEGACY-001" in rules


def test_load_rules_rejects_directories_outside_rules_base(tmp_path: Path) -> None:
    """load_rules rejects traversal to directories outside the repository rules base."""
    outside_rules_dir = tmp_path / "rules"
    outside_rules_dir.mkdir()
    (outside_rules_dir / "outside.yaml").write_text("rules: []\n", encoding="utf-8")

    with pytest.raises(ValueError, match="load_rules path must stay within"):
        load_rules(outside_rules_dir)


def test_load_rules_reports_missing_rules_base(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_rules_base = tmp_path / "missing-rules-base"
    rules_dir = missing_rules_base / "ancient_greek"

    monkeypatch.setattr(explainer_module, "_RULES_BASE_DIR_OVERRIDE", missing_rules_base)

    with pytest.raises(ValueError, match="Configured rules base directory is missing"):
        load_rules(rules_dir)


def test_load_rules_duplicate_error_includes_both_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rules_base = tmp_path / "rules"
    rules_dir = rules_base / "ancient_greek"
    rules_dir.mkdir(parents=True)
    monkeypatch.setattr(explainer_module, "_RULES_BASE_DIR_OVERRIDE", rules_base)

    (rules_dir / "first.yaml").write_text(
        "rules:\n"
        "  - id: DUP-001\n"
        "    name: first\n",
        encoding="utf-8",
    )
    (rules_dir / "second.yaml").write_text(
        "rules:\n"
        "  - id: DUP-001\n"
        "    name: second\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        load_rules(rules_dir)

    message = str(exc_info.value)
    assert "first.yaml" in message
    assert "second.yaml" in message
