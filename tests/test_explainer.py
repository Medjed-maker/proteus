"""Tests for phonology.explainer."""

from pathlib import Path

import pytest

from phonology import explainer as explainer_module
from phonology.distance import load_matrix
from phonology.explainer import (
    Alignment,
    Explanation,
    POSITION_UNKNOWN,
    RuleApplication,
    explain,
    explain_alignment,
    load_rules,
    to_prose,
)
from phonology.ipa_converter import to_ipa, tokenize_ipa
# Intentional private import: these packaged-rule tests need the exact
# Smith-Waterman alignment used at runtime before calling explain(), so they
# depend on _smith_waterman_alignment and may need updates after refactors.
from phonology.search import _smith_waterman_alignment

MATRIX_FILE = "attic_doric.json"


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


def test_declares_position_unknown_in_public_api() -> None:
    """POSITION_UNKNOWN is part of the explicit public API surface."""
    assert "POSITION_UNKNOWN" in explainer_module.__all__
    assert POSITION_UNKNOWN == -1


def test_rule_application_accepts_legacy_alias_fields() -> None:
    """Verify legacy alias fields still populate the canonical rule fields."""
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
    """Verify an explicit rule_name wins over any name parsed from description text."""
    application = RuleApplication(
        rule_id="EXPLICIT-001",
        description="Parsed Name: /a/ → /b/",
        rule_name="Explicit Name",
        position=0,
    )

    assert application.description == "Parsed Name: /a/ → /b/"
    assert application.rule_name == "Explicit Name"


def test_rule_application_defaults_description_and_rule_name_together() -> None:
    """Verify missing description metadata defaults both display fields to empty strings."""
    application = RuleApplication(
        rule_id="DEFAULT-001",
        description=None,
        rule_name=None,
        position=0,
    )

    assert application.description == ""
    assert application.rule_name == ""


def test_explain_detects_single_token_substitution_with_japanese_description() -> None:
    """Verify explain emits one substitution step with the expected Japanese rule text."""
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
    """Verify explain prefers the longest matching multi-token rule over shorter overlaps."""
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
    """Verify deletions are reported at the lemma-side position where the phone disappears."""
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
    """Verify deletion rules do not fire unless the alignment actually contains a query gap.

    The catalogued deletion rule CCH-DEL should NOT match (no query gap), but an
    observed-substitution annotation is still generated for the mismatch.
    """
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

    # The catalogued rule should not match; only the fallback observation fires.
    assert len(applications) == 1
    assert applications[0].rule_id == "OBS-SUB"
    assert applications[0].input_phoneme == "s"
    assert applications[0].output_phoneme == "a"


def test_explain_ignores_empty_input_rule_without_insertion_flag() -> None:
    """Verify empty-input rules are ignored unless they are explicitly marked as insertions.

    The unflagged insertion rule should NOT match, but an observed-insertion
    annotation is still generated for the unmatched gap.
    """
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

    # The unflagged rule should not match; only the fallback observation fires.
    assert len(applications) == 1
    assert applications[0].rule_id == "OBS-INS"
    assert applications[0].input_phoneme == ""
    assert applications[0].output_phoneme == "a"


def test_explain_matches_empty_input_rule_only_when_explicitly_allowed() -> None:
    """Verify insertion rules match only when the rule is explicitly flagged as an insertion."""
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
    """Verify context-sensitive competing rules resolve to the rule matching each environment."""
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
    """Verify a more specific contextual rule outranks a generic fallback rule."""
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
    """Verify primary context shorthand notations map to the intended matching rule."""
    applications = explain(
        query_ipa=query_ipa,
        lemma_ipa=lemma_ipa,
        alignment=alignment,
        rules=[rule],
    )

    assert [application.rule_id for application in applications] == [expected_rule_id]


def test_explain_supports_word_final_context_with_exact_tail_inside_suffix() -> None:
    """Accept CTX-FINAL when the exact remaining tail is contained within a suffix alignment."""
    applications = explain(
        query_ipa=["b", "a", "s", "i", "l", "e", "o", "s"],
        lemma_ipa=["b", "a", "s", "i", "l", "eu", "s"],
        alignment=Alignment(
            aligned_query=("b", "a", "s", "i", "l", "e", "o", "s"),
            aligned_lemma=("b", "a", "s", "i", "l", "eu", None, "s"),
        ),
        rules=[
            _rule(
                rule_id="CTX-FINAL",
                input_phoneme="eus",
                output_phoneme="eos",
                context="_#",
                name_ja="語末規則",
            )
        ],
    )

    assert [application.rule_id for application in applications] == ["CTX-FINAL"]


def test_explain_rejects_word_final_context_when_tokens_remain_after_suffix() -> None:
    """Reject CTX-FINAL when tokens remain after the candidate word-final suffix."""
    applications = explain(
        query_ipa=["b", "a", "s", "i", "l", "e", "o", "s", "u"],
        lemma_ipa=["b", "a", "s", "i", "l", "eu", "s", "u"],
        alignment=Alignment(
            aligned_query=("b", "a", "s", "i", "l", "e", "o", "s", "u"),
            aligned_lemma=("b", "a", "s", "i", "l", "eu", None, "s", "u"),
        ),
        rules=[
            _rule(
                rule_id="CTX-FINAL",
                input_phoneme="eus",
                output_phoneme="eos",
                context="_#",
                name_ja="語末規則",
            )
        ],
    )

    assert [application.rule_id for application in applications] == ["OBS-SUB", "OBS-INS"]


def test_explain_rejects_word_final_context_when_later_mismatch_block_remains() -> None:
    """Reject CTX-FINAL when the candidate would need a later mismatch block."""
    applications = explain(
        query_ipa=["x", "b", "y"],
        lemma_ipa=["a", "b", "c"],
        alignment=Alignment(
            aligned_query=("x", "b", "y"),
            aligned_lemma=("a", "b", "c"),
        ),
        rules=[
            _rule(
                rule_id="CTX-FINAL",
                input_phoneme="abc",
                output_phoneme="xby",
                context="_#",
                name_ja="語末規則",
            )
        ],
    )

    assert [application.rule_id for application in applications] == ["OBS-SUB", "OBS-SUB"]
    assert [application.position for application in applications] == [0, 2]


def test_explain_supports_gap_spanning_one_to_two_token_rule() -> None:
    """Verify alignment matches GAP-1TO2 when one lemma token expands into two query tokens."""
    applications = explain(
        query_ipa=["e", "o"],
        lemma_ipa=["eu"],
        alignment=Alignment(
            aligned_query=("e", "o"),
            aligned_lemma=("eu", None),
        ),
        rules=[_rule(rule_id="GAP-1TO2", input_phoneme="eu", output_phoneme="eo", context=None)],
    )

    assert [application.rule_id for application in applications] == ["GAP-1TO2"]


def test_explain_supports_gap_spanning_two_to_one_token_rule() -> None:
    """Verify alignment matches GAP-2TO1 when two lemma tokens contract into one query token."""
    applications = explain(
        query_ipa=["ɛː"],
        lemma_ipa=["e", "a"],
        alignment=Alignment(
            aligned_query=("ɛː", None),
            aligned_lemma=("e", "a"),
        ),
        rules=[_rule(rule_id="GAP-2TO1", input_phoneme="ea", output_phoneme="ɛː", context=None)],
    )

    assert [application.rule_id for application in applications] == ["GAP-2TO1"]


def test_explain_supports_gap_spanning_two_to_three_token_expansion() -> None:
    """Verify a 2->3 rule matches across a lemma gap when all columns are mismatches.

    All alignment columns must differ so they remain inside a single mismatch
    block; exact-match columns would split the block and prevent the rule from
    spanning the gap.
    """
    # tokenize_ipa("ɛːp") = ["ɛː", "p"], tokenize_ipa("eab") = ["e", "a", "b"]
    applications = explain(
        query_ipa=["e", "a", "b"],
        lemma_ipa=["ɛː", "p"],
        alignment=Alignment(
            aligned_query=("e", "a", "b"),
            aligned_lemma=("ɛː", None, "p"),
        ),
        rules=[_rule(rule_id="GAP-2TO3", input_phoneme="ɛːp", output_phoneme="eab", context=None)],
    )

    assert [application.rule_id for application in applications] == ["GAP-2TO3"]


def test_explain_supports_gap_spanning_three_to_two_token_contraction() -> None:
    """Verify a 3->2 rule matches across a query gap when all columns are mismatches."""
    # tokenize_ipa("ean") = ["e", "a", "n"], tokenize_ipa("ɛːm") = ["ɛː", "m"]
    applications = explain(
        query_ipa=["ɛː", "m"],
        lemma_ipa=["e", "a", "n"],
        alignment=Alignment(
            aligned_query=("ɛː", None, "m"),
            aligned_lemma=("e", "a", "n"),
        ),
        rules=[_rule(rule_id="GAP-3TO2", input_phoneme="ean", output_phoneme="ɛːm", context=None)],
    )

    assert [application.rule_id for application in applications] == ["GAP-3TO2"]


def test_explain_rejects_crossing_gap_match_for_multi_token_rule() -> None:
    """Reject a candidate when insertion and deletion gaps cross in one match."""
    applications = explain(
        query_ipa=["t", "t"],
        lemma_ipa=["s", "s"],
        alignment=Alignment(
            aligned_query=("t", None, "t"),
            aligned_lemma=(None, "s", "s"),
        ),
        rules=[_rule(rule_id="CCH-LONG", input_phoneme="ss", output_phoneme="tt", context=None)],
    )

    assert [application.rule_id for application in applications] == ["OBS-INS", "OBS-DEL", "OBS-SUB"]
    assert [application.position for application in applications] == [0, 0, 1]


def test_explain_rejects_crossing_gap_match_for_two_to_two_rule() -> None:
    """Reject 2->2 rules when they would need both gap directions in one match."""
    applications = explain(
        query_ipa=["e", "oː"],
        lemma_ipa=["eː", "o"],
        alignment=Alignment(
            aligned_query=("e", None, "oː"),
            aligned_lemma=(None, "eː", "o"),
        ),
        rules=[_rule(rule_id="VSH-004", input_phoneme="eːo", output_phoneme="eoː", context=None)],
    )

    assert [application.rule_id for application in applications] == ["OBS-INS", "OBS-DEL", "OBS-SUB"]


def test_explain_rejects_crossing_gap_match_for_word_final_suffix_rule() -> None:
    """Reject `_#` suffix rules when the remaining alignment contains crossing gaps."""
    applications = explain(
        query_ipa=["e", "oː"],
        lemma_ipa=["eː", "o"],
        alignment=Alignment(
            aligned_query=("e", None, "oː"),
            aligned_lemma=(None, "eː", "o"),
        ),
        rules=[_rule(rule_id="CTX-FINAL", input_phoneme="eːo", output_phoneme="eoː", context="_#")],
    )

    assert [application.rule_id for application in applications] == ["OBS-INS", "OBS-DEL", "OBS-SUB"]


def test_explain_supports_nc_context_with_query_side_fallback_after_lemma_end() -> None:
    """Verify _NC context matching can fall back to query-side lookahead beyond lemma length."""
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

    # The catalogued rule matches the first mismatch; the remaining unmatched
    # gaps produce observed-insertion annotations.
    assert applications[0].rule_id == "CTX-NC-FALLBACK"
    obs_ids = [a.rule_id for a in applications[1:]]
    assert all(oid == "OBS-INS" for oid in obs_ids)


def test_explain_generates_observed_substitution_for_unmatched_blocks() -> None:
    """Verify unmatched alignment differences produce observed-substitution annotations."""
    applications = explain(
        query_ipa=["x"],
        lemma_ipa=["a"],
        alignment=Alignment(
            aligned_query=("x",),
            aligned_lemma=("a",),
        ),
        rules=[_rule(rule_id="NOPE", input_phoneme="b", output_phoneme="c")],
    )

    assert len(applications) == 1
    assert applications[0].rule_id == "OBS-SUB"
    assert applications[0].input_phoneme == "a"
    assert applications[0].output_phoneme == "x"
    assert applications[0].rule_name == "観測された置換"
    assert applications[0].rule_name_en == "Observed substitution"


def test_explain_alignment_wraps_rule_ids_into_explanation_steps() -> None:
    """Verify explain_alignment expands rule ids into populated explanation steps."""
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
    """Verify explain_alignment drops non-string dialects and deduplicates the remainder."""
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
    """Verify explain_alignment defaults missing distance data to zero for legacy callers."""
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
    """Verify load_rules resolves a bare relative rules directory name under the rules base."""
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
    """Verify load_rules still accepts the legacy data/rules/... relative path style."""
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


def test_load_rules_reads_all_three_packaged_rule_files() -> None:
    """Verify that all three default packaged rule files are loaded properly.

    Checks for the presence of specific rule IDs across the domains and ensures
    a minimum number of rules are loaded without tying the test to a hardcoded count.
    """
    rules = load_rules("ancient_greek")

    assert len(rules) >= 50
    assert "CCH-015" in rules
    assert "VSH-022" in rules
    assert "MPH-013" in rules


@pytest.mark.parametrize(
    ("query_word", "lemma_word", "expected_rule_id"),
    [
        ("Δαμοσθένας", "Δημοσθένης", "MPH-004"),
        ("βασιλέος", "βασιλεύς", "MPH-013"),
    ],
)
def test_packaged_morphophonemic_rules_match_runtime_ipa_examples(
    query_word: str,
    lemma_word: str,
    expected_rule_id: str,
) -> None:
    """Verify packaged morphophonemic rules yield the expected rule_id for runtime IPA examples.

    ``query_word`` and ``lemma_word`` are orthographic forms converted to runtime
    IPA tokens, and ``expected_rule_id`` is the packaged rule expected after
    building the alignment via ``_smith_waterman_alignment`` before ``explain()``.
    """
    rules = list(load_rules("ancient_greek").values())
    matrix = load_matrix(MATRIX_FILE)
    query_tokens = tokenize_ipa(to_ipa(query_word))
    lemma_tokens = tokenize_ipa(to_ipa(lemma_word))
    _, aligned_query, aligned_lemma = _smith_waterman_alignment(
        query_tokens,
        lemma_tokens,
        matrix,
    )

    applications = explain(
        query_ipa=query_tokens,
        lemma_ipa=lemma_tokens,
        alignment=Alignment(
            aligned_query=tuple(aligned_query),
            aligned_lemma=tuple(aligned_lemma),
        ),
        rules=rules,
    )

    assert expected_rule_id in [application.rule_id for application in applications]


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
    """Verify load_rules reports a missing configured rules base before reading any files."""
    missing_rules_base = tmp_path / "missing-rules-base"
    rules_dir = missing_rules_base / "ancient_greek"

    monkeypatch.setattr(explainer_module, "_RULES_BASE_DIR_OVERRIDE", missing_rules_base)

    with pytest.raises(ValueError, match="Configured rules base directory is missing"):
        load_rules(rules_dir)


def test_load_rules_duplicate_error_includes_both_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify duplicate rule ids report both source files in the validation error."""
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


def test_explain_generates_observed_deletion() -> None:
    """Verify a gap-aligned mismatch (deletion) produces OBS-DEL."""
    applications = explain(
        query_ipa=["a"],
        lemma_ipa=["a", "s"],
        alignment=Alignment(
            aligned_query=("a", None),
            aligned_lemma=("a", "s"),
        ),
        rules=[],
    )
    obs = [a for a in applications if a.rule_id == "OBS-DEL"]
    assert len(obs) == 1
    assert obs[0].input_phoneme == "s"
    assert obs[0].output_phoneme == ""
    assert obs[0].rule_name == "観測された脱落"
    assert obs[0].rule_name_en == "Observed deletion"


def test_explain_generates_observed_insertion() -> None:
    """Verify a gap-aligned mismatch (insertion) produces OBS-INS."""
    applications = explain(
        query_ipa=["a", "s"],
        lemma_ipa=["a"],
        alignment=Alignment(
            aligned_query=("a", "s"),
            aligned_lemma=("a", None),
        ),
        rules=[],
    )
    obs = [a for a in applications if a.rule_id == "OBS-INS"]
    assert len(obs) == 1
    assert obs[0].input_phoneme == ""
    assert obs[0].output_phoneme == "s"
    assert obs[0].rule_name == "観測された挿入"
    assert obs[0].rule_name_en == "Observed insertion"


def test_explain_preserves_column_order_for_leading_query_gap() -> None:
    """Verify OBS fallback consumes mismatch columns in aligned order."""
    applications = explain(
        query_ipa=["x"],
        lemma_ipa=["a", "b"],
        alignment=Alignment(
            aligned_query=(None, "x"),
            aligned_lemma=("a", "b"),
        ),
        rules=[],
    )

    assert [(app.rule_id, app.input_phoneme, app.output_phoneme) for app in applications] == [
        ("OBS-DEL", "a", ""),
        ("OBS-SUB", "b", "x"),
    ]
    assert [app.position for app in applications] == [0, 1]


def test_explain_preserves_column_order_for_leading_lemma_gap() -> None:
    """Verify leading insertions do not shift later observed substitutions."""
    applications = explain(
        query_ipa=["x", "y"],
        lemma_ipa=["a"],
        alignment=Alignment(
            aligned_query=("x", "y"),
            aligned_lemma=(None, "a"),
        ),
        rules=[],
    )

    assert [(app.rule_id, app.input_phoneme, app.output_phoneme) for app in applications] == [
        ("OBS-INS", "", "x"),
        ("OBS-SUB", "a", "y"),
    ]
    assert [app.position for app in applications] == [0, 0]


def test_explain_matches_catalogued_rule_after_leading_query_gap() -> None:
    """Verify OBS fallback does not consume a later catalogued substitution."""
    applications = explain(
        query_ipa=["x"],
        lemma_ipa=["a", "b"],
        alignment=Alignment(
            aligned_query=(None, "x"),
            aligned_lemma=("a", "b"),
        ),
        rules=[
            _rule(
                rule_id="RULE-BX",
                input_phoneme="b",
                output_phoneme="x",
                name_ja="b から x への推移",
            )
        ],
    )

    assert [(app.rule_id, app.input_phoneme, app.output_phoneme) for app in applications] == [
        ("OBS-DEL", "a", ""),
        ("RULE-BX", "b", "x"),
    ]
    assert [app.position for app in applications] == [0, 1]


def test_explain_fails_fast_on_double_gap_mismatch_column() -> None:
    """Verify invalid mismatch blocks with double-gap columns raise a clear error."""
    with pytest.raises(RuntimeError) as exc_info:
        explain(
            query_ipa=["x"],
            lemma_ipa=["a"],
            alignment=Alignment(
                aligned_query=(None, "x"),
                aligned_lemma=(None, "a"),
            ),
            rules=[],
        )

    message = str(exc_info.value)
    assert "lemma_token=None" in message
    assert "query_token=None" in message
    assert "lemma_index=0" in message
    assert "query_index=0" in message


def test_to_prose_exact_match_message() -> None:
    """Verify distance=0 with no steps produces an exact-match message."""
    explanation = Explanation(
        source="loɡos",
        target="loɡos",
        source_ipa="loɡos",
        target_ipa="loɡos",
        distance=0.0,
        steps=[],
    )
    prose = to_prose(explanation)
    assert "exact match" in prose
    assert "No rule applications" not in prose


def test_to_prose_treats_near_zero_distance_as_exact_match() -> None:
    """Verify a tiny floating-point residue still produces the exact-match prose."""
    explanation = Explanation(
        source="loɡos",
        target="loɡos",
        source_ipa="loɡos",
        target_ipa="loɡos",
        distance=1e-12,
        steps=[],
    )
    prose = to_prose(explanation)
    assert "exact match" in prose
    assert "No rule applications" not in prose


def test_rule_application_has_bilingual_names() -> None:
    """Verify RuleApplication stores both Japanese and English names."""
    app = RuleApplication(
        rule_id="VSH-001",
        rule_name="長母音 ā > ē 推移",
        rule_name_en="Long vowel ā > ē shift",
        input_phoneme="aː",
        output_phoneme="ɛː",
        position=0,
    )
    assert app.rule_name == "長母音 ā > ē 推移"
    assert app.rule_name_en == "Long vowel ā > ē shift"


def test_rule_application_rule_name_en_defaults_to_empty() -> None:
    """Verify rule_name_en defaults to empty string when not provided."""
    app = RuleApplication(
        rule_id="X",
        rule_name="test",
        position=0,
    )
    assert app.rule_name_en == ""
