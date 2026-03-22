"""Tests for proteus.phonology.explainer."""

from pathlib import Path

import pytest

from proteus.phonology import explainer as explainer_module
from proteus.phonology.explainer import Explanation, RuleApplication, load_rules, to_prose


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
