"""Tests for the Buck reference-data loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from phonology import buck as buck_module
from phonology.buck import load_buck_data


def _write_buck_fixture(
    buck_dir: Path,
    *,
    grammar_rules: str,
    dialects: str,
    glossary: str,
) -> None:
    """Write Buck fixture files into the provided directory.

    Creates the directory (including parent directories) and writes
    grammar_rules.yaml, dialects.yaml, and glossary.yaml with the
    provided content.

    Args:
        buck_dir: Path to the Buck data directory (created if needed).
        grammar_rules: YAML content for grammar_rules.yaml.
        dialects: YAML content for dialects.yaml.
        glossary: YAML content for glossary.yaml.
    """
    buck_dir.mkdir(parents=True)
    (buck_dir / "grammar_rules.yaml").write_text(grammar_rules, encoding="utf-8")
    (buck_dir / "dialects.yaml").write_text(dialects, encoding="utf-8")
    (buck_dir / "glossary.yaml").write_text(glossary, encoding="utf-8")


@pytest.fixture(autouse=True)
def reset_buck_loader_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PROTEUS_TRUSTED_BUCK_DIR", raising=False)
    buck_module._load_buck_data_cached.cache_clear()


def test_load_buck_data_reads_packaged_documents() -> None:
    data = load_buck_data()

    assert set(data) == {"grammar_rules", "dialects", "glossary"}
    rule_ids = {rule["id"] for rule in data["grammar_rules"]["rules"]}

    assert {"grc_phon_41_4", "grc_phon_60_1", "grc_synt_175", "grc_morph_134_3"} <= rule_ids


def test_load_buck_data_returns_defensive_copy() -> None:
    first = load_buck_data()
    first["grammar_rules"]["rules"].append({"id": "MUTATED"})
    first["dialects"]["dialects"][0]["id"] = "mutated"
    first["glossary"]["words"].clear()

    second = load_buck_data()

    assert all(rule.get("id") != "MUTATED" for rule in second["grammar_rules"]["rules"])
    assert second["dialects"]["dialects"][0]["id"] != "mutated"
    assert second["glossary"]["words"]


def test_load_buck_data_supports_override_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buck_dir = tmp_path / "buck"
    _write_buck_fixture(
        buck_dir,
        grammar_rules=(
            "rules:\n"
            "  - id: TEST-BUCK-001\n"
            "    affected_dialects: [test_dialect]\n"
            "    variants:\n"
            "      - form: x\n"
            "        dialects: [test_dialect]\n"
        ),
        dialects=(
            "dialects:\n"
            "  - id: test_dialect\n"
            "    rules: [TEST-BUCK-001]\n"
        ),
        glossary=(
            "words:\n"
            "  - word: test\n"
            "    dialect: test_dialect\n"
            "    rule_id: TEST-BUCK-001\n"
        ),
    )
    monkeypatch.setenv("PROTEUS_TRUSTED_BUCK_DIR", str(buck_dir))

    data = load_buck_data()

    assert data["grammar_rules"]["rules"][0]["id"] == "TEST-BUCK-001"
    assert data["dialects"]["dialects"][0]["id"] == "test_dialect"
    assert data["glossary"]["words"][0]["rule_id"] == "TEST-BUCK-001"


def test_load_buck_data_supports_relative_override_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buck_dir = tmp_path / "buck"
    _write_buck_fixture(
        buck_dir,
        grammar_rules="rules:\n  - id: TEST-BUCK-REL-001\n",
        dialects="dialects:\n  - id: test_dialect\n    rules: [TEST-BUCK-REL-001]\n",
        glossary=(
            "words:\n"
            "  - word: test\n"
            "    dialect: test_dialect\n"
            "    rule_id: TEST-BUCK-REL-001\n"
        ),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PROTEUS_TRUSTED_BUCK_DIR", "buck")

    data = load_buck_data()

    assert data["grammar_rules"]["rules"][0]["id"] == "TEST-BUCK-REL-001"
    assert data["dialects"]["dialects"][0]["id"] == "test_dialect"
    assert data["glossary"]["words"][0]["rule_id"] == "TEST-BUCK-REL-001"


def test_load_buck_data_rejects_duplicate_rule_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buck_dir = tmp_path / "buck"
    _write_buck_fixture(
        buck_dir,
        grammar_rules=(
            "rules:\n"
            "  - id: DUP-001\n"
            "  - id: DUP-001\n"
        ),
        dialects="dialects:\n  - id: test_dialect\n    rules: []\n",
        glossary="words:\n  - word: test\n    dialect: test_dialect\n",
    )
    monkeypatch.setenv("PROTEUS_TRUSTED_BUCK_DIR", str(buck_dir))

    with pytest.raises(ValueError, match="Duplicate Buck rule id 'DUP-001'"):
        load_buck_data()


def test_load_buck_data_rejects_unknown_dialect_rule_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buck_dir = tmp_path / "buck"
    _write_buck_fixture(
        buck_dir,
        grammar_rules="rules:\n  - id: TEST-001\n",
        dialects="dialects:\n  - id: test_dialect\n    rules: [MISSING-001]\n",
        glossary="words:\n  - word: test\n    dialect: test_dialect\n",
    )
    monkeypatch.setenv("PROTEUS_TRUSTED_BUCK_DIR", str(buck_dir))

    with pytest.raises(ValueError, match="references unknown rule id 'MISSING-001'"):
        load_buck_data()


def test_load_buck_data_rejects_unknown_glossary_rule_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buck_dir = tmp_path / "buck"
    _write_buck_fixture(
        buck_dir,
        grammar_rules="rules:\n  - id: TEST-001\n",
        dialects="dialects:\n  - id: test_dialect\n    rules: [TEST-001]\n",
        glossary=(
            "words:\n"
            "  - word: test\n"
            "    dialect: test_dialect\n"
            "    rule_id: MISSING-001\n"
        ),
    )
    monkeypatch.setenv("PROTEUS_TRUSTED_BUCK_DIR", str(buck_dir))

    with pytest.raises(ValueError, match="references unknown rule id 'MISSING-001'"):
        load_buck_data()


def test_load_buck_data_rejects_unknown_glossary_dialect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buck_dir = tmp_path / "buck"
    _write_buck_fixture(
        buck_dir,
        grammar_rules="rules:\n  - id: TEST-001\n",
        dialects="dialects:\n  - id: test_dialect\n    rules: [TEST-001]\n",
        glossary=(
            "words:\n"
            "  - word: test\n"
            "    dialect: missing_dialect\n"
            "    rule_id: TEST-001\n"
        ),
    )
    monkeypatch.setenv("PROTEUS_TRUSTED_BUCK_DIR", str(buck_dir))

    with pytest.raises(ValueError, match="references unknown dialect id 'missing_dialect'"):
        load_buck_data()


def test_load_buck_data_rejects_unknown_grammar_dialect_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buck_dir = tmp_path / "buck"
    _write_buck_fixture(
        buck_dir,
        grammar_rules="rules:\n  - id: TEST-001\n    affected_dialects: [missing_dialect]\n",
        dialects="dialects:\n  - id: test_dialect\n    rules: [TEST-001]\n",
        glossary="words:\n  - word: test\n    dialect: test_dialect\n    rule_id: TEST-001\n",
    )
    monkeypatch.setenv("PROTEUS_TRUSTED_BUCK_DIR", str(buck_dir))

    with pytest.raises(ValueError, match="references unknown dialect id 'missing_dialect'"):
        load_buck_data()


def test_load_buck_data_rejects_wrong_top_level_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buck_dir = tmp_path / "buck"
    _write_buck_fixture(
        buck_dir,
        grammar_rules="rules: []\n",
        dialects="rules: []\ndialects: []\n",
        glossary="words: []\n",
    )
    monkeypatch.setenv("PROTEUS_TRUSTED_BUCK_DIR", str(buck_dir))

    with pytest.raises(ValueError, match="must not define a top-level 'rules' key"):
        load_buck_data()


def test_load_buck_data_reports_missing_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_dir = tmp_path / "missing-buck"
    monkeypatch.setenv("PROTEUS_TRUSTED_BUCK_DIR", str(missing_dir))

    with pytest.raises(FileNotFoundError, match="Could not find Buck data directory"):
        load_buck_data()


def test_load_buck_data_rejects_symlink_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_dir = tmp_path / "real-buck"
    real_dir.mkdir()
    link = tmp_path / "link-buck"
    link.symlink_to(real_dir)
    monkeypatch.setenv("PROTEUS_TRUSTED_BUCK_DIR", str(link))

    with pytest.raises(ValueError, match="must not contain a symlink"):
        load_buck_data()


def test_load_buck_data_rejects_symlink_in_parent_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    link_parent = tmp_path / "link-parent"
    link_parent.symlink_to(real_parent)
    buck_dir = link_parent / "buck"
    # buck_dir itself is not a symlink, but its parent is
    monkeypatch.setenv("PROTEUS_TRUSTED_BUCK_DIR", str(buck_dir))

    with pytest.raises(ValueError, match="must not contain a symlink"):
        load_buck_data()
