"""Tests for the Buck reference-data loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from phonology import buck as buck_module
from phonology._trusted_paths import TRUSTED_DIR_OVERRIDES_OPT_IN_ENV_VAR
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
    monkeypatch.setenv(TRUSTED_DIR_OVERRIDES_OPT_IN_ENV_VAR, "1")
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
    # Keep "true" here to verify non-"1" truthy opt-in values are also accepted.
    monkeypatch.setenv(TRUSTED_DIR_OVERRIDES_OPT_IN_ENV_VAR, "true")
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


def test_load_yaml_mapping_reports_yaml_parse_errors(tmp_path: Path) -> None:
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("rules: [unterminated\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Failed to parse YAML file"):
        buck_module._load_yaml_mapping(bad_yaml, required_list_key="rules")


def test_load_yaml_mapping_propagates_file_read_oserror(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unreadable = tmp_path / "unreadable.yaml"

    def fail_read_text(self: Path, *args: object, **kwargs: object) -> str:
        del args, kwargs
        if self == unreadable:
            raise OSError("disk failure")
        return "rules: []\n"

    monkeypatch.setattr(Path, "read_text", fail_read_text)

    with pytest.raises(OSError, match="disk failure"):
        buck_module._load_yaml_mapping(unreadable, required_list_key="rules")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not_a_mapping\n", "must contain a top-level mapping"),
        ("metadata: {}\n", "must define a list under 'rules'"),
        ("rules: {}\n", "must define a list under 'rules'"),
    ],
)
def test_load_yaml_mapping_rejects_required_list_shape(
    tmp_path: Path,
    content: str,
    message: str,
) -> None:
    yaml_path = tmp_path / "shape.yaml"
    yaml_path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        buck_module._load_yaml_mapping(yaml_path, required_list_key="rules")


def test_load_buck_data_cache_ignores_env_changes_until_cleared(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_dir = tmp_path / "buck-a"
    second_dir = tmp_path / "buck-b"
    _write_buck_fixture(
        first_dir,
        grammar_rules="rules:\n  - id: FIRST\n",
        dialects="dialects:\n  - id: dialect_a\n    rules: [FIRST]\n",
        glossary="words:\n  - word: first\n    dialect: dialect_a\n    rule_id: FIRST\n",
    )
    _write_buck_fixture(
        second_dir,
        grammar_rules="rules:\n  - id: SECOND\n",
        dialects="dialects:\n  - id: dialect_b\n    rules: [SECOND]\n",
        glossary="words:\n  - word: second\n    dialect: dialect_b\n    rule_id: SECOND\n",
    )

    monkeypatch.setenv("PROTEUS_TRUSTED_BUCK_DIR", str(first_dir))
    first = load_buck_data()
    monkeypatch.setenv("PROTEUS_TRUSTED_BUCK_DIR", str(second_dir))
    still_first = load_buck_data()
    buck_module._load_buck_data_cached.cache_clear()
    second = load_buck_data()

    assert first["grammar_rules"]["rules"][0]["id"] == "FIRST"
    assert still_first["grammar_rules"]["rules"][0]["id"] == "FIRST"
    assert second["grammar_rules"]["rules"][0]["id"] == "SECOND"


@pytest.mark.parametrize(
    ("grammar_rules", "dialects", "glossary", "message"),
    [
        (
            "rules:\n  - not-a-mapping\n",
            "dialects:\n  - id: test_dialect\n    rules: []\n",
            "words:\n  - word: test\n    dialect: test_dialect\n",
            "Buck rule entry 0 .* must be a mapping",
        ),
        (
            "rules:\n  - id: TEST\n",
            "dialects:\n  - not-a-mapping\n",
            "words:\n  - word: test\n    dialect: test_dialect\n",
            "Buck dialect entry 0 .* must be a mapping",
        ),
        (
            "rules:\n  - id: TEST\n",
            "dialects:\n  - id: test_dialect\n    rules: TEST\n",
            "words:\n  - word: test\n    dialect: test_dialect\n",
            "must define 'rules' as a list",
        ),
        (
            "rules:\n  - id: TEST\n",
            "dialects:\n  - id: test_dialect\n    rules: []\n",
            "words:\n  - not-a-mapping\n",
            "Buck glossary entry 0 .* must be a mapping",
        ),
        (
            "rules:\n  - id: TEST\n",
            "dialects:\n  - id: test_dialect\n    rules: []\n",
            "words:\n  - word: test\n",
            "must define a non-empty dialect",
        ),
        (
            "rules:\n  - id: TEST\n",
            "dialects:\n  - id: test_dialect\n    rules: [123]\n",
            "words:\n  - word: test\n    dialect: test_dialect\n",
            "must contain only non-empty string rule ids",
        ),
        (
            "rules:\n  - id: TEST\n",
            "dialects:\n  - id: test_dialect\n    rules: []\n",
            "words:\n  - word: test\n    dialect: test_dialect\n    rule_id: 123\n",
            "must define a non-empty rule_id",
        ),
        (
            "rules:\n  - id: TEST\n    affected_dialects: [123]\n",
            "dialects:\n  - id: test_dialect\n    rules: []\n",
            "words:\n  - word: test\n    dialect: test_dialect\n",
            "must contain only non-empty strings in affected_dialects",
        ),
        (
            "rules:\n  - id: TEST\n    variants:\n      - not-a-mapping\n",
            "dialects:\n  - id: test_dialect\n    rules: []\n",
            "words:\n  - word: test\n    dialect: test_dialect\n",
            "variant 0 .* must be a mapping",
        ),
        (
            "rules:\n  - id: TEST\n    variants:\n      - dialects: [123]\n",
            "dialects:\n  - id: test_dialect\n    rules: []\n",
            "words:\n  - word: test\n    dialect: test_dialect\n",
            "must contain only non-empty strings in dialects",
        ),
        (
            "rules:\n  - id: TEST\n    variants:\n      - dialects: [missing]\n",
            "dialects:\n  - id: test_dialect\n    rules: []\n",
            "words:\n  - word: test\n    dialect: test_dialect\n",
            "variant 0 .* references unknown dialect id 'missing'",
        ),
    ],
)
def test_load_buck_data_rejects_validation_edge_cases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    grammar_rules: str,
    dialects: str,
    glossary: str,
    message: str,
) -> None:
    buck_dir = tmp_path / "buck"
    _write_buck_fixture(
        buck_dir,
        grammar_rules=grammar_rules,
        dialects=dialects,
        glossary=glossary,
    )
    monkeypatch.setenv("PROTEUS_TRUSTED_BUCK_DIR", str(buck_dir))

    with pytest.raises(ValueError, match=message):
        load_buck_data()


def test_load_buck_data_reports_missing_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_dir = tmp_path / "missing-buck"
    monkeypatch.setenv("PROTEUS_TRUSTED_BUCK_DIR", str(missing_dir))

    with pytest.raises(FileNotFoundError, match="Could not find Buck data directory"):
        load_buck_data()


def test_load_buck_data_rejects_override_without_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buck_dir = tmp_path / "buck"
    buck_dir.mkdir()
    monkeypatch.delenv(TRUSTED_DIR_OVERRIDES_OPT_IN_ENV_VAR, raising=False)
    monkeypatch.setenv("PROTEUS_TRUSTED_BUCK_DIR", str(buck_dir))

    with pytest.raises(
        ValueError,
        match="PROTEUS_TRUSTED_BUCK_DIR requires PROTEUS_ALLOW_TRUSTED_DIR_OVERRIDES=1",
    ):
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
    # load_buck_data must reject symlinked parents before it checks existence.
    monkeypatch.setenv("PROTEUS_TRUSTED_BUCK_DIR", str(buck_dir))

    with pytest.raises(ValueError, match="must not contain a symlink"):
        load_buck_data()


def test_load_buck_data_rejects_file_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buck_file = tmp_path / "buck.yaml"
    buck_file.write_text("rules: []\n", encoding="utf-8")
    monkeypatch.setenv("PROTEUS_TRUSTED_BUCK_DIR", str(buck_file))

    with pytest.raises(NotADirectoryError, match="Buck data path is not a directory"):
        load_buck_data()
