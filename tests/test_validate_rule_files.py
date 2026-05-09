"""Tests for tools.validate_rule_files."""

from __future__ import annotations

import builtins
import json
import logging
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml
from jsonschema import Draft202012Validator, FormatChecker, ValidationError

from tools import validate_rule_files

ROOT_DIR = Path(__file__).resolve().parents[1]
RULE_SCHEMA_PATH = ROOT_DIR / "data" / "schemas" / "phonology_rule_file.schema.json"


def _rule_validator() -> Draft202012Validator:
    """Load the rule JSON schema and return a Draft202012Validator."""
    schema = json.loads(RULE_SCHEMA_PATH.read_text(encoding="utf-8"))
    return Draft202012Validator(schema, format_checker=FormatChecker())


RuleDocument = dict[str, Any]


def _valid_rule_document() -> RuleDocument:
    """Return a minimal valid RuleDocument fixture for validation tests."""
    return {
        "meta": {
            "version": "1.0.0",
            "status": "provisional",
            "review_status": "not_expert_reviewed",
            "citation_ready": False,
            "license": "MIT",
            "source_notes": ["test fixture"],
        },
        "rules": [
            {
                "id": "TST-001",
                "name_en": "Test rule",
                "name_ja": "Test rule",
                "input": "p",
                "output": "b",
                "context": None,
                "dialects": ["test"],
                "period": "test",
                "references": ["test"],
                "examples": [
                    {
                        "standard": "pa",
                        "dialect": "ba",
                        "meaning": "test example",
                    }
                ],
            }
        ],
    }


def _validate_document(tmp_path: Path, document: RuleDocument) -> list[object]:
    """Write document under tmp_path and return validate_rule_file errors."""
    path = tmp_path / "rules.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return validate_rule_files.validate_rule_file(path, _rule_validator())


def _run_main(monkeypatch: pytest.MonkeyPatch, *args: Path | str) -> int:
    """Set sys.argv from args via monkeypatch and return main()'s exit code."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_rule_files.py", *(str(arg) for arg in args)],
    )
    return validate_rule_files.main()


def test_load_schema_reports_missing_file_with_context(tmp_path: Path) -> None:
    schema_path = tmp_path / "missing.schema.json"

    with pytest.raises(RuntimeError) as exc_info:
        validate_rule_files.load_schema(schema_path)

    message = str(exc_info.value)
    assert f"Error loading schema {schema_path}:" in message
    assert "FileNotFoundError:" in message


def test_load_schema_reports_invalid_json_with_context(tmp_path: Path) -> None:
    schema_path = tmp_path / "invalid.schema.json"
    schema_path.write_text("{invalid", encoding="utf-8")

    with pytest.raises(RuntimeError) as exc_info:
        validate_rule_files.load_schema(schema_path)

    message = str(exc_info.value)
    assert f"Error loading schema {schema_path}:" in message
    assert "JSONDecodeError:" in message


def test_load_schema_reports_os_error_with_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema_path = Path("schema.json")

    def raise_os_error(*args: object, **kwargs: object) -> object:
        raise OSError("no access")

    monkeypatch.setattr(builtins, "open", raise_os_error)

    with pytest.raises(RuntimeError) as exc_info:
        validate_rule_files.load_schema(schema_path)

    message = str(exc_info.value)
    assert f"Error loading schema {schema_path}:" in message
    assert "OSError: no access" in message


def test_validate_rule_file_reports_empty_yaml_document(
    tmp_path: Path,
) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")

    errors = validate_rule_files.validate_rule_file(path, Draft202012Validator({}))

    assert len(errors) == 1
    assert isinstance(errors[0], ValidationError)
    assert errors[0].message == "Empty or invalid YAML document"


def test_validate_rule_file_reports_missing_file_as_file_read_error(
    tmp_path: Path,
) -> None:
    validator = Draft202012Validator({})

    errors = validate_rule_files.validate_rule_file(
        tmp_path / "missing.yaml",
        validator,
    )

    assert len(errors) == 1
    assert isinstance(errors[0], validate_rule_files.FileReadError)
    assert str(errors[0]).startswith("Error reading file: FileNotFoundError:")


def test_validate_rule_file_reports_yaml_parse_error_as_file_read_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("rules: [unterminated", encoding="utf-8")
    validator = Draft202012Validator({})

    errors = validate_rule_files.validate_rule_file(path, validator)

    assert len(errors) == 1
    assert isinstance(errors[0], validate_rule_files.FileReadError)
    assert str(errors[0]).startswith("YAML parse error:")


def test_validate_rule_file_reports_yaml_error_from_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("rules: []", encoding="utf-8")

    def raise_yaml_error(stream: object) -> object:
        raise yaml.YAMLError("bad yaml")

    monkeypatch.setattr(validate_rule_files.yaml, "safe_load", raise_yaml_error)

    errors = validate_rule_files.validate_rule_file(path, Draft202012Validator({}))

    assert len(errors) == 1
    assert isinstance(errors[0], validate_rule_files.FileReadError)
    assert (
        validate_rule_files.format_error(errors[0])
        == "  - root: YAML parse error: bad yaml"
    )


def test_validate_rule_file_reports_os_error_from_open(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = Path("rules.yaml")

    def raise_os_error(*args: object, **kwargs: object) -> object:
        raise OSError("read failed")

    monkeypatch.setattr(builtins, "open", raise_os_error)
    caplog.set_level(logging.WARNING, logger=validate_rule_files.logger.name)

    errors = validate_rule_files.validate_rule_file(path, Draft202012Validator({}))

    assert len(errors) == 1
    assert isinstance(errors[0], validate_rule_files.FileReadError)
    assert str(errors[0]) == "Error reading file: OSError: read failed"
    assert "Error reading rule file: rules.yaml" in caplog.text


def test_validate_rule_file_formats_schema_validation_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "rules.yaml"
    path.write_text("rules: []", encoding="utf-8")
    schema_error = ValidationError("'meta' is a required property")
    schema_error.path.append("meta")

    class StubValidator:
        def iter_errors(self, document: object) -> Iterator[ValidationError]:
            assert document == {"rules": []}
            return iter([schema_error])

    monkeypatch.setattr(
        validate_rule_files.yaml,
        "safe_load",
        lambda stream: {"rules": []},
    )

    errors = validate_rule_files.validate_rule_file(path, StubValidator())

    assert errors == [schema_error]
    assert (
        validate_rule_files.format_error(errors[0])
        == "  - meta: 'meta' is a required property"
    )


def test_format_error_formats_file_read_error_with_root_path() -> None:
    error = validate_rule_files.FileReadError(
        "Error reading file: PermissionError: denied"
    )

    assert (
        validate_rule_files.format_error(error)
        == "  - root: Error reading file: PermissionError: denied"
    )


def test_main_reports_no_yaml_files_as_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()

    exit_code = _run_main(
        monkeypatch,
        "--schema",
        RULE_SCHEMA_PATH,
        "--rules-dir",
        rules_dir,
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert f"No .yaml or .yml files found in {rules_dir}" in captured.err


def test_main_reports_missing_schema_as_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    schema_path = tmp_path / "missing.schema.json"

    exit_code = _run_main(
        monkeypatch,
        "--schema",
        schema_path,
        "--rules-dir",
        rules_dir,
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert f"Error: Schema file not found or not a file: {schema_path}" in captured.err
    assert captured.out == ""


def test_main_reports_schema_directory_as_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    schema_dir = tmp_path / "schema-dir"
    schema_dir.mkdir()

    exit_code = _run_main(
        monkeypatch,
        "--schema",
        schema_dir,
        "--rules-dir",
        rules_dir,
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert f"Error: Schema file not found or not a file: {schema_dir}" in captured.err
    assert captured.out == ""


def test_main_reports_invalid_schema_json_as_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    schema_path = tmp_path / "invalid.schema.json"
    schema_path.write_text("{invalid", encoding="utf-8")

    exit_code = _run_main(
        monkeypatch,
        "--schema",
        schema_path,
        "--rules-dir",
        rules_dir,
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert f"Error: Error loading schema {schema_path}:" in captured.err
    assert "JSONDecodeError:" in captured.err
    assert captured.out == ""


def test_main_reports_load_schema_runtime_error_as_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    schema_path = tmp_path / "schema.json"
    schema_path.write_text("{}", encoding="utf-8")

    def raise_runtime_error(_path: Path) -> object:
        raise RuntimeError("load failed")

    monkeypatch.setattr(
        validate_rule_files,
        "load_schema",
        raise_runtime_error,
    )

    exit_code = _run_main(
        monkeypatch,
        "--schema",
        schema_path,
        "--rules-dir",
        rules_dir,
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err == "Error: load failed\n"
    assert captured.out == ""


def test_main_reports_rules_dir_file_as_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rules_dir = tmp_path / "rules.yaml"
    rules_dir.write_text("not a directory\n", encoding="utf-8")

    exit_code = _run_main(
        monkeypatch,
        "--schema",
        RULE_SCHEMA_PATH,
        "--rules-dir",
        rules_dir,
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert (
        f"Error: Rules directory not found or not a directory: {rules_dir}"
        in captured.err
    )
    assert captured.out == ""


def test_main_reports_valid_inputs_as_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    rule_path = rules_dir / "rules.yaml"
    rule_path.write_text(
        yaml.safe_dump(_valid_rule_document(), sort_keys=False),
        encoding="utf-8",
    )

    exit_code = _run_main(
        monkeypatch,
        "--schema",
        RULE_SCHEMA_PATH,
        "--rules-dir",
        rules_dir,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert f"OK: {rule_path}" in captured.out
    assert "1 files checked, 0 errors found" in captured.out
    assert captured.err == ""


def test_main_reports_rule_validation_errors_as_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    rule_path = rules_dir / "rules.yaml"
    document = _valid_rule_document()
    document["rules"][0].pop("name_en")
    rule_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    exit_code = _run_main(
        monkeypatch,
        "--schema",
        RULE_SCHEMA_PATH,
        "--rules-dir",
        rules_dir,
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert f"FAIL: {rule_path}" in captured.out
    assert "'name_en' is a required property" in captured.out
    assert "1 files checked, 1 errors found" in captured.out
    assert captured.err == ""


def test_main_ignores_nested_rule_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rules_dir = tmp_path / "rules"
    nested_dir = rules_dir / "nested"
    nested_dir.mkdir(parents=True)
    rule_path = nested_dir / "rules.yml"
    rule_path.write_text(
        yaml.safe_dump(_valid_rule_document(), sort_keys=False),
        encoding="utf-8",
    )

    exit_code = _run_main(
        monkeypatch,
        "--schema",
        RULE_SCHEMA_PATH,
        "--rules-dir",
        rules_dir,
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert f"No .yaml or .yml files found in {rules_dir}" in captured.err


@pytest.mark.parametrize(
    ("mutate", "expected_message"),
    [
        (
            lambda document: document["rules"][0].pop("name_en"),
            "'name_en' is a required property",
        ),
        (
            lambda document: document["rules"][0].__setitem__("references", []),
            "[] should be non-empty",
        ),
        (
            lambda document: document["meta"].__setitem__("citation_ready", "false"),
            "'false' is not of type 'boolean'",
        ),
        (
            lambda document: document["rules"][0].__setitem__(
                "change_type", "assimilation"
            ),
            "'assimilation' is not one of",
        ),
        (
            lambda document: document["rules"][0].__setitem__("output", ""),
            "'change_type' is a required property",
        ),
        (
            lambda document: document["rules"][0].__setitem__(
                "future_rule_key", "ignored"
            ),
            "Additional properties are not allowed",
        ),
        (
            lambda document: document["rules"][0].update(
                {"change_type": "retention", "output": ""}
            ),
            "should be non-empty",
        ),
    ],
)
def test_rule_schema_rejects_invalid_documents(
    tmp_path: Path,
    mutate: Callable[[RuleDocument], None],
    expected_message: str,
) -> None:
    document = _valid_rule_document()
    mutate(document)

    errors = _validate_document(tmp_path, document)

    messages = [error.message for error in errors if isinstance(error, ValidationError)]
    assert any(expected_message in message for message in messages)


def test_rule_schema_allows_extra_meta_and_example_fields(tmp_path: Path) -> None:
    """Rule metadata and examples should remain extensible for future data."""
    document = _valid_rule_document()
    document["meta"]["future_field"] = "ignored"
    document["rules"][0]["examples"][0]["future_example_key"] = "ignored"

    assert _validate_document(tmp_path, document) == []


def test_rule_schema_allows_empty_output_for_explicit_deletion(
    tmp_path: Path,
) -> None:
    document = _valid_rule_document()
    rule = document["rules"][0]
    rule["output"] = ""
    rule["change_type"] = "deletion"

    assert _validate_document(tmp_path, document) == []


def test_rule_schema_rejects_non_empty_output_for_deletion(
    tmp_path: Path,
) -> None:
    document = _valid_rule_document()
    rule = document["rules"][0]
    rule["change_type"] = "deletion"
    rule["output"] = "non-empty"

    errors = _validate_document(tmp_path, document)

    assert errors
    # Multiple error-message patterns are accepted because different JSON Schema
    # validators/versions or localization may produce different messages:
    #   - "'' was expected" (strict string pattern validation)
    #   - "empty" (type-based validation)
    #   - "not allowed" (enum/constraint validation)
    # Any of these indicate the "output" field was rejected for a non-empty value.
    assert any(
        list(e.path) == ["rules", 0, "output"]
        and (
            "'' was expected" in e.message
            or "empty" in e.message.lower()
            or "not allowed" in e.message.lower()
        )
        for e in errors
        if isinstance(e, ValidationError)
    )


def test_rule_schema_rejects_empty_output_for_retention(
    tmp_path: Path,
) -> None:
    document = _valid_rule_document()
    rule = document["rules"][0]
    rule["change_type"] = "retention"
    rule["output"] = ""

    errors = _validate_document(tmp_path, document)

    assert errors
    # jsonschema 4.x commonly emits "'' should be non-empty" or
    # "'' is too short" for minLength violations; accept any message that
    # signals the output failed length validation.
    assert any(
        list(e.path) == ["rules", 0, "output"]
        and (
            "non-empty" in e.message.lower()
            or "too short" in e.message.lower()
            or "minlength" in e.message.lower()
        )
        for e in errors
        if isinstance(e, ValidationError)
    )


def test_rule_schema_allows_non_empty_output_for_retention(
    tmp_path: Path,
) -> None:
    """Retention rules with a non-empty output should validate cleanly."""
    document = _valid_rule_document()
    rule = document["rules"][0]
    rule["change_type"] = "retention"
    rule["output"] = "p"

    assert _validate_document(tmp_path, document) == []
