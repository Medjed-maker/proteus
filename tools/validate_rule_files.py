#!/usr/bin/env python3
"""Standalone validation tool for phonology rule YAML files.

This tool validates rule files against the JSON Schema and prints
file-level validation errors.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, TypeAlias

import yaml
from jsonschema import Draft202012Validator, FormatChecker, ValidationError


logger = logging.getLogger(__name__)


class FileReadError(Exception):
    """Rule file IO failure reported by the validation tool."""


RuleFileError: TypeAlias = ValidationError | FileReadError


def load_schema(schema_path: Path) -> dict[str, Any]:
    """Load a JSON Schema document from disk.

    Args:
        schema_path: Path to the JSON Schema file to read as UTF-8 JSON.

    Returns:
        Parsed schema data as a dictionary suitable for jsonschema validators.

    Raises:
        RuntimeError: If the file cannot be read, or if the file content is not
            valid JSON. The original OSError or JSONDecodeError is chained.
    """
    try:
        with open(schema_path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Error loading schema {schema_path}: {exc.__class__.__name__}: {exc}"
        ) from exc


def validate_rule_file(
    file_path: Path,
    validator: Draft202012Validator,
) -> list[RuleFileError]:
    """Validate a single phonology rule YAML file.

    Args:
        file_path: Path to the YAML rule file to read and validate.
        validator: JSON Schema validator used to produce ValidationError
            instances for schema violations.

    Returns:
        A list of rule-file errors. YAML parse and file-read failures are
        returned as FileReadError; empty documents and schema failures are
        returned as ValidationError.

    Raises:
        No exceptions are expected for YAML parse errors, OSError read failures,
        empty documents, or schema validation errors; they are returned in the
        error list instead.
    """
    errors: list[str] = []
    try:
        with open(file_path, encoding="utf-8") as f:
            document = yaml.safe_load(f)
        if document is None:
            errors.append(ValidationError("Empty or invalid YAML document"))
            return errors

        errors.extend(validator.iter_errors(document))
    except yaml.YAMLError as exc:
        errors.append(FileReadError(f"YAML parse error: {exc}"))
    except OSError as exc:
        logger.warning(
            "Error reading rule file: %s (%s): %s",
            file_path,
            exc.__class__.__name__,
            exc,
        )
        errors.append(
            FileReadError(f"Error reading file: {exc.__class__.__name__}: {exc}")
        )

    return errors


def format_error(error: RuleFileError) -> str:
    """Format a rule-file error for CLI output.

    Args:
        error: ValidationError or FileReadError returned by validate_rule_file.

    Returns:
        A human-readable string containing the schema path, or root for
        file-level errors, and the associated error message.

    Raises:
        No exceptions are raised for supported RuleFileError values.
    """
    if isinstance(error, ValidationError):
        path = (
            "/".join(str(p) for p in error.absolute_path)
            if error.absolute_path
            else "root"
        )
        message = error.message
    else:
        path = "root"
        message = str(error)
    return f"  - {path}: {message}"


# CLI: keep stdout/stderr direct for human consumption; only unexpected file-read
# failures are logged by validate_rule_file().
def main() -> int:
    """Main entry point for the validation tool.

    Returns:
        Exit code: 0 if all files are valid, 1 otherwise.
    """
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s:%(name)s:%(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Validate phonology rule YAML files against JSON Schema"
    )
    parser.add_argument(
        "--rules-dir",
        type=Path,
        default=Path("data/languages/ancient_greek/rules"),
        help="Directory containing rule YAML files (default: data/languages/ancient_greek/rules)",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("data/schemas/phonology_rule_file.schema.json"),
        help="Path to JSON Schema file (default: data/schemas/phonology_rule_file.schema.json)",
    )
    args = parser.parse_args()

    if not args.schema.is_file():
        print(
            f"Error: Schema file not found or not a file: {args.schema}",
            file=sys.stderr,
        )
        return 1

    if not args.rules_dir.is_dir():
        print(
            f"Error: Rules directory not found or not a directory: {args.rules_dir}",
            file=sys.stderr,
        )
        return 1

    try:
        schema = load_schema(args.schema)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    validator = Draft202012Validator(schema, format_checker=FormatChecker())

    rule_files = sorted(
        path
        for pattern in ("*.yaml", "*.yml")
        for path in args.rules_dir.glob(pattern)
        if path.is_file()
    )
    if not rule_files:
        print(f"No .yaml or .yml files found in {args.rules_dir}", file=sys.stderr)
        return 1

    all_valid = True
    total_errors = 0

    for file_path in rule_files:
        errors = validate_rule_file(file_path, validator)
        if errors:
            all_valid = False
            total_errors += len(errors)
            print(f"FAIL: {file_path}")
            for error in errors:
                print(format_error(error))
        else:
            print(f"OK: {file_path}")

    print(f"\n{len(rule_files)} files checked, {total_errors} errors found")
    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())
