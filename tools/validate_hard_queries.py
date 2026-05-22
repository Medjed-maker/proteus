"""Validate Phase 3 hard query case files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

from jsonschema import Draft202012Validator
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA_PATH = REPO_ROOT / "data" / "schemas" / "hard_query_case.schema.json"
PRIVATE_VISIBILITIES = {"private_collaborator", "embargoed"}
URL_LIKE_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
FORBIDDEN_TEXT_KEYS = {"source_text", "evidence_excerpt", "quote", "quotation"}
# Keys whose URL policy is enforced by the JSON Schema's own pattern. The
# validator skips its `_iter_strings` URL check on these to avoid duplicate
# errors; defense-in-depth coverage is preserved by the schema pattern.
SCHEMA_URL_PATTERN_KEYS = {"source_id", "short_citation"}


class HardQueryValidationError(ValueError):
    """Raised when a hard query file cannot be validated."""


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Validate Proteus Phase 3 hard query case files."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="YAML or JSON hard query case files to validate.",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=DEFAULT_SCHEMA_PATH,
        help="Path to the hard query case JSON Schema.",
    )
    parser.add_argument(
        "--public-only",
        action="store_true",
        help="Reject private_collaborator and embargoed cases.",
    )
    parser.add_argument(
        "--min-cases",
        type=int,
        default=None,
        help="Require at least this many total cases across all files.",
    )
    return parser.parse_args()


def _load_document(path: Path) -> Any:
    """Load a YAML or JSON document."""
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise HardQueryValidationError(f"{path}: cannot read file: {exc}") from exc

    try:
        if path.suffix.lower() == ".json":
            return json.loads(raw_text)
        return yaml.safe_load(raw_text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise HardQueryValidationError(f"{path}: cannot parse document: {exc}") from exc


def _extract_cases(document: Any, *, path: Path) -> list[dict[str, Any]]:
    """Return hard query cases from a loaded single case, list, or wrapper."""
    cases: Any
    if isinstance(document, list):
        cases = document
    elif isinstance(document, dict) and "cases" in document:
        cases = document.get("cases")
    elif isinstance(document, dict) and "case_id" in document:
        cases = [document]
    else:
        cases = None
    if not isinstance(cases, list):
        raise HardQueryValidationError(
            f"{path}: expected a single case object, a list of cases, "
            "or an object with a 'cases' list"
        )

    normalized_cases: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise HardQueryValidationError(f"{path}: case {index} must be an object")
        normalized_cases.append(case)
    return normalized_cases


def load_cases(path: Path) -> list[dict[str, Any]]:
    """Return hard query cases from a single case, list, or ``cases`` object."""
    return _extract_cases(_load_document(path), path=path)


def load_schema(path: Path = DEFAULT_SCHEMA_PATH) -> dict[str, Any]:
    """Load the hard query case JSON Schema."""
    schema = _load_document(path)
    if not isinstance(schema, dict):
        raise HardQueryValidationError(f"{path}: schema must be an object")
    return schema


def _iter_strings(value: Any, *, path: str = "$") -> list[tuple[str, str]]:
    """Return string leaves with dotted paths for policy checks.

    Skips dict values whose key is in :data:`SCHEMA_URL_PATTERN_KEYS` because
    the JSON Schema already enforces a URL-rejection pattern on those fields;
    yielding them here would surface duplicate errors for the same violation.
    """
    if isinstance(value, str):
        return [(path, value)]
    if isinstance(value, list):
        strings: list[tuple[str, str]] = []
        for index, item in enumerate(value):
            strings.extend(_iter_strings(item, path=f"{path}[{index}]"))
        return strings
    if isinstance(value, dict):
        strings = []
        for key, item in value.items():
            if key in SCHEMA_URL_PATTERN_KEYS:
                continue
            strings.extend(_iter_strings(item, path=f"{path}.{key}"))
        return strings
    return []


def _iter_forbidden_keys(value: Any, *, path: str = "$") -> list[str]:
    """Return paths containing forbidden source-text fields."""
    if isinstance(value, list):
        found: list[str] = []
        for index, item in enumerate(value):
            found.extend(_iter_forbidden_keys(item, path=f"{path}[{index}]"))
        return found
    if isinstance(value, dict):
        found = []
        for key, item in value.items():
            child_path = f"{path}.{key}"
            if key in FORBIDDEN_TEXT_KEYS:
                found.append(child_path)
            found.extend(_iter_forbidden_keys(item, path=child_path))
        return found
    return []


def _iter_all_strings(value: Any, *, path: str = "$") -> list[tuple[str, str]]:
    """Return all string leaves with dotted paths for document-wide checks."""
    if isinstance(value, str):
        return [(path, value)]
    if isinstance(value, list):
        strings: list[tuple[str, str]] = []
        for index, item in enumerate(value):
            strings.extend(_iter_all_strings(item, path=f"{path}[{index}]"))
        return strings
    if isinstance(value, dict):
        strings = []
        for key, item in value.items():
            strings.extend(_iter_all_strings(item, path=f"{path}.{key}"))
        return strings
    return []


def validate_document_policy(document: Any, *, source_label: str) -> list[str]:
    """Validate repository-safety policy against the entire loaded document."""
    errors: list[str] = []
    for forbidden_path in _iter_forbidden_keys(document):
        errors.append(
            f"{source_label}:document:{forbidden_path}: long source text fields are not allowed"
        )

    for string_path, text in _iter_all_strings(document):
        if URL_LIKE_RE.search(text):
            errors.append(
                f"{source_label}:document:{string_path}: URLs are not allowed in hard query document text"
            )
    return errors


def validate_cases(
    cases: list[dict[str, Any]],
    *,
    schema: dict[str, Any] | None = None,
    public_only: bool = False,
    source_label: str = "<memory>",
) -> list[str]:
    """Validate cases and return human-readable error messages."""
    validator = Draft202012Validator(
        schema or load_schema(),
        format_checker=Draft202012Validator.FORMAT_CHECKER,
    )
    errors: list[str] = []

    for index, case in enumerate(cases):
        case_id = case.get("case_id", f"case[{index}]")
        for schema_error in sorted(
            validator.iter_errors(case), key=lambda error: list(error.path)
        ):
            location = ".".join(str(part) for part in schema_error.path) or "$"
            errors.append(f"{source_label}:{case_id}:{location}: {schema_error.message}")

        visibility = case.get("visibility")
        if public_only and visibility in PRIVATE_VISIBILITIES:
            errors.append(
                f"{source_label}:{case_id}: visibility {visibility!r} is not allowed in public-only mode"
            )

        for forbidden_path in _iter_forbidden_keys(case):
            errors.append(
                f"{source_label}:{case_id}:{forbidden_path}: long source text fields are not allowed"
            )

        for string_path, text in _iter_strings(case):
            if URL_LIKE_RE.search(text):
                errors.append(
                    f"{source_label}:{case_id}:{string_path}: URLs are not allowed in hard query case text"
                )
    return errors


def validate_files(
    paths: list[Path],
    *,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    public_only: bool = False,
    min_cases: int | None = None,
) -> list[str]:
    """Validate hard query files and return all errors."""
    schema = load_schema(schema_path)
    errors: list[str] = []
    total_cases = 0
    for path in paths:
        try:
            document = _load_document(path)
            errors.extend(validate_document_policy(document, source_label=str(path)))
            cases = _extract_cases(document, path=path)
        except HardQueryValidationError as exc:
            errors.append(str(exc))
            continue
        total_cases += len(cases)
        errors.extend(
            validate_cases(
                cases,
                schema=schema,
                public_only=public_only,
                source_label=str(path),
            )
        )

    if min_cases is not None and total_cases < min_cases:
        errors.append(
            f"expected at least {min_cases} hard query cases, found {total_cases}"
        )
    return errors


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    if args.min_cases is not None and args.min_cases < 1:
        print("--min-cases must be >= 1", file=sys.stderr)
        return 2

    errors = validate_files(
        args.paths,
        schema_path=args.schema,
        public_only=args.public_only,
        min_cases=args.min_cases,
    )
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"validated {len(args.paths)} hard query file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
