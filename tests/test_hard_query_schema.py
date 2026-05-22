"""Tests for Phase 3 hard query validation and evaluation helpers."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from tools import evaluate_hard_queries, validate_hard_queries


REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SEED_PATH = (
    REPO_ROOT / "data" / "evaluation" / "hard_queries" / "public_seed_cases.yaml"
)
VALIDATOR_SCRIPT = REPO_ROOT / "tools" / "validate_hard_queries.py"
SUBPROCESS_TIMEOUT_SECONDS = 60


def _valid_case(**overrides: object) -> dict[str, Any]:
    case: dict[str, Any] = {
        "case_id": "hq-test-0001",
        "visibility": "public_seed",
        "input_form": "λόγος",
        "language": "ancient_greek",
        "dialect_hint": "attic",
        "max_candidates": 5,
        "expected_candidates": [
            {
                "headword": "λόγος",
                "acceptable_forms": [],
                "note": "Control expected candidate.",
            }
        ],
        "reasoning": "Control case for validator and evaluator tests.",
        "source_notes": [
            {
                "source_id": "lsj-logos-control",
                "short_citation": "LSJ, λόγος",
                "note": "Short public source note without source text.",
            }
        ],
        "existing_search_failure": {
            "tool": "not applicable",
            "query": "λόγος",
            "failure_type": "not_yet_checked",
            "notes": "Control case.",
        },
        "manual_search": {
            "checked_by": "test",
            "checked_at": "2026-05-22",
            "summary": "Packaged lexicon contains the expected headword.",
        },
        "tool_assisted_search": {
            "checked_at": "2026-05-22",
            "summary": "Expected to return the headword.",
        },
        "review_status": "collected",
        "reviewer": "",
        "false_positive_notes": [],
        "false_negative_notes": [],
    }
    case.update(overrides)
    return case


def _write_cases(path: Path, cases: list[dict[str, Any]]) -> None:
    path.write_text(
        yaml.safe_dump({"cases": cases}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def test_public_seed_cases_validate_in_public_only_mode() -> None:
    """Committed public seed data should satisfy the hard-query schema."""
    cases = validate_hard_queries.load_cases(PUBLIC_SEED_PATH)

    errors = validate_hard_queries.validate_cases(cases, public_only=True)

    assert errors == []


def test_missing_required_field_is_rejected() -> None:
    """Schema validation should reject incomplete cases."""
    case = _valid_case()
    del case["expected_candidates"]

    errors = validate_hard_queries.validate_cases([case], public_only=True)

    assert any("'expected_candidates' is a required property" in error for error in errors)


@pytest.mark.parametrize("visibility", ["private_collaborator", "embargoed"])
def test_public_only_rejects_private_visibility(visibility: str) -> None:
    """Public validation should stop private or embargoed cases from leaking."""
    case = _valid_case(visibility=visibility)

    errors = validate_hard_queries.validate_cases([case], public_only=True)

    assert any("is not allowed in public-only mode" in error for error in errors)


def test_source_note_urls_are_rejected_by_schema() -> None:
    """The schema pattern should reject URLs in source identifiers."""
    case = _valid_case()
    case["source_notes"][0]["source_id"] = "https://example.test/record"

    errors = validate_hard_queries.validate_cases([case], public_only=True)

    assert any("source_id" in error and "does not match" in error for error in errors)


def test_uppercase_url_is_rejected_by_schema() -> None:
    """The schema pattern uses an inline ``(?i)`` flag and rejects upper-case URLs."""
    case = _valid_case()
    case["source_notes"][0]["source_id"] = "HTTPS://EXAMPLE.TEST/record"

    errors = validate_hard_queries.validate_cases([case], public_only=True)

    assert any("source_id" in error and "does not match" in error for error in errors)


def test_validator_rejects_urls_outside_schema_protected_fields() -> None:
    """Defense in depth: validator catches URLs in fields without a schema pattern."""
    case = _valid_case()
    case["reasoning"] = "See https://example.test for the published note."

    errors = validate_hard_queries.validate_cases([case], public_only=True)

    assert any(
        "reasoning" in error and "URLs are not allowed" in error for error in errors
    )


def test_validator_does_not_duplicate_schema_url_error() -> None:
    """Schema and validator should not both report URLs in source_id / short_citation."""
    case = _valid_case()
    case["source_notes"][0]["short_citation"] = "https://example.test/citation"

    errors = validate_hard_queries.validate_cases([case], public_only=True)

    # Schema pattern still flags the violation as a field-level format error...
    assert any(
        "short_citation" in error and "does not match" in error for error in errors
    )
    # ...but the validator-level "URLs are not allowed" rule must not double-report
    # the same path covered by the schema pattern.
    assert not any(
        "short_citation" in error and "URLs are not allowed" in error
        for error in errors
    )


def test_long_source_text_fields_are_rejected() -> None:
    """Runtime cases should store identifiers and summaries, not source excerpts."""
    case = _valid_case()
    case["source_text"] = "Do not commit source text excerpts here."

    errors = validate_hard_queries.validate_cases([case], public_only=True)

    assert any("long source text fields are not allowed" in error for error in errors)


@pytest.mark.parametrize(
    ("section", "expected_path"),
    [
        ("manual_search", "manual_search.checked_at"),
        ("tool_assisted_search", "tool_assisted_search.checked_at"),
    ],
)
def test_checked_at_fields_must_be_valid_dates(
    section: str, expected_path: str
) -> None:
    """Schema format checks should reject impossible checked_at dates."""
    case = _valid_case()
    case[section]["checked_at"] = "2026-99-99"

    errors = validate_hard_queries.validate_cases([case], public_only=True)

    assert any(expected_path in error and "is not a 'date'" in error for error in errors)


def test_load_cases_accepts_single_case_document(tmp_path: Path) -> None:
    """A one-case collection file should match the published template shape."""
    case = _valid_case()
    cases_path = tmp_path / "single_case.yaml"
    cases_path.write_text(
        yaml.safe_dump(case, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    cases = validate_hard_queries.load_cases(cases_path)

    assert cases == [case]


def test_load_cases_rejects_unknown_mapping_shape(tmp_path: Path) -> None:
    """Mappings must be either a single case or a wrapper with a cases list."""
    cases_path = tmp_path / "invalid_shape.yaml"
    cases_path.write_text(
        yaml.safe_dump({"metadata": {"dataset_id": "missing-cases"}}, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(validate_hard_queries.HardQueryValidationError) as exc_info:
        validate_hard_queries.load_cases(cases_path)

    assert "expected a single case object, a list of cases" in str(exc_info.value)


def test_validate_files_rejects_urls_in_wrapper_metadata(tmp_path: Path) -> None:
    """Public file validation should inspect metadata before extracting cases."""
    cases_path = tmp_path / "cases.yaml"
    cases_path.write_text(
        yaml.safe_dump(
            {
                "metadata": {"dataset_url": "https://example.test/private-log"},
                "cases": [_valid_case()],
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    errors = validate_hard_queries.validate_files([cases_path], public_only=True)

    assert any(
        ":document:$.metadata.dataset_url:" in error
        and "URLs are not allowed" in error
        for error in errors
    )


def test_validate_files_rejects_source_text_in_wrapper_metadata(tmp_path: Path) -> None:
    """Document-wide policy checks should cover non-case wrapper fields."""
    cases_path = tmp_path / "cases.yaml"
    cases_path.write_text(
        yaml.safe_dump(
            {
                "metadata": {"source_text": "Do not publish source excerpts here."},
                "cases": [_valid_case()],
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    errors = validate_hard_queries.validate_files([cases_path], public_only=True)

    assert any(
        ":document:$.metadata.source_text:" in error
        and "long source text fields are not allowed" in error
        for error in errors
    )


def test_validate_hard_queries_cli_public_only_smoke(tmp_path: Path) -> None:
    """The public-only validator CLI should accept valid public files."""
    cases_path = tmp_path / "cases.yaml"
    _write_cases(cases_path, [_valid_case()])

    completed = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR_SCRIPT),
            "--public-only",
            str(cases_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )

    assert completed.returncode == 0, completed.stderr
    assert "validated 1 hard query file" in completed.stdout


def test_validate_hard_queries_cli_accepts_single_case_document(
    tmp_path: Path,
) -> None:
    """The validator CLI should accept one-case files from the template."""
    cases_path = tmp_path / "single_case.yaml"
    cases_path.write_text(
        yaml.safe_dump(_valid_case(), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR_SCRIPT),
            "--public-only",
            str(cases_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )

    assert completed.returncode == 0, completed.stderr
    assert "validated 1 hard query file" in completed.stdout


def test_validate_hard_queries_cli_min_cases(tmp_path: Path) -> None:
    """The collection target can be checked with --min-cases."""
    cases_path = tmp_path / "cases.yaml"
    _write_cases(cases_path, [_valid_case()])

    completed = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR_SCRIPT),
            "--min-cases",
            "2",
            str(cases_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )

    assert completed.returncode == 1
    assert "expected at least 2 hard query cases, found 1" in completed.stderr


def test_evaluate_hard_queries_reports_wrong_candidates_between_expected_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-expected hits before any expected hit should be reported."""
    case = _valid_case(case_id="hq-test-multiple-expected")
    case["expected_candidates"] = [
        {
            "headword": "λόγος",
            "acceptable_forms": [],
            "note": "First expected hit.",
        },
        {
            "headword": "late-primary",
            "acceptable_forms": ["late-accepted"],
            "note": "Second expected hit matched by an accepted form.",
        },
    ]

    def fake_hit(headword: str) -> SimpleNamespace:
        return SimpleNamespace(
            headword=headword,
            confidence="high",
            distance=0.0,
            match_type="exact",
            candidate_bucket="strong",
            rules_applied=[],
            orthographic_notes=[],
        )

    def fake_run_search(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            response=SimpleNamespace(
                hits=[
                    fake_hit("λόγος"),
                    fake_hit("wrong-mid"),
                    fake_hit("other-mid"),
                    fake_hit("filler-mid"),
                    fake_hit("late-accepted"),
                ],
                query_ipa="logos",
                query_mode="lexical",
                meta=SimpleNamespace(
                    request_id="test-request",
                    engine_version="test-engine",
                    schema_version="test-schema",
                ),
            )
        )

    monkeypatch.setattr(evaluate_hard_queries, "run_search", fake_run_search)
    monkeypatch.setattr(evaluate_hard_queries, "build_ruleset_versions", lambda _language: {})

    report = evaluate_hard_queries.evaluate_case(
        case,
        deps_cache={
            "ancient_greek": SimpleNamespace(
                profile=SimpleNamespace(language_id="ancient_greek")
            )
        },
    )

    assert report["matched"] is True
    assert [hit["rank"] for hit in report["wrong_outranking_candidates"]] == [2, 3, 4]
    assert all(
        hit["headword"] != "late-accepted"
        for hit in report["non_expected_top_candidates"]
    )


@pytest.mark.integration
def test_evaluate_hard_queries_reports_expected_hit_and_miss() -> None:
    """The evaluator should record expected-candidate hits and misses."""
    hit_case = _valid_case(case_id="hq-test-hit")
    miss_case = copy.deepcopy(hit_case)
    miss_case["case_id"] = "hq-test-miss"
    miss_case["expected_candidates"] = [
        {
            "headword": "παιδίον",
            "acceptable_forms": [],
            "note": "Deliberate miss for evaluator regression coverage.",
        }
    ]

    report = evaluate_hard_queries.evaluate_cases(
        [hit_case, miss_case],
        max_candidates_override=3,
    )

    assert report["summary"]["case_count"] == 2
    assert report["summary"]["matched_count"] == 1
    assert report["summary"]["missed_count"] == 1
    hit_result, miss_result = report["cases"]
    assert hit_result["matched"] is True
    assert hit_result["matched_expected_candidates"] == [
        {"headword": "λόγος", "matched": True, "rank": 1}
    ]
    # Hit case: λόγος at rank 1 means no candidate outranks the expected one.
    assert hit_result["wrong_outranking_candidates"] == []
    # "false_positive_candidates" has been replaced by the strict
    # "wrong_outranking_candidates" field and the informational
    # "non_expected_top_candidates" field.
    assert "false_positive_candidates" not in hit_result
    assert miss_result["matched"] is False
    assert miss_result["false_negative_candidates"] == ["παιδίον"]
    # When no expected candidate is matched, every non-expected top hit
    # outranks every missing expected entry by definition.
    assert miss_result["non_expected_top_candidates"]
    assert miss_result["wrong_outranking_candidates"]
    assert (
        miss_result["wrong_outranking_candidates"]
        == miss_result["non_expected_top_candidates"]
    )


@pytest.mark.integration
def test_evaluate_hard_queries_reports_partial_match_wrong_candidates() -> None:
    """Partial expected matches should still report top-N wrong candidates."""
    case = _valid_case(case_id="hq-test-partial")
    case["expected_candidates"] = [
        {
            "headword": "λόγος",
            "acceptable_forms": [],
            "note": "Expected hit for the partial-match regression case.",
        },
        {
            "headword": "παιδίον",
            "acceptable_forms": [],
            "note": "Deliberate miss for the partial-match regression case.",
        },
    ]

    report = evaluate_hard_queries.evaluate_cases([case], max_candidates_override=3)

    (result,) = report["cases"]
    assert result["matched"] is False
    assert result["false_negative_candidates"] == ["παιδίον"]
    assert result["non_expected_top_candidates"]
    assert (
        result["wrong_outranking_candidates"]
        == result["non_expected_top_candidates"]
    )


@pytest.mark.integration
def test_evaluate_hard_queries_accepts_acceptable_forms() -> None:
    """acceptable_forms should count as alternative matches for the headword."""
    case = _valid_case(case_id="hq-test-acceptable")
    # Primary headword is intentionally absent from the lexicon, but the
    # acceptable_forms list contains λόγος, which the runtime does return.
    case["expected_candidates"] = [
        {
            "headword": "not-in-lexicon-headword",
            "acceptable_forms": ["λόγος"],
            "note": "Tests that acceptable_forms count as matches.",
        }
    ]

    report = evaluate_hard_queries.evaluate_cases([case], max_candidates_override=3)

    (result,) = report["cases"]
    assert result["matched"] is True
    assert result["matched_expected_candidates"] == [
        {"headword": "not-in-lexicon-headword", "matched": True, "rank": 1}
    ]
    assert result["false_negative_candidates"] == []
    # The hit for λόγος belongs to the accepted-forms set of the expected
    # entry, so it must not appear in non_expected_top_candidates.
    assert all(
        hit["headword"] != "λόγος"
        for hit in result["non_expected_top_candidates"]
    )


@pytest.mark.integration
def test_evaluate_hard_queries_cli_writes_json(tmp_path: Path) -> None:
    """Evaluation reports should be written as machine-readable JSON."""
    cases_path = tmp_path / "cases.yaml"
    output_path = tmp_path / "evaluation.json"
    _write_cases(cases_path, [_valid_case()])

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "evaluate_hard_queries.py"),
            "--cases",
            str(cases_path),
            "--output-json",
            str(output_path),
            "--max-candidates",
            "3",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["summary"]["case_count"] == 1
    assert payload["cases"][0]["matched"] is True
