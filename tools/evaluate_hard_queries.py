"""Evaluate Phase 3 hard query cases against the Proteus search runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any, Literal, cast

from api.main import (
    APP_VERSION,
    SearchHit,
    SearchRequest,
    build_ruleset_versions,
    generate_request_id,
    load_search_dependencies,
    run_search,
)

if TYPE_CHECKING:
    from tools.validate_hard_queries import (
        HardQueryValidationError,
        load_cases,
        validate_cases,
    )
else:
    try:
        from tools.validate_hard_queries import (
            HardQueryValidationError,
            load_cases,
            validate_cases,
        )
    except ModuleNotFoundError:  # pragma: no cover - exercised by direct script CLI.
        from validate_hard_queries import (
            HardQueryValidationError,
            load_cases,
            validate_cases,
        )


DEFAULT_CASES_PATH = Path("data/evaluation/hard_queries/public_seed_cases.yaml")
ResponseLanguage = Literal["en", "ja"]
# Cache key is the language id passed to load_search_dependencies; the value
# is the opaque SearchDependencies instance. Typed as Any to avoid pulling a
# private type into a tools-layer signature.
DepsCache = dict[str, Any]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Proteus hard query cases against the search engine."
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help="YAML or JSON hard query case file.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Path to write the evaluation JSON report.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Override max_candidates for every case.",
    )
    parser.add_argument(
        "--response-language",
        choices=["en", "ja"],
        default="en",
        help="Response prose language for generated explanations.",
    )
    return parser.parse_args()


def _expected_match_sets(
    case: dict[str, Any],
) -> list[tuple[str, frozenset[str]]]:
    """Return (primary_headword, accepted_forms) pairs for each expected entry.

    ``accepted_forms`` is the headword plus any ``acceptable_forms`` surface
    variants. A hit counts as matching the expected entry when its headword
    appears in the accepted_forms set.
    """
    entries: list[tuple[str, frozenset[str]]] = []
    for candidate in case["expected_candidates"]:
        if not isinstance(candidate, dict):
            continue
        primary = candidate["headword"]
        if not isinstance(primary, str):
            continue
        accepted: set[str] = {primary}
        for form in candidate.get("acceptable_forms") or []:
            if isinstance(form, str):
                accepted.add(form)
        entries.append((primary, frozenset(accepted)))
    return entries


def _serialize_hit(rank: int, hit: SearchHit) -> dict[str, Any]:
    """Return the machine-readable subset needed for quality review."""
    return {
        "rank": rank,
        "headword": hit.headword,
        "confidence": hit.confidence,
        "distance": hit.distance,
        "match_type": hit.match_type,
        "candidate_bucket": hit.candidate_bucket,
        "rules_applied": [step.rule_id for step in hit.rules_applied],
        "orthographic_notes": [note.kind for note in hit.orthographic_notes],
    }


def _resolve_deps(language: str, deps_cache: DepsCache | None) -> Any:
    """Return cached search dependencies for ``language``, loading if needed."""
    if deps_cache is None:
        return load_search_dependencies(language)
    deps = deps_cache.get(language)
    if deps is None:
        deps = load_search_dependencies(language)
        deps_cache[language] = deps
    return deps


def evaluate_case(
    case: dict[str, Any],
    *,
    max_candidates_override: int | None = None,
    response_language: ResponseLanguage = "en",
    deps_cache: DepsCache | None = None,
) -> dict[str, Any]:
    """Evaluate one hard query case and return a JSON-serializable report."""
    max_candidates = int(max_candidates_override or case.get("max_candidates") or 20)
    request = SearchRequest(
        query_form=case["input_form"],
        language=case["language"],
        dialect_hint=case["dialect_hint"],
        max_candidates=max_candidates,
        response_language=response_language,
    )
    deps = _resolve_deps(request.language, deps_cache)
    profile = deps.profile
    language_id = profile.language_id if profile is not None else request.language
    outcome = run_search(
        request,
        deps=deps,
        request_id=generate_request_id(),
        base_url=None,
        engine_version=APP_VERSION,
        ruleset_versions=build_ruleset_versions(language_id),
    )

    expected_entries = _expected_match_sets(case)
    all_accepted_forms: set[str] = {
        form for _, forms in expected_entries for form in forms
    }

    ranked_hits = [
        _serialize_hit(rank, hit)
        for rank, hit in enumerate(outcome.response.hits, start=1)
    ]

    matched_expected: list[dict[str, Any]] = []
    matched_ranks: list[int] = []
    for primary, accepted in expected_entries:
        matching_ranks = [
            hit["rank"]
            for hit in ranked_hits
            if isinstance(hit["headword"], str) and hit["headword"] in accepted
        ]
        if matching_ranks:
            best_rank = min(matching_ranks)
            matched_expected.append(
                {"headword": primary, "matched": True, "rank": best_rank}
            )
            matched_ranks.append(best_rank)
        else:
            matched_expected.append(
                {"headword": primary, "matched": False, "rank": None}
            )
    matched_expected.sort(key=lambda item: item["headword"])

    false_negative_candidates = [
        item["headword"] for item in matched_expected if not item["matched"]
    ]
    non_expected_top_candidates = [
        hit
        for hit in ranked_hits
        if isinstance(hit["headword"], str)
        and hit["headword"] not in all_accepted_forms
    ]
    if false_negative_candidates:
        # If any expected candidate is absent from the requested top-N, every
        # non-expected hit in that top-N outranks the missing expected entry.
        wrong_outranking_candidates = list(non_expected_top_candidates)
    elif matched_ranks:
        # A non-expected candidate is wrong-outranking when it appears before
        # at least one expected candidate, not only before the best expected hit.
        worst_matched_rank = max(matched_ranks)
        wrong_outranking_candidates = [
            hit
            for hit in non_expected_top_candidates
            if hit["rank"] < worst_matched_rank
        ]
    else:
        wrong_outranking_candidates = []

    return {
        "case_id": case["case_id"],
        "input_form": case["input_form"],
        "language": request.language,
        "dialect_hint": request.dialect_hint,
        "max_candidates": request.max_candidates,
        "query_ipa": outcome.response.query_ipa,
        "query_mode": outcome.response.query_mode,
        "matched": not false_negative_candidates,
        "matched_expected_candidates": matched_expected,
        "false_negative_candidates": false_negative_candidates,
        "non_expected_top_candidates": non_expected_top_candidates,
        "wrong_outranking_candidates": wrong_outranking_candidates,
        "top_candidates": ranked_hits,
        "meta": {
            "request_id": outcome.response.meta.request_id,
            "engine_version": outcome.response.meta.engine_version,
            "schema_version": outcome.response.meta.schema_version,
        },
    }


def evaluate_cases(
    cases: list[dict[str, Any]],
    *,
    max_candidates_override: int | None = None,
    response_language: ResponseLanguage = "en",
    deps_cache: DepsCache | None = None,
) -> dict[str, Any]:
    """Evaluate cases and return a summary report."""
    # Share a deps cache across cases of the same language to avoid the
    # per-case reload cost (load_search_dependencies is not memoized upstream).
    shared_cache: DepsCache = {} if deps_cache is None else deps_cache
    case_results = [
        evaluate_case(
            case,
            max_candidates_override=max_candidates_override,
            response_language=response_language,
            deps_cache=shared_cache,
        )
        for case in cases
    ]
    matched_count = sum(1 for result in case_results if result["matched"])
    return {
        "summary": {
            "case_count": len(case_results),
            "matched_count": matched_count,
            "missed_count": len(case_results) - matched_count,
        },
        "cases": case_results,
    }


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    if args.max_candidates is not None and args.max_candidates < 1:
        print("--max-candidates must be >= 1", file=sys.stderr)
        return 2

    try:
        cases = load_cases(args.cases)
    except HardQueryValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    validation_errors = validate_cases(cases, source_label=str(args.cases))
    if validation_errors:
        for error in validation_errors:
            print(error, file=sys.stderr)
        return 1

    # argparse ``choices`` guarantees one of the two literal values; the cast
    # narrows ``str`` to the Literal type for the typed evaluate_cases signature.
    response_language = cast(ResponseLanguage, args.response_language)
    report = evaluate_cases(
        cases,
        max_candidates_override=args.max_candidates,
        response_language=response_language,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "evaluated "
        f"{report['summary']['case_count']} hard query case(s); "
        f"{report['summary']['matched_count']} matched"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
