"""Benchmark packaged phonological search latency and compare against a baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from phonology.distance import load_matrix
from phonology.search import build_ipa_index, build_kmer_index, build_lexicon_map, search

DEFAULT_LEXICON_PATH = Path("data/lexicon/greek_lemmas.json")
DEFAULT_QUERIES_PATH = Path("tests/fixtures/search_benchmark_queries.txt")
DEFAULT_OUTPUT_PATH = Path("search-latency-benchmark.json")
DEFAULT_MATRIX_NAME = "attic_doric.json"
DEFAULT_WARMUP_ROUNDS = 1
DEFAULT_MEASUREMENT_ROUNDS = 5
DEFAULT_MAX_RESULTS = 8
MEAN_THRESHOLD_RATIO = 0.10
P95_THRESHOLD_RATIO = 0.10
MAX_THRESHOLD_RATIO = 0.15


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Measure Proteus packaged-data search latency and optionally compare "
            "the result to a saved baseline JSON."
        )
    )
    parser.add_argument(
        "--lexicon",
        type=Path,
        default=DEFAULT_LEXICON_PATH,
        help="Path to the lexicon JSON document to benchmark.",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_QUERIES_PATH,
        help="Path to a newline-delimited query file.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to write the benchmark summary JSON.",
    )
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=None,
        help="Optional prior benchmark JSON to compare against.",
    )
    parser.add_argument(
        "--warmup-rounds",
        type=int,
        default=DEFAULT_WARMUP_ROUNDS,
        help="Number of warmup passes over the query list before timing.",
    )
    parser.add_argument(
        "--measurement-rounds",
        type=int,
        default=DEFAULT_MEASUREMENT_ROUNDS,
        help="Number of timed passes over the query list.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=DEFAULT_MAX_RESULTS,
        help="max_results forwarded to phonology.search.search().",
    )
    args = parser.parse_args()
    if args.warmup_rounds < 0:
        parser.error("--warmup-rounds must be >= 0")
    if args.measurement_rounds <= 0:
        parser.error("--measurement-rounds must be > 0")
    if args.max_results <= 0:
        parser.error("--max-results must be > 0")
    return args


def load_lexicon_entries(lexicon_path: Path) -> tuple[dict[str, Any], ...]:
    """Load a benchmark lexicon document."""
    with lexicon_path.open(encoding="utf-8") as lexicon_file:
        document = json.load(lexicon_file)
    lemmas = document.get("lemmas")
    if not isinstance(lemmas, list):
        raise ValueError(f"Lexicon file {lexicon_path} must define a list under 'lemmas'")
    entries: list[dict[str, Any]] = []
    for index, entry in enumerate(lemmas):
        if not isinstance(entry, dict):
            raise ValueError(f"Lexicon entry {index} in {lexicon_path} must be an object")
        entries.append(entry)
    return tuple(entries)


def load_queries(queries_path: Path) -> list[str]:
    """Load newline-delimited benchmark queries, ignoring blank lines and comments."""
    queries: list[str] = []
    for raw_line in queries_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        queries.append(line)
    if not queries:
        raise ValueError(f"Query file {queries_path} did not contain any benchmark queries")
    return queries


def percentile(values: list[float], fraction: float) -> float:
    """Return the requested percentile using linear interpolation for 0.0-1.0 fractions."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be between 0.0 and 1.0, got {fraction}")
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    weight = position - lower_index
    return lower_value + (upper_value - lower_value) * weight


def benchmark_search(
    *,
    lexicon_path: Path,
    queries_path: Path,
    warmup_rounds: int,
    measurement_rounds: int,
    max_results: int,
) -> dict[str, Any]:
    """Run repeated packaged-data searches and return timing metrics."""
    lexicon = load_lexicon_entries(lexicon_path)
    matrix = load_matrix(DEFAULT_MATRIX_NAME)
    kmer_index = build_kmer_index(lexicon)
    lexicon_map = build_lexicon_map(lexicon)
    ipa_index = build_ipa_index(lexicon_map)
    queries = load_queries(queries_path)

    for _ in range(warmup_rounds):
        for query in queries:
            search(
                query,
                lexicon,
                matrix,
                max_results=max_results,
                dialect="attic",
                index=kmer_index,
                prebuilt_lexicon_map=lexicon_map,
                prebuilt_ipa_index=ipa_index,
            )

    durations_ms: list[float] = []
    for _ in range(measurement_rounds):
        for query in queries:
            started = time.perf_counter()
            search(
                query,
                lexicon,
                matrix,
                max_results=max_results,
                dialect="attic",
                index=kmer_index,
                prebuilt_lexicon_map=lexicon_map,
                prebuilt_ipa_index=ipa_index,
            )
            durations_ms.append((time.perf_counter() - started) * 1000.0)

    query_count = len(queries)
    sample_count = len(durations_ms)
    total_ms = sum(durations_ms)
    return {
        "lexicon_path": str(lexicon_path),
        "queries_path": str(queries_path),
        "matrix_name": DEFAULT_MATRIX_NAME,
        "warmup_rounds": warmup_rounds,
        "measurement_rounds": measurement_rounds,
        "query_count": query_count,
        "sample_count": sample_count,
        "max_results": max_results,
        "mean_ms": total_ms / sample_count,
        "p95_ms": percentile(durations_ms, 0.95),
        "max_ms": max(durations_ms),
    }


def compare_against_baseline(
    result: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    """Compare benchmark metrics against threshold ratios."""
    comparisons: dict[str, dict[str, float | bool]] = {}
    failed_metrics: list[str] = []
    metric_thresholds = {
        "mean_ms": MEAN_THRESHOLD_RATIO,
        "p95_ms": P95_THRESHOLD_RATIO,
        "max_ms": MAX_THRESHOLD_RATIO,
    }
    for metric_name, threshold_ratio in metric_thresholds.items():
        if metric_name not in baseline:
            raise ValueError(
                "Baseline benchmark payload is missing required metric "
                f"{metric_name!r}: {baseline}"
            )
        if metric_name not in result:
            raise ValueError(
                "Result benchmark payload is missing required metric "
                f"{metric_name!r}: {result}"
            )
        current_value = float(result[metric_name])
        baseline_value = float(baseline[metric_name])
        if baseline_value <= 0:
            raise ValueError(f"Baseline metric {metric_name} must be > 0, got {baseline_value}")
        ratio = (current_value - baseline_value) / baseline_value
        passed = ratio <= threshold_ratio
        comparisons[metric_name] = {
            "baseline": baseline_value,
            "current": current_value,
            "delta_ratio": ratio,
            "allowed_ratio": threshold_ratio,
            "passed": passed,
        }
        if not passed:
            failed_metrics.append(metric_name)
    return {
        "baseline_json": baseline,
        "comparisons": comparisons,
        "passed": not failed_metrics,
        "failed_metrics": failed_metrics,
    }


def write_output(output_path: Path, payload: dict[str, Any]) -> None:
    """Write a benchmark payload as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    """Run the benchmark CLI."""
    args = parse_args()
    result = benchmark_search(
        lexicon_path=args.lexicon,
        queries_path=args.queries,
        warmup_rounds=args.warmup_rounds,
        measurement_rounds=args.measurement_rounds,
        max_results=args.max_results,
    )
    output_payload: dict[str, Any] = {"benchmark": result}

    if args.baseline_json is not None:
        baseline_payload = json.loads(args.baseline_json.read_text(encoding="utf-8"))
        baseline_benchmark = baseline_payload.get("benchmark", baseline_payload)
        if "benchmark" not in baseline_payload:
            print(
                "Warning: baseline file "
                f"{args.baseline_json} does not contain a top-level 'benchmark' key; "
                "falling back to the entire JSON payload. Expected format: "
                "{'benchmark': {...}}."
            )
        if not isinstance(baseline_benchmark, dict):
            raise ValueError(
                f"Baseline file {args.baseline_json} must contain a benchmark object"
            )
        output_payload["comparison"] = compare_against_baseline(result, baseline_benchmark)

    write_output(args.output_json, output_payload)

    print(f"Wrote benchmark results to {args.output_json}")
    print(
        f"mean_ms={result['mean_ms']:.3f} "
        f"p95_ms={result['p95_ms']:.3f} "
        f"max_ms={result['max_ms']:.3f}"
    )

    comparison = output_payload.get("comparison")
    if isinstance(comparison, dict) and not comparison.get("passed", False):
        failed_metrics = ", ".join(comparison.get("failed_metrics", []))
        print(f"Baseline comparison failed for: {failed_metrics}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
