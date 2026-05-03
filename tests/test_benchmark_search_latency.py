"""Tests for the search latency benchmark helper."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools import benchmark_search_latency


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "tools" / "benchmark_search_latency.py"
SUBPROCESS_TIMEOUT_SECONDS = 300


def test_benchmark_script_writes_json(tmp_path: Path) -> None:
    """CLI smoke test should produce benchmark JSON with expected keys."""
    output_path = tmp_path / "benchmark.json"

    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--queries",
                str(REPO_ROOT / "tests" / "fixtures" / "search_benchmark_queries.txt"),
                "--output-json",
                str(output_path),
                "--warmup-rounds",
                "0",
                "--measurement-rounds",
                "1",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            f"benchmark CLI timed out in test_benchmark_script_writes_json: {exc}"
        )

    assert completed.returncode == 0, completed.stderr
    assert output_path.exists()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    benchmark = payload["benchmark"]
    assert benchmark["query_count"] >= 1
    assert benchmark["sample_count"] == benchmark["query_count"]
    assert benchmark["mean_ms"] >= 0.0
    assert benchmark["p95_ms"] >= 0.0
    assert benchmark["max_ms"] >= 0.0


def test_benchmark_script_fails_when_baseline_threshold_is_exceeded(
    tmp_path: Path,
) -> None:
    """CLI should exit non-zero when the saved baseline is much faster."""
    baseline_path = tmp_path / "baseline.json"
    output_path = tmp_path / "result.json"
    baseline_payload = {
        "benchmark": {
            "mean_ms": 0.001,
            "p95_ms": 0.001,
            "max_ms": 0.001,
        }
    }
    baseline_path.write_text(json.dumps(baseline_payload), encoding="utf-8")

    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--queries",
                str(REPO_ROOT / "tests" / "fixtures" / "search_benchmark_queries.txt"),
                "--output-json",
                str(output_path),
                "--baseline-json",
                str(baseline_path),
                "--warmup-rounds",
                "0",
                "--measurement-rounds",
                "1",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            "benchmark CLI timed out in "
            f"test_benchmark_script_fails_when_baseline_threshold_is_exceeded: {exc}"
        )

    assert completed.returncode == 1
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["comparison"]["passed"] is False
    assert payload["comparison"]["failed_metrics"] == ["mean_ms", "p95_ms", "max_ms"]


@pytest.mark.parametrize(
    ("args", "expected_message"),
    [
        (["--warmup-rounds", "-1"], "--warmup-rounds must be >= 0"),
        (["--measurement-rounds", "0"], "--measurement-rounds must be > 0"),
        (["--max-results", "0"], "--max-results must be > 0"),
    ],
)
def test_benchmark_script_rejects_invalid_arguments(
    tmp_path: Path,
    args: list[str],
    expected_message: str,
) -> None:
    """Argument validation should fail fast with a clear message."""
    missing_queries_path = tmp_path / "missing-queries.txt"

    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--queries",
                str(missing_queries_path),
                *args,
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            "benchmark CLI timed out in "
            f"test_benchmark_script_rejects_invalid_arguments: {exc}"
        )

    assert completed.returncode != 0
    assert expected_message in completed.stderr


def test_percentile_rejects_fraction_out_of_range() -> None:
    """percentile() should reject fractions outside the inclusive 0.0-1.0 range."""
    with pytest.raises(ValueError, match="fraction must be between 0.0 and 1.0"):
        benchmark_search_latency.percentile([1.0, 2.0, 3.0], -0.1)

    with pytest.raises(ValueError, match="fraction must be between 0.0 and 1.0"):
        benchmark_search_latency.percentile([1.0, 2.0, 3.0], 1.1)


def test_percentile_rejects_empty_list() -> None:
    """percentile() should reject empty inputs before attempting interpolation."""
    with pytest.raises(ValueError, match="percentile requires at least one value"):
        benchmark_search_latency.percentile([], 0.5)


def test_percentile_accepts_boundary_fractions() -> None:
    """percentile() should accept 0.0 and 1.0 and return min/max respectively."""
    values = [1.0, 2.0, 3.0]

    result_min = benchmark_search_latency.percentile(values, 0.0)
    result_max = benchmark_search_latency.percentile(values, 1.0)

    assert result_min == 1.0
    assert result_max == 3.0


def test_compare_against_baseline_rejects_missing_metric() -> None:
    """Baseline comparisons should fail clearly when a required metric is absent."""
    with pytest.raises(ValueError, match="missing required metric 'p95_ms'"):
        benchmark_search_latency.compare_against_baseline(
            {
                "mean_ms": 1.0,
                "p95_ms": 1.5,
                "max_ms": 2.0,
            },
            {
                "mean_ms": 1.0,
                "max_ms": 2.0,
            },
        )


def test_benchmark_script_warns_when_baseline_json_lacks_benchmark_key(
    tmp_path: Path,
) -> None:
    """CLI should warn when falling back to a legacy whole-payload baseline format."""
    baseline_path = tmp_path / "baseline.json"
    output_path = tmp_path / "result.json"
    baseline_path.write_text(
        json.dumps(
            {
                "mean_ms": 9999.0,
                "p95_ms": 9999.0,
                "max_ms": 9999.0,
            }
        ),
        encoding="utf-8",
    )

    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--queries",
                str(REPO_ROOT / "tests" / "fixtures" / "search_benchmark_queries.txt"),
                "--output-json",
                str(output_path),
                "--baseline-json",
                str(baseline_path),
                "--warmup-rounds",
                "0",
                "--measurement-rounds",
                "1",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            "benchmark CLI timed out in "
            "test_benchmark_script_warns_when_baseline_json_lacks_benchmark_key: "
            f"{exc}"
        )

    assert completed.returncode == 0, completed.stderr
    assert "does not contain a top-level 'benchmark' key" in completed.stdout
