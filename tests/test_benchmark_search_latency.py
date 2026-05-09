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
    assert set(payload["comparison"]["failed_metrics"]) == {
        "mean_ms",
        "p95_ms",
        "max_ms",
    }


@pytest.mark.parametrize(
    ("args", "expected_message"),
    [
        (["--warmup-rounds", "-1"], "--warmup-rounds must be >= 0"),
        (["--measurement-rounds", "0"], "--measurement-rounds must be > 0"),
        (["--max-results", "0"], "--max-results must be > 0"),
        (["--matrix-name", ""], "argument --matrix-name: value must be a non-empty string"),
        (["--dialect", ""], "argument --dialect: value must be a non-empty string"),
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


def test_percentile_returns_single_element_for_any_fraction() -> None:
    """percentile() should return the only value without interpolation."""
    assert benchmark_search_latency.percentile([4.2], 0.0) == pytest.approx(4.2)
    assert benchmark_search_latency.percentile([4.2], 0.5) == pytest.approx(4.2)
    assert benchmark_search_latency.percentile([4.2], 1.0) == pytest.approx(4.2)


@pytest.mark.parametrize(
    ("values", "fraction", "expected"),
    [
        ([1.0, 3.0, 5.0], 0.5, 3.0),
        ([1.0, 3.0, 5.0, 7.0], 0.5, 4.0),
        ([10.0, 20.0, 30.0, 40.0], 0.25, 17.5),
        ([40.0, 10.0, 30.0, 20.0], 0.75, 32.5),
    ],
)
def test_percentile_interpolates_known_inputs(
    values: list[float], fraction: float, expected: float
) -> None:
    """percentile() should sort values and linearly interpolate between indexes."""
    assert benchmark_search_latency.percentile(values, fraction) == pytest.approx(
        expected
    )


def test_benchmark_search_propagates_matrix_and_dialect_parameters(
    isolated_language_registry: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """benchmark_search() should not depend on main() for profile registration."""
    lexicon = ({"id": "L1", "headword": "target", "ipa": "pa", "dialect": "attic"},)
    queries_path = tmp_path / "queries.txt"
    queries_path.write_text("target\n", encoding="utf-8")
    captured: dict[str, object] = {}
    captured_dialects: list[object] = []

    monkeypatch.setattr(
        benchmark_search_latency,
        "load_lexicon_entries",
        lambda _path: lexicon,
    )

    def fake_load_matrix(matrix_name: str) -> dict[str, dict[str, float]]:
        captured["matrix_name"] = matrix_name
        return {}

    monkeypatch.setattr(benchmark_search_latency, "load_matrix", fake_load_matrix)
    monkeypatch.setattr(benchmark_search_latency, "build_kmer_index", lambda *_, **__: {})
    monkeypatch.setattr(benchmark_search_latency, "build_lexicon_map", lambda *_, **__: {})
    monkeypatch.setattr(benchmark_search_latency, "build_ipa_index", lambda _map: {})
    monkeypatch.setattr(benchmark_search_latency, "load_queries", lambda _path: ["target"])

    def fake_search(*_args: object, **kwargs: object) -> list[object]:
        assert "dialect" in kwargs, "missing 'dialect' kwarg in fake_search"
        captured_dialects.append(kwargs["dialect"])
        return []

    monkeypatch.setattr(benchmark_search_latency, "search", fake_search)

    result = benchmark_search_latency.benchmark_search(
        lexicon_path=tmp_path / "lexicon.json",
        queries_path=queries_path,
        warmup_rounds=0,
        measurement_rounds=1,
        max_results=1,
        matrix_name="custom_matrix.json",
        dialect="koine",
    )

    assert result["query_count"] == 1
    assert result["sample_count"] == 1
    assert result["matrix_name"] == "custom_matrix.json"
    assert captured["matrix_name"] == "custom_matrix.json"
    assert captured_dialects == ["koine"]


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


@pytest.mark.parametrize("metric_name", ["mean_ms", "p95_ms", "max_ms"])
@pytest.mark.parametrize("baseline_value", [0.0, -1.0])
def test_compare_against_baseline_rejects_non_positive_baseline(
    baseline_value: float, metric_name: str
) -> None:
    """Baseline comparisons should reject zero or negative metric values."""
    baseline = {
        "mean_ms": 1.0,
        "p95_ms": 1.0,
        "max_ms": 1.0,
    }
    baseline[metric_name] = baseline_value

    with pytest.raises(
        ValueError, match=f"Baseline metric {metric_name} must be > 0"
    ):
        benchmark_search_latency.compare_against_baseline(
            {
                "mean_ms": 1.0,
                "p95_ms": 1.0,
                "max_ms": 1.0,
            },
            baseline,
        )


def test_compare_against_baseline_passes_at_threshold_boundaries() -> None:
    """Current metrics exactly at allowed thresholds should still pass."""
    result = {
        "mean_ms": 110.0,
        "p95_ms": 220.0,
        "max_ms": 345.0,
    }
    baseline = {
        "mean_ms": 100.0,
        "p95_ms": 200.0,
        "max_ms": 300.0,
    }

    comparison = benchmark_search_latency.compare_against_baseline(result, baseline)

    assert comparison["passed"] is True
    assert comparison["failed_metrics"] == []
    assert comparison["comparisons"]["mean_ms"] == {
        "baseline": 100.0,
        "current": 110.0,
        "delta_ratio": pytest.approx(0.10),
        "allowed_ratio": benchmark_search_latency.MEAN_THRESHOLD_RATIO,
        "passed": True,
    }
    assert comparison["comparisons"]["p95_ms"]["delta_ratio"] == pytest.approx(0.10)
    assert comparison["comparisons"]["max_ms"]["delta_ratio"] == pytest.approx(0.15)


def test_compare_against_baseline_reports_failed_metrics() -> None:
    """Baseline comparisons should aggregate every metric over its threshold."""
    result = {
        "mean_ms": 111.0,
        "p95_ms": 219.0,
        "max_ms": 346.0,
    }
    baseline = {
        "mean_ms": 100.0,
        "p95_ms": 200.0,
        "max_ms": 300.0,
    }

    comparison = benchmark_search_latency.compare_against_baseline(result, baseline)

    assert comparison["baseline_json"] == baseline
    assert comparison["passed"] is False
    assert set(comparison["failed_metrics"]) == {"mean_ms", "max_ms"}
    assert comparison["comparisons"]["mean_ms"]["passed"] is False
    assert comparison["comparisons"]["p95_ms"]["passed"] is True
    assert comparison["comparisons"]["max_ms"]["passed"] is False


def test_benchmark_script_warns_when_baseline_json_lacks_benchmark_key(
    tmp_path: Path,
) -> None:
    """CLI should warn when falling back to a legacy whole-payload baseline format."""
    baseline_path = tmp_path / "baseline.json"
    output_path = tmp_path / "result.json"
    baseline_path.write_text(
        json.dumps(
            {
                # Deliberately huge so the regression check never trips on environment
                # variance — we only verify the missing-baseline warning path here.
                "mean_ms": 1_000_000_000.0,
                "p95_ms": 1_000_000_000.0,
                "max_ms": 1_000_000_000.0,
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
