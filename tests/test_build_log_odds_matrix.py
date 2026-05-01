"""Integration tests for scripts/build_log_odds_matrix.py CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_log_odds_matrix import DEFAULT_OUTPUT
from scripts.build_log_odds_matrix import main


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_GREEK_PAIRS = [
    # original, regularized — real Ancient Greek pairs from test corpus
    {"original": "νομου", "regularized": "νόμου", "source_file": "f1.xml", "tm_id": "1"},
    {"original": "θιου", "regularized": "θεοῦ", "source_file": "f2.xml", "tm_id": "2"},
    {"original": "πατρος", "regularized": "πατρός", "source_file": "f3.xml", "tm_id": "3"},
    {"original": "λογου", "regularized": "λόγου", "source_file": "f4.xml", "tm_id": "4"},
    {"original": "νομον", "regularized": "νόμον", "source_file": "f5.xml", "tm_id": "5"},
]


def test_default_output_uses_language_scoped_matrix_dir():
    assert DEFAULT_OUTPUT == Path(
        "data/languages/ancient_greek/matrices/log_odds_epidoc_choices.json"
    )


@pytest.fixture()
def input_file(tmp_path) -> Path:
    p = tmp_path / "input.json"
    p.write_text(json.dumps(_GREEK_PAIRS), encoding="utf-8")
    return p


@pytest.fixture()
def output_file(tmp_path) -> Path:
    return tmp_path / "out" / "matrix.json"


# ---------------------------------------------------------------------------
# Roundtrip
# ---------------------------------------------------------------------------


class TestRoundtrip:
    def test_produces_valid_output(self, input_file, output_file):
        main(["--input", str(input_file), "--output", str(output_file)])

        assert output_file.exists()
        with output_file.open(encoding="utf-8") as f:
            doc = json.load(f)

        assert "_meta" in doc
        assert "scores" in doc

    def test_meta_fields_present(self, input_file, output_file):
        main(["--input", str(input_file), "--output", str(output_file)])
        with output_file.open(encoding="utf-8") as f:
            doc = json.load(f)

        meta = doc["_meta"]
        for field in ("description", "version", "generated_at", "method",
                      "source_pair_count", "alignments_used", "alphabet",
                      "smoothing", "nw_params", "indel_counts", "totals"):
            assert field in meta, f"Missing _meta.{field}"

    def test_source_pair_count_recorded(self, input_file, output_file):
        main(["--input", str(input_file), "--output", str(output_file)])
        with output_file.open(encoding="utf-8") as f:
            doc = json.load(f)
        assert doc["_meta"]["source_pair_count"] == len(_GREEK_PAIRS)

    def test_scores_symmetric(self, input_file, output_file):
        main(["--input", str(input_file), "--output", str(output_file)])
        with output_file.open(encoding="utf-8") as f:
            doc = json.load(f)

        scores = doc["scores"]
        for phone_i, row in scores.items():
            for phone_j, score in row.items():
                assert scores[phone_j][phone_i] == pytest.approx(score), (
                    f"S({phone_i},{phone_j}) != S({phone_j},{phone_i})"
                )

    def test_output_dir_created_automatically(self, input_file, tmp_path):
        nested = tmp_path / "a" / "b" / "c" / "matrix.json"
        main(["--input", str(input_file), "--output", str(nested)])
        assert nested.exists()

    def test_alphabet_is_sorted(self, input_file, output_file):
        main(["--input", str(input_file), "--output", str(output_file)])
        with output_file.open(encoding="utf-8") as f:
            doc = json.load(f)
        alphabet = doc["_meta"]["alphabet"]
        assert alphabet == sorted(alphabet)

    def test_auto_reads_jsonl_input(self, tmp_path, output_file):
        jsonl_input = tmp_path / "input.jsonl"
        jsonl_input.write_text(
            "\n".join(json.dumps(pair) for pair in _GREEK_PAIRS) + "\n",
            encoding="utf-8",
        )

        main(["--input", str(jsonl_input), "--output", str(output_file)])

        with output_file.open(encoding="utf-8") as f:
            doc = json.load(f)
        assert doc["_meta"]["source_pair_count"] == len(_GREEK_PAIRS)

    def test_explicit_jsonl_reads_non_jsonl_extension(self, tmp_path, output_file):
        jsonl_input = tmp_path / "input.json"
        jsonl_input.write_text(
            "\n".join(json.dumps(pair) for pair in _GREEK_PAIRS) + "\n",
            encoding="utf-8",
        )

        main([
            "--input", str(jsonl_input), "--output", str(output_file),
            "--input-format", "jsonl",
        ])

        with output_file.open(encoding="utf-8") as f:
            doc = json.load(f)
        assert doc["_meta"]["source_pair_count"] == len(_GREEK_PAIRS)

    def test_explicit_input_format_auto_reads_jsonl_input(self, tmp_path, output_file):
        jsonl_input = tmp_path / "input.jsonl"
        jsonl_input.write_text(
            "\n".join(json.dumps(pair) for pair in _GREEK_PAIRS) + "\n",
            encoding="utf-8",
        )

        main([
            "--input", str(jsonl_input), "--output", str(output_file),
            "--input-format", "auto",
        ])

        with output_file.open(encoding="utf-8") as f:
            doc = json.load(f)
        assert doc["_meta"]["source_pair_count"] == len(_GREEK_PAIRS)

    def test_auto_rejects_jsonl_content_with_json_extension(self, tmp_path, output_file):
        json_input = tmp_path / "input.json"
        json_input.write_text(
            "\n".join(json.dumps(pair) for pair in _GREEK_PAIRS) + "\n",
            encoding="utf-8",
        )

        with pytest.raises(SystemExit) as exc_info:
            main([
                "--input", str(json_input), "--output", str(output_file),
                "--input-format", "auto",
            ])
        assert exc_info.value.code != 0

    def test_input_format_json_rejects_jsonl_extension(self, tmp_path, output_file):
        jsonl_input = tmp_path / "input.jsonl"
        jsonl_input.write_text(
            "\n".join(json.dumps(pair) for pair in _GREEK_PAIRS) + "\n",
            encoding="utf-8",
        )

        with pytest.raises(SystemExit) as exc_info:
            main([
                "--input", str(jsonl_input), "--output", str(output_file),
                "--input-format", "json",
            ])
        assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# --limit flag
# ---------------------------------------------------------------------------


class TestLimitFlag:
    def test_limit_reduces_alignments_used(self, input_file, output_file):
        main(["--input", str(input_file), "--output", str(output_file), "--limit", "2"])
        with output_file.open(encoding="utf-8") as f:
            doc = json.load(f)
        assert doc["_meta"]["alignments_used"] <= 2

    def test_limit_larger_than_input_processes_all(self, input_file, output_file):
        main(["--input", str(input_file), "--output", str(output_file), "--limit", "999"])
        with output_file.open(encoding="utf-8") as f:
            doc = json.load(f)
        assert doc["_meta"]["alignments_used"] == len(_GREEK_PAIRS)


# ---------------------------------------------------------------------------
# --smoothing flag
# ---------------------------------------------------------------------------


class TestSmoothingFlag:
    def test_laplace_smoothing_strategy_recorded(self, input_file, output_file):
        main([
            "--input", str(input_file), "--output", str(output_file),
            "--smoothing", "laplace",
        ])
        with output_file.open(encoding="utf-8") as f:
            doc = json.load(f)
        assert doc["_meta"]["smoothing"]["strategy"] == "laplace"

    def test_lidstone_smoothing_strategy_recorded(self, input_file, output_file):
        main([
            "--input", str(input_file), "--output", str(output_file),
            "--smoothing", "lidstone", "--lidstone-alpha", "0.3",
        ])
        with output_file.open(encoding="utf-8") as f:
            doc = json.load(f)
        meta_smoothing = doc["_meta"]["smoothing"]
        assert meta_smoothing["strategy"] == "lidstone"
        assert meta_smoothing["params"]["alpha"] == pytest.approx(0.3)

    def test_floor_smoothing_strategy_recorded(self, input_file, output_file):
        main([
            "--input", str(input_file), "--output", str(output_file),
            "--smoothing", "floor", "--floor", "-5.0",
        ])
        with output_file.open(encoding="utf-8") as f:
            doc = json.load(f)
        meta_smoothing = doc["_meta"]["smoothing"]
        assert meta_smoothing["strategy"] == "floor"
        assert meta_smoothing["params"]["floor"] == pytest.approx(-5.0)

    def test_zero_lidstone_alpha_is_allowed(self, input_file, output_file):
        main([
            "--input", str(input_file), "--output", str(output_file),
            "--smoothing", "lidstone", "--lidstone-alpha", "0",
        ])
        with output_file.open(encoding="utf-8") as f:
            doc = json.load(f)
        assert doc["_meta"]["smoothing"]["params"]["alpha"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Min count flag
# ---------------------------------------------------------------------------


class TestMinCountFlag:
    def test_min_count_zero_keeps_all(self, input_file, output_file):
        main([
            "--input", str(input_file), "--output", str(output_file),
            "--min-count", "0",
        ])
        with output_file.open(encoding="utf-8") as f:
            doc = json.load(f)
        # With min_count=0 every observed phone pair should be present
        assert len(doc["scores"]) > 0

    def test_min_count_filters_rare_pairs(self, input_file, output_file):
        # First run with min_count=0 to get baseline
        main([
            "--input", str(input_file), "--output", str(output_file),
            "--min-count", "0",
        ])
        with output_file.open(encoding="utf-8") as f:
            baseline = json.load(f)

        # Run with a high min_count — most cells should disappear
        filtered_output = output_file.parent / "filtered.json"
        main([
            "--input", str(input_file), "--output", str(filtered_output),
            "--min-count", "999",
        ])
        with filtered_output.open(encoding="utf-8") as f:
            filtered = json.load(f)

        baseline_cells = sum(len(row) for row in baseline["scores"].values())
        filtered_cells = sum(len(row) for row in filtered["scores"].values())
        assert filtered_cells <= baseline_cells


# ---------------------------------------------------------------------------
# --gap-cost flag
# ---------------------------------------------------------------------------


class TestGapCostFlag:
    def test_gap_cost_recorded_in_meta(self, input_file, output_file):
        main([
            "--input", str(input_file), "--output", str(output_file),
            "--gap-cost", "2.5",
        ])
        with output_file.open(encoding="utf-8") as f:
            doc = json.load(f)
        assert doc["_meta"]["nw_params"]["gap"] == pytest.approx(2.5)

    def test_default_gap_cost(self, input_file, output_file):
        main(["--input", str(input_file), "--output", str(output_file)])
        with output_file.open(encoding="utf-8") as f:
            doc = json.load(f)
        assert doc["_meta"]["nw_params"]["gap"] == pytest.approx(1.25)


# ---------------------------------------------------------------------------
# Invalid argument handling
# ---------------------------------------------------------------------------


class TestInvalidArgs:
    def test_negative_min_count_exits(self, input_file, output_file):
        with pytest.raises(SystemExit) as exc_info:
            main([
                "--input", str(input_file), "--output", str(output_file),
                "--min-count", "-1",
            ])
        assert exc_info.value.code != 0

    def test_non_numeric_gap_cost_exits(self, input_file, output_file):
        with pytest.raises(SystemExit) as exc_info:
            main([
                "--input", str(input_file), "--output", str(output_file),
                "--gap-cost", "abc",
            ])
        assert exc_info.value.code != 0

    @pytest.mark.parametrize("gap_cost", ["-1", "0"])
    def test_non_positive_gap_cost_exits(self, input_file, output_file, gap_cost):
        with pytest.raises(SystemExit) as exc_info:
            main([
                "--input", str(input_file), "--output", str(output_file),
                "--gap-cost", gap_cost,
            ])
        assert exc_info.value.code != 0

    @pytest.mark.parametrize("gap_cost", ["nan", "inf", "-inf"])
    def test_non_finite_gap_cost_exits(self, input_file, output_file, gap_cost):
        with pytest.raises(SystemExit) as exc_info:
            main([
                "--input", str(input_file), "--output", str(output_file),
                "--gap-cost", gap_cost,
            ])
        assert exc_info.value.code != 0

    def test_negative_lidstone_alpha_exits(self, input_file, output_file):
        with pytest.raises(SystemExit) as exc_info:
            main([
                "--input", str(input_file), "--output", str(output_file),
                "--smoothing", "lidstone", "--lidstone-alpha", "-0.1",
            ])
        assert exc_info.value.code != 0

    @pytest.mark.parametrize("alpha", ["nan", "inf", "-inf"])
    def test_non_finite_lidstone_alpha_exits(self, input_file, output_file, alpha):
        with pytest.raises(SystemExit) as exc_info:
            main([
                "--input", str(input_file), "--output", str(output_file),
                "--smoothing", "lidstone", "--lidstone-alpha", alpha,
            ])
        assert exc_info.value.code != 0

    @pytest.mark.parametrize("floor", ["nan", "inf", "-inf"])
    def test_non_finite_floor_exits(self, input_file, output_file, floor):
        with pytest.raises(SystemExit) as exc_info:
            main([
                "--input", str(input_file), "--output", str(output_file),
                "--smoothing", "floor", "--floor", floor,
            ])
        assert exc_info.value.code != 0

    def test_zero_limit_exits(self, input_file, output_file):
        with pytest.raises(SystemExit) as exc_info:
            main([
                "--input", str(input_file), "--output", str(output_file),
                "--limit", "0",
            ])
        assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_missing_input_exits_with_error(self, tmp_path):
        with pytest.raises(SystemExit) as exc_info:
            main([
                "--input", str(tmp_path / "nonexistent.json"),
                "--output", str(tmp_path / "out.json"),
            ])
        assert exc_info.value.code != 0

    def test_non_array_input_exits_with_error(self, tmp_path):
        bad_input = tmp_path / "bad.json"
        bad_input.write_text('{"key": "value"}', encoding="utf-8")
        with pytest.raises(SystemExit) as exc_info:
            main([
                "--input", str(bad_input),
                "--output", str(tmp_path / "out.json"),
            ])
        assert exc_info.value.code != 0

    def test_empty_input_produces_empty_scores(self, tmp_path, output_file):
        empty_input = tmp_path / "empty.json"
        empty_input.write_text("[]", encoding="utf-8")
        main(["--input", str(empty_input), "--output", str(output_file)])
        with output_file.open(encoding="utf-8") as f:
            doc = json.load(f)
        assert doc["scores"] == {}
        assert doc["_meta"]["source_pair_count"] == 0
        assert doc["_meta"]["alignments_used"] == 0

    def test_invalid_jsonl_line_exits(self, tmp_path, output_file):
        bad_input = tmp_path / "bad.jsonl"
        bad_input.write_text(
            json.dumps(_GREEK_PAIRS[0]) + "\nnot json\n",
            encoding="utf-8",
        )

        with pytest.raises(SystemExit) as exc_info:
            main(["--input", str(bad_input), "--output", str(output_file)])
        assert exc_info.value.code != 0
