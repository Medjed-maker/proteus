"""Tests for tools.validate_matrix."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tools import validate_matrix as validate_matrix_tool


ROOT_DIR = Path(__file__).resolve().parents[1]
COMMITTED_MATRIX_PATH = ROOT_DIR / "data" / "languages" / "ancient_greek" / "matrices" / "attic_doric.json"


def _valid_matrix_document() -> dict[str, object]:
    return {
        "_meta": {"description": "test"},
        "sound_classes": {
            "vowels": {
                "a": {"a": 0.0, "e": 0.4},
                "e": {"a": 0.4, "e": 0.0},
            },
            "stops": {
                "p": {"p": 0.0, "t": 0.5},
                "t": {"p": 0.5, "t": 0.0},
            },
            "dialect_pairs": {"attic_doric": {"a_e": 0.4}},
        },
    }


def _write_matrix(tmp_path: Path, document: dict[str, object]) -> Path:
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return path


class TestValidateHelpers:
    def test_matrix_symmetry_accepts_symmetric_rows(self) -> None:
        validate_matrix_tool.test_matrix_symmetry(
            {"a": {"a": 0.0, "e": 0.4}, "e": {"a": 0.4, "e": 0.0}}
        )

    def test_matrix_symmetry_accepts_close_float_rows(self) -> None:
        validate_matrix_tool.test_matrix_symmetry(
            {"a": {"a": 0.0, "e": 0.4}, "e": {"a": 0.4000000001, "e": 0.0}}
        )

    def test_matrix_symmetry_rejects_asymmetric_rows(self) -> None:
        with pytest.raises(ValueError, match="Matrix must be symmetric"):
            validate_matrix_tool.test_matrix_symmetry(
                {"a": {"a": 0.0, "e": 0.4}, "e": {"a": 0.2, "e": 0.0}}
            )

    def test_matrix_symmetry_rejects_float_difference_beyond_tolerance(self) -> None:
        with pytest.raises(ValueError, match="Matrix must be symmetric"):
            validate_matrix_tool.test_matrix_symmetry(
                {"a": {"a": 0.0, "e": 0.4}, "e": {"a": 0.40001, "e": 0.0}}
            )

    def test_matrix_symmetry_rejects_missing_reverse_entry(self) -> None:
        with pytest.raises(ValueError, match="must define reverse distance"):
            validate_matrix_tool.test_matrix_symmetry(
                {"a": {"a": 0.0, "e": 0.4}, "e": {"e": 0.0}}
            )

    def test_matrix_completeness_accepts_complete_rows(self) -> None:
        validate_matrix_tool.test_matrix_completeness(
            {"a": {"a": 0.0, "e": 0.4}, "e": {"a": 0.4, "e": 0.0}}
        )

    def test_matrix_completeness_rejects_missing_pairs(self) -> None:
        with pytest.raises(ValueError, match="must define all phone pairs"):
            validate_matrix_tool.test_matrix_completeness(
                {"a": {"a": 0.0}, "e": {"a": 0.4, "e": 0.0}}
            )

    def test_value_bounds_accepts_values_within_range(self) -> None:
        validate_matrix_tool.test_value_bounds(
            {"a": {"a": 0.0, "e": 0.4}, "e": {"a": 0.4, "e": 1.0}}
        )

    def test_value_bounds_rejects_out_of_range_values(self) -> None:
        with pytest.raises(ValueError, match=r"within \[0.0, 1.0\]"):
            validate_matrix_tool.test_value_bounds(
                {"a": {"a": 0.0, "e": 1.2}, "e": {"a": 1.2, "e": 0.0}}
            )
        with pytest.raises(ValueError, match=r"within \[0.0, 1.0\]"):
            validate_matrix_tool.test_value_bounds(
                {"a": {"a": 0.0, "e": -0.1}, "e": {"a": -0.1, "e": 0.0}}
            )

    def test_value_bounds_rejects_non_finite_values(self) -> None:
        with pytest.raises(ValueError, match=r"within \[0.0, 1.0\]"):
            validate_matrix_tool.test_value_bounds(
                {"a": {"a": 0.0, "e": math.nan}, "e": {"a": math.nan, "e": 0.0}}
            )
        with pytest.raises(ValueError, match=r"within \[0.0, 1.0\]"):
            validate_matrix_tool.test_value_bounds(
                {"a": {"a": 0.0, "e": math.inf}, "e": {"a": math.inf, "e": 0.0}}
            )


class TestValidateMatrix:
    def test_accepts_committed_attic_doric_matrix(self) -> None:
        validate_matrix_tool.validate_matrix(COMMITTED_MATRIX_PATH)

    def test_rejects_asymmetric_matrix_document(self, tmp_path: Path) -> None:
        document = _valid_matrix_document()
        document["sound_classes"]["vowels"]["e"]["a"] = 0.2  # type: ignore[index]

        with pytest.raises(ValueError, match="Matrix must be symmetric"):
            validate_matrix_tool.validate_matrix(_write_matrix(tmp_path, document))

    def test_rejects_incomplete_matrix_document(self, tmp_path: Path) -> None:
        document = _valid_matrix_document()
        del document["sound_classes"]["stops"]["p"]["t"]  # type: ignore[index]

        with pytest.raises(ValueError, match="must define all phone pairs"):
            validate_matrix_tool.validate_matrix(_write_matrix(tmp_path, document))

    def test_rejects_out_of_bounds_matrix_document(self, tmp_path: Path) -> None:
        document = _valid_matrix_document()
        document["sound_classes"]["stops"]["p"]["t"] = -0.1  # type: ignore[index]
        document["sound_classes"]["stops"]["t"]["p"] = -0.1  # type: ignore[index]

        with pytest.raises(ValueError, match=r"within \[0.0, 1.0\]"):
            validate_matrix_tool.validate_matrix(_write_matrix(tmp_path, document))

    def test_rejects_missing_sound_classes_object(self, tmp_path: Path) -> None:
        path = _write_matrix(tmp_path, {"_meta": {"description": "test"}})

        with pytest.raises(ValueError, match="must define sound_classes"):
            validate_matrix_tool.validate_matrix(path)

    def test_rejects_non_mapping_class_rows(self, tmp_path: Path) -> None:
        document = _valid_matrix_document()
        document["sound_classes"]["vowels"] = []  # type: ignore[index]

        with pytest.raises(ValueError, match="sound_classes.vowels must be a JSON object"):
            validate_matrix_tool.validate_matrix(_write_matrix(tmp_path, document))

    def test_uses_sound_classes_constant(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        document = _valid_matrix_document()
        del document["sound_classes"]["stops"]  # type: ignore[index]
        monkeypatch.setattr(validate_matrix_tool, "SOUND_CLASSES", ("vowels",))

        validate_matrix_tool.validate_matrix(_write_matrix(tmp_path, document))


class TestRunCli:
    def test_returns_zero_and_prints_success_message(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert validate_matrix_tool.run_cli([str(COMMITTED_MATRIX_PATH)]) == 0
        captured = capsys.readouterr()
        assert f"Validated matrix: {COMMITTED_MATRIX_PATH}" in captured.out

    def test_returns_one_and_prints_error_on_failure(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = _write_matrix(tmp_path, {"sound_classes": {"vowels": [], "stops": {}}})

        assert validate_matrix_tool.run_cli([str(path)]) == 1
        captured = capsys.readouterr()
        assert "Error: sound_classes.vowels must be a JSON object" in captured.err

    def test_returns_one_and_prints_error_when_no_args(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert validate_matrix_tool.run_cli([]) == 1
        captured = capsys.readouterr()
        assert captured.err.startswith("Error:")
        assert "usage:" not in captured.err

    def test_returns_one_and_prints_error_for_missing_file(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert validate_matrix_tool.run_cli(["/nonexistent/path.json"]) == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err

    def test_returns_one_and_prints_error_for_invalid_json(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "invalid.json"
        path.write_text("{invalid json", encoding="utf-8")

        assert validate_matrix_tool.run_cli([str(path)]) == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err

    def test_propagates_unexpected_exceptions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise_runtime_error(argv: list[str] | None = None) -> int:
            raise RuntimeError("unexpected failure")

        monkeypatch.setattr(validate_matrix_tool, "main", _raise_runtime_error)

        with pytest.raises(RuntimeError, match="unexpected failure"):
            validate_matrix_tool.run_cli([])
