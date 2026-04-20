"""Tests for phonology.matrix_generator."""

from collections.abc import Generator
from datetime import datetime, timedelta, timezone
import json
import re
from pathlib import Path

import pytest

from phonology import matrix_generator


@pytest.fixture
def clear_seed_document_cache() -> Generator[None, None, None]:
    """Reset the cached seed document around tests that override MATRIX_PATH."""
    matrix_generator._load_seed_document.cache_clear()

    yield

    matrix_generator._load_seed_document.cache_clear()


def _make_legacy_rows(
    order: tuple[str, ...] | None = None,
) -> dict[str, dict[str, float]]:
    legacy_order = order if order is not None else matrix_generator._LEGACY_STOP_ORDER
    return {
        row_phone: {
            column_phone: abs(row_index - column_index) / 10
            for column_index, column_phone in enumerate(legacy_order)
        }
        for row_index, row_phone in enumerate(legacy_order)
    }


class TestOverlaySeedRows:
    def test_logs_unknown_seed_rows_and_columns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level("WARNING", logger="phonology.matrix_generator")
        base_rows = {
            "a": {"a": 0.0, "b": 0.1},
            "b": {"a": 0.1, "b": 0.0},
        }

        matrix_generator._overlay_seed_rows(
            base_rows,
            {"x": {"a": 0.2}, "a": {"y": 0.3}},
            ["a", "b"],
            seed_source="seed.json",
        )

        assert "Skipping unknown row 'x' from seed.json" in caplog.text
        assert "Skipping unknown column 'y' for row 'a' from seed.json" in caplog.text

    def test_raises_clear_error_when_reverse_distance_is_missing(self) -> None:
        base_rows = {
            "a": {"a": 0.0},
            "b": {"b": 0.0},
        }

        with pytest.raises(ValueError, match="Missing symmetric distance"):
            matrix_generator._overlay_seed_rows(base_rows, {}, ["a", "b"], seed_source="seed.json")


class TestExpandKoineStopRows:
    def test_expands_koine_rows_from_legacy_inventory(self) -> None:
        legacy_rows = _make_legacy_rows()

        rows = matrix_generator._expand_koine_stop_rows(legacy_rows)

        assert set(rows) == set(matrix_generator.STOP_ORDER)
        assert rows["f"]["f"] == pytest.approx(0.0)
        assert rows["f"]["pʰ"] == pytest.approx(matrix_generator._KOINE_DIRECT_DISTANCE)
        assert rows["f"]["p"] == pytest.approx(legacy_rows["pʰ"]["p"])
        assert rows["f"]["θ"] == pytest.approx(legacy_rows["pʰ"]["tʰ"])
        assert rows["p"]["f"] == pytest.approx(rows["f"]["p"])

    def test_raises_clear_error_for_unmapped_stop_order_phone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        legacy_rows = _make_legacy_rows()
        monkeypatch.setattr(
            matrix_generator,
            "STOP_ORDER",
            matrix_generator.STOP_ORDER + ("z",),
        )

        with pytest.raises(
            ValueError,
            match=r"unknown phone.*'z'.*_KOINE_STOP_BASES=.*STOP_ORDER=",
        ):
            matrix_generator._expand_koine_stop_rows(legacy_rows)

    def test_raises_clear_error_when_base_phone_is_missing(self) -> None:
        legacy_order = tuple(
            phone for phone in matrix_generator._LEGACY_STOP_ORDER if phone != "pʰ"
        )
        legacy_rows = _make_legacy_rows(legacy_order)

        with pytest.raises(
            ValueError,
            match=r"_expand_koine_stop_rows.*_KOINE_STOP_BASES.*missing base_phone 'pʰ'.*_coerce_seed_rows",
        ):
            matrix_generator._expand_koine_stop_rows(legacy_rows)


class TestLoadBaseSoundClassRows:
    def test_reads_base_rows_from_committed_seed(self) -> None:
        vowels, stops = matrix_generator._load_base_sound_class_rows()

        assert vowels["a"]["aː"] == pytest.approx(0.1)
        assert vowels["oi"]["i"] == pytest.approx(0.15)
        assert vowels["ɔː"]["o"] == pytest.approx(0.1)
        assert stops["ɡ"]["k"] == pytest.approx(0.2)

    def test_raises_clear_error_when_required_keys_are_missing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        clear_seed_document_cache: None,
    ) -> None:
        bad_seed = tmp_path / "missing.json"
        bad_seed.write_text(json.dumps({"_meta": {}}), encoding="utf-8")
        monkeypatch.setattr(matrix_generator, "MATRIX_PATH", bad_seed)

        with pytest.raises(RuntimeError, match="missing required key"):
            matrix_generator._load_base_sound_class_rows()

    def test_raises_clear_error_when_seed_rows_are_malformed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        clear_seed_document_cache: None,
    ) -> None:
        bad_seed = tmp_path / "malformed.json"
        bad_seed.write_text(
            json.dumps({"sound_classes": {"vowels": [], "stops": {}}}),
            encoding="utf-8",
        )
        monkeypatch.setattr(matrix_generator, "MATRIX_PATH", bad_seed)

        with pytest.raises(ValueError, match="is malformed"):
            matrix_generator._load_base_sound_class_rows()


class TestValidateCompleteMatrix:
    def test_accepts_nearly_equal_symmetric_floats(self) -> None:
        rows = {
            "a": {"a": 0.0, "b": 0.3},
            "b": {"a": 0.3000000000001, "b": 0.0},
        }

        matrix_generator._validate_complete_matrix(rows, ["a", "b"])

    def test_rejects_symmetric_values_outside_tolerance(self) -> None:
        rows = {
            "a": {"a": 0.0, "b": 0.3},
            "b": {"a": 0.3001, "b": 0.0},
        }

        with pytest.raises(ValueError, match="Matrix must be symmetric"):
            matrix_generator._validate_complete_matrix(rows, ["a", "b"])


class TestGetBaseRows:
    def test_loads_rows_lazily_and_returns_independent_copies(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        matrix_generator._get_base_rows.cache_clear()
        calls = 0
        vowels = {"a": {"a": 0.0}}
        stops = {"p": {"p": 0.0}}

        def fake_load() -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
            nonlocal calls
            calls += 1
            return vowels, stops

        monkeypatch.setattr(matrix_generator, "_load_base_sound_class_rows", fake_load)

        try:
            first = matrix_generator._get_base_rows()
            first[0]["a"]["a"] = 9.9
            second = matrix_generator._get_base_rows()

            assert calls == 1
            assert first[0]["a"]["a"] == 9.9
            assert second == (vowels, stops)
            assert second[0] is not vowels
            assert second[0]["a"] is not vowels["a"]
            assert second[0] is not first[0]
            assert second[0]["a"] is not first[0]["a"]
        finally:
            matrix_generator._get_base_rows.cache_clear()


class TestValidateDialectPairs:
    def test_rejects_non_numeric_distance(self) -> None:
        with pytest.raises(ValueError, match="must be numeric"):
            matrix_generator._validate_dialect_pairs(
                {"attic_doric": {"ɛː_aː": "0.3"}}
            )

    def test_rejects_negative_distance(self) -> None:
        with pytest.raises(ValueError, match=r"within \[0.0, 1.0\]"):
            matrix_generator._validate_dialect_pairs(
                {"attic_doric": {"ɛː_aː": -0.1}}
            )

    def test_rejects_distance_outside_unit_interval(self) -> None:
        with pytest.raises(ValueError, match=r"within \[0.0, 1.0\]"):
            matrix_generator._validate_dialect_pairs(
                {"attic_doric": {"ɛː_aː": 1.5}}
            )

    @pytest.mark.parametrize("distance", [0.0, 1.0])
    def test_accepts_boundary_distances(self, distance: float) -> None:
        validated = matrix_generator._validate_dialect_pairs(
            {"attic_doric": {"ɛː_aː": distance}}
        )

        assert validated == {"attic_doric": {"ɛː_aː": distance}}


class TestBuildAtticDoricMatrix:
    def test_returns_expected_document_structure(self) -> None:
        document = matrix_generator.build_attic_doric_matrix()

        assert set(document) == {"_meta", "sound_classes"}

        meta = document["_meta"]
        assert isinstance(meta, dict)
        assert isinstance(meta["generated_at"], str)

        sound_classes = document["sound_classes"]
        assert isinstance(sound_classes, dict)
        assert isinstance(sound_classes["vowels"], dict)
        assert isinstance(sound_classes["stops"], dict)
        assert isinstance(sound_classes["dialect_pairs"], dict)

    def test_generated_at_includes_timezone_offset_and_milliseconds(self) -> None:
        document = matrix_generator.build_attic_doric_matrix(
            generated_at="2026-04-20T09:10:11.123456Z"
        )

        generated_at = document["_meta"]["generated_at"]

        parsed = datetime.fromisoformat(generated_at)
        assert parsed.tzinfo is not None
        assert parsed.utcoffset() == timedelta(0)
        assert generated_at == "2026-04-20T09:10:11.123+00:00"
        assert re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}\+00:00",
            generated_at,
        ), f"generated_at format mismatch: {generated_at}"

    def test_normalizes_explicit_generated_at(self) -> None:
        document = matrix_generator.build_attic_doric_matrix(
            generated_at="2026-04-20T09:10:11Z"
        )

        assert document["_meta"]["generated_at"] == "2026-04-20T09:10:11.000+00:00"

    def test_rejects_invalid_generated_at(self) -> None:
        invalid_generated_at = "not-a-date"

        with pytest.raises(
            ValueError,
            match=(
                "generated_at must be a valid ISO-8601 timestamp: "
                f"{invalid_generated_at!r}"
            ),
        ):
            matrix_generator.build_attic_doric_matrix(
                generated_at=invalid_generated_at
            )

    def test_rejects_non_string_generated_at(self) -> None:
        invalid_generated_at = 123

        with pytest.raises(
            ValueError,
            match=f"generated_at must be a string or None: {invalid_generated_at!r}",
        ):
            matrix_generator.build_attic_doric_matrix(
                generated_at=invalid_generated_at  # type: ignore[arg-type]
            )


class TestRunCli:
    def test_main_prints_success_message(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(matrix_generator, "write_attic_doric_matrix", lambda: matrix_generator.MATRIX_PATH)

        assert matrix_generator.main() == 0
        captured = capsys.readouterr()
        assert "Attic-Doric matrix regenerated successfully." in captured.out

    def test_returns_main_exit_code_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(matrix_generator, "main", lambda: 7)

        assert matrix_generator.run_cli() == 7

    def test_prints_concise_error_and_returns_one_on_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        def fail() -> int:
            raise RuntimeError("boom")

        monkeypatch.setattr(matrix_generator, "main", fail)

        assert matrix_generator.run_cli() == 1
        captured = capsys.readouterr()
        assert "Error: boom" in captured.err
