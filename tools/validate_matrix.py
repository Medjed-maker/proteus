"""Validate committed matrix JSON files used by CI."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, NoReturn

_SYMMETRY_REL_TOL = 1e-9
_SYMMETRY_ABS_TOL = 1e-9
SOUND_CLASSES = ("vowels", "stops")


class _CliArgumentParser(argparse.ArgumentParser):
    """ArgumentParser variant that reports errors without exiting the process."""

    def error(self, message: str) -> NoReturn:
        raise ValueError(message)


def test_matrix_symmetry(matrix: dict[str, dict[str, float]]) -> None:
    """Raise when a matrix contains asymmetric distances."""
    for row_phone, row in matrix.items():
        for column_phone, distance in row.items():
            try:
                reverse_distance = matrix[column_phone][row_phone]
            except KeyError as exc:
                raise ValueError(
                    "Matrix must define reverse distance "
                    f"for {column_phone!r}->{row_phone!r} while checking "
                    f"{row_phone!r}->{column_phone!r}"
                ) from exc
            if not math.isclose(
                distance,
                reverse_distance,
                rel_tol=_SYMMETRY_REL_TOL,
                abs_tol=_SYMMETRY_ABS_TOL,
            ):
                raise ValueError(
                    f"Matrix must be symmetric for {row_phone!r} and {column_phone!r}: "
                    f"{distance} != {reverse_distance}"
                )


def test_matrix_completeness(matrix: dict[str, dict[str, float]]) -> None:
    """Raise when any row omits a phone or contains an unexpected one."""
    inventory = set(matrix)
    for row_phone, row in matrix.items():
        row_inventory = set(row)
        if row_inventory != inventory:
            missing = sorted(inventory - row_inventory)
            extra = sorted(row_inventory - inventory)
            raise ValueError(
                f"Matrix row {row_phone!r} must define all phone pairs; "
                f"missing={missing}, extra={extra}"
            )


def test_value_bounds(
    matrix: dict[str, dict[str, float]], min_value: float = 0.0, max_value: float = 1.0
) -> None:
    """Raise when a distance falls outside the configured inclusive bounds."""
    for row_phone, row in matrix.items():
        for column_phone, distance in row.items():
            if not math.isfinite(distance):
                raise ValueError(
                    f"Matrix value {row_phone!r}->{column_phone!r}={distance} "
                    f"must be within [{min_value}, {max_value}]"
                )
            if not min_value <= distance <= max_value:
                raise ValueError(
                    f"Matrix value {row_phone!r}->{column_phone!r}={distance} "
                    f"must be within [{min_value}, {max_value}]"
                )


def _load_sound_class_rows(document: dict[str, Any], class_name: str) -> dict[str, dict[str, float]]:
    """Load a sound-class matrix from the JSON document."""
    sound_classes = document.get("sound_classes")
    if not isinstance(sound_classes, dict):
        raise ValueError("Matrix document must define sound_classes as a JSON object")

    raw_rows = sound_classes.get(class_name)
    if not isinstance(raw_rows, dict):
        raise ValueError(f"sound_classes.{class_name} must be a JSON object")

    rows: dict[str, dict[str, float]] = {}
    for row_phone, raw_row in raw_rows.items():
        if not isinstance(row_phone, str) or not row_phone:
            raise ValueError(f"sound_classes.{class_name} row keys must be non-empty strings")
        if not isinstance(raw_row, dict):
            raise ValueError(
                f"sound_classes.{class_name}.{row_phone} must be a JSON object of distances"
            )

        row: dict[str, float] = {}
        for column_phone, distance in raw_row.items():
            if not isinstance(column_phone, str) or not column_phone:
                raise ValueError(
                    f"sound_classes.{class_name}.{row_phone} column keys must be non-empty strings"
                )
            if not isinstance(distance, (int, float)):
                raise ValueError(
                    f"sound_classes.{class_name}.{row_phone}.{column_phone} must be numeric"
                )
            row[column_phone] = float(distance)
        rows[row_phone] = row

    return rows


def validate_matrix(file_path: str | Path) -> None:
    """Validate matrix sound classes used by the search runtime."""
    path = Path(file_path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Matrix file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Matrix file is not valid JSON: {path}") from exc

    for class_name in SOUND_CLASSES:
        rows = _load_sound_class_rows(document, class_name)
        test_matrix_completeness(rows)
        test_matrix_symmetry(rows)
        test_value_bounds(rows)


def main(argv: list[str] | None = None) -> int:
    """Run the CLI validator."""
    parser = _CliArgumentParser(description="Validate a matrix JSON file.")
    parser.add_argument("file_path", help="Path to the matrix JSON file to validate")
    args = parser.parse_args(argv)

    validate_matrix(args.file_path)
    print(f"Validated matrix: {args.file_path}")
    return 0


def run_cli(argv: list[str] | None = None) -> int:
    """Run the CLI with concise error reporting."""
    try:
        return main(argv)
    except (ValueError, OSError) as exc:  # pragma: no cover - exercised through tests
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(run_cli())
