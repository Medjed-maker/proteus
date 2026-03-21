"""Generate the canonical Attic-Doric phone distance matrix."""

from __future__ import annotations

import functools
import json
import logging
import math
import sys
from copy import deepcopy
from numbers import Real
from pathlib import Path
from typing import Any

from ._paths import resolve_repo_data_dir

MATRIX_PATH = resolve_repo_data_dir("matrices") / "attic_doric.json"
logger = logging.getLogger(__name__)

VOWEL_ORDER = ["a", "aː", "e", "eː", "ɛː", "i", "o", "ɔː", "oː", "y", "ai", "oi", "au", "eu"]
STOP_ORDER = ["p", "b", "pʰ", "t", "d", "tʰ", "k", "ɡ", "kʰ"]


def _load_seed_document() -> dict[str, Any]:
    """Load the committed matrix as the seed document for regeneration."""
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8"))


def _coerce_seed_rows(
    raw_rows: Any, order: list[str], *, label: str
) -> dict[str, dict[str, float]]:
    """Normalize JSON-backed matrix rows into validated float mappings."""
    if not isinstance(raw_rows, dict):
        raise ValueError(f"{label} must be a JSON object of row mappings")

    expected_phones = set(order)
    actual_phones = set(raw_rows)
    if actual_phones != expected_phones:
        missing = sorted(expected_phones - actual_phones)
        extra = sorted(actual_phones - expected_phones)
        raise ValueError(f"{label} must define exactly {order}; missing={missing}, extra={extra}")

    rows: dict[str, dict[str, float]] = {}
    for row_phone in order:
        raw_row = raw_rows[row_phone]
        if not isinstance(raw_row, dict):
            raise ValueError(f"{label}.{row_phone} must be a JSON object of column distances")

        actual_columns = set(raw_row)
        if actual_columns != expected_phones:
            missing = sorted(expected_phones - actual_columns)
            extra = sorted(actual_columns - expected_phones)
            raise ValueError(
                f"{label}.{row_phone} must define exactly {order}; missing={missing}, extra={extra}"
            )

        row: dict[str, float] = {}
        for column_phone in order:
            distance = raw_row[column_phone]
            if not isinstance(distance, Real):
                raise ValueError(
                    f"{label}.{row_phone}.{column_phone} must be numeric, got {type(distance).__name__}"
                )
            row[column_phone] = float(distance)
        rows[row_phone] = row

    return rows


def _load_base_sound_class_rows() -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Load the base vowel/stop rows from the committed JSON seed file."""
    try:
        seed_document = _load_seed_document()
    except FileNotFoundError as exc:
        raise RuntimeError(f"Matrix seed file not found at {MATRIX_PATH}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Matrix seed file at {MATRIX_PATH} is not valid JSON") from exc

    try:
        sound_classes = seed_document["sound_classes"]
        vowels = _coerce_seed_rows(
            sound_classes["vowels"],
            VOWEL_ORDER,
            label="sound_classes.vowels",
        )
        stops = _coerce_seed_rows(
            sound_classes["stops"],
            STOP_ORDER,
            label="sound_classes.stops",
        )
    except KeyError as exc:
        raise RuntimeError(
            f"Matrix seed file at {MATRIX_PATH} is missing required key {exc.args[0]!r}"
        ) from exc
    except ValueError as exc:
        raise ValueError(f"Matrix seed file at {MATRIX_PATH} is malformed: {exc}") from exc

    return vowels, stops


@functools.lru_cache(maxsize=1)
def _get_cached_base_rows() -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Load and cache the committed base vowel/stop rows on first use."""
    return _load_base_sound_class_rows()


def _get_base_rows() -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Return deep-copied cached base rows safe for caller mutation.

    ``_overlay_seed_rows()`` deep-copies its input before mutating it. This helper
    returns independent deep copies as well so callers cannot corrupt the cached
    base rows held by ``_get_cached_base_rows()``.

    ``cache_clear`` and ``cache_info`` are forwarded from the underlying
    ``_get_cached_base_rows`` LRU cache so callers can invalidate or inspect
    the cache through this public-facing wrapper.
    """
    base_vowels, base_stops = _get_cached_base_rows()
    return deepcopy(base_vowels), deepcopy(base_stops)


#: Forwarded from _get_cached_base_rows so callers (especially tests)
#: can manage the underlying LRU cache through this public-facing wrapper.
_get_base_rows.cache_clear = _get_cached_base_rows.cache_clear  # type: ignore[attr-defined]
_get_base_rows.cache_info = _get_cached_base_rows.cache_info  # type: ignore[attr-defined]


def _overlay_seed_rows(
    base_rows: dict[str, dict[str, float]],
    seed_rows: dict[str, Any],
    order: list[str],
    seed_source: str = "committed matrix seed",
) -> dict[str, dict[str, float]]:
    """Overlay committed seed distances onto a complete base matrix."""
    rows = deepcopy(base_rows)

    for row_phone, row in seed_rows.items():
        if row_phone not in rows:
            logger.warning(
                "Skipping unknown row %r from %s", row_phone, seed_source
            )
            continue
        for column_phone, distance in row.items():
            if column_phone not in rows[row_phone]:
                logger.warning(
                    "Skipping unknown column %r for row %r from %s",
                    column_phone,
                    row_phone,
                    seed_source,
                )
                continue
            rows[row_phone][column_phone] = float(distance)
            rows[column_phone][row_phone] = float(distance)

    for row_phone in order:
        for column_phone in order:
            if row_phone == column_phone:
                rows[row_phone][column_phone] = 0.0
                continue

            if column_phone not in rows[row_phone]:
                reverse_row = rows.get(column_phone)
                if reverse_row is None or row_phone not in reverse_row:
                    raise ValueError(
                        "Missing symmetric distance for "
                        f"{row_phone!r} <-> {column_phone!r} in {seed_source}"
                    )
                rows[row_phone][column_phone] = reverse_row[row_phone]

    return rows


def _validate_complete_matrix(rows: dict[str, dict[str, float]], order: list[str]) -> None:
    """Ensure matrix rows are complete, symmetric, and normalized."""
    for row_phone in order:
        row = rows[row_phone]
        if set(row) != set(order):
            raise ValueError(f"Row {row_phone!r} does not cover the full inventory")

        for column_phone in order:
            distance = row[column_phone]
            if not 0.0 <= distance <= 1.0:
                raise ValueError(
                    f"Distance {row_phone!r}->{column_phone!r} must be within [0.0, 1.0]"
                )
            if not math.isclose(
                distance,
                rows[column_phone][row_phone],
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"Matrix must be symmetric for {row_phone!r} and {column_phone!r}"
                )
            if row_phone == column_phone and distance != 0.0:
                raise ValueError(f"Self-distance for {row_phone!r} must be 0.0")


def _validate_dialect_pairs(dialect_pairs: Any) -> dict[str, dict[str, float]]:
    """Validate dialect-pair metadata before copying it into the generated document."""
    if not isinstance(dialect_pairs, dict):
        raise ValueError("sound_classes.dialect_pairs must be a JSON object")

    validated_pairs: dict[str, dict[str, float]] = {}
    for dialect_pair_name, raw_phone_pairs in dialect_pairs.items():
        if not isinstance(dialect_pair_name, str) or not dialect_pair_name.strip():
            raise ValueError("sound_classes.dialect_pairs keys must be non-empty strings")
        if not isinstance(raw_phone_pairs, dict):
            raise ValueError(
                f"sound_classes.dialect_pairs.{dialect_pair_name} must be a JSON object"
            )

        validated_phone_pairs: dict[str, float] = {}
        for phone_pair, distance in raw_phone_pairs.items():
            if not isinstance(phone_pair, str) or not phone_pair.strip():
                raise ValueError(
                    f"sound_classes.dialect_pairs.{dialect_pair_name} keys must be non-empty strings"
                )
            if not isinstance(distance, Real):
                raise ValueError(
                    "sound_classes.dialect_pairs."
                    f"{dialect_pair_name}.{phone_pair} must be numeric"
                )

            normalized_distance = float(distance)
            if not 0.0 <= normalized_distance <= 1.0:
                raise ValueError(
                    "sound_classes.dialect_pairs."
                    f"{dialect_pair_name}.{phone_pair} must be within [0.0, 1.0]"
                )
            validated_phone_pairs[phone_pair] = normalized_distance

        validated_pairs[dialect_pair_name] = validated_phone_pairs

    return validated_pairs


def build_attic_doric_matrix() -> dict[str, Any]:
    """Build the canonical Attic-Doric matrix document."""
    seed_document = _load_seed_document()
    sound_classes = seed_document["sound_classes"]
    base_vowels, base_stops = _get_base_rows()

    vowels = _overlay_seed_rows(
        base_vowels,
        sound_classes["vowels"],
        VOWEL_ORDER,
        seed_source=f"{MATRIX_PATH}::sound_classes.vowels",
    )
    stops = _overlay_seed_rows(
        base_stops,
        sound_classes["stops"],
        STOP_ORDER,
        seed_source=f"{MATRIX_PATH}::sound_classes.stops",
    )

    _validate_complete_matrix(vowels, VOWEL_ORDER)
    _validate_complete_matrix(stops, STOP_ORDER)

    return {
        "_meta": {
            "description": "Phonological distance matrix between Attic and Doric Greek",
            "method": "weighted edit distance based on sound classes (Dolgopolsky/ASJP)",
            "range": "0.0 (identical) to 1.0 (maximally distant)",
            "note": (
                "Generated from the repository sound-class inventory and validated for "
                "symmetry, completeness, and 0.0-1.0 bounds."
            ),
        },
        "sound_classes": {
            "vowels": vowels,
            "stops": stops,
            "dialect_pairs": deepcopy(
                _validate_dialect_pairs(sound_classes["dialect_pairs"])
            ),
        },
    }


def write_attic_doric_matrix(path: Path = MATRIX_PATH) -> Path:
    """Write the canonical Attic-Doric matrix JSON document."""
    document = build_attic_doric_matrix()
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def main() -> int:
    """Regenerate the canonical Attic-Doric matrix in-place."""
    write_attic_doric_matrix()
    print("Attic-Doric matrix regenerated successfully.")
    return 0


def run_cli() -> int:
    """Run the CLI entrypoint with concise error reporting."""
    try:
        return main()
    except Exception as exc:  # pragma: no cover - covered via monkeypatched tests
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(run_cli())
