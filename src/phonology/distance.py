"""Phonological distance calculation between word forms.

Uses weighted edit distance over full IPA phone sequences, with segment
distances loaded from pre-computed matrices in ``data/matrices/``.
The low-level distance functions return raw total cost, and companion
helpers expose a 0.0-1.0 normalized scale for API-facing consumers.
"""

from __future__ import annotations

import errno
import functools
import importlib.resources as resources
import json
import os
import stat
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Sequence, TypeGuard

from ._paths import resolve_repo_data_dir
from .ipa_converter import (
    get_known_phones,
    strip_ignored_ipa_combining_marks,
    tokenize_ipa,
)

__all__ = [
    "MatrixData",
    "DEFAULT_COST",
    "load_matrix",
    "phone_distance",
    "phonological_distance",
    "normalized_phonological_distance",
    "sequence_distance",
    "normalized_sequence_distance",
    "word_distance",
    "normalized_word_distance",
    "UNKNOWN_SUBSTITUTION_COST",
]

MatrixData = dict[str, dict[str, float]]
DEFAULT_COST = 5.0
UNKNOWN_SUBSTITUTION_COST = 1.0
_TRUSTED_MATRICES_DIR_ENV_VAR = "PROTEUS_TRUSTED_MATRICES_DIR"


def _normalize_ipa_phone(phone: str) -> str:
    """Normalize an IPA phone by removing accent/stress marks only."""
    return strip_ignored_ipa_combining_marks(phone)


def _normalize_ipa_sequence(seq: Sequence[str]) -> list[str]:
    """Normalize each IPA phone so pre-tokenized inputs match word-level APIs."""
    return [_normalize_ipa_phone(phone) for phone in seq]


@functools.lru_cache(maxsize=1)
def _get_known_ipa_phones() -> frozenset[str]:
    """Return the cached set of known IPA phones, loading on first use."""
    return frozenset(get_known_phones())


def _get_trusted_matrices_dir() -> Path:
    """Resolve the trusted matrices directory from env, package data, or repo layout."""
    override = os.environ.get(_TRUSTED_MATRICES_DIR_ENV_VAR)
    if override:
        override_path = Path(override).expanduser()
        if override_path.is_symlink():
            raise ValueError(
                f"{_TRUSTED_MATRICES_DIR_ENV_VAR} must not be a symlink, got {override}"
            )
        if not override_path.exists() or not override_path.is_dir():
            raise FileNotFoundError(override_path)
        return override_path

    try:
        resource_dir = resources.files("phonology").joinpath("data", "matrices")
    except (ModuleNotFoundError, FileNotFoundError):
        resource_dir = None

    # resources.files() may return a Traversable that does not implement
    # os.PathLike (e.g. when loaded from a zip archive), so verify the
    # interface before attempting filesystem resolution.
    if (
        resource_dir is not None
        and isinstance(resource_dir, os.PathLike)
        and resource_dir.is_dir()
    ):
        try:
            return Path(os.fspath(resource_dir))
        except TypeError:
            pass

    return resolve_repo_data_dir("matrices")


def _is_numeric_row(candidate: Any) -> TypeGuard[dict[str, Real]]:
    """Return True for dict-valued rows whose values are all ``Real`` numbers.

    ``_flatten_rows()`` uses this to distinguish terminal matrix rows from
    nested namespaces while only accepting numeric phone-distance mappings.
    """
    if not isinstance(candidate, dict):
        return False
    if not candidate:
        return False
    return all(isinstance(value, Real) for value in candidate.values())


def _flatten_rows(data: dict[str, Any], matrix: MatrixData) -> None:
    """Recursively extract numeric phone rows from a nested matrix JSON object.

    Keys starting with ``_`` are treated as metadata and skipped entirely.
    The special-case ``dialect_pairs`` key is also ignored because it records
    dialect metadata rather than phone distances. Numeric rows are only
    accepted when the key contains no underscore so compound or namespaced
    keys are not mistaken for canonical phone rows.
    """
    for key, value in data.items():
        if key.startswith("_") or key == "dialect_pairs":
            continue

        # Underscore-joined keys (e.g. "ɛː_aː" inside dialect_pairs) are
        # metadata pairs, not canonical phone rows — skip them.
        if _is_numeric_row(value) and "_" not in key:
            matrix[key] = {column: float(distance) for column, distance in value.items()}
            continue

        if isinstance(value, dict):
            _flatten_rows(value, matrix)


def _resolve_matrix_path(path: Path | str, trusted_dir: Path) -> Path:
    """Resolve matrix paths relative to the trusted matrices directory."""
    candidate_path = Path(path)
    if candidate_path.is_absolute():
        return candidate_path

    parts = candidate_path.parts
    if len(parts) >= 2 and parts[:2] == ("data", "matrices"):
        candidate_path = Path(*parts[2:])

    return trusted_dir / candidate_path


def load_matrix(path: Path | str) -> MatrixData:
    """Load a phonological distance matrix from a JSON file.

    Args:
        path: Path or string naming a matrix JSON file. Relative inputs are
            resolved from the trusted matrices directory, so both
            ``"attic_doric.json"`` and ``"data/matrices/attic_doric.json"``
            resolve to the packaged runtime asset.

    Returns:
        Nested dict mapping phone pair -> distance float.

    Security:
        Mitigates TOCTOU races via three layers:
        (1) Pre-open lstat walk rejects symlinks in path components.
        (2) O_NOFOLLOW open prevents kernel from following late-injected symlinks.
        (3) Post-open fstat/stat comparison detects replacement between open and read.
    """
    trusted_dir = Path(os.path.abspath(_get_trusted_matrices_dir()))
    if not trusted_dir.is_dir():
        raise FileNotFoundError(trusted_dir)
    requested_path = _resolve_matrix_path(path, trusted_dir)
    requested_path = Path(os.path.abspath(requested_path))
    if not requested_path.is_relative_to(trusted_dir):
        raise ValueError(
            f"Matrix path must stay within {trusted_dir}, got {path}"
        )
    candidate_path = requested_path.resolve(strict=True)
    if not candidate_path.is_relative_to(trusted_dir):
        raise ValueError(
            f"Matrix path must stay within {trusted_dir}, got {path}"
        )

    relative_parts = requested_path.relative_to(trusted_dir).parts

    current_path = trusted_dir
    for part in relative_parts:
        current_path = current_path / part
        current_stat = os.lstat(current_path)
        if stat.S_ISLNK(current_stat.st_mode):
            raise ValueError(f"Matrix path must not traverse symlinks, got {path}")

    candidate_lstat = os.lstat(candidate_path)
    if not stat.S_ISREG(candidate_lstat.st_mode):
        raise ValueError(f"Matrix path must resolve to a regular file, got {path}")

    open_flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        open_flags |= os.O_NOFOLLOW

    try:
        file_descriptor = os.open(candidate_path, open_flags)
    except OSError as exc:
        if getattr(os, "O_NOFOLLOW", None) is not None and exc.errno == errno.ELOOP:
            raise ValueError(f"Matrix path must not traverse symlinks, got {path}") from exc
        raise

    with os.fdopen(file_descriptor, encoding="utf-8") as matrix_file:
        fd_stat = os.fstat(matrix_file.fileno())
        if not stat.S_ISREG(fd_stat.st_mode):
            raise ValueError(f"Matrix path must resolve to a regular file, got {path}")

        opened_target = os.stat(candidate_path, follow_symlinks=False)
        if (
            fd_stat.st_dev != opened_target.st_dev
            or fd_stat.st_ino != opened_target.st_ino
            or not stat.S_ISREG(opened_target.st_mode)
        ):
            raise ValueError(f"Matrix path changed during validation, got {path}")

        raw_data = json.load(matrix_file)

    matrix: MatrixData = {}
    _flatten_rows(raw_data, matrix)
    return matrix


def phone_distance(p1: str, p2: str, matrix: MatrixData) -> float:
    """Return the distance between two IPA phones.

    Args:
        p1: IPA phone string.
        p2: IPA phone string.
        matrix: Distance matrix as returned by load_matrix().

    Returns:
        Raw substitution cost. The following logic applies:
        (1) Identical phones (p1 == p2) have cost 0.0.
        (2) For a pair present in the distance matrix, return the matrix value.
        (3) If both phones are known to the inventory (retrieved via
            get_known_phones()) but the pair is missing from the matrix, return
            UNKNOWN_SUBSTITUTION_COST (1.0).
        (4) If at least one phone is unknown to the inventory, return
            DEFAULT_COST (5.0).
    """
    p1 = _normalize_ipa_phone(p1)
    p2 = _normalize_ipa_phone(p2)
    return _phone_distance_raw(p1, p2, matrix)


def _phone_distance_raw(p1: str, p2: str, matrix: MatrixData) -> float:
    """Return phone distance for IPA phones that are already normalized."""
    if p1 == p2:
        return 0.0
    if p1 in matrix and p2 in matrix[p1]:
        return matrix[p1][p2]
    if p2 in matrix and p1 in matrix[p2]:
        return matrix[p2][p1]
    known_phones = _get_known_ipa_phones()
    if p1 in known_phones and p2 in known_phones:
        return UNKNOWN_SUBSTITUTION_COST
    return DEFAULT_COST


def _edit_distance(
    seq1: Sequence[str],
    seq2: Sequence[str],
    matrix: MatrixData,
    gap_cost: float,
    sub_fn: Callable[[str, str, MatrixData], float],
) -> float:
    """Core Needleman-Wunsch edit distance.

    Args:
        seq1: First phone sequence.
        seq2: Second phone sequence.
        matrix: Nested phone distance matrix.
        gap_cost: Cost charged for insertions and deletions.
        sub_fn: Substitution cost function ``(p1, p2, matrix) -> float``.

    Returns:
        Total cost accumulated across the full-sequence alignment.
    """
    if not seq1 and not seq2:
        return 0.0

    rows = len(seq1) + 1
    cols = len(seq2) + 1
    costs = [[0.0] * cols for _ in range(rows)]

    for i in range(1, rows):
        costs[i][0] = i * gap_cost
    for j in range(1, cols):
        costs[0][j] = j * gap_cost

    for i in range(1, rows):
        for j in range(1, cols):
            substitution = costs[i - 1][j - 1] + sub_fn(
                seq1[i - 1], seq2[j - 1], matrix
            )
            deletion = costs[i - 1][j] + gap_cost
            insertion = costs[i][j - 1] + gap_cost
            costs[i][j] = min(substitution, deletion, insertion)

    return costs[-1][-1]


def phonological_distance(seq1: list[str], seq2: list[str], matrix: MatrixData) -> float:
    """Compute a raw phonological distance using weighted edit distance.

    Args:
        seq1: First phone sequence.
        seq2: Second phone sequence.
        matrix: Nested phone distance matrix.

    Returns:
        Raw total cost accumulated across the full-sequence alignment.
    """
    normalized_seq1 = _normalize_ipa_sequence(seq1)
    normalized_seq2 = _normalize_ipa_sequence(seq2)
    return _edit_distance(
        normalized_seq1,
        normalized_seq2,
        matrix,
        DEFAULT_COST,
        _phone_distance_raw,
    )


def _normalization_denominator(seq1: Sequence[str], seq2: Sequence[str]) -> float:
    """Return the unit-scale denominator for normalized sequence comparison."""
    return float(max(len(seq1), len(seq2), 1))


def _normalized_substitution_cost(p1: str, p2: str, matrix: MatrixData) -> float:
    """Return unit-scale substitution cost for normalized distance scoring.

    Unlike ``phone_distance()``, normalized scoring treats any non-identical
    substitution as at most ``1.0`` so the resulting edit distance remains on
    a meaningful 0.0-1.0 scale after division by the longer sequence length.

    Inputs are expected to already be normalized by ``_normalize_ipa_sequence``.
    Matrix-backed substitutions are clamped to at most ``1.0``; substitutions
    missing from the matrix return ``UNKNOWN_SUBSTITUTION_COST`` (not
    ``DEFAULT_COST``).
    """
    if p1 == p2:
        return 0.0
    if p1 in matrix and p2 in matrix[p1]:
        return min(matrix[p1][p2], 1.0)
    if p2 in matrix and p1 in matrix[p2]:
        return min(matrix[p2][p1], 1.0)
    return UNKNOWN_SUBSTITUTION_COST


def _normalized_edit_distance(
    seq1: Sequence[str], seq2: Sequence[str], matrix: MatrixData
) -> float:
    """Compute unit-scale edit distance for normalized API consumers."""
    return _edit_distance(seq1, seq2, matrix, 1.0, _normalized_substitution_cost)


def _normalize_raw_distance(
    raw_distance: float, seq1: Sequence[str], seq2: Sequence[str]
) -> float:
    """Map a unit-scale edit distance onto the inclusive 0.0-1.0 interval."""
    if not seq1 and not seq2:
        return 0.0

    normalized = raw_distance / _normalization_denominator(seq1, seq2)
    return min(max(normalized, 0.0), 1.0)


def normalized_phonological_distance(
    seq1: list[str], seq2: list[str], matrix: MatrixData
) -> float:
    """Compute a normalized phonological distance in the 0.0-1.0 range.

    This uses a unit-scale edit distance tailored for API/search consumers:
    matrix-backed substitutions keep their 0.0-1.0 values, while insertions,
    deletions, and unmapped substitutions cost 1.0. The total is divided by
    the longer sequence length so 0.0 means identical and 1.0 means maximally
    dissimilar under this normalized scoring model.
    """
    normalized_seq1 = _normalize_ipa_sequence(seq1)
    normalized_seq2 = _normalize_ipa_sequence(seq2)
    normalized_distance = _normalized_edit_distance(normalized_seq1, normalized_seq2, matrix)
    return _normalize_raw_distance(normalized_distance, normalized_seq1, normalized_seq2)


def sequence_distance(seq1: Sequence[str], seq2: Sequence[str], matrix: MatrixData) -> float:
    """Compute phonological distance between two phone sequences.

    Args:
        seq1: First phone sequence (list of IPA strings).
        seq2: Second phone sequence.
        matrix: Distance matrix.

    Returns:
        Raw total cost from full-sequence weighted edit distance.
    """
    return phonological_distance(list(seq1), list(seq2), matrix)


def normalized_sequence_distance(
    seq1: Sequence[str], seq2: Sequence[str], matrix: MatrixData
) -> float:
    """Compute normalized phonological distance between two phone sequences."""
    seq1_list = list(seq1)
    seq2_list = list(seq2)
    return normalized_phonological_distance(seq1_list, seq2_list, matrix)


def word_distance(word1_ipa: str, word2_ipa: str, matrix: MatrixData) -> float:
    """Convenience wrapper: distance between two IPA-transcribed words.

    Args:
        word1_ipa: Compact or space-separated IPA phones for word 1.
        word2_ipa: Compact or space-separated IPA phones for word 2.
        matrix: Distance matrix.

    Returns:
        Raw total cost from full-sequence weighted edit distance.
    """
    return phonological_distance(tokenize_ipa(word1_ipa), tokenize_ipa(word2_ipa), matrix)


def normalized_word_distance(word1_ipa: str, word2_ipa: str, matrix: MatrixData) -> float:
    """Compute normalized phonological distance for IPA-transcribed words."""
    seq1 = tokenize_ipa(word1_ipa)
    seq2 = tokenize_ipa(word2_ipa)
    return normalized_phonological_distance(seq1, seq2, matrix)
