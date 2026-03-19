"""Phonological distance calculation between word forms.

Uses weighted edit distance over IPA segments, with distances
loaded from pre-computed matrices (data/matrices/).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


MatrixData = dict[str, dict[str, float]]


def load_matrix(path: Path) -> MatrixData:
    """Load a phonological distance matrix from a JSON file.

    Args:
        path: Path to a matrix JSON file (e.g. data/matrices/attic_doric.json).

    Returns:
        Nested dict mapping phone pair -> distance float.
    """
    raise NotImplementedError


def phone_distance(p1: str, p2: str, matrix: MatrixData) -> float:
    """Return the distance between two IPA phones.

    Args:
        p1: IPA phone string.
        p2: IPA phone string.
        matrix: Distance matrix as returned by load_matrix().

    Returns:
        Float in [0.0, 1.0]. 0.0 = identical, 1.0 = maximally distant.
    """
    if p1 == p2:
        return 0.0
    raise NotImplementedError


def sequence_distance(seq1: Sequence[str], seq2: Sequence[str], matrix: MatrixData) -> float:
    """Compute weighted edit distance between two phone sequences.

    Implements Needleman-Wunsch alignment with phone-level distances
    from the supplied matrix.

    Args:
        seq1: First phone sequence (list of IPA strings).
        seq2: Second phone sequence.
        matrix: Distance matrix.

    Returns:
        Normalised distance in [0.0, 1.0].
    """
    raise NotImplementedError


def word_distance(word1_ipa: str, word2_ipa: str, matrix: MatrixData) -> float:
    """Convenience wrapper: distance between two IPA-transcribed words.

    Args:
        word1_ipa: Space-separated IPA phones for word 1.
        word2_ipa: Space-separated IPA phones for word 2.
        matrix: Distance matrix.

    Returns:
        Normalised distance in [0.0, 1.0].
    """
    raise NotImplementedError
