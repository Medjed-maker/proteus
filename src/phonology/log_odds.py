"""BLOSUM-style log-odds phonological substitution matrix computation.

Workflow:
1. Collect (original, regularized) IPA phone sequences.
2. Align each pair with Needleman-Wunsch (identity scoring by default).
3. Accumulate aligned-column counts across all pairs.
4. Compute log-odds: S(i, j) = log2(q_ij / expected_ij).
5. Build a JSON-serialisable matrix document.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from .explainer import Alignment

__all__ = [
    "CountTables",
    "NWParams",
    "accumulate_counts",
    "build_matrix_document",
    "compute_log_odds",
    "needleman_wunsch",
]

_DIAG = 0
_UP = 1
_LEFT = 2


@dataclass(frozen=True)
class NWParams:
    """Scoring parameters for identity-based Needleman-Wunsch alignment."""

    match: float = 0.0
    mismatch: float = 1.0
    gap: float = 1.25


@dataclass(frozen=True)
class CountTables:
    """Aligned-column count tables accumulated across an alignment corpus."""

    pair_counts: dict[tuple[str, str], int]
    phone_totals: Counter[str]
    indel_counts: Counter[str]
    pair_total: int
    phone_total: int


def needleman_wunsch(
    seq1: Sequence[str],
    seq2: Sequence[str],
    params: NWParams | None = None,
) -> Alignment:
    """Global alignment with full traceback.

    Uses identity scoring: zero cost for matches, ``params.mismatch`` for
    substitutions, ``params.gap`` for insertions/deletions.  Tie-break order
    is DIAG > UP > LEFT for determinism.

    Returns an :class:`~phonology.explainer.Alignment` where ``None`` entries
    represent gaps.
    """
    if params is None:
        params = NWParams()
    rows = len(seq1) + 1
    cols = len(seq2) + 1

    costs = [[0.0] * cols for _ in range(rows)]
    ptrs = [[_DIAG] * cols for _ in range(rows)]

    for i in range(1, rows):
        costs[i][0] = i * params.gap
        ptrs[i][0] = _UP
    for j in range(1, cols):
        costs[0][j] = j * params.gap
        ptrs[0][j] = _LEFT

    for i in range(1, rows):
        for j in range(1, cols):
            sub_cost = params.match if seq1[i - 1] == seq2[j - 1] else params.mismatch
            diag = costs[i - 1][j - 1] + sub_cost
            up = costs[i - 1][j] + params.gap
            left = costs[i][j - 1] + params.gap

            if diag <= up and diag <= left:
                costs[i][j] = diag
                ptrs[i][j] = _DIAG
            elif up <= left:
                costs[i][j] = up
                ptrs[i][j] = _UP
            else:
                costs[i][j] = left
                ptrs[i][j] = _LEFT

    aligned_q: list[str | None] = []
    aligned_l: list[str | None] = []
    i, j = len(seq1), len(seq2)
    while i > 0 or j > 0:
        ptr = ptrs[i][j]
        if ptr == _DIAG:
            aligned_q.append(seq1[i - 1])
            aligned_l.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif ptr == _UP:
            aligned_q.append(seq1[i - 1])
            aligned_l.append(None)
            i -= 1
        else:
            aligned_q.append(None)
            aligned_l.append(seq2[j - 1])
            j -= 1

    aligned_q.reverse()
    aligned_l.reverse()
    return Alignment(
        aligned_query=tuple(aligned_q),
        aligned_lemma=tuple(aligned_l),
    )


def accumulate_counts(alignments: Iterable[Alignment]) -> CountTables:
    """Aggregate aligned-column statistics from an iterable of alignments.

    Each matched column (both phones non-None) increments ``pair_counts`` and
    ``phone_totals`` for both phones.  Gap columns increment ``indel_counts``
    for the non-gap phone.  The canonical pair key is ``(min(a, b), max(a, b))``,
    so counts are symmetrised at accumulation time.
    """
    pair_counts: dict[tuple[str, str], int] = {}
    phone_totals: Counter[str] = Counter()
    indel_counts: Counter[str] = Counter()
    pair_total = 0
    phone_total = 0

    for alignment in alignments:
        for q, lemma in zip(
            alignment.aligned_query, alignment.aligned_lemma, strict=True
        ):
            if q is not None and lemma is not None:
                key = (min(q, lemma), max(q, lemma))
                pair_counts[key] = pair_counts.get(key, 0) + 1
                phone_totals[q] += 1
                phone_totals[lemma] += 1
                pair_total += 1
                phone_total += 2
            elif q is not None:
                indel_counts[q] += 1
            elif lemma is not None:
                indel_counts[lemma] += 1
            else:
                # Both q and lemma are None.  The Needleman-Wunsch traceback
                # guarantees this cannot happen: every alignment column
                # advances at least one of the two sequences.  Guard here
                # defensively so that a future algorithm change surfaces the
                # issue immediately rather than silently skipping data.
                raise RuntimeError(
                    "Alignment produced a column where both phones are None"
                )

    return CountTables(
        pair_counts=pair_counts,
        phone_totals=phone_totals,
        indel_counts=indel_counts,
        pair_total=pair_total,
        phone_total=phone_total,
    )


def compute_log_odds(
    counts: CountTables,
    *,
    smoothing: Literal["laplace", "lidstone", "floor"] = "laplace",
    lidstone_alpha: float = 0.5,
    floor: float = -10.0,
    min_count: int = 0,
) -> dict[str, dict[str, float]]:
    """Compute a symmetric log-odds substitution score matrix.

    Formula::

        S(i, j) = log2(q_ij / expected_ij)

    where ``q_ij`` is the smoothed observed joint frequency of the unordered
    pair ``{i, j}``, and ``expected_ij = p_i ** 2`` for ``i == j`` and
    ``2 * p_i * p_j`` for ``i != j`` (the factor of 2 accounts for symmetric
    counting of unordered pairs).

    Smoothing strategies:

    - ``"laplace"``: add 1 to every unordered-pair count before computing
      frequencies.
    - ``"lidstone"``: add ``lidstone_alpha`` to every count.
    - ``"floor"``: no additive smoothing; zero-count cells are clamped to
      ``floor`` rather than yielding ``-inf``.

    Cells whose raw symmetrised count falls below ``min_count`` are omitted
    from the output; callers may fall back to a default for missing cells.
    """
    if counts.phone_total == 0:
        return {}

    alphabet = sorted(counts.phone_totals)
    n = len(alphabet)
    n_pairs = n * (n + 1) // 2

    p: dict[str, float] = {
        ph: counts.phone_totals[ph] / counts.phone_total for ph in alphabet
    }

    if smoothing == "laplace":
        alpha = 1.0
    elif smoothing == "lidstone":
        alpha = lidstone_alpha
    else:
        alpha = 0.0

    smoothed_total = counts.pair_total + alpha * n_pairs

    scores: dict[str, dict[str, float]] = {ph: {} for ph in alphabet}

    for idx_i, phone_i in enumerate(alphabet):
        for phone_j in alphabet[idx_i:]:
            key = (phone_i, phone_j)
            raw_count = counts.pair_counts.get(key, 0)

            if raw_count < min_count:
                continue

            smoothed_count = raw_count + alpha

            q_ij = smoothed_count / smoothed_total if smoothed_total > 0 else 0.0

            p_i = p[phone_i]
            p_j = p[phone_j]
            expected = p_i * p_j if phone_i == phone_j else 2.0 * p_i * p_j

            if q_ij <= 0.0 or expected <= 0.0:
                s = floor
            else:
                s = math.log2(q_ij / expected)
                if not math.isfinite(s):
                    s = floor

            scores[phone_i][phone_j] = s
            scores[phone_j][phone_i] = s

    return scores


def build_matrix_document(
    counts: CountTables,
    scores: dict[str, dict[str, float]],
    *,
    source_path: str,
    smoothing: str,
    smoothing_params: dict[str, Any],
    nw_params: NWParams,
    source_pair_count: int,
    alignments_used: int,
) -> dict[str, Any]:
    """Assemble the final JSON-serialisable log-odds matrix document."""
    now = datetime.now(timezone.utc)
    generated_at = now.isoformat(timespec="milliseconds")

    alphabet = sorted(counts.phone_totals)

    return {
        "_meta": {
            "description": "BLOSUM-style log-odds phonological substitution matrix",
            "version": "1.0.0",
            "generated_at": generated_at,
            "method": (
                f"Needleman-Wunsch (identity scoring) + "
                f"log2(q_ij / expected_ij), where expected_ij = p_i^2 for "
                f"i == j and 2 * p_i * p_j for i != j; {smoothing} smoothing"
            ),
            "range": "real numbers; positive = more likely than chance, negative = less",
            "source": source_path,
            "source_pair_count": source_pair_count,
            "alignments_used": alignments_used,
            "alphabet": alphabet,
            "smoothing": {"strategy": smoothing, "params": smoothing_params},
            "nw_params": {
                "match": nw_params.match,
                "mismatch": nw_params.mismatch,
                "gap": nw_params.gap,
            },
            "indel_counts": dict(counts.indel_counts),
            "totals": {
                "pair_total": counts.pair_total,
                "phone_total": counts.phone_total,
            },
        },
        "scores": scores,
    }
