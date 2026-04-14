"""Scoring and alignment functions for the search pipeline.

Contains the pure computation functions: Smith-Waterman alignment, substitution
scoring, confidence normalization, rule marker overlay, and the score stage.

Functions that depend on monkeypatched names (``load_rules``,
``tokenize_rules_for_matching``) remain in ``__init__`` to preserve test
isolation via ``search_module`` patching.
"""

from __future__ import annotations

from collections.abc import Iterable
import logging
import math
from typing import Literal

from ..distance import phone_distance
from ..explainer import (
    Alignment,
    RuleApplication,
)
from ..ipa_converter import tokenize_ipa
from ._annotation import (
    _build_dialect_attribution,
    _candidate_dialect,
    _is_observed_application,
)
from ._lookup import (
    _entry_ipa,
    _lemma_label,
    _lookup_entry,
)
from ._types import (
    DistanceMatrix,
    LexiconLookup,
    LexiconLookupValue,
    LexiconRecord,
    SearchResult,
)

logger = logging.getLogger(__name__)

_GAP_PENALTY: float = -1.0
_MATCH_SCORE: float = 2.0


def _resolve_entry_tokens(record_or_entry: LexiconLookupValue) -> tuple[str, ...]:
    """Return cached IPA tokens when available, tokenizing only as a fallback.

    Unlike ``phonology.search.__init__._lookup_entry_tokens``, this copy
    resolves ``tokenize_ipa`` from ``phonology.ipa_converter`` at import
    time and is therefore **not affected** by monkeypatching
    ``search_module.tokenize_ipa``.  Tests that exercise the scoring
    internals directly (e.g. ``extend_stage``) should pass pre-tokenized
    ``LexiconRecord`` entries.
    """
    if isinstance(record_or_entry, LexiconRecord) and len(record_or_entry.ipa_tokens) > 0:
        return record_or_entry.ipa_tokens
    entry = _lookup_entry(record_or_entry)
    return tuple(tokenize_ipa(_entry_ipa(entry)))


def _substitution_score(lemma_phone: str, query_phone: str, matrix: DistanceMatrix) -> float:
    """Return a Smith-Waterman substitution score for two phones."""
    if lemma_phone == query_phone:
        return _MATCH_SCORE
    return 1.0 - phone_distance(lemma_phone, query_phone, matrix)


def _align_edge_tokens(
    query_tokens: list[str],
    lemma_tokens: list[str],
    *,
    side: Literal["prefix", "suffix"],
) -> tuple[list[str | None], list[str | None]]:
    """Align unmatched edge tokens around a local-alignment core.

    Preserve any exact-match run on the outer edge, then align the remaining
    unmatched phones nearest the local-alignment core.
    """
    if side == "prefix":
        shared_length = 0
        while (
            shared_length < len(query_tokens)
            and shared_length < len(lemma_tokens)
            and query_tokens[shared_length] == lemma_tokens[shared_length]
        ):
            shared_length += 1
        shared_query = list(query_tokens[:shared_length])
        shared_lemma = list(lemma_tokens[:shared_length])
        remaining_query = list(query_tokens[shared_length:])
        remaining_lemma = list(lemma_tokens[shared_length:])
        max_length = max(len(remaining_query), len(remaining_lemma))
        return (
            shared_query + ([None] * (max_length - len(remaining_query))) + remaining_query,
            shared_lemma + ([None] * (max_length - len(remaining_lemma))) + remaining_lemma,
        )
    if side == "suffix":
        shared_length = 0
        while (
            shared_length < len(query_tokens)
            and shared_length < len(lemma_tokens)
            and query_tokens[-(shared_length + 1)] == lemma_tokens[-(shared_length + 1)]
        ):
            shared_length += 1
        if shared_length == 0:
            shared_query = []
            shared_lemma = []
            remaining_query = list(query_tokens)
            remaining_lemma = list(lemma_tokens)
        else:
            shared_query = list(query_tokens[-shared_length:])
            shared_lemma = list(lemma_tokens[-shared_length:])
            remaining_query = list(query_tokens[:-shared_length])
            remaining_lemma = list(lemma_tokens[:-shared_length])
        max_length = max(len(remaining_query), len(remaining_lemma))
        return (
            remaining_query + ([None] * (max_length - len(remaining_query))) + shared_query,
            remaining_lemma + ([None] * (max_length - len(remaining_lemma))) + shared_lemma,
        )
    raise ValueError(f"Unknown edge-alignment side {side!r}")


def _smith_waterman_alignment(
    query_tokens: list[str],
    lemma_tokens: list[str],
    matrix: DistanceMatrix,
) -> tuple[float, list[str | None], list[str | None]]:
    """Compute the best local alignment between lemma and query phone sequences."""
    if not query_tokens or not lemma_tokens:
        return 0.0, [], []

    rows = len(lemma_tokens) + 1
    cols = len(query_tokens) + 1
    scores = [[0.0] * cols for _ in range(rows)]
    directions: list[list[str | None]] = [[None] * cols for _ in range(rows)]
    best_score = 0.0
    best_position = (0, 0)

    for row in range(1, rows):
        for col in range(1, cols):
            diag = scores[row - 1][col - 1] + _substitution_score(
                lemma_tokens[row - 1], query_tokens[col - 1], matrix
            )
            up = scores[row - 1][col] + _GAP_PENALTY
            left = scores[row][col - 1] + _GAP_PENALTY
            # Tie-breaking priority: diag > up > left.
            # Use math.isclose to avoid FP misselection of direction.
            cell_score = max(0.0, diag, up, left)
            scores[row][col] = cell_score
            if cell_score == 0.0:
                directions[row][col] = None
            elif math.isclose(cell_score, diag, rel_tol=1e-9, abs_tol=1e-9):
                directions[row][col] = "diag"
            elif math.isclose(cell_score, up, rel_tol=1e-9, abs_tol=1e-9):
                directions[row][col] = "up"
            elif math.isclose(cell_score, left, rel_tol=1e-9, abs_tol=1e-9):
                directions[row][col] = "left"
            else:
                raise RuntimeError(
                    f"Smith-Waterman direction selection failed at ({row}, {col}): "
                    f"cell_score={cell_score!r} did not match "
                    f"diag={diag!r}, up={up!r}, left={left!r}"
                )

            if cell_score > best_score:
                best_score = cell_score
                best_position = (row, col)

    aligned_query: list[str | None] = []
    aligned_lemma: list[str | None] = []
    end_row, end_col = best_position
    row, col = best_position
    while row > 0 and col > 0 and scores[row][col] > 0.0:
        direction = directions[row][col]
        if direction == "diag":
            aligned_lemma.append(lemma_tokens[row - 1])
            aligned_query.append(query_tokens[col - 1])
            row -= 1
            col -= 1
        elif direction == "up":
            aligned_lemma.append(lemma_tokens[row - 1])
            aligned_query.append(None)
            row -= 1
        elif direction == "left":
            aligned_lemma.append(None)
            aligned_query.append(query_tokens[col - 1])
            col -= 1
        else:
            break

    aligned_query.reverse()
    aligned_lemma.reverse()

    prefix_query, prefix_lemma = _align_edge_tokens(
        query_tokens[:col],
        lemma_tokens[:row],
        side="prefix",
    )
    suffix_query, suffix_lemma = _align_edge_tokens(
        query_tokens[end_col:],
        lemma_tokens[end_row:],
        side="suffix",
    )
    return (
        best_score,
        prefix_query + aligned_query + suffix_query,
        prefix_lemma + aligned_lemma + suffix_lemma,
    )


def _normalized_confidence(
    best_local_score: float,
    query_tokens: list[str],
    lemma_tokens: list[str],
) -> float:
    """Normalize a Smith-Waterman score into the 0.0-1.0 interval."""
    denominator = 2.0 * max(len(query_tokens), len(lemma_tokens), 1)
    confidence = best_local_score / denominator
    if query_tokens == lemma_tokens:
        return 1.0
    return max(0.0, min(1.0, confidence))


def _apply_rule_markers(
    markers: list[str],
    aligned_query: list[str | None],
    aligned_lemma: list[str | None],
    applications: list[RuleApplication],
) -> list[str]:
    """Overlay rule spans on top of baseline alignment markers."""
    local_markers = list(markers)
    if not applications:
        return local_markers

    lemma_alignment_indices = [
        aligned_index
        for aligned_index, lemma_token in enumerate(aligned_lemma)
        if lemma_token is not None
    ]
    for application in applications:
        if application.position < 0 or _is_observed_application(application):
            continue
        input_tokens = tokenize_ipa(application.input_phoneme)
        for offset in range(len(input_tokens)):
            lemma_position = application.position + offset
            if lemma_position >= len(lemma_alignment_indices):
                continue
            aligned_index = lemma_alignment_indices[lemma_position]
            if aligned_query[aligned_index] is not None:
                local_markers[aligned_index] = ":"
    return local_markers


def _score_stage(
    query_ipa: str,
    candidates: Iterable[str],
    lexicon_map: LexiconLookup,
    matrix: DistanceMatrix,
) -> list[SearchResult]:
    """Stage 2a: score candidates without explanation or visualization work."""
    query_tokens = tokenize_ipa(query_ipa)
    results: list[SearchResult] = []

    for candidate_id in candidates:
        record_or_entry = lexicon_map.get(candidate_id)
        if record_or_entry is None:
            logger.debug(
                "Skipping candidate_id %r not found in lexicon_map (size=%d)",
                candidate_id,
                len(lexicon_map),
            )
            continue
        entry = _lookup_entry(record_or_entry)
        lemma = _lemma_label(entry)
        lemma_ipa = _entry_ipa(entry)
        lemma_tokens = list(_resolve_entry_tokens(record_or_entry))
        best_score, aligned_query, aligned_lemma = _smith_waterman_alignment(
            query_tokens, lemma_tokens, matrix
        )
        confidence = _normalized_confidence(best_score, query_tokens, lemma_tokens)
        result = SearchResult(
            lemma=lemma,
            confidence=confidence,
            dialect_attribution=_build_dialect_attribution(_candidate_dialect(entry)),
            ipa=lemma_ipa,
            entry_id=candidate_id,
            alignment=Alignment(
                aligned_query=tuple(aligned_query),
                aligned_lemma=tuple(aligned_lemma),
            ),
        )
        results.append(result)

    return results
