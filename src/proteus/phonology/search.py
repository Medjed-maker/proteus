"""BLAST-like three-stage phonological search over a Greek lemma lexicon."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
import logging
import math
from typing import Any, TypeAlias

import yaml

from ._phones import VOWEL_PHONES
from .distance import MatrixData, phone_distance
from .explainer import Alignment, RuleApplication, explain, load_rules
from .ipa_converter import to_ipa, tokenize_ipa

DistanceMatrix: TypeAlias = MatrixData
KmerIndex: TypeAlias = dict[str, list[str]]

logger = logging.getLogger(__name__)

__all__ = [
    "SearchResult",
    "build_kmer_index",
    "seed_stage",
    "extend_stage",
    "filter_stage",
    "search",
]

_DEFAULT_KMER_SIZE = 2
_SEED_MULTIPLIER = 10
_MIN_STAGE2_CANDIDATES = 25
_GAP_PENALTY = -1.0
_MATCH_SCORE = 2.0


@dataclass
class SearchResult:
    """Single ranked hit returned by phonological search.

    ``dialect_attribution`` is always set to a descriptive string by
    ``extend_stage``, but defaults to ``None`` so that callers constructing
    instances outside the pipeline (e.g. tests, future stages) are not
    forced to supply a value.  Consumers should handle ``None`` gracefully.
    """

    lemma: str
    confidence: float
    dialect_attribution: str | None = None
    applied_rules: list[str] = field(default_factory=list)
    rule_applications: list[RuleApplication] = field(default_factory=list)
    alignment_visualization: str = ""
    ipa: str | None = None


def _entry_id(entry: dict[str, Any]) -> str:
    """Return a stable id for a lexicon entry."""
    raw_id = entry.get("id") or entry.get("headword")
    if not isinstance(raw_id, str) or not raw_id.strip():
        raise ValueError("Lexicon entries must define a non-empty 'id' or 'headword'")
    return raw_id


def _lemma_label(entry: dict[str, Any]) -> str:
    """Return the display lemma for a lexicon entry."""
    raw_lemma = entry.get("headword")
    if not isinstance(raw_lemma, str) or not raw_lemma.strip():
        raise ValueError("Lexicon entries must define a non-empty 'headword'")
    return raw_lemma


def _entry_ipa(entry: dict[str, Any]) -> str:
    """Return the IPA field for a lexicon entry."""
    raw_ipa = entry.get("ipa")
    if not isinstance(raw_ipa, str) or not raw_ipa.strip():
        raise ValueError("Lexicon entries must define a non-empty 'ipa'")
    return raw_ipa


def _extract_consonant_skeleton(ipa_text: str) -> list[str]:
    """Drop vowels from a tokenized IPA string to form a consonant skeleton."""
    return [token for token in tokenize_ipa(ipa_text) if token not in VOWEL_PHONES]


def _iter_kmers(tokens: list[str], k: int) -> list[str]:
    """Return space-joined k-mers for a token sequence."""
    if k <= 0:
        raise ValueError(f"k-mer size must be positive, got {k}")
    if len(tokens) < k:
        return []
    return [" ".join(tokens[index : index + k]) for index in range(len(tokens) - k + 1)]


def build_kmer_index(
    lexicon: Sequence[dict[str, Any]],
    k: int = _DEFAULT_KMER_SIZE,
) -> KmerIndex:
    """Build a consonant-skeleton k-mer index over the lexicon."""
    if k <= 0:
        raise ValueError(f"build_kmer_index requires k > 0 for k-mer size, got {k}")

    index: KmerIndex = {}
    for entry in lexicon:
        entry_id = _entry_id(entry)
        skeleton = _extract_consonant_skeleton(_entry_ipa(entry))
        for kmer in _iter_kmers(skeleton, k):
            index.setdefault(kmer, []).append(entry_id)
    return index


def seed_stage(
    query_ipa: str,
    index: KmerIndex,
    k: int = _DEFAULT_KMER_SIZE,
) -> list[str]:
    """Stage 1: rank candidate ids by shared consonant-skeleton k-mers."""
    if k <= 0:
        raise ValueError(f"seed_stage requires k > 0 for k-mer size, got {k}")

    query_skeleton = _extract_consonant_skeleton(query_ipa)
    query_kmers = _iter_kmers(query_skeleton, k)
    if not query_kmers:
        return []

    counts: Counter[str] = Counter()
    for kmer, weight in Counter(query_kmers).items():
        for candidate_id in index.get(kmer, []):
            counts[candidate_id] += weight

    return [
        candidate_id
        for candidate_id, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _substitution_score(lemma_phone: str, query_phone: str, matrix: DistanceMatrix) -> float:
    """Return a Smith-Waterman substitution score for two phones."""
    if lemma_phone == query_phone:
        return _MATCH_SCORE
    return 1.0 - phone_distance(lemma_phone, query_phone, matrix)


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
    return best_score, aligned_query, aligned_lemma


def _normalized_confidence(
    best_local_score: float,
    query_tokens: list[str],
    lemma_tokens: list[str],
) -> float:
    """Normalize a Smith-Waterman score into the 0.0-1.0 interval."""
    denominator = 2.0 * max(len(query_tokens), len(lemma_tokens), 1)
    confidence = best_local_score / denominator
    # Short-circuit for exact matches: the Smith-Waterman score divided by
    # the denominator may not equal exactly 1.0 due to floating-point
    # rounding, so return 1.0 directly to avoid confusing API consumers.
    if query_tokens == lemma_tokens:
        return 1.0
    return max(0.0, min(1.0, confidence))


@lru_cache(maxsize=8)
def _get_rules_registry(language: str = "ancient_greek") -> dict[str, dict[str, Any]]:
    """Load the packaged rule registry once per process."""
    try:
        return load_rules(language)
    except (OSError, ValueError, yaml.YAMLError) as err:
        raise ValueError(
            f"_get_rules_registry failed to load rules for language {language!r}: {err}"
        ) from err


def _build_alignment_markers(
    aligned_query: list[str | None],
    aligned_lemma: list[str | None],
) -> list[str]:
    """Build baseline visualization markers for an aligned token sequence."""
    markers: list[str] = []
    for query_token, lemma_token in zip(aligned_query, aligned_lemma):
        if query_token is not None and lemma_token is not None and query_token == lemma_token:
            markers.append("|")
        elif query_token is None or lemma_token is None:
            markers.append(" ")
        else:
            markers.append(".")
    return markers


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
        if application.position < 0:
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


def _collect_application_dialects(applications: list[RuleApplication]) -> list[str]:
    """Return dialect labels from rule applications in first-seen order."""
    matched_dialects: list[str] = []
    seen_dialects: set[str] = set()
    for application in applications:
        for dialect in application.dialects:
            if dialect in seen_dialects:
                continue
            seen_dialects.add(dialect)
            matched_dialects.append(dialect)
    return matched_dialects


def _format_alignment_visualization(
    aligned_query: list[str | None],
    aligned_lemma: list[str | None],
    markers: list[str],
) -> str:
    """Render a fixed three-line ASCII alignment visualization."""
    query_cells = [token if token is not None else "-" for token in aligned_query]
    lemma_cells = [token if token is not None else "-" for token in aligned_lemma]

    if not query_cells and not lemma_cells:
        return "query:\n       \nlemma:"

    widths = [
        max(len(query_cell), len(lemma_cell), 1)
        for query_cell, lemma_cell in zip(query_cells, lemma_cells)
    ]
    query_line = "query: " + " ".join(
        f"{cell:<{width}}" for cell, width in zip(query_cells, widths)
    )
    marker_line = "       " + " ".join(
        f"{marker:<{width}}" for marker, width in zip(markers, widths)
    )
    lemma_line = "lemma: " + " ".join(
        f"{cell:<{width}}" for cell, width in zip(lemma_cells, widths)
    )
    return "\n".join([query_line.rstrip(), marker_line.rstrip(), lemma_line.rstrip()])


def extend_stage(
    query_ipa: str,
    candidates: Iterable[str],
    lexicon_map: dict[str, dict[str, Any]],
    matrix: DistanceMatrix,
    language: str = "ancient_greek",
) -> list[SearchResult]:
    """Stage 2: run Smith-Waterman on candidate IPA forms and assemble results.

    For each candidate, compute a local alignment score, detect matching
    phonological rules, attribute dialects, and build a three-line ASCII
    visualization.

    Args:
        query_ipa: IPA transcription of the search query (space-separated or
            compact notation accepted by ``tokenize_ipa``).
        candidates: Iterable of lexicon entry ids produced by the seed stage.
        lexicon_map: Mapping from entry id to full lexicon entry dict.  Each
            entry must contain ``"headword"``, ``"ipa"``, and optionally
            ``"dialect"`` keys.
        matrix: Phonological distance matrix used for substitution scoring.
        language: Language identifier selecting the phonological rule set.
            Defaults to ``"ancient_greek"``.

    Returns:
        Unranked list of ``SearchResult`` objects, one per successfully
        resolved candidate.  Callers should pass them through
        ``filter_stage`` for ranking and truncation.
    """
    query_tokens = tokenize_ipa(query_ipa)
    results: list[SearchResult] = []
    rules_registry = _get_rules_registry(language)
    rules = list(rules_registry.values())

    for candidate_id in candidates:
        entry = lexicon_map.get(candidate_id)
        if entry is None:
            # Skip stale candidate ids if the seed index and lexicon map drift apart.
            continue
        lemma = _lemma_label(entry)
        lemma_ipa = _entry_ipa(entry)
        lemma_tokens = tokenize_ipa(lemma_ipa)
        best_score, aligned_query, aligned_lemma = _smith_waterman_alignment(
            query_tokens, lemma_tokens, matrix
        )
        confidence = _normalized_confidence(best_score, query_tokens, lemma_tokens)
        applications = explain(
            query_ipa=query_tokens,
            lemma_ipa=lemma_tokens,
            alignment=Alignment(
                aligned_query=tuple(aligned_query),
                aligned_lemma=tuple(aligned_lemma),
            ),
            rules=rules,
        )
        matched_dialects = _collect_application_dialects(applications)
        markers = _apply_rule_markers(
            _build_alignment_markers(aligned_query, aligned_lemma),
            aligned_query,
            aligned_lemma,
            applications,
        )
        dialect = entry.get("dialect")
        candidate_dialect = (
            "unknown" if dialect is None or str(dialect).strip() == "" else str(dialect)
        )
        if matched_dialects:
            dialect_attribution = (
                f"lemma dialect: {candidate_dialect}; "
                f"query-compatible dialects: {', '.join(matched_dialects)}"
            )
        else:
            dialect_attribution = f"lemma dialect: {candidate_dialect}"

        results.append(
            SearchResult(
                lemma=lemma,
                confidence=confidence,
                dialect_attribution=dialect_attribution,
                applied_rules=[application.rule_id for application in applications],
                rule_applications=list(applications),
                alignment_visualization=_format_alignment_visualization(
                    aligned_query, aligned_lemma, markers
                ),
                ipa=lemma_ipa,
            )
        )

    return results


def filter_stage(results: list[SearchResult], max_results: int) -> list[SearchResult]:
    """Stage 3: sort by confidence and keep the top N results."""
    if max_results <= 0:
        raise ValueError("max_results must be a positive integer")
    return sorted(results, key=lambda result: (-result.confidence, result.lemma))[:max_results]


def search(
    query: str,
    lexicon: Sequence[dict[str, Any]],
    matrix: DistanceMatrix,
    max_results: int = 5,
    dialect: str = "attic",
    index: KmerIndex | None = None,
    language: str = "ancient_greek",
) -> list[SearchResult]:
    """Run full three-stage search for a Greek query word.

    Args:
        query: Greek query string to normalize and search.
        lexicon: Lexicon entries to search over.
        matrix: Distance matrix used for phone substitution scoring.
        max_results: Maximum number of ranked hits to return.
        dialect: Dialect/model used for IPA conversion. Defaults to ``"attic"``.
        index: Optional precomputed k-mer index to reuse for faster searches.
        language: Language identifier selecting the phonological rule set
            passed to ``extend_stage``. Defaults to ``"ancient_greek"``.

    Returns:
        Ranked search results ordered by descending confidence.
    """
    if not query.strip():
        raise ValueError("query must be a non-empty string")
    if max_results <= 0:
        raise ValueError("max_results must be a positive integer")

    query_ipa = to_ipa(query, dialect=dialect)
    lexicon_map = {_entry_id(entry): entry for entry in lexicon}
    search_index = (
        index if index is not None else build_kmer_index(lexicon, k=_DEFAULT_KMER_SIZE)
    )
    seed_candidates = seed_stage(query_ipa, search_index, k=_DEFAULT_KMER_SIZE)
    stage2_limit = max(_MIN_STAGE2_CANDIDATES, max_results * _SEED_MULTIPLIER)

    if seed_candidates:
        candidate_ids = seed_candidates[:stage2_limit]
    else:
        candidate_ids = lexicon_map.keys()
        logger.warning(
            "No seed candidates found for query IPA %r; falling back to full-lexicon "
            "scan (%d entries)",
            query_ipa,
            len(lexicon_map),
        )

    return filter_stage(
        extend_stage(
            query_ipa=query_ipa,
            candidates=candidate_ids,
            lexicon_map=lexicon_map,
            matrix=matrix,
            language=language,
        ),
        max_results=max_results,
    )
