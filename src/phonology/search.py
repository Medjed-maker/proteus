"""BLAST-like three-stage phonological search over a Greek lemma lexicon."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
import heapq
import logging
import math
from typing import Any, Literal, NamedTuple, TypeAlias

import yaml  # type: ignore[import-untyped]

from ._phones import VOWEL_PHONES
from .distance import MatrixData, phone_distance
from .explainer import Alignment, RuleApplication, explain, load_rules
from .ipa_converter import apply_koine_consonant_shifts, to_ipa, tokenize_ipa

DistanceMatrix: TypeAlias = MatrixData
KmerIndex: TypeAlias = dict[str, list[str]]
LexiconEntry: TypeAlias = dict[str, Any]


class LexiconRecord(NamedTuple):
    """A lexicon entry paired with its cached IPA token count."""

    entry: LexiconEntry
    token_count: int


LexiconMap: TypeAlias = dict[str, LexiconRecord]
LexiconLookupValue: TypeAlias = LexiconEntry | LexiconRecord
LexiconLookup: TypeAlias = dict[str, LexiconLookupValue]

logger = logging.getLogger(__name__)

__all__ = [
    "LexiconEntry",
    "LexiconMap",
    "LexiconRecord",
    "SearchResult",
    "build_kmer_index",
    "build_lexicon_map",
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

# Prefix for synthetic observed-difference annotations generated during alignment.
# These pseudo-rules mark raw phoneme mismatches that were not explained by any
# phonological rule in the loaded rule set.
OBSERVED_PREFIX = "OBS-"


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


def _entry_id(entry: LexiconEntry) -> str:
    """Return a stable id for a lexicon entry."""
    raw_id = entry.get("id") or entry.get("headword")
    if not isinstance(raw_id, str) or not raw_id.strip():
        raise ValueError("Lexicon entries must define a non-empty 'id' or 'headword'")
    return raw_id


def _lemma_label(entry: LexiconEntry) -> str:
    """Return the display lemma for a lexicon entry."""
    raw_lemma = entry.get("headword")
    if not isinstance(raw_lemma, str) or not raw_lemma.strip():
        raise ValueError("Lexicon entries must define a non-empty 'headword'")
    return raw_lemma


def _entry_ipa(entry: LexiconEntry) -> str:
    """Return the IPA field for a lexicon entry."""
    raw_ipa = entry.get("ipa")
    if not isinstance(raw_ipa, str) or not raw_ipa.strip():
        raise ValueError("Lexicon entries must define a non-empty 'ipa'")
    return raw_ipa


def _build_entry_lookup(lexicon: Sequence[LexiconEntry]) -> dict[str, LexiconEntry]:
    """Build an entry-id lookup without tokenizing IPA strings."""
    result: dict[str, LexiconEntry] = {}
    for entry in lexicon:
        entry_id = _entry_id(entry)
        if entry_id in result:
            raise ValueError(
                f"Duplicate entry ID {entry_id!r} in lexicon; "
                f"existing entry: {result[entry_id]!r}, "
                f"new duplicate entry: {entry!r}"
            )
        result[entry_id] = entry
    return result


def _lookup_entry(record_or_entry: LexiconLookupValue) -> LexiconEntry:
    """Return the underlying lexicon entry from a lookup value."""
    if isinstance(record_or_entry, LexiconRecord):
        return record_or_entry.entry
    return record_or_entry


def _extract_consonant_skeleton(tokens: list[str]) -> list[str]:
    """Drop vowels from a tokenized IPA sequence to form a consonant skeleton."""
    return [token for token in tokens if token not in VOWEL_PHONES]


def _build_entry_kmers(ipa_text: str, k: int) -> list[str]:
    """Return stable, de-duplicated seed k-mers for one lexicon IPA form.

    Each entry contributes both its stored Attic-oriented skeleton and the
    Koine-compatible skeleton produced by the shared consonant-shift logic.
    
    Args:
        ipa_text: IPA transcription string for a lexicon entry.
        k: Size of each consonant-skeleton k-mer.

    Returns:
        De-duplicated list of space-joined k-mer strings.
    """
    original_tokens = tokenize_ipa(ipa_text)
    original_skeleton = _extract_consonant_skeleton(original_tokens)
    koine_skeleton = _extract_consonant_skeleton(apply_koine_consonant_shifts(original_tokens))
    return list(
        dict.fromkeys(
            [
                *_iter_kmers(original_skeleton, k),
                *_iter_kmers(koine_skeleton, k),
            ]
        )
    )


def _iter_kmers(tokens: list[str], k: int) -> list[str]:
    """Return space-joined k-mers for a token sequence.
    
    Args:
        tokens: List of IPA tokens.
        k: Size of each k-mer.
        
    Returns:
        List of space-joined k-mer strings.
    """
    if k <= 0:
        raise ValueError(f"k-mer size must be positive, got {k}")
    if len(tokens) < k:
        return []
    return [" ".join(tokens[index : index + k]) for index in range(len(tokens) - k + 1)]


def build_kmer_index(
    lexicon: Sequence[dict[str, Any]],
    k: int = _DEFAULT_KMER_SIZE,
) -> KmerIndex:
    """Build a consonant-skeleton k-mer index for lexicon lookup.

    Args:
        lexicon: Sequence of lexicon entry dicts to index.
        k: Size of each consonant-skeleton k-mer. Defaults to ``2``.

    Returns:
        KmerIndex mapping each space-joined k-mer to a list of matching
        lexicon entry ids.

    Raises:
        ValueError: If ``k <= 0``.
    """
    if k <= 0:
        raise ValueError(f"build_kmer_index requires k > 0 for k-mer size, got {k}")

    index: KmerIndex = {}
    for entry in lexicon:
        entry_id = _entry_id(entry)
        for kmer in _build_entry_kmers(_entry_ipa(entry), k):
            index.setdefault(kmer, []).append(entry_id)
    return index


def _rank_by_token_count_proximity(
    query_ipa: str,
    lexicon_map: LexiconMap,
    *,
    max_candidates: int | None = None,
    query_token_count: int | None = None,
) -> list[str]:
    """Rank candidates whose IPA token count is closest to the query's.

    Used as a last-resort fallback when no consonant k-mers can be
    generated (e.g. pure-vowel queries). Returns entry IDs sorted by
    ascending token-count difference, then exact-IPA matches, then entry ID.
    By default callers can evaluate the full ranked list; ``max_candidates``
    is only an explicit override for a capped fallback scan.
    
    If ``query_token_count`` is provided, it is used directly instead of
    re-tokenizing ``query_ipa``.
    """
    if max_candidates is not None and max_candidates <= 0:
        raise ValueError("max_candidates must be a positive integer when provided")

    query_length = query_token_count
    if query_length is None:
        query_length = len(tokenize_ipa(query_ipa))

    def _sort_key(item: tuple[str, LexiconRecord]) -> tuple[int, bool, str]:
        entry_id, record = item
        return (
            abs(record.token_count - query_length),
            _entry_ipa(record.entry) != query_ipa,
            entry_id,
        )

    if max_candidates is not None:
        top = heapq.nsmallest(max_candidates, lexicon_map.items(), key=_sort_key)
        return [entry_id for entry_id, _ in top]

    scored = sorted(lexicon_map.items(), key=_sort_key)
    return [entry_id for entry_id, _ in scored]


def seed_stage(
    query_ipa: str,
    index: KmerIndex,
    k: int = _DEFAULT_KMER_SIZE,
) -> list[str]:
    """Stage 1: rank candidate ids by shared consonant-skeleton k-mers.

    Args:
        query_ipa: String of IPA phones representing the search query.
        index: KmerIndex mapping k-mers to lists of candidate IDs.
        k: Size of consonant-skeleton k-mers. Defaults to `_DEFAULT_KMER_SIZE`.

    Returns:
        List of candidate IDs ranked by number of shared consonant-skeleton k-mers.

    Raises:
        ValueError: If `k <= 0`.
    """
    if k <= 0:
        raise ValueError(f"seed_stage requires k > 0 for k-mer size, got {k}")

    query_skeleton = _extract_consonant_skeleton(tokenize_ipa(query_ipa))
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
    # Preserve the end-of-local-alignment position before traceback mutates
    # row/col.  When no match is found (best_position stays at (0, 0)), the
    # traceback loop is skipped and all tokens fall into the suffix edge
    # alignment, which is the correct degenerate-case behaviour.
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


def build_lexicon_map(lexicon: Sequence[LexiconEntry]) -> LexiconMap:
    """Build a lexicon map with cached IPA token counts for each entry.

    Args:
        lexicon: Sequence of lexicon entry dicts to index.

    Returns:
        LexiconMap mapping entry ids to LexiconRecord instances.
    
    Raises:
        ValueError: If duplicate entry IDs are found in the lexicon.
    """
    result: LexiconMap = {}
    for entry_id, entry in _build_entry_lookup(lexicon).items():
        result[entry_id] = LexiconRecord(
            entry=entry,
            token_count=len(tokenize_ipa(_entry_ipa(entry))),
        )
    return result


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


def _is_observed_application(application: RuleApplication) -> bool:
    """Return True for synthetic observed-difference annotations."""
    rule_id = application.rule_id
    if not isinstance(rule_id, str) or not rule_id:
        return False
    return rule_id.startswith(OBSERVED_PREFIX)


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
    lexicon_map: LexiconLookup,
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
        lexicon_map: Mapping from entry id to either full lexicon entry dicts
            or ``LexiconRecord`` instances. Each entry must contain
            ``"headword"``, ``"ipa"``, and optionally ``"dialect"`` keys.
        matrix: Phonological distance matrix used for substitution scoring.
        language: Language identifier selecting the phonological rule set.
            Defaults to ``"ancient_greek"``.

    Returns:
        Unranked list of ``SearchResult`` objects, one per successfully
        resolved candidate.  Callers should pass them through
        ``filter_stage`` for ranking and truncation.

    Raises:
        ValueError: If a candidate lexicon entry is missing a non-empty
            ``"headword"`` or ``"ipa"`` field and ``_lemma_label`` or
            ``_entry_ipa`` rejects it.
    """
    query_tokens = tokenize_ipa(query_ipa)
    results: list[SearchResult] = []
    rules_registry = _get_rules_registry(language)
    rules = list(rules_registry.values())

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

        # applied_rules: rule IDs excluding observed-difference annotations (OBS-*).
        # Used by api.main.explain_alignment for rule-based explanations.
        # rule_applications: full application list including observed annotations.
        # Used by api.main for Explanation construction when no rule-based explanation exists.
        # This asymmetry is intentional: applied_rules captures phonological rules only,
        # while rule_applications preserves the complete alignment history.
        results.append(
            SearchResult(
                lemma=lemma,
                confidence=confidence,
                dialect_attribution=dialect_attribution,
                applied_rules=[
                    application.rule_id
                    for application in applications
                    if not _is_observed_application(application)
                ],
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
    lexicon: Sequence[LexiconEntry],
    matrix: DistanceMatrix,
    max_results: int = 5,
    dialect: str = "attic",
    index: KmerIndex | None = None,
    unigram_index: KmerIndex | None = None,
    prebuilt_lexicon_map: LexiconMap | None = None,
    language: str = "ancient_greek",
    similarity_fallback_limit: int | None = None,
) -> list[SearchResult]:
    """Run full three-stage search for a Greek query word.

    Args:
        query: Greek query string to normalize and search.
        lexicon: Lexicon entries to search over.
        matrix: Distance matrix used for phone substitution scoring.
        max_results: Maximum number of ranked hits to return.
        dialect: Dialect/model used for IPA conversion. Supports ``"attic"``
            and query-side ``"koine"`` normalization. Defaults to ``"attic"``.
        index: Optional precomputed k-mer index to reuse for faster searches.
        unigram_index: Optional precomputed k=1 index used as fallback when
            the default k=2 index produces no seed candidates. This fallback
            only reorders stage-2 work for short queries; it does not prune
            the candidate set.
        prebuilt_lexicon_map: Optional cached entry-id map with token counts
            to reuse across repeated searches over the same lexicon.
        language: Language identifier selecting the phonological rule set
            passed to ``extend_stage``. Defaults to ``"ancient_greek"``.
        similarity_fallback_limit: Optional explicit cap for the token-count
            fallback path used when both k=2 and k=1 seeds are empty.
            Defaults to uncapped evaluation of the ranked fallback list.

    Returns:
        Ranked search results ordered by descending confidence.

    Raises:
        ValueError: If ``query`` is empty/whitespace-only, ``max_results``
            is non-positive, or a lexicon entry lacks a valid ``"id"`` or
            ``"headword"`` (raised by ``_entry_id`` during lexicon map
            construction).
    """
    if not query.strip():
        raise ValueError("query must be a non-empty string")
    if max_results <= 0:
        raise ValueError("max_results must be a positive integer")
    if similarity_fallback_limit is not None and similarity_fallback_limit <= 0:
        raise ValueError("similarity_fallback_limit must be a positive integer")

    query_ipa = to_ipa(query, dialect=dialect)
    query_tokens = tokenize_ipa(query_ipa)
    query_skeleton = _extract_consonant_skeleton(query_tokens)
    entry_lookup: dict[str, LexiconEntry] | None = None
    lexicon_map = prebuilt_lexicon_map

    def _get_entry_lookup() -> dict[str, LexiconEntry]:
        nonlocal entry_lookup
        if entry_lookup is None:
            entry_lookup = _build_entry_lookup(lexicon)
        return entry_lookup

    def _get_lexicon_lookup() -> LexiconLookup:
        if lexicon_map is not None:
            return lexicon_map
        return _get_entry_lookup()

    def _get_tokenized_lexicon_map() -> LexiconMap:
        nonlocal lexicon_map
        if lexicon_map is None:
            lexicon_map = build_lexicon_map(lexicon)
        return lexicon_map

    search_index = (
        index if index is not None else build_kmer_index(lexicon, k=_DEFAULT_KMER_SIZE)
    )
    seed_candidates = seed_stage(query_ipa, search_index, k=_DEFAULT_KMER_SIZE)
    stage2_limit = max(_MIN_STAGE2_CANDIDATES, max_results * _SEED_MULTIPLIER)
    token_proximity_limit = similarity_fallback_limit

    candidate_ids: list[str]
    lexicon_lookup: LexiconLookup
    if seed_candidates:
        candidate_ids = seed_candidates[:stage2_limit]
        lexicon_lookup = _get_lexicon_lookup()
    else:
        unigram_candidates: list[str] = []
        if query_skeleton:
            fallback_unigram_index = (
                unigram_index if unigram_index is not None else build_kmer_index(lexicon, k=1)
            )
            unigram_candidates = seed_stage(query_ipa, fallback_unigram_index, k=1)
        if unigram_candidates:
            # Keep unigram hits first for lightweight ranking, but preserve full
            # lexicon coverage so short-query recall still comes from stage 2.
            lexicon_lookup = _get_lexicon_lookup()
            unigram_candidate_set = set(unigram_candidates)
            candidate_ids = unigram_candidates + [
                entry_id for entry_id in lexicon_lookup if entry_id not in unigram_candidate_set
            ]
            logger.info(
                "k=2 seed empty for query IPA %r; k=1 fallback evaluating %d candidates",
                query_ipa,
                len(candidate_ids),
            )
        else:
            tokenized_map = _get_tokenized_lexicon_map()
            lexicon_lookup = tokenized_map
            candidate_ids = _rank_by_token_count_proximity(
                query_ipa,
                tokenized_map,
                max_candidates=token_proximity_limit,
                query_token_count=len(query_tokens),
            )
            if token_proximity_limit is None:
                logger.info(
                    "k=2 and k=1 seeds empty for query IPA %r; evaluating all %d candidates "
                    "ranked by token-count proximity",
                    query_ipa,
                    len(candidate_ids),
                )
            else:
                logger.info(
                    "k=2 and k=1 seeds empty for query IPA %r; evaluating %d of %d candidates "
                    "ranked by token-count proximity",
                    query_ipa,
                    len(candidate_ids),
                    len(tokenized_map),
                )

    return filter_stage(
        extend_stage(
            query_ipa=query_ipa,
            candidates=candidate_ids,
            lexicon_map=lexicon_lookup,
            matrix=matrix,
            language=language,
        ),
        max_results=max_results,
    )
