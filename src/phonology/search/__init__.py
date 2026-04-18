"""BLAST-like three-stage phonological search over a Greek lemma lexicon."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import replace
from typing import Any, NamedTuple, Protocol
from functools import lru_cache
import heapq
import logging

import yaml  # type: ignore[import-untyped]

from ..explainer import (
    Alignment,
    TokenizedRule,
    explain,
    explain_with_tokenized_rules,
    load_rules,
    tokenize_rules_for_matching,
)
from ..ipa_converter import to_ipa, tokenize_ipa
from ._constants import (
    _annotation_candidate_limit,
    _DEFAULT_FALLBACK_CANDIDATE_LIMIT,
    _DEFAULT_KMER_SIZE,
    _MIN_PARTIAL_STAGE2_CANDIDATES,
    _MIN_STAGE2_CANDIDATES,
    _partial_candidate_limit,
    _PARTIAL_QUERY_CONFIDENCE_THRESHOLD,
    _SEED_MULTIPLIER,
    _SHORT_QUERY_CONFIDENCE_THRESHOLD,
    _SHORT_QUERY_MAX_ANNOTATION_BATCHES,
    OBSERVED_PREFIX,
)
from ._lookup import (
    _build_entry_lookup,
    _entry_ipa,
    _inject_exact_ipa_matches,
    _lemma_label,
    _lookup_entry,
    build_ipa_index,
    IpaIndex,
)
from ._indexing import _iter_kmers, build_kmer_index
from ._annotation import (
    _build_alignment_markers,
    _build_dialect_attribution,
    _candidate_dialect,
    _collect_application_dialects,
    _format_alignment_visualization,
    _is_observed_application,
)
from ._scoring import (
    _apply_rule_markers,
    _normalized_confidence,
    _score_stage,
    _smith_waterman_alignment,
    _substitution_score,
)
from ._overlap import (
    _merge_bounded_candidate_ids,
    _token_count_proximity_key,
)
from ._partial import (
    _match_partial_query,
    _select_partial_fallback_candidates,
    _select_partial_seed_candidates,
)
from ._filtering import (
    _apply_mode_quality_filter,
    _rank_short_query_annotation_candidates,
    _select_annotation_candidates,
)
from ._dedup import (
    _deduplicate_by_headword,
    _deduplicate_by_headword_common,
)
from ._quality import (
    _rank_short_query_results,
)
from ._query import (
    _classify_non_partial_query,
    _extract_consonant_skeleton,
    _normalize_query_with_pattern,
    _parse_partial_query,
    classify_query_mode,
    normalize_query_for_search,
    prepare_query,
)
from ._types import (
    DistanceMatrix,
    KmerIndex,
    LexiconEntry,
    LexiconLookup,
    LexiconLookupValue,
    LexiconMap,
    LexiconRecord,
    PartialQueryPattern,
    PartialQueryTokens,
    QueryMode,
    SearchResult,
)

logger = logging.getLogger(__name__)

# Private tuning attributes remain available for internal tests but are
# intentionally excluded from the public star-import surface.
__all__ = [
    "LexiconEntry",
    "LexiconMap",
    "LexiconRecord",
    "IpaIndex",
    "OBSERVED_PREFIX",
    "QueryMode",
    "SearchResult",
    "build_ipa_index",
    "build_kmer_index",
    "build_lexicon_map",
    "classify_query_mode",
    "explain",  # re-exported from .explainer for test/consumer access
    "get_rules_registry",
    "seed_stage",
    "extend_stage",
    "filter_stage",
    "normalize_query_for_search",
    "prepare_query",
    "prepare_query_ipa",
    "search",
]


class IpaConverter(Protocol):
    """Protocol for IPA conversion functions."""

    def __call__(self, text: str, *, dialect: str) -> str: ...


class PreparedQueryIpa(NamedTuple):
    """Query classification, normalization, and IPA data for one search."""

    query_mode: QueryMode
    normalized_query: str
    partial_query: PartialQueryPattern | None
    query_ipa: str
    partial_query_tokens: PartialQueryTokens | None


def _lookup_entry_tokens(record_or_entry: LexiconLookupValue) -> tuple[str, ...]:
    """Return cached IPA tokens when available, tokenizing only as a fallback.

    Stays in this module (not ``_lookup``) because tests monkeypatch
    ``search_module.tokenize_ipa`` and bare-name resolution only picks up
    the patched value in the module where the call is defined.
    """
    if isinstance(record_or_entry, LexiconRecord) and len(record_or_entry.ipa_tokens) > 0:
        return record_or_entry.ipa_tokens
    return tuple(tokenize_ipa(_entry_ipa(_lookup_entry(record_or_entry))))


def build_lexicon_map(lexicon: Sequence[LexiconEntry]) -> LexiconMap:
    """Build a lexicon map with cached IPA token counts for each entry.

    Stays in this module (not ``_lookup``) because tests monkeypatch
    ``search_module.tokenize_ipa`` and bare-name resolution only picks up
    the patched value in the module where the call is defined.

    Args:
        lexicon: Sequence of lexicon entry dicts to index.

    Returns:
        LexiconMap mapping entry ids to LexiconRecord instances.

    Raises:
        ValueError: If duplicate entry IDs are found in the lexicon.
    """
    result: LexiconMap = {}
    for entry_id, entry in _build_entry_lookup(lexicon).items():
        ipa_tokens = tuple(tokenize_ipa(_entry_ipa(entry)))
        result[entry_id] = LexiconRecord(
            entry=entry,
            token_count=len(ipa_tokens),
            ipa_tokens=ipa_tokens,
        )
    return result


def _convert_partial_query_fragments(
    partial_query: PartialQueryPattern,
    *,
    dialect: str,
    converter: IpaConverter | None = None,
) -> tuple[str, str]:
    """Convert partial-query fragments independently to preserve boundaries."""
    ipa_converter = converter if converter is not None else to_ipa
    left_ipa = (
        ipa_converter(partial_query.left_fragment, dialect=dialect)
        if partial_query.left_fragment
        else ""
    )
    right_ipa = (
        ipa_converter(partial_query.right_fragment, dialect=dialect)
        if partial_query.right_fragment
        else ""
    )
    return left_ipa, right_ipa


def _tokenize_partial_query(
    partial_query: PartialQueryPattern,
    *,
    dialect: str,
    converter: IpaConverter | None = None,
    converted_fragments: tuple[str, str] | None = None,
) -> PartialQueryTokens:
    """Convert partial-query fragments into IPA token sequences."""
    left_ipa, right_ipa = (
        converted_fragments
        if converted_fragments is not None
        else _convert_partial_query_fragments(
            partial_query,
            dialect=dialect,
            converter=converter,
        )
    )
    return PartialQueryTokens(
        shape=partial_query.shape,
        left_tokens=tuple(tokenize_ipa(left_ipa)) if left_ipa else (),
        right_tokens=tuple(tokenize_ipa(right_ipa)) if right_ipa else (),
    )


def _partial_query_ipa_from_fragments(
    partial_query: PartialQueryPattern,
    left_ipa: str,
    right_ipa: str,
) -> str:
    """Build the main query IPA while preserving wildcard fragment boundaries."""
    if partial_query.shape == "infix":
        return " ".join(part for part in (left_ipa, right_ipa) if part)
    if partial_query.shape == "prefix":
        return left_ipa
    if partial_query.shape == "suffix":
        return right_ipa
    raise ValueError(f"Unknown partial query shape: {partial_query.shape!r}")


def prepare_query_ipa(
    query: str,
    *,
    dialect: str = "attic",
    converter: IpaConverter | None = None,
    query_ipa: str | None = None,
) -> PreparedQueryIpa:
    """Classify, normalize, and convert a query without crossing wildcard gaps.

    This function prepares a search query for phonological matching by:
    1. Classifying the query type (exact vs partial with wildcards)
    2. Normalizing the query string
    3. Converting to IPA, preserving wildcard fragment boundaries

    Conversion does NOT cross wildcard gaps - each fragment is converted
    independently to maintain phonological accuracy.

    Args:
        query: The search query string (Greek text with optional wildcards).
            Supports ``*`` for any characters and ``?`` for single character.
        dialect: The Greek dialect to use for IPA conversion.
            Defaults to "attic". Other valid values include "ionic", "doric", etc.
        converter: An optional IPA converter function conforming to the
            ``IpaConverter`` Protocol (``(str, *, dialect: str) -> str``).
            Defaults to the built-in ``to_ipa`` function if not provided.
        query_ipa: Optional precomputed IPA to skip conversion.
            When provided, the converter is bypassed entirely.

    Returns:
        PreparedQueryIpa: A namedtuple containing:
            - query_mode: Classification ("Exact-form" or "Partial-form")
            - normalized_query: Cleaned/normalized query string
            - partial_query: Pattern info if wildcards present, else None
            - query_ipa: The IPA-converted query string
            - partial_query_tokens: Tokenized fragments for partial queries

    Raises:
        ValueError: If ``query`` is empty or contains only whitespace,
            or if normalization results in an empty string.

    See Also:
        IpaConverter: Protocol defining the converter interface.
        PreparedQueryIpa: Container for query preparation results.
    """
    if not query.strip():
        raise ValueError("query must be a non-empty string")

    partial_query = _parse_partial_query(query)
    query_mode: QueryMode = (
        "Partial-form" if partial_query is not None else _classify_non_partial_query(query)
    )
    normalized_query = _normalize_query_with_pattern(query, partial_query)
    if not normalized_query.strip():
        raise ValueError("query must be a non-empty string")

    ipa_converter = converter if converter is not None else to_ipa
    partial_query_tokens: PartialQueryTokens | None = None
    if query_ipa is None:
        if partial_query is None:
            query_ipa = ipa_converter(normalized_query, dialect=dialect)
        else:
            converted_fragments = _convert_partial_query_fragments(
                partial_query,
                dialect=dialect,
                converter=ipa_converter,
            )
            partial_query_tokens = _tokenize_partial_query(
                partial_query,
                dialect=dialect,
                converted_fragments=converted_fragments,
            )
            query_ipa = _partial_query_ipa_from_fragments(
                partial_query,
                *converted_fragments,
            )
    elif partial_query is not None:
        partial_query_tokens = _tokenize_partial_query(
            partial_query,
            dialect=dialect,
            converter=ipa_converter,
        )

    return PreparedQueryIpa(
        query_mode=query_mode,
        normalized_query=normalized_query,
        partial_query=partial_query,
        query_ipa=query_ipa,
        partial_query_tokens=partial_query_tokens,
    )


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

    **Performance note**: When ``max_candidates`` is ``None`` the full
    ``lexicon_map`` is sorted (O(n log n) for ~63k entries). Callers should
    pass an explicit cap unless uncapped evaluation is intentional.
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






def _select_partial_token_fallback_candidates(
    partial_query: PartialQueryTokens,
    query_ipa: str,
    query_token_count: int,
    lexicon_map: LexiconMap,
    *,
    max_results: int,
    explicit_limit: int,
) -> list[str]:
    """Select bounded partial-form token fallback candidates.

    Rank only ``explicit_limit`` candidates by token-count proximity, then
    narrow the expensive partial matching to the stage-2 window. Callers
    must pass a positive integer cap; the full-scan uncapped variant was
    removed because every call site now enforces a default cap.
    """
    ranked_candidates = _rank_by_token_count_proximity(
        query_ipa,
        lexicon_map,
        max_candidates=explicit_limit,
        query_token_count=query_token_count,
    )
    return _select_partial_fallback_candidates(
        partial_query,
        ranked_candidates,
        lexicon_map,
        max_results=max_results,
        explicit_limit=explicit_limit,
    )






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


@lru_cache(maxsize=8)
def get_rules_registry(language: str = "ancient_greek") -> dict[str, dict[str, Any]]:
    """Load the packaged rule registry for the specified language.

    Loads packaged phonological rules via ``load_rules`` and caches the result
    once per process. Subsequent calls with the same language return the
    cached registry.

    Args:
        language: The language identifier for which to load rules.
            Defaults to "ancient_greek".

    Returns:
        A dictionary mapping rule IDs to rule definitions. Each rule definition
        is a dict[str, Any] containing the rule's metadata and patterns.

    Raises:
        OSError: Propagated from ``load_rules`` if file system errors occur.
        ValueError: Propagated from ``load_rules`` if rule validation fails,
            or raised if the registry cannot be loaded for the given language.
        yaml.YAMLError: Propagated from ``load_rules`` if YAML parsing fails.
    """
    try:
        return load_rules(language)
    except (OSError, ValueError, yaml.YAMLError) as err:
        raise ValueError(
            f"get_rules_registry failed to load rules for language {language!r}: {err}"
        ) from err


@lru_cache(maxsize=8)
def _get_tokenized_rules(language: str = "ancient_greek") -> tuple[TokenizedRule, ...]:
    """Get tokenized rules from the registry for matching."""
    rules_registry = get_rules_registry(language)
    return tuple(tokenize_rules_for_matching(list(rules_registry.values())))


def _annotate_search_results(
    query_ipa: str,
    results: list[SearchResult],
    lexicon_map: LexiconLookup,
    matrix: DistanceMatrix,
    language: str = "ancient_greek",
) -> list[SearchResult]:
    """Stage 2b: annotate ranked hits with explanations and alignments."""
    if not results:
        return []

    query_tokens = tokenize_ipa(query_ipa)
    tokenized_rules = _get_tokenized_rules(language)
    annotated: list[SearchResult] = []

    for result in results:
        candidate_id = result.entry_id
        if candidate_id is None:
            annotated.append(result)
            continue

        record_or_entry = lexicon_map.get(candidate_id)
        if record_or_entry is None:
            logger.debug(
                "Skipping annotation for candidate_id %r not found in lexicon_map (size=%d)",
                candidate_id,
                len(lexicon_map),
            )
            annotated.append(result)
            continue

        entry = _lookup_entry(record_or_entry)
        lemma_tokens = list(_lookup_entry_tokens(record_or_entry))
        alignment = result.alignment
        if alignment is None:
            _best_score, aligned_query, aligned_lemma = _smith_waterman_alignment(
                query_tokens, lemma_tokens, matrix
            )
            alignment = Alignment(
                aligned_query=tuple(aligned_query),
                aligned_lemma=tuple(aligned_lemma),
            )
        else:
            aligned_query = list(alignment.aligned_query)
            aligned_lemma = list(alignment.aligned_lemma)
        applications = explain_with_tokenized_rules(
            query_tokens=query_tokens,
            lemma_tokens=lemma_tokens,
            alignment=alignment,
            tokenized_rules=tokenized_rules,
            lemma_metadata=entry,
        )
        matched_dialects = _collect_application_dialects(applications)
        markers = _apply_rule_markers(
            _build_alignment_markers(aligned_query, aligned_lemma),
            aligned_query,
            aligned_lemma,
            applications,
        )
        annotated_result = SearchResult(
            lemma=result.lemma,
            confidence=result.confidence,
            dialect_attribution=_build_dialect_attribution(
                _candidate_dialect(entry),
                matched_dialects,
            ),
            applied_rules=[
                application.rule_id
                for application in applications
                if not _is_observed_application(application)
            ],
            rule_applications=list(applications),
            alignment_visualization=_format_alignment_visualization(
                aligned_query, aligned_lemma, markers
            ),
            ipa=result.ipa,
            entry_id=candidate_id,
            alignment=alignment,
        )
        annotated.append(annotated_result)

    return annotated


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
    scored_results = _score_stage(
        query_ipa=query_ipa,
        candidates=candidates,
        lexicon_map=lexicon_map,
        matrix=matrix,
    )
    return _annotate_search_results(
        query_ipa=query_ipa,
        results=scored_results,
        lexicon_map=lexicon_map,
        matrix=matrix,
        language=language,
    )


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
    query_ipa: str | None = None,
    prepared_query: PreparedQueryIpa | None = None,
    prebuilt_ipa_index: IpaIndex | None = None,
    similarity_fallback_limit: int | None = _DEFAULT_FALLBACK_CANDIDATE_LIMIT,
    unigram_fallback_limit: int | None = _DEFAULT_FALLBACK_CANDIDATE_LIMIT,
) -> list[SearchResult]:
    """Run full three-stage search for a Greek query word.

    The query mode is classified before search:
      - "Partial-form": Query contains exactly one wildcard marker ('-' or '*').
      - "Short-query": Query is short (<= 3 characters, minus markers).
      - "Full-form": All other queries.

    This categorization influences intermediate candidate filtering and fallback logic:
      - similarity_fallback_limit and unigram_fallback_limit default to a
        conservative fallback exploration cap for every mode. Passing ``None``
        explicitly leaves the corresponding fallback path uncapped.
      - Short queries restrict unfiltered candidates to those exceeding
        _SHORT_QUERY_CONFIDENCE_THRESHOLD or exact/supported rules.
      - Partial queries prioritize full wildcard matches and candidates above
        _PARTIAL_QUERY_CONFIDENCE_THRESHOLD.
      - Non-full-form queries annotate only a bounded, mode-aware window before
        applying mode-specific filters, limiting expensive rule explanation
        and visualization work on large fallback candidate sets.

    The stages operate via cascading fallbacks:
      1. Default seed stage (k=2) returns candidates if found.
      2. If empty, a unigram fallback seed (k=1) is applied when the query
         has a consonant skeleton. The unigram fallback limit truncates the
         unigram-hit set before scoring unless callers explicitly pass ``None``.
      3. If still empty, a token-count similarity fallback ranks candidates
         by length proximity. The similarity fallback limit bounds this scan
         unless callers explicitly pass ``None``.
    After seeding, full-form queries preserve the lightweight top-N annotation
    path, while short and partial queries annotate a bounded ranked subset
    before applying mode-specific quality filters and final deduplication.

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
            reorders stage-2 work for mode-aware follow-up ranking.
        prebuilt_lexicon_map: Optional cached entry-id map with token counts
            to reuse across repeated searches over the same lexicon.
        language: Language identifier selecting the phonological rule set
            passed to ``extend_stage``. Defaults to ``"ancient_greek"``.
        query_ipa: Optional pre-computed IPA transcription of the normalized
            query. When provided, the internal ``to_ipa`` call for the main
            query is skipped. Fragment-level conversions for partial queries
            are still performed independently.
        prepared_query: Optional pre-computed query preparation bundle returned
            by ``prepare_query_ipa``. When provided, both main-query and
            fragment-level conversions are reused.
        prebuilt_ipa_index: Optional precomputed IPA-to-entry-ids map to reuse
            across repeated searches. When provided, the internal ``build_ipa_index``
            call is skipped. Useful for callers managing their own IPA index lifecycle.
            Must be consistent with other prebuilt inputs (``index``, ``unigram_index``,
            ``prebuilt_lexicon_map``) derived from the same lexicon. Defaults to
            ``None``, which triggers automatic index construction.
        similarity_fallback_limit: Optional explicit exploration cap for the
            token-count fallback path used when both k=2 and k=1 seeds are
            empty. Defaults to ``_DEFAULT_FALLBACK_CANDIDATE_LIMIT``. Passing
            ``None`` is treated as a safety net: it falls back to
            ``_DEFAULT_FALLBACK_CANDIDATE_LIMIT`` (with a warning) to prevent
            unbounded scans over the full lexicon. Pass an explicit positive
            integer to override the default cap.
        unigram_fallback_limit: Optional explicit exploration cap for the k=1
            fallback path used when the default k=2 seed is empty but
            single-consonant unigram matches exist. Defaults to
            ``_DEFAULT_FALLBACK_CANDIDATE_LIMIT``. Passing ``None`` is treated
            as a safety net: it falls back to ``_DEFAULT_FALLBACK_CANDIDATE_LIMIT``
            (with a warning) to bound worst-case exploration. Pass an explicit
            positive integer to override the default cap.

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
    if unigram_fallback_limit is not None and unigram_fallback_limit <= 0:
        raise ValueError("unigram_fallback_limit must be a positive integer")
    if prepared_query is not None and query_ipa is not None:
        raise ValueError(
            "Pass either prepared_query or query_ipa, not both; "
            "prepared_query already carries the query IPA."
        )

    if prepared_query is None:
        prepared_query = prepare_query_ipa(
            query,
            dialect=dialect,
            query_ipa=query_ipa,
        )
    partial_query = prepared_query.partial_query
    query_mode = prepared_query.query_mode
    query_ipa = prepared_query.query_ipa
    query_tokens = tokenize_ipa(query_ipa)
    query_skeleton = _extract_consonant_skeleton(query_tokens)
    partial_query_tokens = prepared_query.partial_query_tokens
    # Keep the two fallback caps distinct in local naming: the unigram cap only
    # applies to the k=1 rescue path, while the similarity cap only applies
    # after both k=2 and k=1 seeds are empty. ``None`` is not an uncapped
    # escape hatch — it falls back to ``_DEFAULT_FALLBACK_CANDIDATE_LIMIT`` so
    # that every mode enforces a predictable ceiling on candidate exploration.
    if similarity_fallback_limit is None:
        effective_similarity_fallback_limit = _DEFAULT_FALLBACK_CANDIDATE_LIMIT
        logger.warning(
            "similarity_fallback_limit=None for query IPA %r; applying default cap %d. "
            "Pass an explicit positive integer to silence this warning.",
            query_ipa,
            _DEFAULT_FALLBACK_CANDIDATE_LIMIT,
        )
    else:
        effective_similarity_fallback_limit = similarity_fallback_limit
    if unigram_fallback_limit is None:
        effective_unigram_fallback_limit = _DEFAULT_FALLBACK_CANDIDATE_LIMIT
        logger.warning(
            "unigram_fallback_limit=None for query IPA %r; applying default cap %d. "
            "Pass an explicit positive integer to silence this warning.",
            query_ipa,
            _DEFAULT_FALLBACK_CANDIDATE_LIMIT,
        )
    else:
        effective_unigram_fallback_limit = unigram_fallback_limit
    entry_lookup: dict[str, LexiconEntry] | None = None
    ipa_index: IpaIndex | None = None
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

    def _get_ipa_index() -> IpaIndex:
        nonlocal ipa_index
        if prebuilt_ipa_index is not None:
            return prebuilt_ipa_index
        if ipa_index is None:
            ipa_index = build_ipa_index(_get_lexicon_lookup())
        return ipa_index

    search_index = (
        index if index is not None else build_kmer_index(lexicon, k=_DEFAULT_KMER_SIZE)
    )
    seed_candidates = seed_stage(query_ipa, search_index, k=_DEFAULT_KMER_SIZE)
    stage2_limit = max(_MIN_STAGE2_CANDIDATES, max_results * _SEED_MULTIPLIER)

    candidate_ids: list[str]
    lexicon_lookup: LexiconLookup
    if seed_candidates:
        if query_mode == "Partial-form":
            if partial_query_tokens is None:
                raise ValueError("partial_query_tokens must not be None when query_mode == 'Partial-form'")
            tokenized_map = _get_tokenized_lexicon_map()
            partial_candidate_window = _annotation_candidate_limit(max_results)
            supplemental_unigram_candidates: list[str] = []
            if query_skeleton:
                fallback_unigram_index = (
                    unigram_index if unigram_index is not None else build_kmer_index(lexicon, k=1)
                )
                supplemental_unigram_candidates = seed_stage(
                    query_ipa,
                    fallback_unigram_index,
                    k=1,
                )
            partial_candidates = _merge_bounded_candidate_ids(
                seed_candidates,
                supplemental_unigram_candidates,
                limit=partial_candidate_window,
            )
            candidate_ids = _select_partial_seed_candidates(
                partial_query_tokens,
                partial_candidates,
                tokenized_map,
                stage2_limit,
            )
            lexicon_lookup = tokenized_map
        else:
            candidate_ids = seed_candidates[:stage2_limit]
            lexicon_lookup = _get_lexicon_lookup()
            candidate_ids = _inject_exact_ipa_matches(
                query_ipa,
                candidate_ids,
                lexicon_lookup,
                ipa_index=_get_ipa_index(),
                limit=stage2_limit,
            )
    else:
        unigram_candidates: list[str] = []
        if query_skeleton:
            fallback_unigram_index = (
                unigram_index if unigram_index is not None else build_kmer_index(lexicon, k=1)
            )
            unigram_candidates = seed_stage(query_ipa, fallback_unigram_index, k=1)
        if unigram_candidates:
            if query_mode == "Partial-form":
                if partial_query_tokens is None:
                    raise ValueError("partial_query_tokens must not be None for Partial-form query")
                tokenized_map = _get_tokenized_lexicon_map()
                ranked_candidates = _rank_by_token_count_proximity(
                    query_ipa,
                    {
                        entry_id: tokenized_map[entry_id]
                        for entry_id in unigram_candidates
                        if entry_id in tokenized_map
                    },
                    max_candidates=effective_unigram_fallback_limit,
                    query_token_count=len(query_tokens),
                )
                # Partial-form follow-up work is always narrowed again inside
                # _select_partial_fallback_candidates().
                candidate_ids = _select_partial_fallback_candidates(
                    partial_query_tokens,
                    ranked_candidates,
                    tokenized_map,
                    max_results=max_results,
                    explicit_limit=effective_unigram_fallback_limit,
                )
                lexicon_lookup = tokenized_map
                logger.info(
                    "k=2 seed empty for query IPA %r; partial k=1 fallback selected %d "
                    "stage-2 candidates from %d unigram hits (cap=%s)",
                    query_ipa,
                    len(candidate_ids),
                    len(unigram_candidates),
                    effective_unigram_fallback_limit,
                )
            else:
                tokenized_map = _get_tokenized_lexicon_map()
                missing_unigram_candidates = list(
                    dict.fromkeys(
                        entry_id
                        for entry_id in unigram_candidates
                        if entry_id not in tokenized_map
                    )
                )
                if missing_unigram_candidates:
                    _truncated = missing_unigram_candidates[:10]
                    _remaining = len(missing_unigram_candidates) - 10
                    if _remaining > 0:
                        _msg = f"{_truncated} (+{_remaining} more)"
                    else:
                        _msg = str(_truncated)
                    logger.warning(
                        "k=2 seed empty for query IPA %r; ignoring unigram fallback "
                        "candidates missing from tokenized lexicon map: %s",
                        query_ipa,
                        _msg,
                    )
                unigram_candidate_map = {
                    entry_id: tokenized_map[entry_id]
                    for entry_id in unigram_candidates
                    if entry_id in tokenized_map
                }
                candidate_ids = _rank_by_token_count_proximity(
                    query_ipa,
                    unigram_candidate_map,
                    max_candidates=effective_unigram_fallback_limit,
                    query_token_count=len(query_tokens),
                )
                lexicon_lookup = tokenized_map
                logger.info(
                    "k=2 seed empty for query IPA %r; capped k=1 fallback evaluating %d of %d "
                    "unigram candidates ranked by token-count proximity",
                    query_ipa,
                    len(candidate_ids),
                    len(unigram_candidates),
                )
        else:
            tokenized_map = _get_tokenized_lexicon_map()
            lexicon_lookup = tokenized_map
            if query_mode == "Partial-form":
                if partial_query_tokens is None:
                    raise ValueError("partial_query_tokens must not be None for Partial-form query")
                candidate_ids = _select_partial_token_fallback_candidates(
                    partial_query_tokens,
                    query_ipa,
                    len(query_tokens),
                    tokenized_map,
                    max_results=max_results,
                    explicit_limit=effective_similarity_fallback_limit,
                )
                logger.info(
                    "k=2 and k=1 seeds empty for query IPA %r; partial token fallback "
                    "selected %d stage-2 candidates from %d tokenized entries (cap=%s)",
                    query_ipa,
                    len(candidate_ids),
                    len(tokenized_map),
                    effective_similarity_fallback_limit,
                )
            else:
                candidate_ids = _rank_by_token_count_proximity(
                    query_ipa,
                    tokenized_map,
                    max_candidates=effective_similarity_fallback_limit,
                    query_token_count=len(query_tokens),
                )
                logger.info(
                    "k=2 and k=1 seeds empty for query IPA %r; capped fallback evaluating %d of %d candidates "
                    "ranked by token-count proximity",
                    query_ipa,
                    len(candidate_ids),
                    len(tokenized_map),
                )

    scored_results = _score_stage(
        query_ipa=query_ipa,
        candidates=candidate_ids,
        lexicon_map=lexicon_lookup,
        matrix=matrix,
    )
    ranked_scored = sorted(scored_results, key=lambda r: (-r.confidence, r.lemma))
    if query_mode == "Full-form":
        # Keep the lightweight full-form path: deduplicate and truncate before
        # rule explanation/visualization work so only the final visible hits
        # are annotated.
        unique_scored = _deduplicate_by_headword(ranked_scored, check_sorted=False)
        return _annotate_search_results(
            query_ipa=query_ipa,
            results=unique_scored[:max_results],
            lexicon_map=lexicon_lookup,
            matrix=matrix,
            language=language,
        )

    # Non-full-form modes apply post-annotation quality filtering, so annotate
    # a broader mode-aware window first and let the mode-specific filter decide
    # which supported/exact candidates survive into the visible results.
    annotation_limit = _annotation_candidate_limit(max_results)
    if query_mode == "Short-query":
        ordered_annotation_candidates = _rank_short_query_annotation_candidates(
            query_tokens,
            ranked_scored,
            lexicon_lookup,
        )
        annotated_results: list[SearchResult] = []
        deduplicated_results: list[SearchResult] = []
        # Cap the number of annotation batches to bound worst-case work when
        # most candidates are filtered out (e.g. vowel-only short queries
        # where the confidence floor drops nearly every hit).
        truncated_by_batch_cap = False
        for batch_index, offset in enumerate(
            range(0, len(ordered_annotation_candidates), annotation_limit)
        ):
            if batch_index >= _SHORT_QUERY_MAX_ANNOTATION_BATCHES:
                truncated_by_batch_cap = True
                break
            batch = ordered_annotation_candidates[offset : offset + annotation_limit]
            if not batch:
                break
            batch_annotated = _annotate_search_results(
                query_ipa=query_ipa,
                results=batch,
                lexicon_map=lexicon_lookup,
                matrix=matrix,
                language=language,
            )
            batch_filtered = _apply_mode_quality_filter(
                query_mode,
                query_tokens,
                partial_query_tokens,
                batch_annotated,
                lexicon_lookup,
            )
            batch_filtered = _rank_short_query_results(
                query_tokens,
                batch_filtered,
                lexicon_lookup,
                _lookup_entry_tokens,
            )
            annotated_results.extend(batch_filtered)
            deduplicated_results = _deduplicate_by_headword_common(
                annotated_results
            )
            if len(deduplicated_results) >= max_results:
                return deduplicated_results[:max_results]
        # Loop exited without finding enough results; the last iteration's
        # deduplicated_results, or the initial empty list when no candidates
        # survived filtering, is the best we have.
        if truncated_by_batch_cap and len(deduplicated_results) < max_results:
            logger.warning(
                "Short-query search for query IPA %r returned %d/%d results; "
                "annotation batches capped at %d (query_mode=%s). "
                "Remaining %d candidates were not annotated.",
                query_ipa,
                len(deduplicated_results),
                max_results,
                _SHORT_QUERY_MAX_ANNOTATION_BATCHES,
                query_mode,
                max(
                    0,
                    len(ordered_annotation_candidates)
                    - _SHORT_QUERY_MAX_ANNOTATION_BATCHES * annotation_limit,
                ),
            )
            # Propagate truncation warning to the response
            return [
                replace(r, truncated=True) for r in deduplicated_results[:max_results]
            ]
        return deduplicated_results[:max_results]

    annotation_candidates = _select_annotation_candidates(
        query_mode,
        query_tokens,
        partial_query_tokens,
        ranked_scored,
        lexicon_lookup,
        annotation_limit,
    )
    annotated_results = _annotate_search_results(
        query_ipa=query_ipa,
        results=annotation_candidates,
        lexicon_map=lexicon_lookup,
        matrix=matrix,
        language=language,
    )
    filtered_results = _apply_mode_quality_filter(
        query_mode,
        query_tokens,
        partial_query_tokens,
        annotated_results,
        lexicon_lookup,
    )
    deduplicated_results = _deduplicate_by_headword_common(filtered_results)
    return deduplicated_results[:max_results]
