"""BLAST-like three-stage phonological search over a Greek lemma lexicon."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
import logging
from pathlib import Path
from typing import Any

# Test-seam re-exports: tests monkeypatch ``phonology.search.<name>`` for these
# so the symbols must remain attributes of the package even when callers no
# longer reach them directly from ``__init__``.
from ..explainer import (
    Alignment,
    TokenizedRule,  # noqa: F401 (test/typing access via search_module)
    explain,
    explain_with_tokenized_rules,
    load_rules,  # noqa: F401 (test access via search_module)
    tokenize_rules_for_matching,  # noqa: F401 (test access via search_module)
)
from ..core.ports.profiles import (  # noqa: F401 (test access via search_module)
    LanguageProfile,
    get_default_language_profile,
    get_language_profile,
)
from ._constants import (
    _annotation_candidate_limit,  # noqa: F401 (test access via search_module)
    _DEFAULT_FALLBACK_CANDIDATE_LIMIT,  # noqa: F401 (test/typing access via search_module)
    _MIN_PARTIAL_STAGE2_CANDIDATES,  # noqa: F401 (test access via search_module)
    _MIN_STAGE2_CANDIDATES,  # noqa: F401 (test access via search_module)
    _partial_candidate_limit,  # noqa: F401 (test access via search_module)
    _PARTIAL_QUERY_CONFIDENCE_THRESHOLD,  # noqa: F401 (test access via search_module)
    _SHORT_QUERY_CONFIDENCE_THRESHOLD,  # noqa: F401 (test access via search_module)
    OBSERVED_PREFIX,  # noqa: F401 (test/public access via search_module)
)
from ._lookup import (
    _build_entry_lookup,
    _entry_ipa,
    _inject_exact_ipa_matches,  # noqa: F401 (test access via search_module)
    _lookup_entry,
    build_ipa_index,
    IpaIndex,
)
from ._indexing import _iter_kmers, build_kmer_index as _build_generic_kmer_index
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
    _score_stage,
    _smith_waterman_alignment,
)
from ._partial import (
    _match_partial_query,  # noqa: F401 (test access via search_module)
    _select_partial_seed_candidates,  # noqa: F401 (test access via search_module)
)
from ._filtering import (
    _apply_mode_quality_filter,  # noqa: F401 (test access via search_module)
    _rank_short_query_annotation_candidates,  # noqa: F401 (test access via search_module)
    _select_annotation_candidates,  # noqa: F401 (test access via search_module)
)
from ._debug_logging import (
    log_candidate_selection as _log_candidate_selection,  # noqa: F401
    log_finalization as _log_finalization,  # noqa: F401
    log_scoring as _log_scoring,  # noqa: F401
    perf_counter_if_debug as _perf_counter_if_debug,  # noqa: F401
    summarize_query_ipa_for_logs as _summarize_query_ipa_for_logs,  # noqa: F401
)
from ._dedup import (
    _deduplicate_by_headword,  # noqa: F401 (test access via search_module)
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
from ._tokenization import (
    resolve_entry_tokens,
    tokenize_for_inventory,
    tokenize_ipa,  # noqa: F401 (public/test compatibility seam)
)
from ._types import (
    DistanceMatrix,
    KmerIndex,
    LexiconEntry,
    LexiconLookup,
    LexiconMap,
    LexiconRecord,
    PartialQueryPattern,
    PartialQueryTokens,
    PhoneInventory,
    QueryMode,
    SearchResult,
)
from ._dependencies import (
    IpaConverter,
    PreparedQueryIpa,
    SearchExecutionResult,
    _FallbackLimits,
    _LazySearchDependencies,  # noqa: F401 (test/typing access via search_module)
)
from ._registry import (
    _get_tokenized_rules,
    _load_rules_cached,  # noqa: F401 (test access via search_module)
    get_rules_registry,
)
from ._selection import (
    _select_partial_token_fallback_candidates,  # noqa: F401
    _select_seeded_candidates,  # noqa: F401 (test access via search_module)
    _select_token_proximity_fallback_candidates,  # noqa: F401
    _select_unigram_fallback_candidates,  # noqa: F401
)
from .compat import (
    build_kmer_index,
    build_lexicon_map,
    extend_stage,
    prepare_query_ipa,
    search,
    search_execution,
    seed_stage,
)

logger = logging.getLogger(__name__)


def _legacy_to_ipa(text: str, *, dialect: str | None = None) -> str:
    """Convert text to IPA via the default language profile.

    Exists as a module-level seam so tests can monkeypatch
    ``search_module.to_ipa`` to inject deterministic IPA strings without
    touching the language registry. Production callers should normally use
    ``LanguageProfile.converter`` directly; this helper is invoked only when
    the public search API uses the default profile without an explicit
    converter.

    Note: calling this after ``isolated_language_registry`` resets the
    registry will rebuild the default profile as a side effect.
    """
    profile = get_default_language_profile()
    return profile.converter(
        text,
        dialect=profile.default_dialect if dialect is None else dialect,
    )


# Module-level alias kept as a monkeypatch seam for tests; see ``_legacy_to_ipa``
# above for the rationale and runtime call conditions.
to_ipa = _legacy_to_ipa


def _backfill_defaults_from_profile(
    profile: LanguageProfile,
    *,
    phone_inventory: Iterable[str] | None,
    vowel_phones: Iterable[str] | None,
    dialect_skeleton_builders: Iterable[Callable[[list[str]], list[str]]] | None,
) -> tuple[
    Iterable[str] | None,
    Iterable[str] | None,
    Iterable[Callable[[list[str]], list[str]]] | None,
]:
    """Fill any unset public defaults from a resolved language profile."""
    return (
        profile.phone_inventory if phone_inventory is None else phone_inventory,
        profile.vowel_phones if vowel_phones is None else vowel_phones,
        (
            profile.dialect_skeleton_builders
            if dialect_skeleton_builders is None
            else dialect_skeleton_builders
        ),
    )


def _public_compatibility_search_defaults(
    *,
    language: str | None,
    phone_inventory: Iterable[str] | None,
    vowel_phones: Iterable[str] | None = None,
    dialect_skeleton_builders: Iterable[Callable[[list[str]], list[str]]] | None,
    allow_fallback: bool = False,
) -> tuple[
    Iterable[str] | None,
    Iterable[str] | None,
    Iterable[Callable[[list[str]], list[str]]] | None,
]:
    """Supply profile defaults at the public compatibility boundary.

    Args:
        language: Registered language id to resolve. ``None`` uses the default
            language profile.
        phone_inventory: Explicit inventory to keep, or ``None`` to backfill
            from the resolved profile.
        vowel_phones: Explicit vowel phones to keep, or ``None`` to backfill
            from the resolved profile.
        dialect_skeleton_builders: Explicit builders to keep, or ``None`` to
            backfill from the resolved profile.
        allow_fallback: When ``False`` (default), unrecognized language ids
            re-raise the profile lookup ``ValueError``. When ``True``, the
            function preserves the legacy silent fallback and returns the
            original ``(phone_inventory, vowel_phones,
            dialect_skeleton_builders)`` tuple unchanged.
    """
    if language is None:
        profile = get_default_language_profile()
    else:
        normalized_language = language.strip().lower()
        try:
            profile = get_language_profile(normalized_language)
        except ValueError as exc:
            logger.debug(
                "Ignoring unsupported normalized language %r when resolving public search defaults: %s",
                normalized_language,
                exc,
            )
            if not allow_fallback:
                raise
            return phone_inventory, vowel_phones, dialect_skeleton_builders

    return _backfill_defaults_from_profile(
        profile,
        phone_inventory=phone_inventory,
        vowel_phones=vowel_phones,
        dialect_skeleton_builders=dialect_skeleton_builders,
    )


def _with_phone_inventory(
    kwargs: dict[str, Any],
    phone_inventory: PhoneInventory,
) -> dict[str, Any]:
    """Add already-resolved phone inventory to delegated helper kwargs."""
    updated_kwargs = dict(kwargs)
    updated_kwargs["phone_inventory"] = phone_inventory
    return updated_kwargs


def _build_kmer_index_for_inventory(
    lexicon: Sequence[LexiconEntry],
    *,
    k: int,
    phone_inventory: PhoneInventory,
    vowel_phones: Iterable[str],
    dialect_skeleton_builders: Iterable[Callable[[list[str]], list[str]]] | None = None,
) -> KmerIndex:
    """Call k-mer builder, forwarding phone inventory and dialect skeleton builders."""
    return _build_generic_kmer_index(
        lexicon,
        k=k,
        phone_inventory=phone_inventory,
        vowel_phones=vowel_phones,
        dialect_skeleton_builders=dialect_skeleton_builders,
    )


def _build_lexicon_map_for_inventory(
    lexicon: Sequence[LexiconEntry],
    *,
    phone_inventory: PhoneInventory,
) -> LexiconMap:
    """Call the lexicon-map core directly, bypassing public compatibility defaults."""
    return _build_lexicon_map_core(lexicon, phone_inventory=phone_inventory)


def _seed_stage_for_inventory(
    query_ipa: str,
    index: KmerIndex,
    *,
    k: int,
    phone_inventory: PhoneInventory,
    vowel_phones: Iterable[str],
) -> list[str]:
    """Call seed-stage core with already-resolved inventory settings."""
    return _seed_stage_core(
        query_ipa,
        index,
        k=k,
        phone_inventory=phone_inventory,
        vowel_phones=vowel_phones,
    )


# Private tuning attributes remain available for internal tests but are
# intentionally excluded from the public star-import surface.
__all__ = [
    "IpaIndex",
    "KmerIndex",
    "LexiconEntry",
    "LexiconMap",
    "LexiconRecord",
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
    "search_execution",
    "SearchExecutionResult",
]


def _validate_search_arguments(
    *,
    query: str,
    max_results: int,
    similarity_fallback_limit: int | None,
    unigram_fallback_limit: int | None,
    prepared_query: PreparedQueryIpa | None,
    query_ipa: str | None,
) -> None:
    """Validate search arguments before query preparation and candidate selection."""
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


def _resolve_fallback_limits(
    *,
    query_log_label: str,
    similarity_fallback_limit: int | None,
    unigram_fallback_limit: int | None,
) -> _FallbackLimits:
    """Return effective fallback caps, warning when callers pass ``None``."""
    if similarity_fallback_limit is None:
        effective_similarity_fallback_limit = _DEFAULT_FALLBACK_CANDIDATE_LIMIT
        logger.warning(
            "similarity_fallback_limit=None for query %s; applying default cap %d. "
            "Pass an explicit positive integer to silence this warning.",
            query_log_label,
            _DEFAULT_FALLBACK_CANDIDATE_LIMIT,
        )
    else:
        effective_similarity_fallback_limit = similarity_fallback_limit

    if unigram_fallback_limit is None:
        effective_unigram_fallback_limit = _DEFAULT_FALLBACK_CANDIDATE_LIMIT
        logger.warning(
            "unigram_fallback_limit=None for query %s; applying default cap %d. "
            "Pass an explicit positive integer to silence this warning.",
            query_log_label,
            _DEFAULT_FALLBACK_CANDIDATE_LIMIT,
        )
    else:
        effective_unigram_fallback_limit = unigram_fallback_limit

    return _FallbackLimits(
        similarity=effective_similarity_fallback_limit,
        unigram=effective_unigram_fallback_limit,
    )


def _build_lexicon_map_core(
    lexicon: Sequence[LexiconEntry],
    *,
    phone_inventory: PhoneInventory,
) -> LexiconMap:
    """Build a lexicon map without applying public Ancient Greek defaults.

    Internal callers that have already resolved their own phone inventory
    (e.g. profile-aware search paths) call this directly. Generic
    character-by-character tokenization is represented by an empty tuple,
    never by ``None``.
    """
    result: LexiconMap = {}
    for entry_id, entry in _build_entry_lookup(lexicon).items():
        ipa_tokens = tuple(tokenize_for_inventory(_entry_ipa(entry), phone_inventory))
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
    phone_inventory: PhoneInventory,
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
        left_tokens=tuple(tokenize_for_inventory(left_ipa, phone_inventory))
        if left_ipa
        else (),
        right_tokens=tuple(tokenize_for_inventory(right_ipa, phone_inventory))
        if right_ipa
        else (),
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


def _resolve_profile_and_converter(
    dialect: str | None,
    converter: IpaConverter | None,
) -> tuple[str, IpaConverter]:
    """Resolve the effective dialect and IPA converter for query preparation.

    The default profile is consulted only when the dialect must be defaulted or
    when no explicit converter is supplied and the module ``to_ipa`` seam is
    still the built-in ``_legacy_to_ipa`` (so tests that monkeypatch
    ``search_module.to_ipa`` keep their injected converter). ``to_ipa`` is read
    as a module global at call time to preserve that seam.

    Returns the resolved dialect and converter. When ``dialect`` is ``None`` the
    profile-resolution condition guarantees a default profile is available, so
    the returned dialect is always a concrete string.
    """
    default_profile = (
        get_default_language_profile()
        if dialect is None or (converter is None and to_ipa is _legacy_to_ipa)
        else None
    )
    if dialect is not None:
        resolved_dialect = dialect
    else:
        # dialect is None ⟹ the condition above resolved a default profile.
        assert default_profile is not None
        resolved_dialect = default_profile.default_dialect

    if converter is not None:
        ipa_converter: IpaConverter = converter
    elif default_profile is not None and to_ipa is _legacy_to_ipa:
        ipa_converter = default_profile.converter
    else:
        ipa_converter = to_ipa
    return resolved_dialect, ipa_converter


def _prepare_query_ipa_core(
    query: str,
    *,
    dialect: str | None = None,
    converter: IpaConverter | None = None,
    phone_inventory: PhoneInventory,
    query_ipa: str | None = None,
) -> PreparedQueryIpa:
    """Classify, normalize, and convert a query without crossing wildcard gaps.

    This function prepares a search query for phonological matching by:
    1. Classifying the query type (exact vs partial with wildcards)
    2. Normalizing the query string
    3. Converting to IPA, preserving wildcard fragment boundaries

    Conversion does NOT cross wildcard gaps - each fragment is converted
    independently to maintain phonological accuracy. Public calls preserve
    profile defaults: when ``phone_inventory`` is omitted, the selected
    profile's phone inventory is used so multi-character IPA phones are
    tokenized consistently with ``build_kmer_index``.

    Args:
        query: The search query string (optionally with wildcards).
            Supports ``*`` for any characters and ``?`` for single character.
        dialect: Dialect/model identifier passed to the IPA converter.
            ``None`` resolves to the selected profile's default dialect.
        converter: An optional IPA converter function conforming to the
            ``IpaConverter`` Protocol (``(str, *, dialect: str) -> str``).
            Defaults to the built-in ``to_ipa`` function if not provided.
        phone_inventory: Resolved phone inventory used by partial-query
            tokenization. Pass an empty tuple for generic character fallback.
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
        "Partial-form"
        if partial_query is not None
        else _classify_non_partial_query(query)
    )
    normalized_query = _normalize_query_with_pattern(query, partial_query)
    if not normalized_query.strip():
        raise ValueError("query must be a non-empty string")

    resolved_dialect, ipa_converter = _resolve_profile_and_converter(
        dialect,
        converter,
    )
    partial_query_tokens: PartialQueryTokens | None = None
    if query_ipa is None:
        if partial_query is None:
            query_ipa = ipa_converter(normalized_query, dialect=resolved_dialect)
        else:
            converted_fragments = _convert_partial_query_fragments(
                partial_query,
                dialect=resolved_dialect,
                converter=ipa_converter,
            )
            partial_query_tokens = _tokenize_partial_query(
                partial_query,
                dialect=resolved_dialect,
                phone_inventory=phone_inventory,
                converted_fragments=converted_fragments,
            )
            query_ipa = _partial_query_ipa_from_fragments(
                partial_query,
                *converted_fragments,
            )
    elif partial_query is not None:
        partial_query_tokens = _tokenize_partial_query(
            partial_query,
            dialect=resolved_dialect,
            converter=ipa_converter,
            phone_inventory=phone_inventory,
        )

    return PreparedQueryIpa(
        query_mode=query_mode,
        normalized_query=normalized_query,
        partial_query=partial_query,
        query_ipa=query_ipa,
        partial_query_tokens=partial_query_tokens,
    )


from ._orchestration import _rank_by_token_count_proximity  # noqa: E402, F401 (test-seam re-export)


def _seed_stage_core(
    query_ipa: str,
    index: KmerIndex,
    *,
    k: int,
    phone_inventory: PhoneInventory,
    vowel_phones: Iterable[str],
) -> list[str]:
    """Rank candidate ids using caller-resolved tokenization settings.

    Generic character-by-character tokenization is represented by an empty
    tuple, never by ``None``.
    """
    if k <= 0:
        raise ValueError(f"seed_stage requires k > 0 for k-mer size, got {k}")

    query_skeleton = _extract_consonant_skeleton(
        tokenize_for_inventory(query_ipa, phone_inventory),
        vowel_phones=tuple(vowel_phones or ()),
    )
    query_kmers = _iter_kmers(query_skeleton, k)
    if not query_kmers:
        return []

    counts: Counter[str] = Counter()
    for kmer, weight in Counter(query_kmers).items():
        for candidate_id in index.get(kmer, []):
            counts[candidate_id] += weight

    return [
        candidate_id
        for candidate_id, _ in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]


def _annotate_search_results(
    query_ipa: str,
    results: list[SearchResult],
    lexicon_map: LexiconLookup,
    matrix: DistanceMatrix,
    phone_inventory: PhoneInventory,
    vowel_phones: Iterable[str] | None = None,
    phone_matcher: Callable[[str, str], bool] | None = None,
    always_match_contexts: Iterable[str] | None = None,
    language: str | Path | None = None,
) -> list[SearchResult]:
    """Stage 2b: annotate ranked hits with explanations and alignments."""
    if not results:
        return []

    query_tokens = tokenize_for_inventory(query_ipa, phone_inventory)
    resolved_always_match_contexts = tuple(always_match_contexts or ())
    tokenized_rules = _get_tokenized_rules(
        language,
        phone_inventory,
        resolved_always_match_contexts,
    )
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
        lemma_tokens = list(
            resolve_entry_tokens(record_or_entry, phone_inventory=phone_inventory)
        )
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
            phone_matcher=phone_matcher,
            phone_inventory=phone_inventory,
            vowel_phones=tuple(vowel_phones or ()),
            always_match_contexts=resolved_always_match_contexts,
        )
        matched_dialects = _collect_application_dialects(applications)
        markers = _apply_rule_markers(
            _build_alignment_markers(aligned_query, aligned_lemma),
            aligned_query,
            aligned_lemma,
            applications,
            phone_inventory=phone_inventory,
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


def _score_stage_for_inventory(
    *,
    query_ipa: str,
    candidates: Iterable[str],
    lexicon_map: LexiconLookup,
    matrix: DistanceMatrix,
    phone_inventory: PhoneInventory,
) -> list[SearchResult]:
    """Call score stage without custom-inventory kwargs on the default path."""
    kwargs: dict[str, Any] = {
        "query_ipa": query_ipa,
        "candidates": candidates,
        "lexicon_map": lexicon_map,
        "matrix": matrix,
    }
    return _score_stage(**_with_phone_inventory(kwargs, phone_inventory))


def _annotate_search_results_for_inventory(
    *,
    query_ipa: str,
    results: list[SearchResult],
    lexicon_map: LexiconLookup,
    matrix: DistanceMatrix,
    language: str | Path | None,
    phone_inventory: PhoneInventory,
    vowel_phones: Iterable[str] | None = None,
    phone_matcher: Callable[[str, str], bool] | None = None,
    always_match_contexts: Iterable[str] | None = None,
) -> list[SearchResult]:
    """Call annotation without custom-inventory kwargs on the default path."""
    kwargs: dict[str, Any] = {
        "query_ipa": query_ipa,
        "results": results,
        "lexicon_map": lexicon_map,
        "matrix": matrix,
        "language": language,
    }
    kwargs = _with_phone_inventory(kwargs, phone_inventory)
    kwargs["vowel_phones"] = vowel_phones
    kwargs["phone_matcher"] = phone_matcher
    kwargs["always_match_contexts"] = always_match_contexts
    return _annotate_search_results(**kwargs)


def _extend_stage_core(
    query_ipa: str,
    candidates: Iterable[str],
    lexicon_map: LexiconLookup,
    matrix: DistanceMatrix,
    language: str | Path | None = None,
    *,
    phone_inventory: PhoneInventory,
    vowel_phones: Iterable[str] | None = None,
    phone_matcher: Callable[[str, str], bool] | None = None,
    always_match_contexts: Iterable[str] | None = None,
) -> list[SearchResult]:
    """Stage 2: run Smith-Waterman on candidate IPA forms and assemble results.

    For each candidate, compute a local alignment score, detect matching
    phonological rules, attribute dialects, and build a three-line ASCII
    visualization. Public calls preserve profile defaults: when
    ``phone_inventory`` is omitted, the selected profile's phone inventory is
    used so multi-character IPA phones are tokenized consistently with
    ``build_kmer_index``.

    Args:
        query_ipa: IPA transcription of the search query (space-separated or
            compact notation accepted by ``tokenize_ipa``).
        candidates: Iterable of lexicon entry ids produced by the seed stage.
        lexicon_map: Mapping from entry id to either full lexicon entry dicts
            or ``LexiconRecord`` instances. Each entry must contain
            ``"headword"``, ``"ipa"``, and optionally ``"dialect"`` keys.
        matrix: Phonological distance matrix used for substitution scoring.
        language: Language identifier selecting the phonological rule set,
            and the public compatibility default for ``phone_inventory``.
            ``None`` resolves through the default language profile. ``Path``
            values are forwarded unchanged to the rule loader and skip the
            public-default backfill.
        phone_inventory: Resolved phone inventory used for greedy
            longest-match tokenization. Pass an empty tuple for generic
            character fallback.

    Returns:
        Unranked list of ``SearchResult`` objects, one per successfully
        resolved candidate.  Callers should pass them through
        ``filter_stage`` for ranking and truncation.

    Raises:
        ValueError: If a candidate lexicon entry is missing a non-empty
            ``"headword"`` or ``"ipa"`` field and ``_lemma_label`` or
            ``_entry_ipa`` rejects it.
    """
    scored_results = _score_stage_for_inventory(
        query_ipa=query_ipa,
        candidates=candidates,
        lexicon_map=lexicon_map,
        matrix=matrix,
        phone_inventory=phone_inventory,
    )
    return _annotate_search_results_for_inventory(
        query_ipa=query_ipa,
        results=scored_results,
        lexicon_map=lexicon_map,
        matrix=matrix,
        language=language,
        phone_inventory=phone_inventory,
        vowel_phones=vowel_phones,
        phone_matcher=phone_matcher,
        always_match_contexts=always_match_contexts,
    )


def filter_stage(results: list[SearchResult], max_results: int) -> list[SearchResult]:
    """Stage 3: sort by confidence and keep the top N results."""
    if max_results <= 0:
        raise ValueError("max_results must be a positive integer")
    return sorted(results, key=lambda result: (-result.confidence, result.lemma))[
        :max_results
    ]


from ._orchestration import (  # noqa: E402
    _execute_search,  # noqa: F401 (test access via search_module)
    _finalize_short_query_results,  # noqa: F401 (test access via search_module)
)
