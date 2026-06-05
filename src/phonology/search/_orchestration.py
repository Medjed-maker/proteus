"""Search-pipeline orchestration: stage routing, finalization, and proximity fallback.

This module contains the three mode-aware finalization functions, the
token-count proximity ranker used by the partial-form fallback path, and the
top-level ``_execute_search`` orchestrator that wires query preparation,
candidate selection, scoring, and finalization together.

Test-seam policy: every callable that ``_execute_search`` and the
finalizers depend on remains an attribute of ``phonology.search``. Helpers
defined in the package ``__init__`` (``_validate_search_arguments``,
``_prepare_query_ipa_core``, ``_resolve_fallback_limits``,
``_build_kmer_index_for_inventory``, ``_seed_stage_for_inventory``,
``_select_*_candidates``, ``_with_phone_inventory``,
``_annotate_search_results_for_inventory``, ``_score_stage``) are resolved
through late ``from . import ...`` blocks below so that
``monkeypatch.setattr(search_module, ...)`` continues to work after the
split. New cross-module dependencies should follow the same pattern.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace
import heapq
import logging
from pathlib import Path
import time

from ._constants import (
    _DEFAULT_FALLBACK_CANDIDATE_LIMIT,
    _DEFAULT_KMER_SIZE,
    _MIN_STAGE2_CANDIDATES,
    _SEED_MULTIPLIER,
    _SHORT_QUERY_MAX_ANNOTATION_BATCHES,
    _annotation_candidate_limit,
)
from ._debug_logging import (
    log_candidate_selection as _log_candidate_selection,
    log_finalization as _log_finalization,
    log_scoring as _log_scoring,
    perf_counter_if_debug as _perf_counter_if_debug,
)
from ._dedup import _deduplicate_by_headword, _deduplicate_by_headword_common
from ._dependencies import (
    IpaConverter,
    PreparedQueryIpa,
    SearchExecutionResult,
    _FinalizationResult,
    _LazySearchDependencies,
)
from ._filtering import (
    _apply_mode_quality_filter,
    _rank_short_query_annotation_candidates,
    _select_annotation_candidates,
)
from ._lookup import IpaIndex, _entry_ipa
from ._quality import _rank_short_query_results
from ._query import _extract_consonant_skeleton
from ._tokenization import resolve_entry_tokens, tokenize_for_inventory
from ._types import (
    DistanceMatrix,
    KmerIndex,
    LexiconEntry,
    LexiconMap,
    LexiconRecord,
    PartialQueryTokens,
    PhoneInventory,
    SearchResult,
    _CandidateSelectionResult,
)

logger = logging.getLogger(__name__)


def _rank_by_token_count_proximity(
    query_ipa: str,
    lexicon_map: LexiconMap,
    *,
    max_candidates: int,
    phone_inventory: PhoneInventory,
    query_token_count: int | None = None,
) -> list[str]:
    """Rank candidates whose IPA token count is closest to the query's.

    Used as a last-resort fallback when no consonant k-mers can be
    generated (e.g. pure-vowel queries). Returns entry IDs sorted by
    ascending token-count difference, then exact-IPA matches, then entry ID.
    Callers must pass an explicit positive cap so fallback scans remain bounded.

    If ``query_token_count`` is provided, it is used directly instead of
    re-tokenizing ``query_ipa``.
    """
    if max_candidates <= 0:
        raise ValueError("max_candidates must be a positive integer")

    query_length = query_token_count
    if query_length is None:
        query_length = len(tokenize_for_inventory(query_ipa, phone_inventory))

    def _sort_key(item: tuple[str, LexiconRecord]) -> tuple[int, bool, str]:
        entry_id, record = item
        return (
            abs(record.token_count - query_length),
            _entry_ipa(record.entry) != query_ipa,
            entry_id,
        )

    top = heapq.nsmallest(max_candidates, lexicon_map.items(), key=_sort_key)
    return [entry_id for entry_id, _ in top]


def _finalize_full_form_results(
    *,
    query_ipa: str,
    ranked_scored: list[SearchResult],
    selection: _CandidateSelectionResult,
    matrix: DistanceMatrix,
    max_results: int,
    language: str | Path | None,
    phone_inventory: PhoneInventory,
    vowel_phones: tuple[str, ...],
    phone_matcher: Callable[[str, str], bool] | None,
    always_match_contexts: tuple[str, ...] = (),
) -> _FinalizationResult:
    """Finalize Full-form results by annotating only visible deduplicated hits."""
    from . import _annotate_search_results_for_inventory

    unique_scored = _deduplicate_by_headword(ranked_scored, check_sorted=False)
    final_results = _annotate_search_results_for_inventory(
        query_ipa=query_ipa,
        results=unique_scored[:max_results],
        lexicon_map=selection.lexicon_lookup,
        matrix=matrix,
        language=language,
        phone_inventory=phone_inventory,
        vowel_phones=vowel_phones,
        phone_matcher=phone_matcher,
        always_match_contexts=always_match_contexts,
    )
    return _FinalizationResult(
        results=final_results,
        annotated_count=len(unique_scored[:max_results]),
        returned_count=len(final_results),
        truncated=False,
    )


def _finalize_short_query_results(
    *,
    query_ipa: str,
    query_log_label: str,
    ranked_scored: list[SearchResult],
    selection: _CandidateSelectionResult,
    partial_query_tokens: PartialQueryTokens | None,
    matrix: DistanceMatrix,
    max_results: int,
    language: str | Path | None,
    phone_inventory: PhoneInventory,
    vowel_phones: tuple[str, ...],
    phone_matcher: Callable[[str, str], bool] | None,
    always_match_contexts: tuple[str, ...] = (),
) -> _FinalizationResult:
    """Finalize Short-query results with batched annotation and quality filtering."""
    from . import _annotate_search_results_for_inventory

    annotation_limit = _annotation_candidate_limit(max_results)
    annotation_window_limit = annotation_limit * _SHORT_QUERY_MAX_ANNOTATION_BATCHES
    ordered_annotation_candidates = _rank_short_query_annotation_candidates(
        selection.query_tokens,
        ranked_scored,
        selection.lexicon_lookup,
        annotation_limit=annotation_window_limit,
        phone_inventory=phone_inventory,
    )
    deduplicated_results: list[SearchResult] = []
    seen_lemmas: set[str | None] = set()
    total_annotated_count = 0
    window_truncated = len(ranked_scored) > len(ordered_annotation_candidates)

    def resolve_entry_tokens_with_inventory(
        record: LexiconRecord | LexiconEntry,
    ) -> tuple[str, ...]:
        return resolve_entry_tokens(record, phone_inventory=phone_inventory)

    for batch_start in range(0, len(ordered_annotation_candidates), annotation_limit):
        batch = ordered_annotation_candidates[batch_start : batch_start + annotation_limit]
        batch_annotated = _annotate_search_results_for_inventory(
            query_ipa=query_ipa,
            results=batch,
            lexicon_map=selection.lexicon_lookup,
            matrix=matrix,
            language=language,
            phone_inventory=phone_inventory,
            vowel_phones=vowel_phones,
            phone_matcher=phone_matcher,
            always_match_contexts=always_match_contexts,
        )
        total_annotated_count += len(batch)
        batch_filtered = _apply_mode_quality_filter(
            selection.query_mode,
            selection.query_tokens,
            partial_query_tokens,
            batch_annotated,
            selection.lexicon_lookup,
            phone_inventory=phone_inventory,
        )
        batch_filtered = _rank_short_query_results(
            selection.query_tokens,
            batch_filtered,
            selection.lexicon_lookup,
            resolve_entry_tokens_with_inventory,
        )
        for result in batch_filtered:
            if result.lemma not in seen_lemmas:
                seen_lemmas.add(result.lemma)
                deduplicated_results.append(result)
        if len(deduplicated_results) >= max_results:
            final_results = deduplicated_results[:max_results]
            if window_truncated:
                final_results = [
                    replace(result, truncated=True) for result in final_results
                ]
            return _FinalizationResult(
                results=final_results,
                annotated_count=total_annotated_count,
                returned_count=len(final_results),
                truncated=window_truncated,
            )

    if window_truncated and len(deduplicated_results) < max_results:
        logger.warning(
            "Short-query search for query %s returned %d/%d results; "
            "annotation window capped at %d (query_mode=%s). "
            "Remaining %d candidates were not annotated.",
            query_log_label,
            len(deduplicated_results),
            max_results,
            annotation_window_limit,
            selection.query_mode,
            max(
                0,
                len(ranked_scored) - len(ordered_annotation_candidates),
            ),
        )
        final_results = [
            replace(r, truncated=True) for r in deduplicated_results[:max_results]
        ]
        return _FinalizationResult(
            results=final_results,
            annotated_count=total_annotated_count,
            returned_count=len(final_results),
            truncated=True,
        )
    final_results = deduplicated_results[:max_results]
    return _FinalizationResult(
        results=final_results,
        annotated_count=total_annotated_count,
        returned_count=len(final_results),
        truncated=False,
    )


def _finalize_partial_form_results(
    *,
    query_ipa: str,
    ranked_scored: list[SearchResult],
    selection: _CandidateSelectionResult,
    partial_query_tokens: PartialQueryTokens | None,
    matrix: DistanceMatrix,
    max_results: int,
    language: str | Path | None,
    phone_inventory: PhoneInventory,
    vowel_phones: tuple[str, ...],
    phone_matcher: Callable[[str, str], bool] | None,
    always_match_contexts: tuple[str, ...] = (),
) -> _FinalizationResult:
    """Finalize Partial-form results after bounded annotation and filtering."""
    from . import _annotate_search_results_for_inventory

    annotation_candidates = _select_annotation_candidates(
        selection.query_mode,
        selection.query_tokens,
        partial_query_tokens,
        ranked_scored,
        selection.lexicon_lookup,
        _annotation_candidate_limit(max_results),
        phone_inventory=phone_inventory,
    )
    annotated_results = _annotate_search_results_for_inventory(
        query_ipa=query_ipa,
        results=annotation_candidates,
        lexicon_map=selection.lexicon_lookup,
        matrix=matrix,
        language=language,
        phone_inventory=phone_inventory,
        vowel_phones=vowel_phones,
        phone_matcher=phone_matcher,
        always_match_contexts=always_match_contexts,
    )
    filtered_results = _apply_mode_quality_filter(
        selection.query_mode,
        selection.query_tokens,
        partial_query_tokens,
        annotated_results,
        selection.lexicon_lookup,
        phone_inventory=phone_inventory,
    )
    deduplicated_results = _deduplicate_by_headword_common(filtered_results)
    final_results = deduplicated_results[:max_results]
    return _FinalizationResult(
        results=final_results,
        annotated_count=len(annotation_candidates),
        returned_count=len(final_results),
        truncated=False,
    )


def _execute_search(
    query: str,
    lexicon: Sequence[LexiconEntry],
    matrix: DistanceMatrix,
    *,
    max_results: int = 5,
    dialect: str | None = None,
    index: KmerIndex | None = None,
    unigram_index: KmerIndex | None = None,
    prebuilt_lexicon_map: LexiconMap | None = None,
    language: str | Path | None = None,
    converter: IpaConverter | None = None,
    phone_inventory: PhoneInventory,
    vowel_phones: tuple[str, ...] = (),
    phone_matcher: Callable[[str, str], bool] | None = None,
    always_match_contexts: tuple[str, ...] = (),
    dialect_skeleton_builders: Iterable[Callable[[list[str]], list[str]]] | None = None,
    query_ipa: str | None = None,
    prepared_query: PreparedQueryIpa | None = None,
    prebuilt_ipa_index: IpaIndex | None = None,
    similarity_fallback_limit: int | None = _DEFAULT_FALLBACK_CANDIDATE_LIMIT,
    unigram_fallback_limit: int | None = _DEFAULT_FALLBACK_CANDIDATE_LIMIT,
) -> SearchExecutionResult:
    """Run full three-stage search for a query form.

    See the module docstring for the test-seam policy. Helpers defined in
    ``phonology.search.__init__`` (``_validate_search_arguments``,
    ``_prepare_query_ipa_core``, ``_resolve_fallback_limits``,
    ``_build_kmer_index_for_inventory``, ``_seed_stage_for_inventory``,
    ``_select_*_candidates``, ``_with_phone_inventory``, ``_score_stage``) are
    resolved via the package namespace below so that test monkeypatches on
    ``phonology.search`` take effect.

    The query mode is classified before search:
      - "Partial-form": Query contains exactly one wildcard marker ('-' or '*').
      - "Short-query": Query is short (<= 3 characters, minus markers).
      - "Full-form": All other queries.

    The stages operate via cascading fallbacks:
      1. Default seed stage (k=2) returns candidates if found.
      2. If empty, a unigram fallback seed (k=1) is applied when the query
         has a consonant skeleton. The unigram fallback limit truncates the
         unigram-hit set before scoring.
      3. If still empty, a token-count similarity fallback ranks candidates
         by length proximity. The similarity fallback limit bounds this scan.
    After seeding, full-form queries preserve the lightweight top-N annotation
    path, while short and partial queries annotate a bounded ranked subset
    before applying mode-specific quality filters and final deduplication.

    Raises:
        ValueError: If ``query`` is empty/whitespace-only, ``max_results``
            is non-positive, or a lexicon entry lacks a valid ``"id"`` or
            ``"headword"`` (raised by ``_entry_id`` during lexicon map
            construction).
    """
    from . import (  # type: ignore[attr-defined]
        _build_kmer_index_for_inventory,
        _prepare_query_ipa_core,
        _resolve_fallback_limits,
        _score_stage,
        _seed_stage_for_inventory,
        _select_seeded_candidates,
        _select_token_proximity_fallback_candidates,
        _select_unigram_fallback_candidates,
        _summarize_query_ipa_for_logs,
        _validate_search_arguments,
        _with_phone_inventory,
    )

    _validate_search_arguments(
        query=query,
        max_results=max_results,
        similarity_fallback_limit=similarity_fallback_limit,
        unigram_fallback_limit=unigram_fallback_limit,
        prepared_query=prepared_query,
        query_ipa=query_ipa,
    )

    if prepared_query is None:
        prepared_query = _prepare_query_ipa_core(
            query,
            dialect=dialect,
            converter=converter,
            phone_inventory=phone_inventory,
            query_ipa=query_ipa,
        )
    query_mode = prepared_query.query_mode
    query_ipa = prepared_query.query_ipa
    query_tokens = tokenize_for_inventory(query_ipa, phone_inventory)
    _debug_enabled = logger.isEnabledFor(logging.DEBUG)
    query_log_label = _summarize_query_ipa_for_logs(
        query_ipa,
        query_token_count=len(query_tokens),
        debug_enabled=_debug_enabled,
    )
    debug_query_log_label = query_log_label if _debug_enabled else ""
    query_skeleton = _extract_consonant_skeleton(
        query_tokens,
        vowel_phones=vowel_phones,
    )
    partial_query_tokens = prepared_query.partial_query_tokens
    fallback_limits = _resolve_fallback_limits(
        query_log_label=query_log_label,
        similarity_fallback_limit=similarity_fallback_limit,
        unigram_fallback_limit=unigram_fallback_limit,
    )
    effective_similarity_fallback_limit = fallback_limits.similarity
    effective_unigram_fallback_limit = fallback_limits.unigram
    dependencies = _LazySearchDependencies(
        lexicon=lexicon,
        prebuilt_lexicon_map=prebuilt_lexicon_map,
        prebuilt_ipa_index=prebuilt_ipa_index,
        phone_inventory=phone_inventory,
        vowel_phones=vowel_phones,
        dialect_skeleton_builders=dialect_skeleton_builders,
    )

    search_index = (
        index
        if index is not None
        else _build_kmer_index_for_inventory(
            lexicon,
            k=_DEFAULT_KMER_SIZE,
            phone_inventory=phone_inventory,
            vowel_phones=vowel_phones,
            dialect_skeleton_builders=dialect_skeleton_builders,
        )
    )
    _t_selection = _perf_counter_if_debug(logger)
    seed_candidates = _seed_stage_for_inventory(
        query_ipa,
        search_index,
        k=_DEFAULT_KMER_SIZE,
        phone_inventory=phone_inventory,
        vowel_phones=vowel_phones,
    )
    stage2_limit = max(_MIN_STAGE2_CANDIDATES, max_results * _SEED_MULTIPLIER)

    if seed_candidates:
        selection = _select_seeded_candidates(
            query_ipa=query_ipa,
            query_log_label=query_log_label,
            query_mode=query_mode,
            query_tokens=query_tokens,
            query_skeleton=query_skeleton,
            partial_query_tokens=partial_query_tokens,
            seed_candidates=seed_candidates,
            lexicon=lexicon,
            dependencies=dependencies,
            unigram_index=unigram_index,
            max_results=max_results,
            stage2_limit=stage2_limit,
        )
    else:
        unigram_candidates: list[str] = []
        if query_skeleton:
            fallback_unigram_index = (
                unigram_index
                if unigram_index is not None
                else _build_kmer_index_for_inventory(
                    lexicon,
                    k=1,
                    phone_inventory=phone_inventory,
                    vowel_phones=vowel_phones,
                    dialect_skeleton_builders=dialect_skeleton_builders,
                )
            )
            unigram_candidates = _seed_stage_for_inventory(
                query_ipa,
                fallback_unigram_index,
                k=1,
                phone_inventory=phone_inventory,
                vowel_phones=vowel_phones,
            )
        if unigram_candidates:
            selection = _select_unigram_fallback_candidates(
                query_ipa=query_ipa,
                query_log_label=query_log_label,
                query_mode=query_mode,
                query_tokens=query_tokens,
                partial_query_tokens=partial_query_tokens,
                unigram_candidates=unigram_candidates,
                dependencies=dependencies,
                max_results=max_results,
                effective_unigram_fallback_limit=effective_unigram_fallback_limit,
            )
        else:
            selection = _select_token_proximity_fallback_candidates(
                query_ipa=query_ipa,
                query_log_label=query_log_label,
                query_mode=query_mode,
                query_tokens=query_tokens,
                partial_query_tokens=partial_query_tokens,
                dependencies=dependencies,
                max_results=max_results,
                effective_similarity_fallback_limit=effective_similarity_fallback_limit,
            )
    if _debug_enabled:
        _log_candidate_selection(
            logger,
            query_label=debug_query_log_label,
            query_mode=selection.query_mode,
            selection_path=selection.selection_path,
            seed_candidate_count=selection.seed_candidate_count,
            unigram_candidate_count=selection.unigram_candidate_count,
            selected_count=len(selection.candidate_ids),
            fallback_limit=selection.fallback_limit,
            elapsed_ms=(time.perf_counter() - _t_selection) * 1000.0,
        )

    _t_scoring = _perf_counter_if_debug(logger)
    score_params = _with_phone_inventory(
        {
            "query_ipa": query_ipa,
            "candidates": selection.candidate_ids,
            "lexicon_map": selection.lexicon_lookup,
            "matrix": matrix,
        },
        phone_inventory,
    )
    scored_results = _score_stage(**score_params)
    ranked_scored = sorted(
        scored_results,
        key=lambda r: (-r.confidence, r.lemma is None, r.lemma or ""),
    )
    if _debug_enabled:
        _log_scoring(
            logger,
            query_label=debug_query_log_label,
            selected_count=len(selection.candidate_ids),
            scored_count=len(ranked_scored),
            elapsed_ms=(time.perf_counter() - _t_scoring) * 1000.0,
        )

    _t_finalization = _perf_counter_if_debug(logger)
    if selection.query_mode == "Full-form":
        finalization = _finalize_full_form_results(
            query_ipa=query_ipa,
            ranked_scored=ranked_scored,
            selection=selection,
            matrix=matrix,
            max_results=max_results,
            language=language,
            phone_inventory=phone_inventory,
            vowel_phones=vowel_phones,
            phone_matcher=phone_matcher,
            always_match_contexts=always_match_contexts,
        )
    elif selection.query_mode == "Short-query":
        finalization = _finalize_short_query_results(
            query_ipa=query_ipa,
            query_log_label=query_log_label,
            ranked_scored=ranked_scored,
            selection=selection,
            partial_query_tokens=partial_query_tokens,
            matrix=matrix,
            max_results=max_results,
            language=language,
            phone_inventory=phone_inventory,
            vowel_phones=vowel_phones,
            phone_matcher=phone_matcher,
            always_match_contexts=always_match_contexts,
        )
    else:
        finalization = _finalize_partial_form_results(
            query_ipa=query_ipa,
            ranked_scored=ranked_scored,
            selection=selection,
            partial_query_tokens=partial_query_tokens,
            matrix=matrix,
            max_results=max_results,
            language=language,
            phone_inventory=phone_inventory,
            vowel_phones=vowel_phones,
            phone_matcher=phone_matcher,
            always_match_contexts=always_match_contexts,
        )
    if _debug_enabled:
        _log_finalization(
            logger,
            query_label=debug_query_log_label,
            query_mode=selection.query_mode,
            annotated_count=finalization.annotated_count,
            returned_count=finalization.returned_count,
            elapsed_ms=(time.perf_counter() - _t_finalization) * 1000.0,
        )
    return SearchExecutionResult(
        results=finalization.results,
        query_ipa=query_ipa,
        query_mode=query_mode,
        truncated=finalization.truncated,
    )
