"""Candidate-selection paths for the search pipeline.

This module owns the four mutually exclusive candidate-selection paths used
by :func:`_execute_search`:

- :func:`_select_seeded_candidates` — primary k=2 seed path (full / short
  forms) and its partial-form variant.
- :func:`_select_unigram_fallback_candidates` — k=1 fallback when the
  primary seed stage is empty.
- :func:`_select_token_proximity_fallback_candidates` — token-count
  proximity fallback when both k=2 and k=1 are empty.
- :func:`_select_partial_token_fallback_candidates` — partial-form variant
  of the token-proximity fallback.

It also exposes :func:`_inject_length_proximate_candidates`, the helper that
back-fills the Stage 2 window with length-near seeds for full-form queries.

Test-seam policy: ``_rank_by_token_count_proximity``, ``_with_phone_inventory``,
``_build_kmer_index_for_inventory``, and ``_seed_stage_for_inventory`` are
resolved through late ``from . import ...`` blocks below so that
``monkeypatch.setattr(search_module, ...)`` (used by tests for the rank helper)
continues to work after this split. The remaining helpers in those blocks are
also looked up via the package namespace for symmetry.
"""

from __future__ import annotations

from collections.abc import Sequence
import logging

from ._constants import (
    _LENGTH_PROXIMATE_LIMIT_MULTIPLIER,
    _annotation_candidate_limit,
)
from ._dependencies import _LazySearchDependencies
from ._lookup import _inject_exact_ipa_matches, _normalize_ipa_lookup_key
from ._overlap import _merge_bounded_candidate_ids
from ._partial import (
    _select_partial_fallback_candidates as _default_select_partial_fallback_candidates,
    _select_partial_seed_candidates as _default_select_partial_seed_candidates,
)
from ._tokenization import resolve_entry_tokens
from ._types import (
    KmerIndex,
    LexiconEntry,
    LexiconLookup,
    LexiconMap,
    LexiconRecord,
    PartialQueryTokens,
    PhoneInventory,
    QueryMode,
    _CandidateSelectionPath,
    _CandidateSelectionResult,
)

logger = logging.getLogger(__name__)


def _inject_length_proximate_candidates(
    *,
    query_token_count: int,
    seed_candidates: Sequence[str],
    candidate_ids: list[str],
    lexicon_lookup: LexiconLookup,
    limit: int,
    phone_inventory: PhoneInventory,
    max_delta: int = 2,
) -> list[str]:
    """Append seed candidates whose IPA token count is close to the query's.

    This preserves the caller's existing Stage 2 window first, then draws a
    bounded supplement from the original seed order so tied k-mer groups do not
    drop short, length-near full-form candidates before scoring.
    """
    if limit <= 0:
        return []

    merged_ids = list(candidate_ids[:limit])
    seen = set(merged_ids)
    if len(merged_ids) >= limit:
        return merged_ids

    for entry_id in seed_candidates:
        if entry_id in seen:
            continue
        record_or_entry = lexicon_lookup.get(entry_id)
        if record_or_entry is None:
            continue
        if isinstance(record_or_entry, LexiconRecord):
            token_count = record_or_entry.token_count
        else:
            token_count = len(
                resolve_entry_tokens(
                    record_or_entry,
                    phone_inventory=phone_inventory,
                )
            )
        if abs(token_count - query_token_count) > max_delta:
            continue

        merged_ids.append(entry_id)
        seen.add(entry_id)
        if len(merged_ids) >= limit:
            break

    return merged_ids


def _select_seeded_candidates(
    *,
    query_ipa: str,
    query_log_label: str,
    query_mode: QueryMode,
    query_tokens: list[str],
    query_skeleton: list[str],
    partial_query_tokens: PartialQueryTokens | None,
    seed_candidates: list[str],
    lexicon: Sequence[LexiconEntry],
    dependencies: _LazySearchDependencies,
    unigram_index: KmerIndex | None,
    max_results: int,
    stage2_limit: int,
) -> _CandidateSelectionResult:
    """Select stage-2 candidates when the primary k=2 seed stage has hits."""
    import sys

    from . import (
        _build_kmer_index_for_inventory,
        _seed_stage_for_inventory,
    )

    package = sys.modules.get("phonology.search")
    _select_partial_seed_candidates = getattr(
        package,
        "_select_partial_seed_candidates",
        _default_select_partial_seed_candidates,
    )

    if query_mode == "Partial-form":
        if partial_query_tokens is None:
            raise ValueError(
                "partial_query_tokens must not be None when query_mode == 'Partial-form'"
            )
        tokenized_map = dependencies.tokenized_lexicon_map()
        partial_candidate_window = _annotation_candidate_limit(max_results)
        supplemental_unigram_candidates: list[str] = []
        if query_skeleton:
            fallback_unigram_index = (
                unigram_index
                if unigram_index is not None
                else _build_kmer_index_for_inventory(
                    lexicon,
                    k=1,
                    phone_inventory=dependencies.phone_inventory,
                    dialect_skeleton_builders=dependencies.dialect_skeleton_builders,
                )
            )
            supplemental_unigram_candidates = _seed_stage_for_inventory(
                query_ipa,
                fallback_unigram_index,
                k=1,
                phone_inventory=dependencies.phone_inventory,
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
        logger.info(
            "k=2 seed hit for query %s (mode=%s); selected %d stage-2 candidates "
            "from %d seeds (+%d unigram supplements)",
            query_log_label,
            query_mode,
            len(candidate_ids),
            len(seed_candidates),
            len(supplemental_unigram_candidates),
        )
        return _CandidateSelectionResult(
            candidate_ids=candidate_ids,
            lexicon_lookup=tokenized_map,
            query_mode=query_mode,
            query_tokens=query_tokens,
            selection_path="partial-seed",
            seed_candidate_count=len(seed_candidates),
            unigram_candidate_count=len(supplemental_unigram_candidates),
            fallback_limit=None,
        )

    candidate_ids = seed_candidates[:stage2_limit]
    lexicon_lookup = dependencies.lexicon_lookup()
    ipa_index_lookup = dependencies.ipa_index()
    normalized_query_ipa = _normalize_ipa_lookup_key(query_ipa)
    has_exact_ipa_match = (
        query_mode == "Full-form" and normalized_query_ipa in ipa_index_lookup
    )
    candidate_ids = _inject_exact_ipa_matches(
        query_ipa,
        candidate_ids,
        lexicon_lookup,
        ipa_index=ipa_index_lookup,
        limit=stage2_limit,
    )
    if query_mode == "Full-form" and not has_exact_ipa_match:
        candidate_ids = _inject_length_proximate_candidates(
            query_token_count=len(query_tokens),
            seed_candidates=seed_candidates,
            candidate_ids=candidate_ids,
            lexicon_lookup=lexicon_lookup,
            limit=stage2_limit * _LENGTH_PROXIMATE_LIMIT_MULTIPLIER,
            phone_inventory=dependencies.phone_inventory,
        )

    logger.info(
        "k=2 seed hit for query %s (mode=%s); selected %d stage-2 candidates from %d seeds",
        query_log_label,
        query_mode,
        len(candidate_ids),
        len(seed_candidates),
    )
    return _CandidateSelectionResult(
        candidate_ids=candidate_ids,
        lexicon_lookup=lexicon_lookup,
        query_mode=query_mode,
        query_tokens=query_tokens,
        selection_path="seed",
        seed_candidate_count=len(seed_candidates),
        unigram_candidate_count=0,
        fallback_limit=None,
    )


def _select_unigram_fallback_candidates(
    *,
    query_ipa: str,
    query_log_label: str,
    query_mode: QueryMode,
    query_tokens: list[str],
    partial_query_tokens: PartialQueryTokens | None,
    unigram_candidates: list[str],
    dependencies: _LazySearchDependencies,
    max_results: int,
    effective_unigram_fallback_limit: int,
) -> _CandidateSelectionResult:
    """Select stage-2 candidates from k=1 fallback hits."""
    import sys

    from . import (  # type: ignore[attr-defined]
        _rank_by_token_count_proximity,
        _with_phone_inventory,
    )

    package = sys.modules.get("phonology.search")
    _select_partial_fallback_candidates = getattr(
        package,
        "_select_partial_fallback_candidates",
        _default_select_partial_fallback_candidates,
    )

    if query_mode == "Partial-form":
        if partial_query_tokens is None:
            raise ValueError(
                "partial_query_tokens must not be None for Partial-form query"
            )
        tokenized_map = dependencies.tokenized_lexicon_map()
        ranked_candidates = _rank_by_token_count_proximity(
            query_ipa,
            {
                entry_id: tokenized_map[entry_id]
                for entry_id in unigram_candidates
                if entry_id in tokenized_map
            },
            **_with_phone_inventory(
                {
                    "max_candidates": effective_unigram_fallback_limit,
                    "query_token_count": len(query_tokens),
                },
                dependencies.phone_inventory,
            ),
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
        logger.info(
            "k=2 seed empty for query %s; partial k=1 fallback selected %d "
            "stage-2 candidates from %d unigram hits (cap=%s)",
            query_log_label,
            len(candidate_ids),
            len(unigram_candidates),
            effective_unigram_fallback_limit,
        )
        return _CandidateSelectionResult(
            candidate_ids=candidate_ids,
            lexicon_lookup=tokenized_map,
            query_mode=query_mode,
            query_tokens=query_tokens,
            selection_path="partial-unigram-fallback",
            seed_candidate_count=0,
            unigram_candidate_count=len(unigram_candidates),
            fallback_limit=effective_unigram_fallback_limit,
        )

    tokenized_map = dependencies.tokenized_lexicon_map()
    missing_unigram_candidates = list(
        dict.fromkeys(
            entry_id for entry_id in unigram_candidates if entry_id not in tokenized_map
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
            "k=2 seed empty for query %s; ignoring unigram fallback "
            "candidates missing from tokenized lexicon map: %s",
            query_log_label,
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
        **_with_phone_inventory(
            {
                "max_candidates": effective_unigram_fallback_limit,
                "query_token_count": len(query_tokens),
            },
            dependencies.phone_inventory,
        ),
    )
    logger.info(
        "k=2 seed empty for query %s; capped k=1 fallback evaluating %d of %d "
        "unigram candidates ranked by token-count proximity",
        query_log_label,
        len(candidate_ids),
        len(unigram_candidates),
    )
    return _CandidateSelectionResult(
        candidate_ids=candidate_ids,
        lexicon_lookup=tokenized_map,
        query_mode=query_mode,
        query_tokens=query_tokens,
        selection_path="unigram-fallback",
        seed_candidate_count=0,
        unigram_candidate_count=len(unigram_candidates),
        fallback_limit=effective_unigram_fallback_limit,
    )


def _select_token_proximity_fallback_candidates(
    *,
    query_ipa: str,
    query_log_label: str,
    query_mode: QueryMode,
    query_tokens: list[str],
    partial_query_tokens: PartialQueryTokens | None,
    dependencies: _LazySearchDependencies,
    max_results: int,
    effective_similarity_fallback_limit: int,
) -> _CandidateSelectionResult:
    """Select stage-2 candidates when both k=2 and k=1 seeds are empty."""
    import sys

    from . import (  # type: ignore[attr-defined]
        _rank_by_token_count_proximity,
        _with_phone_inventory,
    )

    package = sys.modules.get("phonology.search")
    _partial_token_fallback = getattr(
        package,
        "_select_partial_token_fallback_candidates",
        _select_partial_token_fallback_candidates,
    )

    tokenized_map = dependencies.tokenized_lexicon_map()
    if query_mode == "Partial-form":
        if partial_query_tokens is None:
            raise ValueError(
                "partial_query_tokens must not be None for Partial-form query"
            )
        candidate_ids = _partial_token_fallback(
            partial_query_tokens,
            query_ipa,
            len(query_tokens),
            tokenized_map,
            **_with_phone_inventory(
                {
                    "max_results": max_results,
                    "explicit_limit": effective_similarity_fallback_limit,
                },
                dependencies.phone_inventory,
            ),
        )
        logger.info(
            "k=2 and k=1 seeds empty for query %s; partial token fallback "
            "selected %d stage-2 candidates from %d tokenized entries (cap=%s)",
            query_log_label,
            len(candidate_ids),
            len(tokenized_map),
            effective_similarity_fallback_limit,
        )
        selection_path: _CandidateSelectionPath = "partial-token-proximity-fallback"
    else:
        candidate_ids = _rank_by_token_count_proximity(
            query_ipa,
            tokenized_map,
            **_with_phone_inventory(
                {
                    "max_candidates": effective_similarity_fallback_limit,
                    "query_token_count": len(query_tokens),
                },
                dependencies.phone_inventory,
            ),
        )
        logger.info(
            "k=2 and k=1 seeds empty for query %s; capped fallback evaluating %d of %d candidates "
            "ranked by token-count proximity",
            query_log_label,
            len(candidate_ids),
            len(tokenized_map),
        )
        selection_path = "token-proximity-fallback"

    return _CandidateSelectionResult(
        candidate_ids=candidate_ids,
        lexicon_lookup=tokenized_map,
        query_mode=query_mode,
        query_tokens=query_tokens,
        selection_path=selection_path,
        seed_candidate_count=0,
        unigram_candidate_count=0,
        fallback_limit=effective_similarity_fallback_limit,
    )


def _select_partial_token_fallback_candidates(
    partial_query: PartialQueryTokens,
    query_ipa: str,
    query_token_count: int,
    lexicon_map: LexiconMap,
    *,
    max_results: int,
    explicit_limit: int,
    phone_inventory: PhoneInventory,
) -> list[str]:
    """Select bounded partial-form token fallback candidates.

    Rank only ``explicit_limit`` candidates by token-count proximity, then
    narrow the expensive partial matching to the stage-2 window. Callers
    must pass a positive integer cap; the full-scan uncapped variant was
    removed because every call site now enforces a default cap.
    """
    import sys

    from . import (  # type: ignore[attr-defined]
        _rank_by_token_count_proximity,
        _with_phone_inventory,
    )

    package = sys.modules.get("phonology.search")
    _select_partial_fallback_candidates = getattr(
        package,
        "_select_partial_fallback_candidates",
        _default_select_partial_fallback_candidates,
    )

    ranked_candidates = _rank_by_token_count_proximity(
        query_ipa,
        lexicon_map,
        **_with_phone_inventory(
            {
                "max_candidates": explicit_limit,
                "query_token_count": query_token_count,
            },
            phone_inventory,
        ),
    )
    return _select_partial_fallback_candidates(
        partial_query,
        ranked_candidates,
        lexicon_map,
        max_results=max_results,
        explicit_limit=explicit_limit,
    )
