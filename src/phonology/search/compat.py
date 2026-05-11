"""Public compatibility layer for phonological search APIs."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from importlib import import_module
from pathlib import Path
from threading import Lock
from types import ModuleType
from typing import TYPE_CHECKING

from .._paths import DEFAULT_LANGUAGE_ID

if TYPE_CHECKING:
    from ..search import (
        IpaConverter,
        PreparedQueryIpa,
        SearchExecutionResult,
    )
from ._constants import _DEFAULT_FALLBACK_CANDIDATE_LIMIT, _DEFAULT_KMER_SIZE
from ._lookup import IpaIndex
from ._types import (
    DistanceMatrix,
    KmerIndex,
    LexiconEntry,
    LexiconLookup,
    LexiconMap,
    SearchResult,
)

__all__ = [
    "build_kmer_index",
    "build_lexicon_map",
    "extend_stage",
    "prepare_query_ipa",
    "search",
    "search_execution",
    "seed_stage",
]

_CACHED_CORE_MODULE: ModuleType | None = None
_CORE_MODULE_LOCK = Lock()


def _core_module(*, force_reload: bool = False) -> ModuleType:
    """Return the package module so monkeypatches on phonology.search are honored.

    A module lock serializes cache reads and writes so concurrent callers see a
    consistent imported module, including explicit ``force_reload`` calls.
    """
    global _CACHED_CORE_MODULE
    with _CORE_MODULE_LOCK:
        if force_reload or _CACHED_CORE_MODULE is None:
            _CACHED_CORE_MODULE = import_module("phonology.search")
        return _CACHED_CORE_MODULE


def _normalize_phone_inventory(
    phone_inventory: Iterable[str] | None,
) -> tuple[str, ...]:
    """Materialize public optional inventory into the core's required tuple."""
    if phone_inventory is None:
        return ()
    return tuple(phone_inventory)


def _resolve_public_defaults(
    *,
    language: str | Path,
    phone_inventory: Iterable[str] | None,
    dialect_skeleton_builders: Iterable[Callable[[list[str]], list[str]]] | None = None,
) -> tuple[
    tuple[str, ...],
    Iterable[Callable[[list[str]], list[str]]] | None,
]:
    """Resolve public defaults before crossing into the internal core."""
    if not isinstance(language, str):
        return _normalize_phone_inventory(phone_inventory), dialect_skeleton_builders

    backfilled_inventory, backfilled_builders = (
        _core_module()._public_compatibility_search_defaults(
            language=language,
            phone_inventory=phone_inventory,
            dialect_skeleton_builders=dialect_skeleton_builders,
        )
    )
    return _normalize_phone_inventory(backfilled_inventory), backfilled_builders


def build_kmer_index(
    lexicon: Sequence[LexiconEntry],
    k: int = _DEFAULT_KMER_SIZE,
    *,
    phone_inventory: Iterable[str] | None = None,
    dialect_skeleton_builders: Iterable[Callable[[list[str]], list[str]]] | None = None,
    language: str = DEFAULT_LANGUAGE_ID,
) -> KmerIndex:
    """Build a k-mer index, preserving public Ancient Greek defaults."""
    resolved_inventory, resolved_builders = _resolve_public_defaults(
        language=language,
        phone_inventory=phone_inventory,
        dialect_skeleton_builders=dialect_skeleton_builders,
    )
    return _core_module()._build_kmer_index_for_inventory(
        lexicon,
        k=k,
        phone_inventory=resolved_inventory,
        dialect_skeleton_builders=resolved_builders,
    )


def build_lexicon_map(
    lexicon: Sequence[LexiconEntry],
    *,
    phone_inventory: Iterable[str] | None = None,
    language: str = DEFAULT_LANGUAGE_ID,
) -> LexiconMap:
    """Build a lexicon map with cached IPA token counts for each entry."""
    resolved_inventory, _resolved_builders = _resolve_public_defaults(
        language=language,
        phone_inventory=phone_inventory,
        dialect_skeleton_builders=None,
    )
    return _core_module()._build_lexicon_map_core(
        lexicon,
        phone_inventory=resolved_inventory,
    )


def prepare_query_ipa(
    query: str,
    *,
    dialect: str = "attic",
    converter: IpaConverter | None = None,
    phone_inventory: Iterable[str] | None = None,
    query_ipa: str | None = None,
    language: str = DEFAULT_LANGUAGE_ID,
) -> PreparedQueryIpa:
    """Classify, normalize, and convert a query without crossing wildcard gaps."""
    resolved_inventory, _resolved_builders = _resolve_public_defaults(
        language=language,
        phone_inventory=phone_inventory,
    )
    return _core_module()._prepare_query_ipa_core(
        query,
        dialect=dialect,
        converter=converter,
        phone_inventory=resolved_inventory,
        query_ipa=query_ipa,
    )


def seed_stage(
    query_ipa: str,
    index: KmerIndex,
    k: int = _DEFAULT_KMER_SIZE,
    *,
    phone_inventory: Iterable[str] | None = None,
    language: str = DEFAULT_LANGUAGE_ID,
) -> list[str]:
    """Stage 1: rank candidate ids by shared consonant-skeleton k-mers."""
    resolved_inventory, _resolved_builders = _resolve_public_defaults(
        language=language,
        phone_inventory=phone_inventory,
    )
    return _core_module()._seed_stage_core(
        query_ipa,
        index,
        k=k,
        phone_inventory=resolved_inventory,
    )


def extend_stage(
    query_ipa: str,
    candidates: Iterable[str],
    lexicon_map: LexiconLookup,
    matrix: DistanceMatrix,
    language: str = DEFAULT_LANGUAGE_ID,
    *,
    phone_inventory: Iterable[str] | None = None,
) -> list[SearchResult]:
    """Stage 2: score and annotate candidate IPA forms."""
    resolved_inventory, _resolved_builders = _resolve_public_defaults(
        language=language,
        phone_inventory=phone_inventory,
    )
    return _core_module()._extend_stage_core(
        query_ipa,
        candidates,
        lexicon_map,
        matrix,
        language=language,
        phone_inventory=resolved_inventory,
    )


def search_execution(
    query: str,
    lexicon: Sequence[LexiconEntry],
    matrix: DistanceMatrix,
    max_results: int = 5,
    dialect: str = "attic",
    index: KmerIndex | None = None,
    unigram_index: KmerIndex | None = None,
    prebuilt_lexicon_map: LexiconMap | None = None,
    language: str = DEFAULT_LANGUAGE_ID,
    converter: IpaConverter | None = None,
    phone_inventory: Iterable[str] | None = None,
    dialect_skeleton_builders: Iterable[Callable[[list[str]], list[str]]] | None = None,
    query_ipa: str | None = None,
    prepared_query: PreparedQueryIpa | None = None,
    prebuilt_ipa_index: IpaIndex | None = None,
    similarity_fallback_limit: int | None = _DEFAULT_FALLBACK_CANDIDATE_LIMIT,
    unigram_fallback_limit: int | None = _DEFAULT_FALLBACK_CANDIDATE_LIMIT,
) -> SearchExecutionResult:
    """Run full three-stage search and return the full execution result."""
    resolved_inventory, resolved_builders = _resolve_public_defaults(
        language=language,
        phone_inventory=phone_inventory,
        dialect_skeleton_builders=dialect_skeleton_builders,
    )
    return _core_module()._execute_search(
        query,
        lexicon=lexicon,
        matrix=matrix,
        max_results=max_results,
        dialect=dialect,
        index=index,
        unigram_index=unigram_index,
        prebuilt_lexicon_map=prebuilt_lexicon_map,
        language=language,
        converter=converter,
        phone_inventory=resolved_inventory,
        dialect_skeleton_builders=resolved_builders,
        query_ipa=query_ipa,
        prepared_query=prepared_query,
        prebuilt_ipa_index=prebuilt_ipa_index,
        similarity_fallback_limit=similarity_fallback_limit,
        unigram_fallback_limit=unigram_fallback_limit,
    )


def search(
    query: str,
    lexicon: Sequence[LexiconEntry],
    matrix: DistanceMatrix,
    max_results: int = 5,
    dialect: str = "attic",
    index: KmerIndex | None = None,
    unigram_index: KmerIndex | None = None,
    prebuilt_lexicon_map: LexiconMap | None = None,
    language: str = DEFAULT_LANGUAGE_ID,
    converter: IpaConverter | None = None,
    phone_inventory: Iterable[str] | None = None,
    dialect_skeleton_builders: Iterable[Callable[[list[str]], list[str]]] | None = None,
    query_ipa: str | None = None,
    prepared_query: PreparedQueryIpa | None = None,
    prebuilt_ipa_index: IpaIndex | None = None,
    similarity_fallback_limit: int | None = _DEFAULT_FALLBACK_CANDIDATE_LIMIT,
    unigram_fallback_limit: int | None = _DEFAULT_FALLBACK_CANDIDATE_LIMIT,
) -> list[SearchResult]:
    """Run full three-stage search for a query form."""
    return search_execution(
        query,
        lexicon=lexicon,
        matrix=matrix,
        max_results=max_results,
        dialect=dialect,
        index=index,
        unigram_index=unigram_index,
        prebuilt_lexicon_map=prebuilt_lexicon_map,
        language=language,
        converter=converter,
        phone_inventory=phone_inventory,
        dialect_skeleton_builders=dialect_skeleton_builders,
        query_ipa=query_ipa,
        prepared_query=prepared_query,
        prebuilt_ipa_index=prebuilt_ipa_index,
        similarity_fallback_limit=similarity_fallback_limit,
        unigram_fallback_limit=unigram_fallback_limit,
    ).results
