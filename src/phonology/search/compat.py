"""Public compatibility layer for phonological search APIs."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from importlib import import_module
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Protocol, cast

from ..core.ports.profiles import LanguageProfile

if TYPE_CHECKING:
    from ._dependencies import (
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
    PhoneInventory,
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

_DialectSkeletonBuilder = Callable[[list[str]], list[str]]


class _CoreSearchModule(Protocol):
    """Typed subset of ``phonology.search`` used by public compat wrappers."""

    def get_default_language_profile(self) -> LanguageProfile: ...

    def get_language_profile(self, language_id: str) -> LanguageProfile: ...

    def _public_compatibility_search_defaults(
        self,
        *,
        language: str | None,
        phone_inventory: Iterable[str] | None,
        vowel_phones: Iterable[str] | None,
        dialect_skeleton_builders: Iterable[_DialectSkeletonBuilder] | None,
        allow_fallback: bool = False,
    ) -> tuple[
        Iterable[str] | None,
        Iterable[str] | None,
        Iterable[_DialectSkeletonBuilder] | None,
    ]: ...

    def _build_kmer_index_for_inventory(
        self,
        lexicon: Sequence[LexiconEntry],
        *,
        k: int,
        phone_inventory: PhoneInventory,
        vowel_phones: Iterable[str] | None = None,
        dialect_skeleton_builders: Iterable[_DialectSkeletonBuilder] | None = None,
    ) -> KmerIndex: ...

    def _build_lexicon_map_core(
        self,
        lexicon: Sequence[LexiconEntry],
        *,
        phone_inventory: PhoneInventory,
    ) -> LexiconMap: ...

    def _prepare_query_ipa_core(
        self,
        query: str,
        *,
        dialect: str | None = None,
        converter: IpaConverter | None = None,
        phone_inventory: PhoneInventory,
        query_ipa: str | None = None,
    ) -> PreparedQueryIpa: ...

    def _seed_stage_core(
        self,
        query_ipa: str,
        index: KmerIndex,
        *,
        k: int,
        phone_inventory: PhoneInventory,
        vowel_phones: Iterable[str] | None = None,
    ) -> list[str]: ...

    def _extend_stage_core(
        self,
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
    ) -> list[SearchResult]: ...

    def _execute_search(
        self,
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
        dialect_skeleton_builders: Iterable[_DialectSkeletonBuilder] | None = None,
        query_ipa: str | None = None,
        prepared_query: PreparedQueryIpa | None = None,
        prebuilt_ipa_index: IpaIndex | None = None,
        similarity_fallback_limit: int | None = _DEFAULT_FALLBACK_CANDIDATE_LIMIT,
        unigram_fallback_limit: int | None = _DEFAULT_FALLBACK_CANDIDATE_LIMIT,
    ) -> SearchExecutionResult: ...


_CACHED_CORE_MODULE: _CoreSearchModule | None = None
_CORE_MODULE_LOCK = Lock()


def _core_module(*, force_reload: bool = False) -> _CoreSearchModule:
    """Return the package module so monkeypatches on phonology.search are honored.

    A module lock serializes cache reads and writes so concurrent callers see a
    consistent imported module, including explicit ``force_reload`` calls.
    """
    global _CACHED_CORE_MODULE
    with _CORE_MODULE_LOCK:
        if force_reload or _CACHED_CORE_MODULE is None:
            _CACHED_CORE_MODULE = cast(
                _CoreSearchModule,
                import_module("phonology.search"),
            )
        return _CACHED_CORE_MODULE


def _normalize_phone_inventory(
    phone_inventory: Iterable[str] | None,
) -> PhoneInventory:
    """Return a canonical ``PhoneInventory`` (deduplicated, sorted longest-first).

    Inputs are deduplicated and sorted by ``(-len(phone), phone)`` so equivalent
    inventories always produce the same tuple. This canonical form is the
    cache key for ``@lru_cache`` helpers (e.g. ``_get_tokenized_rules``) and
    matches the longest-phone-first order the IPA tokenizer expects. A
    ``None`` input yields the empty tuple.
    """
    if phone_inventory is None:
        return PhoneInventory(())
    return PhoneInventory(
        tuple(
            sorted(
                {str(phone) for phone in phone_inventory},
                key=lambda phone: (-len(phone), phone),
            )
        )
    )


def _resolve_defaults_from_profile(
    profile: LanguageProfile,
    *,
    phone_inventory: Iterable[str] | None,
    vowel_phones: Iterable[str] | None,
    dialect_skeleton_builders: Iterable[_DialectSkeletonBuilder] | None,
) -> tuple[
    PhoneInventory,
    tuple[str, ...],
    Iterable[_DialectSkeletonBuilder] | None,
]:
    """Backfill unset public defaults from a resolved language profile."""
    return (
        _normalize_phone_inventory(
            profile.phone_inventory if phone_inventory is None else phone_inventory
        ),
        tuple(profile.vowel_phones if vowel_phones is None else vowel_phones),
        (
            profile.dialect_skeleton_builders
            if dialect_skeleton_builders is None
            else dialect_skeleton_builders
        ),
    )


def _resolve_public_defaults(
    *,
    language: str | Path | None,
    phone_inventory: Iterable[str] | None,
    vowel_phones: Iterable[str] | None = None,
    dialect_skeleton_builders: Iterable[_DialectSkeletonBuilder] | None = None,
    profile: LanguageProfile | None = None,
) -> tuple[
    PhoneInventory,
    tuple[str, ...],
    Iterable[_DialectSkeletonBuilder] | None,
]:
    """Resolve public defaults before crossing into the internal core."""
    if isinstance(language, Path):
        return (
            _normalize_phone_inventory(phone_inventory),
            tuple(vowel_phones or ()),
            dialect_skeleton_builders,
        )

    if profile is None and language is None:
        profile = _core_module().get_default_language_profile()

    if profile is not None:
        return _resolve_defaults_from_profile(
            profile,
            phone_inventory=phone_inventory,
            vowel_phones=vowel_phones,
            dialect_skeleton_builders=dialect_skeleton_builders,
        )

    backfilled_inventory, backfilled_vowels, backfilled_builders = (
        _core_module()._public_compatibility_search_defaults(
            language=language,
            phone_inventory=phone_inventory,
            vowel_phones=vowel_phones,
            dialect_skeleton_builders=dialect_skeleton_builders,
        )
    )
    return (
        _normalize_phone_inventory(backfilled_inventory),
        tuple(backfilled_vowels or ()),
        backfilled_builders,
    )


def _resolve_public_language_profile(
    language: str | Path | None,
    *,
    require_registered: bool,
) -> LanguageProfile | None:
    """Resolve registered profiles for converter-using APIs."""
    if isinstance(language, Path):
        return None
    if language is None:
        return _core_module().get_default_language_profile()

    normalized_language = language.strip().lower()
    try:
        return _core_module().get_language_profile(normalized_language)
    except ValueError:
        if require_registered:
            raise
        return None


def _resolve_public_converter(
    *,
    language: str | Path | None,
    converter: IpaConverter | None,
    profile: LanguageProfile | None = None,
) -> IpaConverter | None:
    """Backfill registered language converters from profiles."""
    if converter is not None or isinstance(language, Path):
        return converter

    if profile is None:
        if language is None:
            profile = _core_module().get_default_language_profile()
        else:
            normalized_language = language.strip().lower()
            profile = _core_module().get_language_profile(normalized_language)
    return profile.converter


def _resolve_public_language_argument(
    language: str | Path | None,
    profile: LanguageProfile | None,
) -> str | Path | None:
    """Return the language value to pass to internal rule-loading helpers."""
    if isinstance(language, Path):
        return language
    if profile is not None:
        profile_language = getattr(profile, "language_id", None)
        if isinstance(profile_language, str):
            return profile_language
    if language is None:
        return None
    return language.strip().lower()


def _resolve_public_dialect(
    dialect: str | None,
    profile: LanguageProfile | None,
) -> str | None:
    """Return an explicit dialect or the resolved profile's default dialect."""
    if dialect is not None:
        return dialect
    if profile is not None:
        profile_dialect = getattr(profile, "default_dialect", None)
        if isinstance(profile_dialect, str):
            return profile_dialect
    return None


def build_kmer_index(
    lexicon: Sequence[LexiconEntry],
    k: int = _DEFAULT_KMER_SIZE,
    *,
    phone_inventory: Iterable[str] | None = None,
    vowel_phones: Iterable[str] | None = None,
    dialect_skeleton_builders: Iterable[_DialectSkeletonBuilder] | None = None,
    language: str | None = None,
) -> KmerIndex:
    """Build a k-mer index, preserving public profile defaults."""
    resolved_inventory, resolved_vowels, resolved_builders = _resolve_public_defaults(
        language=language,
        phone_inventory=phone_inventory,
        vowel_phones=vowel_phones,
        dialect_skeleton_builders=dialect_skeleton_builders,
    )
    return _core_module()._build_kmer_index_for_inventory(
        lexicon,
        k=k,
        phone_inventory=resolved_inventory,
        vowel_phones=resolved_vowels,
        dialect_skeleton_builders=resolved_builders,
    )


def build_lexicon_map(
    lexicon: Sequence[LexiconEntry],
    *,
    phone_inventory: Iterable[str] | None = None,
    language: str | None = None,
) -> LexiconMap:
    """Build a lexicon map with cached IPA token counts for each entry."""
    resolved_inventory, _resolved_vowels, _resolved_builders = _resolve_public_defaults(
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
    dialect: str | None = None,
    converter: IpaConverter | None = None,
    phone_inventory: Iterable[str] | None = None,
    query_ipa: str | None = None,
    language: str | Path | None = None,
) -> PreparedQueryIpa:
    """Classify, normalize, and convert a query without crossing wildcard gaps."""
    profile = _resolve_public_language_profile(language, require_registered=True)
    resolved_inventory, _resolved_vowels, _resolved_builders = _resolve_public_defaults(
        language=language,
        phone_inventory=phone_inventory,
        profile=profile,
    )
    resolved_converter = _resolve_public_converter(
        language=language,
        converter=converter,
        profile=profile,
    )
    resolved_dialect = _resolve_public_dialect(dialect, profile)
    return _core_module()._prepare_query_ipa_core(
        query,
        dialect=resolved_dialect,
        converter=resolved_converter,
        phone_inventory=resolved_inventory,
        query_ipa=query_ipa,
    )


def seed_stage(
    query_ipa: str,
    index: KmerIndex,
    k: int = _DEFAULT_KMER_SIZE,
    *,
    phone_inventory: Iterable[str] | None = None,
    vowel_phones: Iterable[str] | None = None,
    language: str | None = None,
) -> list[str]:
    """Stage 1: rank candidate ids by shared consonant-skeleton k-mers."""
    resolved_inventory, resolved_vowels, _resolved_builders = _resolve_public_defaults(
        language=language,
        phone_inventory=phone_inventory,
        vowel_phones=vowel_phones,
    )
    return _core_module()._seed_stage_core(
        query_ipa,
        index,
        k=k,
        phone_inventory=resolved_inventory,
        vowel_phones=resolved_vowels,
    )


def extend_stage(
    query_ipa: str,
    candidates: Iterable[str],
    lexicon_map: LexiconLookup,
    matrix: DistanceMatrix,
    language: str | Path | None = None,
    *,
    phone_inventory: Iterable[str] | None = None,
) -> list[SearchResult]:
    """Stage 2: score and annotate candidate IPA forms."""
    profile = _resolve_public_language_profile(language, require_registered=False)
    resolved_inventory, resolved_vowels, _resolved_builders = _resolve_public_defaults(
        language=language,
        phone_inventory=phone_inventory,
        profile=profile,
    )
    return _core_module()._extend_stage_core(
        query_ipa,
        candidates,
        lexicon_map,
        matrix,
        language=_resolve_public_language_argument(language, profile),
        phone_inventory=resolved_inventory,
        vowel_phones=resolved_vowels,
        phone_matcher=getattr(profile, "phone_matcher", None)
        if profile is not None
        else None,
        always_match_contexts=getattr(profile, "always_match_contexts", ())
        if profile is not None
        else (),
    )


def search_execution(
    query: str,
    lexicon: Sequence[LexiconEntry],
    matrix: DistanceMatrix,
    max_results: int = 5,
    dialect: str | None = None,
    index: KmerIndex | None = None,
    unigram_index: KmerIndex | None = None,
    prebuilt_lexicon_map: LexiconMap | None = None,
    language: str | Path | None = None,
    converter: IpaConverter | None = None,
    phone_inventory: Iterable[str] | None = None,
    vowel_phones: Iterable[str] | None = None,
    dialect_skeleton_builders: Iterable[_DialectSkeletonBuilder] | None = None,
    query_ipa: str | None = None,
    prepared_query: PreparedQueryIpa | None = None,
    prebuilt_ipa_index: IpaIndex | None = None,
    similarity_fallback_limit: int | None = _DEFAULT_FALLBACK_CANDIDATE_LIMIT,
    unigram_fallback_limit: int | None = _DEFAULT_FALLBACK_CANDIDATE_LIMIT,
) -> SearchExecutionResult:
    """Run full three-stage search and return the full execution result."""
    profile = _resolve_public_language_profile(language, require_registered=True)
    resolved_inventory, resolved_vowels, resolved_builders = _resolve_public_defaults(
        language=language,
        phone_inventory=phone_inventory,
        vowel_phones=vowel_phones,
        dialect_skeleton_builders=dialect_skeleton_builders,
        profile=profile,
    )
    resolved_converter = _resolve_public_converter(
        language=language,
        converter=converter,
        profile=profile,
    )
    resolved_language = _resolve_public_language_argument(language, profile)
    resolved_dialect = _resolve_public_dialect(dialect, profile)
    return _core_module()._execute_search(
        query,
        lexicon=lexicon,
        matrix=matrix,
        max_results=max_results,
        dialect=resolved_dialect,
        index=index,
        unigram_index=unigram_index,
        prebuilt_lexicon_map=prebuilt_lexicon_map,
        language=resolved_language,
        converter=resolved_converter,
        phone_inventory=resolved_inventory,
        vowel_phones=resolved_vowels,
        phone_matcher=getattr(profile, "phone_matcher", None)
        if profile is not None
        else None,
        always_match_contexts=getattr(profile, "always_match_contexts", ())
        if profile is not None
        else (),
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
    dialect: str | None = None,
    index: KmerIndex | None = None,
    unigram_index: KmerIndex | None = None,
    prebuilt_lexicon_map: LexiconMap | None = None,
    language: str | Path | None = None,
    converter: IpaConverter | None = None,
    phone_inventory: Iterable[str] | None = None,
    vowel_phones: Iterable[str] | None = None,
    dialect_skeleton_builders: Iterable[_DialectSkeletonBuilder] | None = None,
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
        vowel_phones=vowel_phones,
        dialect_skeleton_builders=dialect_skeleton_builders,
        query_ipa=query_ipa,
        prepared_query=prepared_query,
        prebuilt_ipa_index=prebuilt_ipa_index,
        similarity_fallback_limit=similarity_fallback_limit,
        unigram_fallback_limit=unigram_fallback_limit,
    ).results
