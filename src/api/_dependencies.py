"""Search dependency loading and data-version metadata.

Public callers should use ``load_search_dependencies``, ``load_lexicon_entries``,
and ``build_ruleset_versions``. Selected private helpers and constants remain in
``__all__`` intentionally because existing tests and backward-compatible
scaffolding patch loader internals such as ``_load_search_dependencies``,
``_load_lexicon_entries``, and ``_aggregate_rules_version``.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
import json
import logging
from pathlib import Path
from typing import Any

import yaml

from packaging.version import InvalidVersion, Version

from phonology import search as phonology_search
from phonology.core.ports.corpus import CorpusSourceDataError, EMPTY_CORPUS_ADAPTER
from phonology.core.ports.orthography_notes import OrthographicNoteDataError
from phonology.core.ports.profiles import (
    LanguageProfile,
    get_default_language_profile,
    get_language_profile,
)
from phonology.distance import MatrixData, load_matrix_document
from phonology.explainer import get_rules_version

from ._models import DataVersions, LanguageInfo
from ._search_runner import (
    SearchDependencies,
    SearchDependenciesNotReadyError,
    UnsupportedLanguageError,
)

logger = logging.getLogger(__name__)

_LSJ_REPO_DIR_ENV_VAR = "PROTEUS_LSJ_REPO_DIR"
_SEARCH_DEPENDENCY_LOAD_ERRORS = (
    ValueError,
    FileNotFoundError,
    OSError,
    yaml.YAMLError,
    OrthographicNoteDataError,
    CorpusSourceDataError,
)
_SEARCH_DEPENDENCIES_GENERIC_NOT_READY_DETAIL = (
    "Search dependencies are not ready. Verify packaged matrices, rules, and lexicon assets "
    "are available."
)
_SEARCH_DEPENDENCIES_LEXICON_NOT_READY_DETAIL = (
    "Search dependencies are not ready. Run uv run --extra extract python -m "
    "phonology.languages.ancient_greek.build_lexicon --if-missing "
    "to generate the lexicon. If you use a non-default LSJ checkout, set "
    f"{_LSJ_REPO_DIR_ENV_VAR} before extraction."
)


def _resolve_language_id(language: str | None) -> str:
    """Resolve an optional language id at call time."""
    if language is None:
        return get_default_language_profile().language_id
    return language


@lru_cache(maxsize=8)
def _load_lexicon_document(language_id: str) -> dict[str, Any]:
    """Load a packaged lexicon document once per process.

    Args:
        language_id: Normalized language identifier (e.g., "ancient_greek").
    """
    profile = get_language_profile(language_id)
    lexicon_path = profile.lexicon_path
    try:
        raw_text = lexicon_path.read_text(encoding="utf-8")
    except FileNotFoundError as err:
        raise ValueError(f"Lexicon file not found at {lexicon_path}: {err}") from err
    try:
        document = json.loads(raw_text)
    except json.JSONDecodeError as err:
        raise ValueError(f"Failed to parse JSON in {lexicon_path}: {err}") from err
    if not isinstance(document, dict):
        raise ValueError(f"Lexicon file {lexicon_path} must contain a top-level object")
    return document


@lru_cache(maxsize=8)
def _load_lexicon_entries(
    language_id: str,
) -> tuple[dict[str, Any], ...]:
    """Load a packaged language lexicon once per process.

    Args:
        language_id: Normalized language identifier (e.g., "ancient_greek").
    """
    document = _load_lexicon_document(language_id)

    raw_entries = document.get("lemmas")
    if not isinstance(raw_entries, list):
        raise ValueError("Lexicon document must define a list under 'lemmas'")

    entries: list[dict[str, Any]] = []
    for index, entry in enumerate(raw_entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Lexicon entry {index} must be an object")
        entries.append(dict(entry))
    return tuple(entries)


def load_lexicon_entries(
    language: str | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return cached packaged lexicon entries for a language profile."""
    return _load_lexicon_entries(_resolve_language_id(language))


@lru_cache(maxsize=8)
def _load_distance_matrix_with_meta(
    language_id: str,
) -> tuple[MatrixData, dict[str, Any]]:
    """Load the packaged search distance matrix and metadata once per process.

    Args:
        language_id: Normalized language identifier (e.g., "ancient_greek").
    """
    profile = get_language_profile(language_id)
    return load_matrix_document(profile.matrix_path)


def _load_distance_matrix(
    language: str | None = None,
) -> MatrixData:
    """Load the packaged search distance matrix once per process."""
    matrix, _ = _load_distance_matrix_with_meta(_resolve_language_id(language))
    return matrix


def _load_rules_registry(
    language: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Return the ``lru_cache``-backed registry owned by ``phonology_search.get_rules_registry``.

    The returned dict is shared process-wide with the phonology search layer.
    Callers must not mutate it.
    """
    return phonology_search.get_rules_registry(_resolve_language_id(language))


@lru_cache(maxsize=8)
def _load_search_index(
    language_id: str,
) -> phonology_search.KmerIndex:
    """Build and cache the k-mer index for the packaged lexicon.

    Args:
        language_id: Normalized language identifier (e.g., "ancient_greek").
    """
    profile = get_language_profile(language_id)
    return phonology_search.build_kmer_index(
        load_lexicon_entries(language_id),
        phone_inventory=profile.phone_inventory,
        vowel_phones=profile.vowel_phones,
        dialect_skeleton_builders=profile.dialect_skeleton_builders,
    )


@lru_cache(maxsize=8)
def _load_unigram_index(
    language_id: str,
) -> phonology_search.KmerIndex:
    """Build the k=1 unigram index for fallback lookup.

    Args:
        language_id: Normalized language identifier (e.g., "ancient_greek").
    """
    profile = get_language_profile(language_id)
    return phonology_search.build_kmer_index(
        load_lexicon_entries(language_id),
        k=1,
        phone_inventory=profile.phone_inventory,
        vowel_phones=profile.vowel_phones,
        dialect_skeleton_builders=profile.dialect_skeleton_builders,
    )


@lru_cache(maxsize=8)
def _load_lexicon_map(
    language_id: str,
) -> phonology_search.LexiconMap:
    """Build and cache the lexicon map with per-entry token counts.

    Args:
        language_id: Normalized language identifier (e.g., "ancient_greek").
    """
    profile = get_language_profile(language_id)
    return phonology_search.build_lexicon_map(
        load_lexicon_entries(language_id),
        phone_inventory=profile.phone_inventory,
    )


@lru_cache(maxsize=8)
def _load_ipa_index(
    language_id: str,
) -> phonology_search.IpaIndex:
    """Build and cache the IPA-to-entry-id index for the packaged lexicon."""
    return phonology_search.build_ipa_index(_load_lexicon_map(language_id))


@lru_cache(maxsize=8)
def _get_rules_version_cached(rules_dir: Path) -> dict[str, str]:
    """Return rules version metadata for a directory, cached per ``rules_dir``."""
    return get_rules_version(rules_dir)


def _aggregate_rules_version(profile: LanguageProfile) -> str | None:
    """Return the max ruleset version for a language profile, or ``None``."""
    rules_versions = _get_rules_version_cached(profile.rules_dir)
    if not rules_versions:
        return None
    return str(max(Version(version) for version in rules_versions.values()))


def _build_data_versions(
    language: str | None = None,
) -> DataVersions:
    """Build DataVersions from loaded data sources."""
    language_id = _resolve_language_id(language)
    profile = get_language_profile(language_id)
    fields: dict[str, str] = {}

    try:
        document = _load_lexicon_document(language_id)
        schema_version = document.get("schema_version")
        if isinstance(schema_version, str) and schema_version.strip():
            fields["lexicon"] = schema_version
        meta = document.get("_meta", {})
        if isinstance(meta, dict):
            last_updated = meta.get("last_updated")
            if isinstance(last_updated, str):
                fields["lexicon_updated_at"] = last_updated
    except (OSError, ValueError) as err:
        logger.exception("Failed to load lexicon data version metadata: %s", err)

    try:
        _, matrix_meta = _load_distance_matrix_with_meta(language_id)
        matrix_version = matrix_meta.get("version")
        if isinstance(matrix_version, str) and matrix_version.strip():
            fields["matrix"] = matrix_version
        generated_at = matrix_meta.get("generated_at")
        if isinstance(generated_at, str):
            fields["matrix_generated_at"] = generated_at
    except (OSError, ValueError, json.JSONDecodeError) as err:
        logger.exception("Failed to load matrix data version metadata: %s", err)

    try:
        rules_version = _aggregate_rules_version(profile)
        if rules_version is not None:
            fields["rules"] = rules_version
    except (OSError, ValueError, yaml.YAMLError, InvalidVersion) as err:
        logger.exception("Failed to load rules data version metadata: %s", err)

    return DataVersions(**fields)


def _build_ruleset_versions(language: str) -> dict[str, str]:
    """Return aggregated ruleset versions keyed by language profile id."""
    try:
        profile = get_language_profile(language)
        rules_version = _aggregate_rules_version(profile)
    except (OSError, ValueError, yaml.YAMLError, InvalidVersion):
        logger.exception("Failed to load ruleset versions for language %s", language)
        return {}
    if rules_version is None:
        return {}
    return {profile.language_id: rules_version}


def _load_version_with_fallback(
    load_fn: Callable[[], str | None], context: str, language_id: str
) -> str:
    """Load a version string, degrading missing or invalid assets to unknown."""
    try:
        version = load_fn()
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        yaml.YAMLError,
        InvalidVersion,
    ) as err:
        logger.warning(
            "Failed to load %s for language %s: %s",
            context,
            language_id,
            err,
            exc_info=True,
        )
        return "unknown"
    if version is None:
        return "unknown"
    normalized = version.strip()
    return normalized or "unknown"


def _build_language_info(profile: LanguageProfile) -> LanguageInfo:
    """Build public language metadata, degrading unavailable assets to unknown."""

    def load_ruleset_version() -> str | None:
        return _aggregate_rules_version(profile)

    def load_lexicon_schema_version() -> str | None:
        document = _load_lexicon_document(profile.language_id)
        schema_version = document.get("schema_version")
        if isinstance(schema_version, str) and schema_version.strip():
            return schema_version
        return None

    def load_matrix_version() -> str | None:
        _, matrix_meta = _load_distance_matrix_with_meta(profile.language_id)
        raw_matrix_version = matrix_meta.get("version")
        if isinstance(raw_matrix_version, str) and raw_matrix_version.strip():
            return raw_matrix_version
        return None

    ruleset_version = _load_version_with_fallback(
        load_ruleset_version,
        "ruleset version",
        profile.language_id,
    )
    lexicon_schema_version = _load_version_with_fallback(
        load_lexicon_schema_version,
        "lexicon schema version",
        profile.language_id,
    )
    matrix_version = _load_version_with_fallback(
        load_matrix_version,
        "matrix version",
        profile.language_id,
    )

    return LanguageInfo(
        language_id=profile.language_id,
        display_name=profile.display_name,
        default_dialect=profile.default_dialect,
        supported_dialects=list(profile.supported_dialects),
        status=profile.status,
        ruleset_version=ruleset_version,
        lexicon_schema_version=lexicon_schema_version,
        matrix_version=matrix_version,
        description=profile.description,
    )


def _load_search_dependencies(
    language: str | None = None,
) -> SearchDependencies:
    """Load all cached search dependencies needed by /ready and /search."""
    try:
        language_id = _resolve_language_id(language)
        profile = get_language_profile(language_id)
    except ValueError as err:
        raise UnsupportedLanguageError(str(err)) from err
    try:
        lexicon = load_lexicon_entries(language_id)
    except _SEARCH_DEPENDENCY_LOAD_ERRORS as err:
        raise SearchDependenciesNotReadyError(
            _SEARCH_DEPENDENCIES_LEXICON_NOT_READY_DETAIL
        ) from err

    try:
        data_versions = _build_data_versions(language_id)
        if profile.orthographic_data_preparer is not None:
            profile.orthographic_data_preparer()
        corpus_adapter = (
            profile.corpus_adapter_factory()
            if profile.corpus_adapter_factory is not None
            else EMPTY_CORPUS_ADAPTER
        )
        return SearchDependencies(
            profile=profile,
            lexicon=lexicon,
            matrix=_load_distance_matrix(language_id),
            rules_registry=_load_rules_registry(language_id),
            search_index=_load_search_index(language_id),
            unigram_index=_load_unigram_index(language_id),
            lexicon_map=_load_lexicon_map(language_id),
            ipa_index=_load_ipa_index(language_id),
            data_versions=data_versions,
            corpus_adapter=corpus_adapter,
        )
    except _SEARCH_DEPENDENCY_LOAD_ERRORS as err:
        raise SearchDependenciesNotReadyError(
            _SEARCH_DEPENDENCIES_GENERIC_NOT_READY_DETAIL
        ) from err


load_search_dependencies = _load_search_dependencies
build_ruleset_versions = _build_ruleset_versions


__all__ = [
    "_LSJ_REPO_DIR_ENV_VAR",
    "_SEARCH_DEPENDENCIES_GENERIC_NOT_READY_DETAIL",
    "_SEARCH_DEPENDENCIES_LEXICON_NOT_READY_DETAIL",
    "_SEARCH_DEPENDENCY_LOAD_ERRORS",
    "_aggregate_rules_version",
    "_build_data_versions",
    "_build_language_info",
    "_build_ruleset_versions",
    "_get_rules_version_cached",
    "_load_distance_matrix",
    "_load_distance_matrix_with_meta",
    "_load_ipa_index",
    "_load_lexicon_document",
    "_load_lexicon_entries",
    "_load_lexicon_map",
    "_load_rules_registry",
    "_load_search_dependencies",
    "_load_search_index",
    "_load_unigram_index",
    "_load_version_with_fallback",
    "_resolve_language_id",
    "build_ruleset_versions",
    "load_lexicon_entries",
    "load_search_dependencies",
]
