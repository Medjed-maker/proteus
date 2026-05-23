"""FastAPI-agnostic search execution shared by REST and MCP adapters.

Both ``api.main.search`` (HTTP) and ``mcp_server._search_adapter`` (stdio) need
the same core flow: validate dialect, run the phonology engine, build hits,
build response meta. Keeping that flow here lets each adapter focus only on
its surface-specific error mapping (HTTPException for REST, ``ValueError`` /
``RuntimeError`` for MCP), and prevents the MCP adapter from reaching into
``api.main`` private symbols.

This module deliberately does *not* import ``api.main`` (avoids circular
imports). The dependency loader and the ``_build_ruleset_versions`` helper
still live in ``api.main`` because they depend on private file-loading
helpers; callers pass already-loaded ``deps`` and ``ruleset_versions`` into
:func:`run_search`.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
import os
from typing import Any, NamedTuple

from phonology import search as phonology_search
from phonology.corpus import CorpusAdapter, EMPTY_CORPUS_ADAPTER, safe_lookup
from phonology.distance import MatrixData
from phonology.profiles import LanguageProfile, get_default_language_profile
from phonology.search._types import QueryMode

from ._hit_formatting import _build_search_hit
from ._models import DataVersions, SearchHit, SearchRequest, SearchResponse
from ._response_meta import build_response_meta

# The runner is part of the api package's search surface; routing logs to
# the ``api.main`` logger keeps server-side log routing and existing tests
# stable regardless of where the helpers physically live.
logger = logging.getLogger("api.main")

_LOG_RAW_QUERY_ENV_VAR = "PROTEUS_LOG_RAW_SEARCH_QUERY"

# Upper bound for both the similarity fallback (stage 2 widening when the
# seed set is too small) and the unigram fallback (k=1 recovery for short
# queries). Kept in one place so the two caps stay coupled; 2000 mirrors
# search._DEFAULT_FALLBACK_CANDIDATE_LIMIT, which was chosen to keep a hard
# ceiling on the candidate set even for permissive partial queries.
_API_FALLBACK_CANDIDATE_LIMIT = 2000


class SearchDependencies(NamedTuple):
    """Search dependencies loaded at startup."""

    lexicon: tuple[dict[str, Any], ...]
    matrix: MatrixData
    rules_registry: dict[str, dict[str, Any]]
    search_index: phonology_search.KmerIndex
    unigram_index: phonology_search.KmerIndex
    lexicon_map: phonology_search.LexiconMap
    ipa_index: phonology_search.IpaIndex
    data_versions: DataVersions
    profile: LanguageProfile | None = None
    corpus_adapter: CorpusAdapter = EMPTY_CORPUS_ADAPTER


class SearchDependenciesNotReadyError(RuntimeError):
    """Represent a dependency-loading failure with an API-safe detail string."""

    detail: str

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class UnsupportedLanguageError(ValueError):
    """Raised when the requested language profile is not registered."""


class InvalidDialectError(ValueError):
    """Raised when ``dialect_hint`` is not in the profile's supported dialects.

    The message intentionally mirrors the legacy format
    ``Invalid dialect_hint '<value>'; expected one of (...)`` so the REST
    surface can forward the detail verbatim in 400 responses.
    """


class InvalidSearchQueryError(ValueError):
    """Raised when the phonology engine rejects the query input."""


class SearchExecutionError(RuntimeError):
    """Raised when an OSError or other internal failure aborts execution."""


@dataclass(frozen=True)
class SearchExecutionOutcome:
    """Result of a successful search execution.

    ``response`` is the fully-populated :class:`SearchResponse` to return.
    ``query_ipa`` and ``query_mode`` are also surfaced individually so MCP
    callers can include them in structured payloads without parsing the
    response model.
    """

    response: SearchResponse
    query_ipa: str
    query_mode: QueryMode
    truncated: bool


def _env_flag_enabled(name: str) -> bool:
    """Return True when an environment flag is set to an affirmative value."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _summarize_query_for_logs(query: str) -> str:
    """Return a redacted search-query identifier safe for routine logs."""
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]
    return f"len={len(query)} sha256={digest}"


def _should_log_raw_search_query() -> bool:
    """Return True only when raw search queries may appear in debug logs."""
    return logger.isEnabledFor(logging.DEBUG) and _env_flag_enabled(
        _LOG_RAW_QUERY_ENV_VAR
    )


def run_search(
    request: SearchRequest,
    *,
    deps: SearchDependencies,
    request_id: str,
    base_url: str | None,
    engine_version: str,
    ruleset_versions: dict[str, str],
) -> SearchExecutionOutcome:
    """Execute a phonology search with FastAPI-agnostic error semantics.

    Caller is responsible for loading ``deps`` (typically via
    ``api.main._load_search_dependencies``) and for producing
    ``ruleset_versions`` (via ``api.main._build_ruleset_versions``).
    Surface-specific exceptions are NOT raised here; callers translate the
    pure exception types declared in this module into their preferred
    representation.
    """
    profile = deps.profile or get_default_language_profile()
    query_log_label = _summarize_query_for_logs(request.query_form)

    if request.dialect_hint not in profile.supported_dialects:
        allowed_dialects = ", ".join(
            repr(dialect) for dialect in profile.supported_dialects
        )
        raise InvalidDialectError(
            f"Invalid dialect_hint {request.dialect_hint!r}; expected one of "
            f"({allowed_dialects})"
        )

    try:
        execution = phonology_search.search_execution(
            request.query_form,
            lexicon=deps.lexicon,
            matrix=deps.matrix,
            max_results=request.max_candidates,
            dialect=request.dialect_hint,
            language=profile.language_id,
            converter=profile.converter,
            index=deps.search_index,
            unigram_index=deps.unigram_index,
            prebuilt_lexicon_map=deps.lexicon_map,
            prebuilt_ipa_index=deps.ipa_index,
            phone_inventory=profile.phone_inventory,
            dialect_skeleton_builders=profile.dialect_skeleton_builders,
            similarity_fallback_limit=_API_FALLBACK_CANDIDATE_LIMIT,
            unigram_fallback_limit=_API_FALLBACK_CANDIDATE_LIMIT,
        )
    except ValueError as err:
        logger.info("Rejected search query (%s): %s", query_log_label, err)
        debug_query = (
            request.query_form if _should_log_raw_search_query() else query_log_label
        )
        logger.debug(
            "Full ValueError details for query %r",
            debug_query,
            exc_info=True,
        )
        raise InvalidSearchQueryError(str(err)) from err
    except OSError as err:
        logger.exception("Search execution failed for query (%s)", query_log_label)
        raise SearchExecutionError(
            "Search failed due to an internal error."
        ) from err

    hits: list[SearchHit] = []
    for result in execution.results:
        source_references = safe_lookup(
            deps.corpus_adapter,
            entry_id=result.entry_id,
            headword=result.lemma,
            language=profile.language_id,
            logger=logger,
        )
        hit = _build_search_hit(
            result,
            query_ipa=execution.query_ipa,
            rules_registry=deps.rules_registry,
            query_mode=execution.query_mode,
            lang=request.response_language,
            query_form=request.query_form,
            orthographic_note_builder=profile.orthographic_note_builder,
            source_references=source_references,
        )
        hits.append(hit)
    meta = build_response_meta(
        request_id=request_id,
        request=request,
        base_url=base_url,
        engine_version=engine_version,
        data_versions=deps.data_versions,
        ruleset_versions=ruleset_versions,
    )

    response = SearchResponse(
        query=request.query_form,
        query_ipa=execution.query_ipa,
        query_mode=execution.query_mode,
        hits=hits,
        truncated=execution.truncated,
        data_versions=deps.data_versions,
        meta=meta,
    )
    return SearchExecutionOutcome(
        response=response,
        query_ipa=execution.query_ipa,
        query_mode=execution.query_mode,
        truncated=execution.truncated,
    )


__all__ = [
    "InvalidDialectError",
    "InvalidSearchQueryError",
    "SearchDependencies",
    "SearchDependenciesNotReadyError",
    "SearchExecutionError",
    "SearchExecutionOutcome",
    "UnsupportedLanguageError",
    "_API_FALLBACK_CANDIDATE_LIMIT",
    "_LOG_RAW_QUERY_ENV_VAR",
    "_should_log_raw_search_query",
    "_summarize_query_for_logs",
    "run_search",
]
