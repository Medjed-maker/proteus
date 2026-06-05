"""Adapter from MCP tool input to the shared Proteus search implementation."""

from __future__ import annotations

import logging
import os
from typing import Any

from pydantic import ValidationError

from api import main as api_main
from api._models import SearchRequest
from api._request_context import (
    PUBLIC_BASE_URL_ENV_VAR,
    build_request_echo,
    generate_request_id,
)
from api._search_runner import (
    InvalidDialectError,
    InvalidSearchQueryError,
    SearchDependencies,
    SearchDependenciesNotReadyError,
    SearchExecutionError,
    SearchExecutionOutcome,
    UnsupportedLanguageError,
    _summarize_query_for_logs,
    run_search,
)
from phonology.core.ports.profiles import get_default_language_profile

from .tools.search import McpSearchInput, McpSearchOutput

logger = logging.getLogger("proteus.mcp")


def _verification_base_url_for_mcp() -> str | None:
    """Return the MCP verification URL base from env, or None when unset."""
    base_url = os.environ.get(PUBLIC_BASE_URL_ENV_VAR, "").strip()
    if base_url:
        return base_url
    logger.warning(
        "%s is unset; MCP search responses will include an empty verification_url",
        PUBLIC_BASE_URL_ENV_VAR,
    )
    return None


def _build_search_request(input_model: McpSearchInput) -> SearchRequest:
    """Convert MCP input into the REST SearchRequest contract."""
    try:
        request_kwargs: dict[str, Any] = {
            "query_form": input_model.query_form,
            "language": input_model.source_language,
            "max_candidates": input_model.max_candidates,
            "response_language": input_model.response_language,
        }
        if input_model.dialect_hint is not None:
            request_kwargs["dialect_hint"] = input_model.dialect_hint
        return SearchRequest(**request_kwargs)
    except ValidationError as exc:
        # Preserve the original validator details so MCP clients see the
        # offending field instead of a generic message.
        raise ValueError(f"Invalid MCP search input: {exc}") from exc


def load_and_validate_search_deps(request_language: str) -> SearchDependencies:
    """Load and validate search dependencies for an MCP request.

    Encapsulates the api_main.load_search_dependencies call and handles
    UnsupportedLanguageError and SearchDependenciesNotReadyError.
    ``api_main.load_search_dependencies`` is the single source of truth that
    populates ``corpus_adapter`` (via ``profile.corpus_adapter_factory``) for
    both REST and MCP paths, so source-reference enrichment behaves
    identically across both surfaces.

    Args:
        request_language: Language code from the search request.

    Returns:
        SearchDependencies object with all required search infrastructure.

    Raises:
        ValueError: If the language is unsupported.
        SearchDependenciesNotReadyError: If search dependencies are not ready.
            ``SearchDependenciesNotReadyError`` is a ``RuntimeError`` subclass,
            so callers that catch ``RuntimeError`` still work, but the
            ``.detail`` attribute is preserved for richer error mapping.
    """
    try:
        return api_main.load_search_dependencies(request_language)
    except UnsupportedLanguageError as err:
        logger.info("Rejected unsupported MCP language %r: %s", request_language, err)
        raise ValueError("Unsupported language") from err
    except SearchDependenciesNotReadyError as err:
        logger.warning(
            "MCP search dependencies are not ready for language %s: %s",
            request_language,
            err.detail,
        )
        # Re-raise so the subclass and the structured ``.detail`` attribute
        # survive for any future MCP-side error mapper.
        raise


def execute_search(
    request: SearchRequest,
    deps: SearchDependencies,
    request_id: str,
    query_log_label: str,
) -> SearchExecutionOutcome:
    """Execute the search and handle domain-specific errors.

    Calls run_search with engine_version and ruleset_versions, and maps
    InvalidDialectError, InvalidSearchQueryError, and SearchExecutionError
    into ValueError/RuntimeError responses.

    Args:
        request: The search request.
        deps: Search dependencies.
        request_id: Request correlation ID.
        query_log_label: Sanitized query string for logging.

    Returns:
        Search execution outcome.

    Raises:
        ValueError: For invalid dialect or search query.
        RuntimeError: For search execution errors.
    """
    try:
        return run_search(
            request,
            deps=deps,
            request_id=request_id,
            base_url=_verification_base_url_for_mcp(),
            engine_version=api_main.APP_VERSION,
            ruleset_versions=api_main.build_ruleset_versions(
                (deps.profile or get_default_language_profile()).language_id
            ),
        )
    except InvalidDialectError as err:
        # Defensive: McpSearchInput/SearchRequest validators already reject
        # unsupported dialects, so this branch is rarely reached today. It
        # guards against future profiles whose runner-side dialect check may
        # outpace the model-level validator.
        raise ValueError(str(err)) from err
    except InvalidSearchQueryError as err:
        logger.info("Rejected MCP search query (%s): %s", query_log_label, err)
        raise ValueError("Invalid search query") from err
    except SearchExecutionError as err:
        raise RuntimeError(str(err)) from err


def build_mcp_response(outcome: SearchExecutionOutcome, request: SearchRequest) -> McpSearchOutput:
    """Build the final MCP search response from execution outcome.

    Handles response.meta.request_echo injection and constructs the final
    McpSearchOutput.

    Args:
        outcome: Search execution outcome.
        request: Original search request.

    Returns:
        McpSearchOutput with candidates, metadata, and request echo.
    """
    response = outcome.response
    if response.meta.request_echo is None:
        response = response.model_copy(
            update={
                "meta": response.meta.model_copy(
                    update={"request_echo": build_request_echo(request)}
                )
            }
        )

    return McpSearchOutput(
        candidates=response.hits,
        query=request.query_form,
        query_ipa=outcome.query_ipa,
        query_mode=outcome.query_mode,
        truncated=outcome.truncated,
        meta=response.meta,
    )


def _run_search_for_mcp(input_model: McpSearchInput) -> McpSearchOutput:
    """Run the shared search engine for an MCP request."""
    request_id = generate_request_id()
    request = _build_search_request(input_model)
    query_log_label = _summarize_query_for_logs(request.query_form)
    logger.info("MCP search request_id=%s query=(%s)", request_id, query_log_label)

    deps = load_and_validate_search_deps(request.language)
    outcome = execute_search(request, deps, request_id, query_log_label)
    return build_mcp_response(outcome, request)


__all__ = ["_run_search_for_mcp"]
