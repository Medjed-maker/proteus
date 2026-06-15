"""FastAPI application entry point.

Exposes the Proteus phonological search engine as a REST API.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
import logging
import os
import threading
from typing import AsyncIterator, Protocol

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

# Re-exported so test seams and non-REST callers can patch shared search
# internals (e.g. ``search_execution``) via ``api.main.phonology_search``.
from phonology import search as phonology_search  # noqa: F401
from phonology.core.ports.profiles import (
    get_default_language_profile,
    list_language_profiles,
    register_default_profiles,
)
from . import _assets, _dependencies, _runtime_metadata
from ._app_version import (
    _APP_VERSION_ENV_VAR as _APP_VERSION_ENV_VAR,
    _load_app_version as _load_app_version,
)
from ._constants import API_VERSION as API_VERSION, SCHEMA_VERSION as SCHEMA_VERSION
from ._dependencies import (
    _LSJ_REPO_DIR_ENV_VAR as _LSJ_REPO_DIR_ENV_VAR,
    _SEARCH_DEPENDENCIES_GENERIC_NOT_READY_DETAIL as _SEARCH_DEPENDENCIES_GENERIC_NOT_READY_DETAIL,
    _SEARCH_DEPENDENCIES_LEXICON_NOT_READY_DETAIL as _SEARCH_DEPENDENCIES_LEXICON_NOT_READY_DETAIL,
    _SEARCH_DEPENDENCY_LOAD_ERRORS as _SEARCH_DEPENDENCY_LOAD_ERRORS,
)
from ._buck_routes import router as buck_router
from ._models import (
    DataVersions as DataVersions,
    LanguagesResponse,
    SearchHit as SearchHit,
    SearchRequest as SearchRequest,
    SearchResponse,
    VersionInfo,
)
from ._request_context import (
    PUBLIC_BASE_URL_ENV_VAR,
    generate_request_id as generate_request_id,
    is_valid_request_id,
    resolve_public_base_url,
    validate_public_base_url,
)
from ._hit_formatting import _build_search_hit as _build_search_hit
from ._search_runner import (
    InvalidDialectError,
    InvalidSearchQueryError,
    SearchDependencies,
    SearchDependenciesNotReadyError,
    SearchExecutionError,
    UnsupportedLanguageError,
    _LOG_RAW_QUERY_ENV_VAR as _LOG_RAW_QUERY_ENV_VAR,
    _summarize_query_for_logs,  # re-export for tests
    run_search as run_search,
)

logger = logging.getLogger(__name__)
register_default_profiles()
_ALLOWED_ORIGINS_ENV_VAR = "PROTEUS_ALLOWED_ORIGINS"
_DISABLE_STARTUP_WARMUP_ENV_VAR = "PROTEUS_DISABLE_STARTUP_WARMUP"
_DISABLE_STARTUP_WARMUP_ATTR = "disable_startup_warmup"
_ENABLE_API_DOCS_ENV_VAR = "PROTEUS_ENABLE_API_DOCS"
_BUILD_TIMESTAMP_ENV_VAR = _runtime_metadata._BUILD_TIMESTAMP_ENV_VAR
_GIT_SHA_ENV_VAR = _runtime_metadata._GIT_SHA_ENV_VAR
_GIT_SHA_MAX_LENGTH = _runtime_metadata._GIT_SHA_MAX_LENGTH
_LEGACY_LANGUAGE_ALIAS_DEPRECATION_MESSAGE = (
    "Use response_language='en'|'ja' for response prose; language now selects "
    "the phonology profile."
)


_APP_VERSION = _runtime_metadata._APP_VERSION


class _SearchDependencyLoader(Protocol):
    """Type protocol for search dependency loaders.
    
    Defines the signature for loading search dependencies, optionally
    scoped to a specific language profile.
    """
    def __call__(self, language: str | None = None) -> SearchDependencies:
        ...


_load_rule_schema_version = _runtime_metadata._load_rule_schema_version
_build_version_info = _runtime_metadata._build_version_info


def _validate_public_base_url_env() -> None:
    """Validate ``PROTEUS_PUBLIC_BASE_URL`` at startup, fail-fast on misconfig.

    Empty or unset env var is valid (the runtime falls back to
    ``request.base_url``). When set, the value must satisfy the same rules as
    :func:`build_verification_url`; otherwise every ``/search`` would 500
    silently. Surfacing the misconfiguration at startup makes the operational
    failure obvious instead of latent.
    """
    raw = os.environ.get(PUBLIC_BASE_URL_ENV_VAR, "").strip()
    if not raw:
        return
    try:
        validate_public_base_url(raw)
    except ValueError as err:
        raise RuntimeError(
            f"{PUBLIC_BASE_URL_ENV_VAR}={raw!r} is invalid: {err}. "
            "Set it to an absolute URL with scheme and host, no query or "
            "fragment, or unset it to fall back to request.base_url."
        ) from err


@asynccontextmanager
async def _app_lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Run best-effort startup warmup while keeping app startup non-blocking."""
    _validate_public_base_url_env()
    startup_warmup_disabled = getattr(app.state, _DISABLE_STARTUP_WARMUP_ATTR, False)
    if not startup_warmup_disabled and not _env_flag_enabled(
        _DISABLE_STARTUP_WARMUP_ENV_VAR,
    ):
        warmup_thread = threading.Thread(
            target=_warm_search_dependencies,
            name="proteus-search-warmup",
            daemon=True,
        )
        app.state.search_warmup_thread = warmup_thread
        warmup_thread.start()
    yield


def _env_flag_enabled(name: str) -> bool:
    """Return True when an environment flag is set to an affirmative value."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _get_allowed_origins() -> list[str]:
    """Return the configured CORS allowlist from environment."""
    raw_origins = os.environ.get(_ALLOWED_ORIGINS_ENV_VAR, "")
    if not raw_origins:
        logger.warning(
            "_get_allowed_origins found %s unset; returning an empty CORS allowlist "
            "that may block cross-origin requests.",
            _ALLOWED_ORIGINS_ENV_VAR,
        )
    return [
        origin for origin in (item.strip() for item in raw_origins.split(",")) if origin
    ]


# _env_flag_enabled is evaluated at import time here, not per-request.
# Export PROTEUS_ENABLE_API_DOCS=1 before starting uvicorn to enable /docs.
app = FastAPI(
    title="Proteus",
    description=(
        "Language-independent historical phonology framework with an "
        "Ancient Greek pilot"
    ),
    version=_APP_VERSION,
    docs_url="/docs" if _env_flag_enabled(_ENABLE_API_DOCS_ENV_VAR) else None,
    openapi_url="/openapi.json"
    if _env_flag_enabled(_ENABLE_API_DOCS_ENV_VAR)
    else None,
    redoc_url=None,
    lifespan=_app_lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_allowed_origins(),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Request-ID"],
    allow_credentials=False,
)
app.include_router(buck_router)


@app.middleware("http")
async def _add_security_headers(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Attach conservative security headers to every response.

    Strict-Transport-Security and CSP are deliberately deferred to the fronting
    proxy, which terminates TLS and knows the deployment's asset origins.
    """
    response: Response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault(
        "Permissions-Policy", "camera=(), microphone=(), geolocation=()"
    )
    return response


@app.middleware("http")
async def _add_request_id(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Attach a validated request correlation id to request state and response.

    ``request.state.request_id`` is populated *before* ``call_next`` so the
    ``Exception`` handler below can read it on truly unhandled errors that
    bypass FastAPI's normal exception path.
    """
    raw_request_id = request.headers.get("X-Request-ID", "").strip()
    # Lowercase accepted client IDs so logs and downstream traces always see
    # a single canonical form regardless of the casing the caller chose.
    request_id = (
        raw_request_id.lower()
        if is_valid_request_id(raw_request_id)
        else generate_request_id()
    )
    request.state.request_id = request_id
    response: Response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(Exception)
async def _unhandled_exception_x_request_id(
    request: Request, exc: Exception
) -> Response:
    """Ensure unhandled exceptions still emit ``X-Request-ID``.

    FastAPI's built-in handlers translate ``HTTPException`` (and request
    validation failures) into ``Response`` objects that flow back through the
    middleware stack — those paths already get ``X-Request-ID``. Truly
    unhandled exceptions would otherwise be caught by Starlette's
    ``ServerErrorMiddleware`` *outside* this middleware, returning a 500
    without the correlation id. This handler closes that gap.
    """
    logger.exception("Unhandled exception during request: %s", exc)
    request_id = getattr(request.state, "request_id", "") or generate_request_id()
    response = JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
    )
    response.headers["X-Request-ID"] = request_id
    return response


_FRONTEND_PATH = _assets._FRONTEND_PATH
_CHANGELOG_PATH = _assets._CHANGELOG_PATH
_STATIC_DIR = _assets._STATIC_DIR


def _load_frontend_html() -> str | None:
    """Load and cache the packaged frontend HTML document."""
    return _assets._load_html_asset(_FRONTEND_PATH, label="frontend")


def _build_deprecation_link_header(base_url: str, docs_path: str) -> str:
    """Return an absolute URI Link header value per RFC 8288."""
    return _assets._build_deprecation_link_header(base_url, docs_path)


def _load_changelog_html() -> str | None:
    """Load and cache the packaged changelog HTML document."""
    return _assets._load_html_asset(_CHANGELOG_PATH, label="changelog")


_resolve_language_id = _dependencies._resolve_language_id
_load_lexicon_document = _dependencies._load_lexicon_document
_load_lexicon_entries = _dependencies._load_lexicon_entries
load_lexicon_entries = _dependencies.load_lexicon_entries
_load_distance_matrix_with_meta = _dependencies._load_distance_matrix_with_meta
_load_distance_matrix = _dependencies._load_distance_matrix
_load_rules_registry = _dependencies._load_rules_registry
_load_search_index = _dependencies._load_search_index
_load_unigram_index = _dependencies._load_unigram_index
_load_lexicon_map = _dependencies._load_lexicon_map

# The canonical search-dependency loaders live in ``api._dependencies`` so that
# the MCP surface can import them without pulling in this FastAPI app module.
# ``api.main`` re-exports them as aliases purely for backward-compatible names;
# there is a single implementation (and a single ``lru_cache``) per loader.
# Tests that need to stub sub-loader behavior must patch ``api._dependencies``;
# whole-function swaps consumed by this module's endpoints/warmup may still
# patch ``api.main`` since those names are read here at call time.
_load_ipa_index = _dependencies._load_ipa_index
_get_rules_version_cached = _dependencies._get_rules_version_cached
_aggregate_rules_version = _dependencies._aggregate_rules_version
_build_data_versions = _dependencies._build_data_versions
_build_ruleset_versions = _dependencies._build_ruleset_versions
_load_version_with_fallback = _dependencies._load_version_with_fallback
_build_language_info = _dependencies._build_language_info
_load_search_dependencies = _dependencies._load_search_dependencies


def _search_dependencies_not_ready_exception(detail: str) -> HTTPException:
    """Return a consistent HTTP error for dependency-loading failures."""
    return HTTPException(
        status_code=503,
        detail=detail,
    )


def _log_search_dependencies_not_ready(
    detail: str,
    *,
    log_level: int,
    message: str,
    query_log_label: str | None = None,
) -> None:
    """Log expected dependency-not-ready states without a traceback."""
    if query_log_label is None:
        logger.log(log_level, "%s: %s", message, detail)
        return

    logger.log(log_level, "%s for query (%s): %s", message, query_log_label, detail)


def _warm_search_dependencies() -> None:
    """Eagerly populate cached search dependencies for faster first queries."""
    try:
        _load_search_dependencies()
    except SearchDependenciesNotReadyError as err:
        _log_search_dependencies_not_ready(
            err.detail,
            log_level=logging.INFO,
            message="Background search warmup skipped; dependencies not ready",
        )
    except UnsupportedLanguageError as err:
        logger.info(
            "Background search warmup skipped; default language unavailable: %s",
            err,
        )


# HTML is read once at import time and cached for the process lifetime;
# dev-mode ``--reload`` (see main()) handles staleness during development.
_FRONTEND_HTML = _load_frontend_html()
if _FRONTEND_HTML is None:
    logger.warning("Frontend HTML asset not found at %s", _FRONTEND_PATH)
_CHANGELOG_HTML = _load_changelog_html()
if _CHANGELOG_HTML is None:
    logger.warning("Changelog HTML asset not found at %s", _CHANGELOG_PATH)


def render_frontend() -> HTMLResponse:
    """Return the packaged frontend HTML document."""
    return _assets.render_html_asset(
        _FRONTEND_HTML,
        app_version=_APP_VERSION,
        missing_detail="Frontend asset not found",
    )


def render_changelog() -> HTMLResponse:
    """Return the packaged changelog HTML document."""
    return _assets.render_html_asset(
        _CHANGELOG_HTML,
        app_version=_APP_VERSION,
        missing_detail="Changelog asset not found",
    )


@app.head("/")
async def root_head() -> Response:
    """Return a lightweight response for uptime probes."""
    if _FRONTEND_HTML is None:
        return Response(status_code=404)
    return Response(status_code=200)


@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    """Serve the frontend."""
    return render_frontend()


@app.get("/changelog", response_class=HTMLResponse)
async def changelog() -> HTMLResponse:
    """Serve the changelog page."""
    return render_changelog()


@app.head("/changelog")
async def changelog_head() -> Response:
    """Return a lightweight response for changelog preflight checks."""
    if _CHANGELOG_HTML is None:
        return Response(status_code=404)
    return Response(status_code=200)


@app.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    response: Response,
    fastapi_request: Request,
) -> SearchResponse:
    """Run phonological search for a query word."""
    query_log_label = _summarize_query_for_logs(request.query_form)

    try:
        deps = _load_search_dependencies(request.language)
    except UnsupportedLanguageError as err:
        logger.info("Rejected unsupported language %r: %s", request.language, err)
        raise HTTPException(status_code=400, detail="Unsupported language") from err
    except SearchDependenciesNotReadyError as err:
        _log_search_dependencies_not_ready(
            err.detail,
            log_level=logging.WARNING,
            message="Search dependencies are not ready",
            query_log_label=query_log_label,
        )
        raise _search_dependencies_not_ready_exception(err.detail) from err

    profile = deps.profile or get_default_language_profile()
    try:
        outcome = run_search(
            request,
            deps=deps,
            request_id=fastapi_request.state.request_id,
            base_url=resolve_public_base_url(fastapi_request),
            engine_version=_APP_VERSION,
            ruleset_versions=_build_ruleset_versions(profile.language_id),
        )
    except InvalidDialectError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    except InvalidSearchQueryError as err:
        raise HTTPException(status_code=400, detail="Invalid search query") from err
    except SearchExecutionError as err:
        raise HTTPException(
            status_code=500,
            detail="Search failed due to an internal error.",
        ) from err

    search_response = outcome.response
    if request.legacy_language_alias_used:
        response.headers["Deprecation"] = "true"
        if app.docs_url:
            response.headers["Link"] = _build_deprecation_link_header(
                str(fastapi_request.base_url), app.docs_url
            )
        response.headers["X-Proteus-Migration"] = (
            _LEGACY_LANGUAGE_ALIAS_DEPRECATION_MESSAGE
        )
    return search_response


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.head("/health")
async def health_head() -> Response:
    """Liveness probe for HEAD-based health checks."""
    return Response(status_code=204)


@app.get("/version", response_model=VersionInfo)
async def version() -> VersionInfo:
    """Return API and runtime version metadata."""
    return _build_version_info()


@app.head("/version")
async def version_head() -> Response:
    """Return a lightweight response for version probes."""
    return Response(status_code=204)


@app.get("/languages", response_model=LanguagesResponse)
async def languages() -> LanguagesResponse:
    """Return registered language profiles and version metadata."""
    profiles = list_language_profiles()
    if not profiles:
        raise HTTPException(status_code=503, detail="No language profiles registered")

    return LanguagesResponse(
        languages=[_build_language_info(profile) for profile in profiles],
        meta=_build_version_info(),
    )


@app.head("/languages")
async def languages_head() -> Response:
    """Return a lightweight response for language-profile probes."""
    if not list_language_profiles():
        return Response(status_code=503)
    return Response(status_code=204)


@app.get("/ready")
async def ready() -> dict[str, str]:
    """Readiness probe."""
    try:
        _load_search_dependencies()
    except SearchDependenciesNotReadyError as err:
        _log_search_dependencies_not_ready(
            err.detail,
            log_level=logging.WARNING,
            message="Readiness probe skipped; dependencies not ready",
        )
        raise _search_dependencies_not_ready_exception(err.detail) from err
    except UnsupportedLanguageError as err:
        logger.warning(
            "Readiness probe skipped; default language unavailable: %s",
            err,
        )
        raise _search_dependencies_not_ready_exception(
            _SEARCH_DEPENDENCIES_GENERIC_NOT_READY_DETAIL
        ) from err
    return {"status": "ok"}


# Mount static files after all route definitions. A StaticFiles mount at "/"
# would shadow every route registered after it; placing all mounts last is a
# safe convention even for prefix-scoped paths like "/static".
if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
else:
    logger.warning("Static assets directory not found at %s", _STATIC_DIR)


# Public re-exports for MCP server and other external callers
# Public engine version shared by REST, MCP, and external callers.
APP_VERSION: str = _APP_VERSION
# Public dependency loader for non-REST search surfaces.
load_search_dependencies: _SearchDependencyLoader = _load_search_dependencies
# Public ruleset-version builder for non-REST response metadata.
build_ruleset_versions: Callable[[str], dict[str, str]] = _build_ruleset_versions


def main() -> None:
    """Run the local development server on a safe loopback default."""
    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
