"""FastAPI application entry point.

Exposes the Proteus phonological search engine as a REST API.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
import html
import importlib.resources as resources
from functools import lru_cache
import json
import logging
import os
from pathlib import Path
import sys
import threading
from typing import Any, AsyncIterator

import yaml

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from packaging.version import InvalidVersion, Version

from phonology import search as phonology_search
from phonology.distance import MatrixData, load_matrix_document
from phonology.explainer import get_rules_version
from phonology.orthography_notes import OrthographicNoteDataError
from phonology.profiles import (
    LanguageProfile,
    get_default_language_profile,
    get_language_profile,
    list_language_profiles,
    register_default_profiles,
)

from ._app_version import (
    _APP_VERSION_ENV_VAR as _APP_VERSION_ENV_VAR,
    _load_app_version,
)
from ._constants import API_VERSION, SCHEMA_VERSION
from ._models import (
    DataVersions,
    LanguageInfo,
    LanguagesResponse,
    SearchRequest,
    SearchResponse,
    VersionInfo,
)
from ._request_context import (
    PUBLIC_BASE_URL_ENV_VAR,
    generate_request_id,
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
    run_search,
)

logger = logging.getLogger(__name__)
register_default_profiles()
_ALLOWED_ORIGINS_ENV_VAR = "PROTEUS_ALLOWED_ORIGINS"
_LSJ_REPO_DIR_ENV_VAR = "PROTEUS_LSJ_REPO_DIR"
_DISABLE_STARTUP_WARMUP_ENV_VAR = "PROTEUS_DISABLE_STARTUP_WARMUP"
_DISABLE_STARTUP_WARMUP_ATTR = "disable_startup_warmup"
_ENABLE_API_DOCS_ENV_VAR = "PROTEUS_ENABLE_API_DOCS"
_BUILD_TIMESTAMP_ENV_VAR = "PROTEUS_BUILD_TIMESTAMP"
_GIT_SHA_ENV_VAR = "PROTEUS_GIT_SHA"
# Limit publicly exposed git SHA to 12 hex chars. Long enough to be unique in
# practical git repos, short enough to limit deployment fingerprinting from
# the unauthenticated /version endpoint.
_GIT_SHA_MAX_LENGTH = 12
_SEARCH_DEPENDENCY_LOAD_ERRORS = (
    ValueError,
    FileNotFoundError,
    OSError,
    yaml.YAMLError,
    OrthographicNoteDataError,
)
_SEARCH_DEPENDENCIES_GENERIC_NOT_READY_DETAIL = (
    "Search dependencies are not ready. Verify packaged matrices, rules, and lexicon assets "
    "are available."
)
_SEARCH_DEPENDENCIES_LEXICON_NOT_READY_DETAIL = (
    "Search dependencies are not ready. Run uv run --extra extract python -m "
    "phonology.build_lexicon --if-missing "
    "to generate the lexicon. If you use a non-default LSJ checkout, set "
    f"{_LSJ_REPO_DIR_ENV_VAR} before extraction."
)
_LEGACY_LANGUAGE_ALIAS_DEPRECATION_MESSAGE = (
    "Use response_language='en'|'ja' for response prose; language now selects "
    "the phonology profile."
)


_APP_VERSION = _load_app_version()


@lru_cache
def _load_rule_schema_version() -> str:
    """Return the rule-file JSON schema identifier, or an empty string."""
    schema_candidates: list[Any] = []
    try:
        schema_candidates.append(
            resources.files("phonology").joinpath(
                "data",
                "schemas",
                "phonology_rule_file.schema.json",
            )
        )
    except (ModuleNotFoundError, FileNotFoundError):
        pass

    schema_candidates.append(
        Path(__file__).resolve().parents[2]
        / "data"
        / "schemas"
        / "phonology_rule_file.schema.json"
    )

    for schema_path in schema_candidates:
        # Restrict to Path-like candidates; future Traversable variants without
        # filesystem semantics (e.g., MultiplexedPath under zipped packages) are
        # skipped explicitly so an AttributeError does not mask unrelated bugs.
        if not isinstance(schema_path, (str, os.PathLike)):
            logger.debug(
                "Skipping non-path-like schema candidate of type %s",
                type(schema_path).__name__,
            )
            continue
        try:
            with open(schema_path, "rb") as schema_file:
                schema = json.load(schema_file)
        except (OSError, json.JSONDecodeError):
            continue
        schema_id = schema.get("$id") if isinstance(schema, dict) else None
        if isinstance(schema_id, str):
            return schema_id

    return ""


def _build_version_info() -> VersionInfo:
    """Return API and runtime version metadata shared by versioned endpoints."""
    python_version = ".".join(str(item) for item in sys.version_info[:3])
    return VersionInfo(
        engine_version=_APP_VERSION,
        api_version=API_VERSION,
        schema_version=SCHEMA_VERSION,
        rule_schema_version=_load_rule_schema_version(),
        build_timestamp=os.environ.get(_BUILD_TIMESTAMP_ENV_VAR, "").strip(),
        git_sha=os.environ.get(_GIT_SHA_ENV_VAR, "").strip()[:_GIT_SHA_MAX_LENGTH],
        python_version=python_version,
        mcp_server_version=_APP_VERSION,
    )


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


_FRONTEND_PATH = Path(__file__).resolve().parents[1] / "web" / "index.html"
_CHANGELOG_PATH = Path(__file__).resolve().parents[1] / "web" / "changelog.html"
_STATIC_DIR = Path(__file__).resolve().parents[1] / "web" / "static"


def _load_frontend_html() -> str | None:
    """Load and cache the packaged frontend HTML document."""
    if not _FRONTEND_PATH.exists():
        return None
    try:
        return _FRONTEND_PATH.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        logger.exception("Failed to read frontend HTML from %s", _FRONTEND_PATH)
        return None


def _build_deprecation_link_header(base_url: str, docs_path: str) -> str:
    """Return an absolute URI Link header value per RFC 8288."""
    base = base_url.rstrip("/")
    path = docs_path.lstrip("/")
    return f'<{base}/{path}>; rel="deprecation"'


def _load_changelog_html() -> str | None:
    """Load and cache the packaged changelog HTML document."""
    if not _CHANGELOG_PATH.exists():
        return None
    try:
        return _CHANGELOG_PATH.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        logger.exception("Failed to read changelog HTML from %s", _CHANGELOG_PATH)
        return None


@lru_cache(maxsize=8)
def _load_lexicon_document(language: str = "ancient_greek") -> dict[str, Any]:
    """Load a packaged lexicon document once per process."""
    profile = get_language_profile(language)
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
    language: str = get_default_language_profile().language_id,
) -> tuple[dict[str, Any], ...]:
    """Load a packaged language lexicon once per process."""
    document = _load_lexicon_document(language)

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
    language: str = get_default_language_profile().language_id,
) -> tuple[dict[str, Any], ...]:
    """Return cached packaged lexicon entries for a language profile."""
    return _load_lexicon_entries(language)


@lru_cache(maxsize=8)
def _load_distance_matrix_with_meta(
    language: str = get_default_language_profile().language_id,
) -> tuple[MatrixData, dict[str, Any]]:
    """Load the packaged search distance matrix and metadata once per process."""
    profile = get_language_profile(language)
    return load_matrix_document(profile.matrix_path)


def _load_distance_matrix(
    language: str = get_default_language_profile().language_id,
) -> MatrixData:
    """Load the packaged search distance matrix once per process."""
    matrix, _ = _load_distance_matrix_with_meta(language)
    return matrix


def _load_rules_registry(
    language: str = get_default_language_profile().language_id,
) -> dict[str, dict[str, Any]]:
    """Return the ``lru_cache``-backed registry owned by ``phonology_search.get_rules_registry``.

    The returned dict is shared process-wide with the phonology search layer.
    Callers must not mutate it.
    """
    return phonology_search.get_rules_registry(language)


@lru_cache(maxsize=8)
def _load_search_index(
    language: str = get_default_language_profile().language_id,
) -> phonology_search.KmerIndex:
    """Build and cache the k-mer index for the packaged lexicon."""
    profile = get_language_profile(language)
    return phonology_search.build_kmer_index(
        load_lexicon_entries(language),
        phone_inventory=profile.phone_inventory,
        dialect_skeleton_builders=profile.dialect_skeleton_builders,
    )


@lru_cache(maxsize=8)
def _load_unigram_index(
    language: str = get_default_language_profile().language_id,
) -> phonology_search.KmerIndex:
    """Build and cache the k=1 unigram index for fallback lookup."""
    profile = get_language_profile(language)
    return phonology_search.build_kmer_index(
        load_lexicon_entries(language),
        k=1,
        phone_inventory=profile.phone_inventory,
        dialect_skeleton_builders=profile.dialect_skeleton_builders,
    )


@lru_cache(maxsize=8)
def _load_lexicon_map(
    language: str = get_default_language_profile().language_id,
) -> phonology_search.LexiconMap:
    """Build and cache the lexicon map with per-entry token counts."""
    profile = get_language_profile(language)
    return phonology_search.build_lexicon_map(
        load_lexicon_entries(language),
        phone_inventory=profile.phone_inventory,
    )


@lru_cache(maxsize=8)
def _load_ipa_index(
    language: str = get_default_language_profile().language_id,
) -> phonology_search.IpaIndex:
    """Build and cache the IPA-to-entry-id index for the packaged lexicon."""
    return phonology_search.build_ipa_index(_load_lexicon_map(language))


@lru_cache(maxsize=8)
def _get_rules_version_cached(rules_dir: Path) -> dict[str, str]:
    """Return rules version metadata for a directory, cached per ``rules_dir``.

    Wraps :func:`get_rules_version` so /languages and /search avoid re-parsing
    YAML rule files on every request. Profile rules directories are immutable
    once registered, so per-process caching is safe; tests clear this cache
    via the autouse ``clear_rule_cache`` fixture for full isolation.
    """
    return get_rules_version(rules_dir)


def _aggregate_rules_version(profile: LanguageProfile) -> str | None:
    """Return the max ruleset version for a language profile, or ``None``.

    Returns ``None`` when the profile has no rule files. Exceptions from rule
    loading or version parsing propagate to the caller so the calling context
    can decide whether to log them as degraded-but-recoverable or fatal.
    """
    rules_versions = _get_rules_version_cached(profile.rules_dir)
    if not rules_versions:
        return None
    return str(max(Version(version) for version in rules_versions.values()))


def _build_data_versions(
    language: str = get_default_language_profile().language_id,
) -> DataVersions:
    """Build DataVersions from loaded data sources.

    Collects version metadata from lexicon, matrix, and rules.
    """
    profile = get_language_profile(language)
    fields: dict[str, str] = {}

    try:
        document = _load_lexicon_document(language)
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
        _, matrix_meta = _load_distance_matrix_with_meta(language)
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
    # Same degraded-but-recoverable path as _build_data_versions; use
    # logger.exception so both surfaces log at ERROR with the traceback.
    except (OSError, ValueError, yaml.YAMLError, InvalidVersion):
        logger.exception(
            "Failed to load ruleset versions for language %s", language
        )
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
        # Degraded-but-recoverable: keep the response 200 and surface "unknown".
        # warning + exc_info preserves the traceback for diagnostics without
        # flooding stderr at error level on every probe poll.
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
    """Build public language metadata, degrading unavailable assets to unknown.

    Descriptions are intentionally English-only in Phase 2; localized language
    metadata can be added later without changing the profile registry contract.
    """
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
    language: str = get_default_language_profile().language_id,
) -> SearchDependencies:
    """Load all cached search dependencies needed by /ready and /search."""
    try:
        profile = get_language_profile(language)
    except ValueError as err:
        # Re-raise as a runner-defined subclass so both REST and MCP adapters
        # can distinguish "unknown language" from other generic ValueErrors.
        raise UnsupportedLanguageError(str(err)) from err
    try:
        lexicon = load_lexicon_entries(language)
    except _SEARCH_DEPENDENCY_LOAD_ERRORS as err:
        raise SearchDependenciesNotReadyError(
            _SEARCH_DEPENDENCIES_LEXICON_NOT_READY_DETAIL
        ) from err

    try:
        data_versions = _build_data_versions(language)
        if profile.orthographic_data_preparer is not None:
            profile.orthographic_data_preparer()
        return SearchDependencies(
            profile=profile,
            lexicon=lexicon,
            matrix=_load_distance_matrix(language),
            rules_registry=_load_rules_registry(language),
            search_index=_load_search_index(language),
            unigram_index=_load_unigram_index(language),
            lexicon_map=_load_lexicon_map(language),
            ipa_index=_load_ipa_index(language),
            data_versions=data_versions,
        )
    except _SEARCH_DEPENDENCY_LOAD_ERRORS as err:
        raise SearchDependenciesNotReadyError(
            _SEARCH_DEPENDENCIES_GENERIC_NOT_READY_DETAIL
        ) from err


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
        _load_search_dependencies(get_default_language_profile().language_id)
    except SearchDependenciesNotReadyError as err:
        _log_search_dependencies_not_ready(
            err.detail,
            log_level=logging.INFO,
            message="Background search warmup skipped; dependencies not ready",
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
    if _FRONTEND_HTML is None:
        raise HTTPException(status_code=404, detail="Frontend asset not found")
    escaped_version = html.escape(_APP_VERSION)
    return HTMLResponse(_FRONTEND_HTML.replace("{{APP_VERSION}}", escaped_version))


def render_changelog() -> HTMLResponse:
    """Return the packaged changelog HTML document."""
    if _CHANGELOG_HTML is None:
        raise HTTPException(status_code=404, detail="Changelog asset not found")
    escaped_version = html.escape(_APP_VERSION)
    return HTMLResponse(_CHANGELOG_HTML.replace("{{APP_VERSION}}", escaped_version))


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
        _load_search_dependencies(get_default_language_profile().language_id)
    except SearchDependenciesNotReadyError as err:
        _log_search_dependencies_not_ready(
            err.detail,
            log_level=logging.WARNING,
            message="Readiness probe skipped; dependencies not ready",
        )
        raise _search_dependencies_not_ready_exception(err.detail) from err
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
load_search_dependencies: Callable[[str], SearchDependencies] = _load_search_dependencies
# Public ruleset-version builder for non-REST response metadata.
build_ruleset_versions: Callable[[str], dict[str, str]] = _build_ruleset_versions


def main() -> None:
    """Run the local development server on a safe loopback default."""
    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
