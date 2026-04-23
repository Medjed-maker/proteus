"""FastAPI application entry point.

Exposes the Proteus phonological search engine as a REST API.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
import hashlib
import html
from importlib import metadata
from functools import lru_cache
import json
import logging
import os
from pathlib import Path
import threading
import tomllib
from typing import Any, AsyncIterator, NamedTuple

import yaml

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from packaging.version import InvalidVersion, Version

from phonology import search as phonology_search
from phonology._paths import resolve_repo_data_dir
from phonology.distance import MatrixData, load_matrix_document
from phonology.explainer import get_rules_version
from phonology.ipa_converter import to_ipa

from ._models import DataVersions, SearchRequest, SearchResponse
from ._hit_formatting import _build_search_hit

logger = logging.getLogger(__name__)
_ALLOWED_ORIGINS_ENV_VAR = "PROTEUS_ALLOWED_ORIGINS"
_LOG_RAW_QUERY_ENV_VAR = "PROTEUS_LOG_RAW_SEARCH_QUERY"
_LSJ_REPO_DIR_ENV_VAR = "PROTEUS_LSJ_REPO_DIR"
_DISABLE_STARTUP_WARMUP_ENV_VAR = "PROTEUS_DISABLE_STARTUP_WARMUP"
_DISABLE_STARTUP_WARMUP_ATTR = "disable_startup_warmup"
_ENABLE_API_DOCS_ENV_VAR = "PROTEUS_ENABLE_API_DOCS"
_APP_VERSION_ENV_VAR = "PROTEUS_APP_VERSION"
# Upper bound for both the similarity fallback (stage 2 widening when the
# seed set is too small) and the unigram fallback (k=1 recovery for short
# queries). Kept in one place so the two caps stay coupled; 2000 mirrors
# search._DEFAULT_FALLBACK_CANDIDATE_LIMIT, which was chosen to keep a hard
# ceiling on the candidate set even for permissive partial queries.
_API_FALLBACK_CANDIDATE_LIMIT = 2000
_SEARCH_DEPENDENCY_LOAD_ERRORS = (ValueError, FileNotFoundError, OSError, yaml.YAMLError)
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


def _load_app_version() -> str:
    """Return the application version from env, package metadata, or pyproject."""
    env_version = os.environ.get(_APP_VERSION_ENV_VAR, "").strip().lstrip("v")
    if env_version:
        return env_version

    try:
        return metadata.version("proteus")
    except metadata.PackageNotFoundError:
        pass

    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        with pyproject_path.open("rb") as pyproject_file:
            pyproject = tomllib.load(pyproject_file)
    except (OSError, tomllib.TOMLDecodeError):
        logger.exception("Failed to read application version from %s", pyproject_path)
        return "unknown"

    project = pyproject.get("project", {})
    if isinstance(project, dict):
        version = project.get("version")
        if isinstance(version, str) and version.strip():
            return version.strip()

    logger.error("Application version is missing from %s", pyproject_path)
    return "unknown"


_APP_VERSION = _load_app_version()


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


class SearchDependenciesNotReadyError(RuntimeError):
    """Represent a dependency-loading failure with an API-safe detail string."""

    detail: str

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


@asynccontextmanager
async def _app_lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Run best-effort startup warmup while keeping app startup non-blocking."""
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
        origin
        for origin in (item.strip() for item in raw_origins.split(","))
        if origin
    ]


def _summarize_query_for_logs(query: str) -> str:
    """Return a redacted search-query identifier safe for routine logs."""
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]
    return f"len={len(query)} sha256={digest}"


def _should_log_raw_search_query() -> bool:
    """Return True only when raw search queries may appear in debug logs."""
    return logger.isEnabledFor(logging.DEBUG) and _env_flag_enabled(_LOG_RAW_QUERY_ENV_VAR)


# _env_flag_enabled is evaluated at import time here, not per-request.
# Export PROTEUS_ENABLE_API_DOCS=1 before starting uvicorn to enable /docs.
app = FastAPI(
    title="Proteus",
    description="Ancient Greek phonological search engine",
    version=_APP_VERSION,
    docs_url="/docs" if _env_flag_enabled(_ENABLE_API_DOCS_ENV_VAR) else None,
    openapi_url="/openapi.json" if _env_flag_enabled(_ENABLE_API_DOCS_ENV_VAR) else None,
    redoc_url=None,
    lifespan=_app_lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_allowed_origins(),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
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


def _load_changelog_html() -> str | None:
    """Load and cache the packaged changelog HTML document."""
    if not _CHANGELOG_PATH.exists():
        return None
    try:
        return _CHANGELOG_PATH.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        logger.exception("Failed to read changelog HTML from %s", _CHANGELOG_PATH)
        return None


@lru_cache(maxsize=1)
def _load_lexicon_document() -> dict[str, Any]:
    """Load the packaged Greek lemma lexicon document once per process."""
    lexicon_path = resolve_repo_data_dir("lexicon") / "greek_lemmas.json"
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


@lru_cache(maxsize=1)
def _load_lexicon_entries() -> tuple[dict[str, Any], ...]:
    """Load the packaged Greek lemma lexicon once per process."""
    document = _load_lexicon_document()

    raw_entries = document.get("lemmas")
    if not isinstance(raw_entries, list):
        raise ValueError("Lexicon document must define a list under 'lemmas'")

    entries: list[dict[str, Any]] = []
    for index, entry in enumerate(raw_entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Lexicon entry {index} must be an object")
        entries.append(dict(entry))
    return tuple(entries)


def load_lexicon_entries() -> tuple[dict[str, Any], ...]:
    """Return the cached packaged Greek lemma lexicon."""
    return _load_lexicon_entries()


@lru_cache(maxsize=1)
def _load_distance_matrix_with_meta() -> tuple[MatrixData, dict[str, Any]]:
    """Load the packaged search distance matrix and its metadata once per process."""
    return load_matrix_document("attic_doric.json")


def _load_distance_matrix() -> MatrixData:
    """Load the packaged search distance matrix once per process."""
    matrix, _ = _load_distance_matrix_with_meta()
    return matrix


def _load_rules_registry() -> dict[str, dict[str, Any]]:
    """Return the ``lru_cache``-backed registry owned by ``phonology_search.get_rules_registry``.

    The returned dict is shared process-wide with the phonology search layer.
    Callers must not mutate it.
    """
    return phonology_search.get_rules_registry("ancient_greek")


@lru_cache(maxsize=1)
def _load_search_index() -> phonology_search.KmerIndex:
    """Build and cache the k-mer index for the packaged lexicon."""
    return phonology_search.build_kmer_index(load_lexicon_entries())


@lru_cache(maxsize=1)
def _load_unigram_index() -> phonology_search.KmerIndex:
    """Build and cache the k=1 unigram index for fallback lookup."""
    return phonology_search.build_kmer_index(load_lexicon_entries(), k=1)


@lru_cache(maxsize=1)
def _load_lexicon_map() -> phonology_search.LexiconMap:
    """Build and cache the lexicon map with per-entry token counts."""
    return phonology_search.build_lexicon_map(load_lexicon_entries())


@lru_cache(maxsize=1)
def _load_ipa_index() -> phonology_search.IpaIndex:
    """Build and cache the IPA-to-entry-id index for the packaged lexicon."""
    return phonology_search.build_ipa_index(_load_lexicon_map())


def _build_data_versions() -> DataVersions:
    """Build DataVersions from loaded data sources.

    Collects version metadata from lexicon, matrix, and rules.
    """
    fields: dict[str, str] = {}

    try:
        document = _load_lexicon_document()
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
        _, matrix_meta = _load_distance_matrix_with_meta()
        matrix_version = matrix_meta.get("version")
        if isinstance(matrix_version, str) and matrix_version.strip():
            fields["matrix"] = matrix_version
        generated_at = matrix_meta.get("generated_at")
        if isinstance(generated_at, str):
            fields["matrix_generated_at"] = generated_at
    except (OSError, ValueError, json.JSONDecodeError) as err:
        logger.exception("Failed to load matrix data version metadata: %s", err)

    try:
        rules_versions = get_rules_version("ancient_greek")
        if rules_versions:
            max_version = max(Version(version) for version in rules_versions.values())
            fields["rules"] = str(max_version)
    except (OSError, ValueError, yaml.YAMLError, InvalidVersion) as err:
        logger.exception("Failed to load rules data version metadata: %s", err)

    return DataVersions(**fields)


def _load_search_dependencies() -> SearchDependencies:
    """Load all cached search dependencies needed by /ready and /search."""
    try:
        lexicon = load_lexicon_entries()
    except _SEARCH_DEPENDENCY_LOAD_ERRORS as err:
        raise SearchDependenciesNotReadyError(
            _SEARCH_DEPENDENCIES_LEXICON_NOT_READY_DETAIL
        ) from err

    try:
        data_versions = _build_data_versions()
        return SearchDependencies(
            lexicon=lexicon,
            matrix=_load_distance_matrix(),
            rules_registry=_load_rules_registry(),
            search_index=_load_search_index(),
            unigram_index=_load_unigram_index(),
            lexicon_map=_load_lexicon_map(),
            ipa_index=_load_ipa_index(),
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
        _load_search_dependencies()
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
async def search(request: SearchRequest) -> SearchResponse:
    """Run phonological search for a Greek query word."""
    query_log_label = _summarize_query_for_logs(request.query_form)

    try:
        deps = _load_search_dependencies()
    except SearchDependenciesNotReadyError as err:
        _log_search_dependencies_not_ready(
            err.detail,
            log_level=logging.WARNING,
            message="Search dependencies are not ready",
            query_log_label=query_log_label,
        )
        raise _search_dependencies_not_ready_exception(err.detail) from err

    try:
        prepared_query = phonology_search.prepare_query_ipa(
            request.query_form,
            dialect=request.dialect_hint,
            converter=to_ipa,
        )
        query_mode = prepared_query.query_mode
        query_ipa = prepared_query.query_ipa
        # The cached dependency bundle is reused across requests, so pass its
        # named fields directly into the core search implementation.
        execution = phonology_search.search_execution(
            request.query_form,
            lexicon=deps.lexicon,
            matrix=deps.matrix,
            max_results=request.max_candidates,
            dialect=request.dialect_hint,
            index=deps.search_index,
            unigram_index=deps.unigram_index,
            prebuilt_lexicon_map=deps.lexicon_map,
            prepared_query=prepared_query,
            prebuilt_ipa_index=deps.ipa_index,
            similarity_fallback_limit=_API_FALLBACK_CANDIDATE_LIMIT,
            unigram_fallback_limit=_API_FALLBACK_CANDIDATE_LIMIT,
        )
        results = execution.results
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
        raise HTTPException(status_code=400, detail="Invalid search query") from err
    except OSError as err:
        logger.exception("Search execution failed for query (%s)", query_log_label)
        raise HTTPException(
            status_code=500,
            detail="Search failed due to an internal error.",
        ) from err

    return SearchResponse(
        query=request.query_form,
        query_ipa=query_ipa,
        query_mode=query_mode,
        hits=[
            _build_search_hit(
                result,
                query_ipa=query_ipa,
                rules_registry=deps.rules_registry,
                query_mode=query_mode,
                lang=request.lang,
            )
            for result in results
        ],
        truncated=execution.truncated,
        data_versions=deps.data_versions,
    )


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.head("/health")
async def health_head() -> Response:
    """Liveness probe for HEAD-based health checks."""
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
    return {"status": "ok"}


# Mount static files after all route definitions. A StaticFiles mount at "/"
# would shadow every route registered after it; placing all mounts last is a
# safe convention even for prefix-scoped paths like "/static".
if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
else:
    logger.warning("Static assets directory not found at %s", _STATIC_DIR)


def main() -> None:
    """Run the local development server on a safe loopback default."""
    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
