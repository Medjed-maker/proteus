"""FastAPI application entry point.

Exposes the Proteus phonological search engine as a REST API.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import hashlib
from functools import lru_cache
import json
import logging
import os
from pathlib import Path
import threading
from typing import Annotated, Any, AsyncIterator, Literal

import yaml

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import AliasChoices, BaseModel, Field, StringConstraints, field_validator

from phonology import search as phonology_search
from phonology._paths import resolve_repo_data_dir
from phonology.distance import MatrixData, load_matrix
from phonology.explainer import Explanation, explain_alignment, load_rules, to_prose
from phonology.ipa_converter import to_ipa

logger = logging.getLogger(__name__)
_ALLOWED_ORIGINS_ENV_VAR = "PROTEUS_ALLOWED_ORIGINS"
_LOG_RAW_QUERY_ENV_VAR = "PROTEUS_LOG_RAW_SEARCH_QUERY"
_DISABLE_STARTUP_WARMUP_ATTR = "disable_startup_warmup"
_API_UNIGRAM_FALLBACK_LIMIT = 2000


@asynccontextmanager
async def _app_lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Run best-effort startup warmup while keeping app startup non-blocking."""
    if not getattr(app.state, _DISABLE_STARTUP_WARMUP_ATTR, False):
        warmup_thread = threading.Thread(
            target=_warm_search_dependencies,
            name="proteus-search-warmup",
            daemon=True,
        )
        app.state.search_warmup_thread = warmup_thread
        warmup_thread.start()
    yield


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
    raw_value = os.environ.get(_LOG_RAW_QUERY_ENV_VAR, "")
    return logger.isEnabledFor(logging.DEBUG) and raw_value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


app = FastAPI(
    title="Proteus",
    description="Ancient Greek phonological search engine",
    version="0.1.0",
    docs_url="/docs",
    lifespan=_app_lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_allowed_origins(),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
    allow_credentials=False,
)

_FRONTEND_PATH = Path(__file__).resolve().parents[1] / "web" / "index.html"
_STATIC_DIR = Path(__file__).resolve().parents[1] / "web" / "static"
QueryForm = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


def _load_frontend_html() -> str | None:
    """Load and cache the packaged frontend HTML document."""
    if not _FRONTEND_PATH.exists():
        return None
    try:
        return _FRONTEND_PATH.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        logger.exception("Failed to read frontend HTML from %s", _FRONTEND_PATH)
        return None


@lru_cache(maxsize=1)
def _load_lexicon_entries() -> tuple[dict[str, Any], ...]:
    """Load the packaged Greek lemma lexicon once per process."""
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

    raw_entries = document.get("lemmas")
    if not isinstance(raw_entries, list):
        raise ValueError(f"Lexicon file {lexicon_path} must define a list under 'lemmas'")

    entries: list[dict[str, Any]] = []
    for index, entry in enumerate(raw_entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Lexicon entry {index} in {lexicon_path} must be an object")
        entries.append(dict(entry))
    return tuple(entries)


def load_lexicon_entries() -> tuple[dict[str, Any], ...]:
    """Return the cached packaged Greek lemma lexicon."""
    return _load_lexicon_entries()


@lru_cache(maxsize=1)
def _load_distance_matrix() -> MatrixData:
    """Load the packaged search distance matrix once per process."""
    return load_matrix("attic_doric.json")


@lru_cache(maxsize=1)
def _load_rules_registry() -> dict[str, dict[str, Any]]:
    """Load the packaged phonological rule registry once per process."""
    return load_rules("ancient_greek")


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


def _warm_search_dependencies() -> None:
    """Eagerly populate cached search dependencies for faster first queries."""
    try:
        _load_lexicon_entries()
        _load_distance_matrix()
        _load_rules_registry()
        _load_search_index()
        _load_unigram_index()
        _load_lexicon_map()
    except (ValueError, FileNotFoundError, OSError, yaml.YAMLError):
        logger.exception("Background search warmup failed")


def _distance_from_confidence(confidence: float) -> float:
    """Convert a normalized confidence score into normalized distance."""
    return max(0.0, min(1.0, 1.0 - confidence))


def _build_search_hit(
    result: phonology_search.SearchResult,
    query_ipa: str,
    rules_registry: dict[str, dict[str, Any]],
) -> SearchHit:
    """Convert a core search result into the public API response shape."""
    source_ipa = result.ipa or ""
    # Both `distance` and `confidence` are included in the API response so
    # that consumers can choose whichever convention they prefer (lower-is-
    # better or higher-is-better).  They are strict inverses:
    # confidence = 1.0 - distance.
    distance = _distance_from_confidence(result.confidence)
    if result.rule_applications:
        explanation = Explanation(
            source=source_ipa,
            target=query_ipa,
            source_ipa=source_ipa,
            target_ipa=query_ipa,
            distance=distance,
            steps=list(result.rule_applications),
        )
    else:
        explanation = explain_alignment(
            source_ipa=source_ipa,
            target_ipa=query_ipa,
            rule_ids=result.applied_rules,
            all_rules=rules_registry,
            distance=distance,
        )
    return SearchHit(
        headword=result.lemma,
        ipa=source_ipa,
        distance=distance,
        confidence=result.confidence,
        dialect_attribution=result.dialect_attribution or "",
        alignment_visualization=result.alignment_visualization or "",
        rules_applied=[
            RuleStep(
                rule_id=step.rule_id,
                rule_name=step.rule_name,
                rule_name_en=step.rule_name_en,
                from_phone=step.from_phone,
                to_phone=step.to_phone,
                position=step.position,
            )
            for step in explanation.steps
        ],
        explanation=to_prose(explanation),
    )


# HTML is read once at import time and cached for the process lifetime;
# dev-mode ``--reload`` (see main()) handles staleness during development.
_FRONTEND_HTML = _load_frontend_html()
if _FRONTEND_HTML is None:
    logger.warning("Frontend HTML asset not found at %s", _FRONTEND_PATH)


class SearchRequest(BaseModel):
    """Client request for a phonological search query."""

    query_form: QueryForm = Field(
        validation_alias=AliasChoices("query_form", "query"),
        description="Greek word to search for (Unicode, polytonic or monotonic).",
    )
    dialect_hint: Literal["attic", "koine"] = Field(
        default="attic",
        validation_alias=AliasChoices("dialect_hint", "dialect"),
        description="Dialect hint for IPA conversion. Supports 'attic' and query-side 'koine'.",
    )
    max_candidates: int = Field(
        default=20,
        ge=1,
        le=100,
        validation_alias=AliasChoices("max_candidates", "max_results"),
        description="Maximum number of hits to return.",
    )

    @field_validator("dialect_hint", mode="before")
    @classmethod
    def _normalize_dialect_hint(cls, value: Any) -> Any:
        if value is None:
            return "attic"
        if isinstance(value, str):
            return value.strip().lower()
        return value


class RuleStep(BaseModel):
    """Single applied rule step in a search explanation."""

    rule_id: str = Field(description="Stable identifier for the phonological rule.")
    rule_name: str = Field(description="Human-readable display name for the rule.")
    rule_name_en: str = Field(default="", description="English display name for the rule.")
    from_phone: str = Field(description="Source IPA phone before the rule applied.")
    to_phone: str = Field(description="Target IPA phone after the rule applied.")
    position: int = Field(description="Zero-based phone position in the alignment.")


class SearchHit(BaseModel):
    """A matched headword returned from phonological search."""

    headword: str = Field(description="Matched lexicon entry in Greek script.")
    ipa: str = Field(description="IPA transcription of the matched headword.")
    distance: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Normalized phonological distance from the query in the 0.0-1.0 range "
            "(0.0 = identical, lower is more similar)."
        ),
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Normalized confidence score (0.0-1.0, higher is more similar). "
            "Inverse of ``distance``: ``confidence = 1.0 - distance``. Both "
            "fields are provided to support different consumer preferences."
        ),
    )
    dialect_attribution: str = Field(
        default="",
        description="Dialect attribution for the match.",
    )
    alignment_visualization: str = Field(
        default="",
        description="Three-line ASCII alignment visualization.",
    )
    rules_applied: list[RuleStep] = Field(
        description="Ordered rule steps explaining the match."
    )
    explanation: str = Field(
        description="Human-readable prose summary of the derivation."
    )


class SearchResponse(BaseModel):
    """Top-level response payload for a phonological search."""

    query: str = Field(description="Original Greek query string.")
    query_ipa: str = Field(description="IPA transcription computed for the query.")
    hits: list[SearchHit] = Field(description="Ranked list of matched headwords.")


def render_frontend() -> HTMLResponse:
    """Return the packaged frontend HTML document."""
    if _FRONTEND_HTML is None:
        raise HTTPException(status_code=404, detail="Frontend asset not found")
    return HTMLResponse(_FRONTEND_HTML)


@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    """Serve the frontend."""
    return render_frontend()


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    """Run phonological search for a Greek query word."""
    query_log_label = _summarize_query_for_logs(request.query_form)

    try:
        lexicon = load_lexicon_entries()
        matrix = _load_distance_matrix()
        rules_registry = _load_rules_registry()
        search_index = _load_search_index()
        unigram_index = _load_unigram_index()
        lexicon_map = _load_lexicon_map()
    except (ValueError, FileNotFoundError, OSError, yaml.YAMLError) as err:
        logger.exception(
            "Failed to load search dependencies for query (%s)",
            query_log_label,
        )
        raise HTTPException(
            status_code=500,
            detail="Search is temporarily unavailable. Please try again later.",
        ) from err

    try:
        query_ipa = to_ipa(request.query_form, dialect=request.dialect_hint)
        # phonology_search.search accepts the cached tuple directly, so avoid copying it.
        results = phonology_search.search(
            request.query_form,
            lexicon=lexicon,
            matrix=matrix,
            max_results=request.max_candidates,
            dialect=request.dialect_hint,
            index=search_index,
            unigram_index=unigram_index,
            prebuilt_lexicon_map=lexicon_map,
            unigram_fallback_limit=_API_UNIGRAM_FALLBACK_LIMIT,
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
        hits=[
            _build_search_hit(result, query_ipa=query_ipa, rules_registry=rules_registry)
            for result in results
        ],
    )


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> dict[str, str]:
    """Readiness probe."""
    try:
        _load_lexicon_entries()
        _load_distance_matrix()
        _load_rules_registry()
        _load_search_index()
        _load_unigram_index()
        _load_lexicon_map()
    except (ValueError, FileNotFoundError, OSError, yaml.YAMLError) as err:
        logger.exception("Readiness probe failed")
        raise HTTPException(
            status_code=503,
            detail="Dependencies not ready",
        ) from err
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
