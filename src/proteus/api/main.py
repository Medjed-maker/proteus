"""FastAPI application entry point.

Exposes the Proteus phonological search engine as a REST API.
"""

from __future__ import annotations

from functools import lru_cache
import json
import logging
import os
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import AliasChoices, BaseModel, Field, StringConstraints, field_validator

from proteus.phonology import search as phonology_search
from proteus.phonology._paths import resolve_repo_data_dir
from proteus.phonology.distance import MatrixData, load_matrix
from proteus.phonology.explainer import Explanation, explain_alignment, load_rules, to_prose
from proteus.phonology.ipa_converter import to_ipa

logger = logging.getLogger(__name__)
_ALLOWED_ORIGINS_ENV_VAR = "PROTEUS_ALLOWED_ORIGINS"


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


app = FastAPI(
    title="Proteus",
    description="Ancient Greek phonological search engine",
    version="0.1.0",
    docs_url="/docs",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_allowed_origins(),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
    allow_credentials=False,
)

_FRONTEND_PATH = Path(__file__).resolve().parents[1] / "web" / "index.html"
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
        document = json.loads(lexicon_path.read_text(encoding="utf-8"))
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
        rules_applied=[
            RuleStep(
                rule_id=step.rule_id,
                rule_name=step.rule_name,
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
    dialect_hint: Literal["attic"] = Field(
        default="attic",
        validation_alias=AliasChoices("dialect_hint", "dialect"),
        description="Dialect hint for IPA conversion. Only 'attic' is currently supported.",
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
            return value.strip()
        return value


class RuleStep(BaseModel):
    """Single applied rule step in a search explanation."""

    rule_id: str = Field(description="Stable identifier for the phonological rule.")
    rule_name: str = Field(description="Human-readable display name for the rule.")
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
    try:
        lexicon = load_lexicon_entries()
        matrix = _load_distance_matrix()
        rules_registry = _load_rules_registry()
        search_index = _load_search_index()
    except (ValueError, FileNotFoundError, OSError, yaml.YAMLError) as err:
        logger.exception(
            "Failed to load search dependencies for query %r",
            request.query_form,
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
        )
    except ValueError as err:
        logger.info("Rejected search query %r: %s", request.query_form, err)
        logger.debug("Full ValueError details for query %r", request.query_form, exc_info=True)
        raise HTTPException(status_code=400, detail="Invalid search query") from err
    except (OSError, KeyError, TypeError) as err:
        logger.exception("Search execution failed for query %r", request.query_form)
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


def main() -> None:
    """Run the local development server on a safe loopback default."""
    import uvicorn

    uvicorn.run("proteus.api.main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
