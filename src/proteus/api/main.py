"""FastAPI application entry point.

Exposes the Proteus phonological search engine as a REST API.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Proteus",
    description="Ancient Greek phonological search engine",
    version="0.1.0",
)

_FRONTEND_PATH = Path(__file__).resolve().parents[1] / "web" / "index.html"


def _load_frontend_html() -> str | None:
    """Load and cache the packaged frontend HTML document."""
    if not _FRONTEND_PATH.exists():
        return None
    try:
        return _FRONTEND_PATH.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        logger.exception("Failed to read frontend HTML from %s", _FRONTEND_PATH)
        return None


# HTML is read once at import time and cached for the process lifetime;
# dev-mode ``--reload`` (see main()) handles staleness during development.
_FRONTEND_HTML = _load_frontend_html()
if _FRONTEND_HTML is None:
    logger.warning("Frontend HTML asset not found at %s", _FRONTEND_PATH)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    """Client request for a phonological search query."""

    query: str = Field(
        description="Greek word to search for (Unicode, polytonic or monotonic)."
    )
    dialect: str = Field(
        default="attic",
        description="Source dialect for phonological rules.",
    )
    max_results: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of hits to return.",
    )


class RuleStep(BaseModel):
    """Single applied rule step in a search explanation.

    Attributes:
        rule_id: Stable identifier for the phonological rule.
        rule_name: Human-readable display name for the rule.
        from_phone: Source IPA phone before the rule applied.
        to_phone: Target IPA phone after the rule applied.
        position: Zero-based phone position in the aligned form.
    """

    rule_id: str = Field(description="Stable identifier for the phonological rule.")
    rule_name: str = Field(description="Human-readable display name for the rule.")
    from_phone: str = Field(description="Source IPA phone before the rule applied.")
    to_phone: str = Field(description="Target IPA phone after the rule applied.")
    position: int = Field(description="Zero-based phone position in the alignment.")


class SearchHit(BaseModel):
    """A matched headword returned from phonological search.

    Attributes:
        headword: Matched lexicon entry in Greek script.
        ipa: IPA transcription of the matched headword.
        distance: Normalized phonological distance from the query.
        rules_applied: Ordered rule steps explaining the match.
        explanation: Human-readable prose summary of the derivation.
    """

    headword: str = Field(description="Matched lexicon entry in Greek script.")
    ipa: str = Field(description="IPA transcription of the matched headword.")
    distance: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Normalized phonological distance from the query in the 0.0-1.0 range "
            "(0.0 = identical, lower is more similar)."
        )
    )
    rules_applied: list[RuleStep] = Field(
        description="Ordered rule steps explaining the match."
    )
    explanation: str = Field(
        description="Human-readable prose summary of the derivation."
    )


class SearchResponse(BaseModel):
    """Top-level response payload for a phonological search.

    Attributes:
        query: Original Greek query string.
        query_ipa: IPA transcription computed for the query.
        hits: Ranked list of matched headwords.
    """

    query: str = Field(description="Original Greek query string.")
    query_ipa: str = Field(description="IPA transcription computed for the query.")
    hits: list[SearchHit] = Field(description="Ranked list of matched headwords.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


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
    """Run phonological search for a Greek query word.

    Returns ranked hits with phonological distance, applied rules,
    and human-readable explanation.
    """
    raise HTTPException(
        status_code=501,
        detail="Phonological search is not implemented yet.",
    )


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Dev server entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the local development server on a safe loopback default.

    Production deployments should configure host and port explicitly outside
    this helper instead of relying on the development defaults.
    """
    import uvicorn

    uvicorn.run("proteus.api.main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
