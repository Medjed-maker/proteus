"""FastAPI application entry point.

Exposes the Proteus phonological search engine as a REST API.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(
    title="Proteus",
    description="Ancient Greek phonological search engine",
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    query: str
    """Greek word to search for (Unicode, polytonic or monotonic)."""

    dialect: str = "attic"
    """Source dialect for phonological rules."""

    max_results: int = 20


class RuleStep(BaseModel):
    rule_id: str
    rule_name: str
    from_phone: str
    to_phone: str
    position: int


class SearchHit(BaseModel):
    headword: str
    ipa: str
    distance: float
    rules_applied: list[RuleStep]
    explanation: str


class SearchResponse(BaseModel):
    query: str
    query_ipa: str
    hits: list[SearchHit]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    """Serve the frontend."""
    raise NotImplementedError


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    """Run phonological search for a Greek query word.

    Returns ranked hits with phonological distance, applied rules,
    and human-readable explanation.
    """
    raise NotImplementedError


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Dev server entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    import uvicorn

    uvicorn.run("proteus.api.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
