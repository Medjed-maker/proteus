"""Public source-reference models for corpus metadata adapters."""

from __future__ import annotations

from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

AccessPolicy = Literal[
    "open_metadata",
    "linked_restricted_text",
    "expert_review_required",
]

# Source-text field denylist shared with ``_adapters.py``. The model layer
# rejects these keys on a single ``SourceReference`` payload; the adapter loader
# rejects them recursively across the full YAML document.
_FORBIDDEN_TEXT_FIELDS = frozenset(
    {
        "evidence_excerpt",
        "excerpt",
        "passage",
        "passage_text",
        "quote",
        "raw_text",
        "source_text",
        "text",
    }
)
_SHORT_CITATION_MAX_CHARS = 200
_SHORT_CITATION_MAX_WORDS = 40


class SourceReference(BaseModel):
    """External source metadata attached to a search candidate.

    The model intentionally carries identifiers, short citations, links, and
    license notes only. It must not carry source text or passage excerpts.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str = Field(description="Corpus-local or CTS-style source identifier.")
    corpus: str = Field(description="Human-readable corpus or provider name.")
    short_citation: str = Field(description="Short display citation.")
    external_url: str = Field(
        description="HTTP(S) landing page for the source.",
        max_length=2048,
    )
    license_note: str = Field(description="Short license and attribution note.")
    access_policy: AccessPolicy = Field(
        description="How clients may access the referenced source."
    )
    citation_ready: bool = Field(
        description="Whether this metadata has been reviewed for citation use."
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_text_payloads(cls, data: Any) -> Any:
        if isinstance(data, dict):
            forbidden = sorted(set(data) & _FORBIDDEN_TEXT_FIELDS)
            if forbidden:
                raise ValueError(
                    "SourceReference must not contain source text fields: "
                    + ", ".join(forbidden)
                )
        return data

    @field_validator(
        "source_id",
        "corpus",
        "short_citation",
        "external_url",
        "license_note",
        mode="after",
    )
    @classmethod
    def _require_non_empty_string(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("source reference string fields must not be empty")
        return stripped

    @field_validator("short_citation", mode="after")
    @classmethod
    def _validate_short_citation_length(cls, value: str) -> str:
        """Enforce character and word limits on short_citation.
    
        Rejects citations exceeding 200 characters or 40 words.
        """
        if len(value) > _SHORT_CITATION_MAX_CHARS:
            raise ValueError(
                "short_citation must not exceed "
                f"{_SHORT_CITATION_MAX_CHARS} characters"
            )
        if len(value.split()) > _SHORT_CITATION_MAX_WORDS:
            raise ValueError(
                "short_citation must not exceed "
                f"{_SHORT_CITATION_MAX_WORDS} words"
            )
        return value

    @field_validator("external_url", mode="after")
    @classmethod
    def _validate_external_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("external_url must be an absolute HTTP(S) URL")
        return value


__all__ = ["AccessPolicy", "SourceReference"]
