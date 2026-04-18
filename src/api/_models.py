"""Pydantic request and response models for the Proteus API."""

from __future__ import annotations

import unicodedata
from typing import Annotated, Any, Literal

from pydantic import AliasChoices, BaseModel, Field, StringConstraints, field_validator

# Ancient Greek headwords rarely exceed ~25 graphemes; 64 gives headroom for
# partial-form patterns with wildcards while bounding the Needleman-Wunsch
# quadratic cost for the /search endpoint.
_MAX_QUERY_LENGTH = 64

QueryForm = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


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
    lang: Literal["en", "ja"] = Field(
        default="en",
        validation_alias=AliasChoices("lang", "language"),
        description="Response language for generated prose text ('en' or 'ja').",
    )

    @field_validator("query_form", mode="after")
    @classmethod
    def _validate_query_form_length(cls, value: str) -> str:
        normalized = unicodedata.normalize("NFC", value)
        if len(normalized) > _MAX_QUERY_LENGTH:
            raise ValueError(
                f"query_form must be at most {_MAX_QUERY_LENGTH} characters after NFC normalization"
            )
        return normalized

    @field_validator("dialect_hint", mode="before")
    @classmethod
    def _normalize_dialect_hint(cls, value: Any) -> Any:
        if value is None:
            return "attic"
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized not in {"attic", "koine"}:
                raise ValueError("dialect_hint must be either 'attic' or 'koine'")
            return normalized
        return value

    @field_validator("lang", mode="before")
    @classmethod
    def _normalize_lang(cls, value: Any) -> Any:
        if value is None:
            return "en"
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized not in {"en", "ja"}:
                raise ValueError(f"invalid lang: {value}; expected one of {{'en', 'ja'}}")
            return normalized
        return value


class RuleStep(BaseModel):
    """Single applied rule step in a search explanation."""

    rule_id: str = Field(description="Stable identifier for the phonological rule.")
    rule_name: str = Field(description="Human-readable display name for the rule.")
    rule_name_en: str = Field(default="", description="English display name for the rule.")
    from_phone: str = Field(description="Source IPA phone before the rule applied.")
    to_phone: str = Field(description="Target IPA phone after the rule applied.")
    position: int = Field(
        ge=-1,
        description="Zero-based phone position in the alignment, or -1 when unknown."
    )


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
            "fields are provided to support different consumer preferences. "
            "Exact and rule-supported short-query matches may be retained even "
            "when this score is below the short-query confidence floor; "
            "partial-form ranking applies separate wildcard-coverage semantics."
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
    match_type: Literal["Exact", "Rule-based", "Distance-only", "Low-confidence"] = Field(
        default="Distance-only",
        description="High-level classification describing how the candidate matched.",
    )
    rule_support: bool = Field(
        default=False,
        description="Whether at least one explicit catalogued rule supports the candidate.",
    )
    applied_rule_count: int = Field(
        default=0,
        ge=0,
        description="Count of explicit catalogued rules applied to explain the match.",
    )
    observed_change_count: int = Field(
        default=0,
        ge=0,
        description="Count of uncatalogued observed changes retained in the explanation.",
    )
    alignment_summary: str = Field(
        default="",
        description="Short one-line summary of the main phonological differences.",
    )
    why_candidate: list[str] = Field(
        default_factory=list,
        description="Short bullet points explaining why the candidate ranked highly.",
    )
    uncertainty: Literal["Low", "Medium", "High"] = Field(
        default="High",
        description="Qualitative uncertainty label for the ranked candidate.",
    )
    candidate_bucket: Literal["Supported", "Exploratory"] = Field(
        default="Exploratory",
        description="Default presentation bucket for the result list.",
    )
    rules_applied: list[RuleStep] = Field(
        default_factory=list,
        description="Ordered rule steps explaining the match.",
    )
    explanation: str = Field(
        description="Human-readable prose summary of the derivation.",
    )


class SearchResponse(BaseModel):
    """Top-level response payload for a phonological search."""

    query: str = Field(description="Original Greek query string.")
    query_ipa: str = Field(description="IPA transcription computed for the query.")
    query_mode: Literal["Full-form", "Short-query", "Partial-form"] = Field(
        description=(
            "Heuristic input classification used for UI messaging. Partial-form "
            "ranking prioritizes full wildcard and fragment coverage before "
            "confidence among candidates that pass the mode filter; Short-query "
            "ranking can retain exact and rule-supported matches below its "
            "confidence floor."
        )
    )
    hits: list[SearchHit] = Field(description="Ranked list of matched headwords.")
    truncated: bool = Field(
        default=False,
        description=(
            "True when annotation was truncated due to batch limits. This occurs "
            "for Short-query searches when many candidates are filtered out, "
            "indicating the result set may be incomplete."
        ),
    )
