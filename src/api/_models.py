"""Pydantic request and response models for the Proteus API."""

from __future__ import annotations

from datetime import datetime
import unicodedata
from typing import Annotated, Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    Field,
    model_validator,
    StringConstraints,
    field_validator,
)
from phonology._paths import DEFAULT_LANGUAGE_ID
from phonology.profiles import get_default_language_profile, get_language_profile


class DataVersions(BaseModel):
    """Version metadata for data sources used in search."""

    lexicon: str = Field(
        default="unknown",
        description="Lexicon schema version (e.g., '2.0.0').",
    )
    lexicon_updated_at: str = Field(
        default="",
        description="ISO 8601 timestamp of lexicon last update.",
    )
    matrix: str = Field(
        default="unknown",
        description="Distance matrix version (e.g., '1.0.0').",
    )
    matrix_generated_at: str = Field(
        default="",
        description="ISO 8601 timestamp of matrix generation.",
    )
    rules: str = Field(
        default="unknown",
        description="Aggregated phonological rules version (max of all rule files).",
    )

    @field_validator("lexicon_updated_at", "matrix_generated_at", mode="after")
    @classmethod
    def _validate_timestamp(cls, value: str) -> str:
        if value == "":
            return value
        normalized = (
            value.removesuffix("Z") + "+00:00" if value.endswith("Z") else value
        )
        try:
            datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise ValueError("timestamp must be a valid ISO 8601 datetime") from exc
        return normalized


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

    @model_validator(mode="before")
    @classmethod
    def _normalize_language_and_dialect(cls, data: Any) -> Any:
        """Normalize profile language and dialect, preserving legacy language aliases."""
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        raw_language = payload.get("language")
        if isinstance(raw_language, str):
            normalized = raw_language.strip().lower()
            if normalized in {"en", "ja"}:
                payload.setdefault("lang", normalized)
                payload["language"] = "ancient_greek"

        language_value = payload.get("language")
        if language_value is None:
            language = "ancient_greek"
        elif isinstance(language_value, str):
            language = language_value.strip().lower() or "ancient_greek"
        else:
            raise ValueError("language must be a string")

        try:
            profile = get_language_profile(language)
        except ValueError as exc:
            if language == DEFAULT_LANGUAGE_ID:
                profile = get_default_language_profile()
            else:
                raise ValueError(
                    f"invalid language profile: {language!r}; "
                    "requested profile must be registered in phonology.profiles"
                ) from exc

        dialect_key = "dialect_hint" if "dialect_hint" in payload else "dialect"
        raw_dialect = payload.get(dialect_key)
        if raw_dialect is None:
            dialect = profile.default_dialect
        elif isinstance(raw_dialect, str):
            dialect = raw_dialect.strip().lower()
            if dialect not in profile.supported_dialects:
                allowed = ", ".join(repr(item) for item in profile.supported_dialects)
                raise ValueError(
                    f"dialect_hint must be one of ({allowed}) for language {language!r}"
                )
        else:
            raise ValueError("dialect_hint must be a string")

        payload["language"] = language
        payload["dialect_hint"] = dialect
        return payload

    query_form: QueryForm = Field(
        validation_alias=AliasChoices("query_form", "query"),
        description="Greek word to search for (Unicode, polytonic or monotonic).",
    )
    dialect_hint: str = Field(
        default="attic",
        validation_alias=AliasChoices("dialect_hint", "dialect"),
        description=(
            "Dialect hint for IPA conversion. Built-in Ancient Greek values are "
            "'attic' and 'koine'."
        ),
    )
    max_candidates: int = Field(
        default=20,
        ge=1,
        le=100,
        validation_alias=AliasChoices("max_candidates", "max_results"),
        description="Maximum number of hits to return.",
    )
    language: str = Field(
        default="ancient_greek",
        description="Language profile used for phonological search.",
    )
    lang: Literal["en", "ja"] = Field(
        default="en",
        validation_alias=AliasChoices("lang"),
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
            if not normalized:
                return "attic"
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
                raise ValueError(
                    f"invalid lang: {value}; expected one of {{'en', 'ja'}}"
                )
            return normalized
        return value

    @field_validator("language", mode="before")
    @classmethod
    def _normalize_language(cls, value: Any) -> Any:
        """Normalize language identifier to a clean string."""
        if value is None:
            return "ancient_greek"
        if not isinstance(value, str):
            raise ValueError("language must be a string")
        normalized = value.strip().lower()
        return normalized or "ancient_greek"


class RuleStep(BaseModel):
    """Single applied rule step in a search explanation."""

    rule_id: str = Field(description="Stable identifier for the phonological rule.")
    rule_name: str = Field(description="Human-readable display name for the rule.")
    rule_name_en: str = Field(
        default="", description="English display name for the rule."
    )
    from_phone: str = Field(description="Source IPA phone before the rule applied.")
    to_phone: str = Field(description="Target IPA phone after the rule applied.")
    position: int = Field(
        ge=-1,
        description="Zero-based phone position in the alignment, or -1 when unknown.",
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
    match_type: Literal["Exact", "Rule-based", "Distance-only", "Low-confidence"] = (
        Field(
            default="Distance-only",
            description="High-level classification describing how the candidate matched.",
        )
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
            "True when annotation was truncated due to candidate-window or batch limits. "
            "This occurs for Short-query searches when many candidates are filtered out, "
            "indicating the result set may be incomplete."
        ),
    )
    data_versions: DataVersions = Field(
        default_factory=DataVersions,
        description="Version metadata for data sources used in this search.",
    )
