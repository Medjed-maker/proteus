"""Pydantic request and response models for the Proteus API."""

from __future__ import annotations

from datetime import datetime
import unicodedata
from typing import Annotated, Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
    StringConstraints,
    field_validator,
)
from pydantic.json_schema import SkipJsonSchema

# SourceReference is re-exported so that ``SearchHit.source_references`` resolves
# to a concrete type in the generated OpenAPI / MCP schema. Application code
# should continue to import it from ``phonology.core.ports.corpus``.
from phonology.core.ports.corpus import SourceReference
from phonology.core.ports.profiles import get_default_language_profile, get_language_profile

__all__ = [
    "DataVersions",
    "ErrorResponse",
    "VersionInfo",
    "RequestEcho",
    "ResponseMeta",
    "LanguageInfo",
    "LanguagesResponse",
    "SearchRequest",
    "RuleStep",
    "OrthographicNote",
    "BuckReferenceMetadata",
    "BuckReferenceAnnotation",
    "BuckMetadata",
    "BuckReferenceInfo",
    "BuckTransformation",
    "BuckVariant",
    "BuckRuleInfo",
    "BuckDialectInfo",
    "BuckGlossaryEntryInfo",
    "BuckRulesResponse",
    "BuckDialectsResponse",
    "BuckGlossaryResponse",
    "BuckRuleResponse",
    "BuckDialectResponse",
    "SourceReference",
    "SearchHit",
    "SearchResponse",
]


def _validate_iso8601_timestamp(value: str) -> str:
    if value == "":
        return value
    normalized = value.removesuffix("Z") + "+00:00" if value.endswith("Z") else value
    try:
        datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("timestamp must be a valid ISO 8601 datetime") from exc
    return normalized


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
        return _validate_iso8601_timestamp(value)


class ErrorResponse(BaseModel):
    """Simple error response produced by HTTPException handlers."""

    detail: str = Field(description="Human-readable error detail.")


class VersionInfo(BaseModel):
    """Version metadata for the Proteus API and runtime."""

    engine_version: str = Field(description="Proteus package version.")
    api_version: str = Field(description="Public REST API version.")
    schema_version: str = Field(description="Public response schema version.")
    rule_schema_version: str = Field(
        default="",
        description="Rule-file JSON schema identifier, when available.",
    )
    build_timestamp: str = Field(
        default="",
        description="Deployment build timestamp, when provided by the environment.",
    )
    git_sha: str = Field(
        default="",
        description="Deployment Git commit SHA, when provided by the environment.",
    )
    python_version: str = Field(description="Python runtime version as X.Y.Z.")
    mcp_server_version: str = Field(
        description="Proteus MCP server version exposed by this deployment."
    )


class RequestEcho(BaseModel):
    """Sanitized request parameters echoed for reproducibility.

    ``query_form`` is the client-supplied query verbatim — it is independent of
    the ``PROTEUS_LOG_RAW_SEARCH_QUERY`` server-log guard, which only redacts
    raw queries from server logs. Operators that need the raw query stripped
    from response bodies should redact at the fronting proxy. See
    ``README.md`` "Deployment & Operations".
    """

    model_config = ConfigDict(frozen=True)

    query_form: str = Field(description="Validated search query string.")
    language: str = Field(description="Validated language profile identifier.")
    dialect_hint: str = Field(description="Validated dialect hint.")
    max_candidates: int = Field(description="Validated maximum number of hits.")
    response_language: Literal["en", "ja"] = Field(
        description="Validated response prose language."
    )


class ResponseMeta(BaseModel):
    """Reproducibility and version metadata for public API responses."""

    api_version: str = Field(description="Public REST API version.")
    schema_version: str = Field(description="Public response schema version.")
    engine_version: str = Field(description="Proteus package version.")
    data_versions: DataVersions = Field(
        description="Version metadata for data sources used in the response."
    )
    ruleset_versions: dict[str, str] = Field(
        default_factory=dict,
        description="Aggregated ruleset version by language profile.",
    )
    request_id: str = Field(description="Request correlation identifier.")
    timestamp: str = Field(description="UTC ISO 8601 response timestamp.")
    verification_url: str = Field(
        default="",
        description="Deterministic URL that can reproduce the request.",
    )
    request_echo: RequestEcho | None = Field(
        default=None,
        description="Sanitized validated request parameters, when applicable.",
    )

    @field_validator("timestamp", mode="after")
    @classmethod
    def _validate_timestamp(cls, value: str) -> str:
        # ``ResponseMeta.timestamp`` is required and is always populated from
        # ``datetime.now(timezone.utc).isoformat()``; reject the empty string
        # so a malformed construction is surfaced at the boundary instead of
        # leaking an empty value to clients.
        if value == "":
            raise ValueError("ResponseMeta.timestamp must not be empty")
        return _validate_iso8601_timestamp(value)


class LanguageInfo(BaseModel):
    """Public metadata for a registered language profile."""

    language_id: str = Field(description="Stable language profile identifier.")
    display_name: str = Field(description="Human-readable language name.")
    default_dialect: str = Field(description="Default dialect used by the profile.")
    supported_dialects: list[str] = Field(
        description="Dialects accepted by the profile."
    )
    status: Literal["pilot", "experimental", "stable"] = Field(
        description="Support maturity for this language profile."
    )
    ruleset_version: str = Field(
        default="unknown",
        description="Aggregated phonological rules version for this language.",
    )
    lexicon_schema_version: str = Field(
        default="unknown",
        description="Lexicon schema version for this language.",
    )
    matrix_version: str = Field(
        default="unknown",
        description="Distance matrix version for this language.",
    )
    description: str = Field(
        default="",
        description="Short English description of this language profile.",
    )


class LanguagesResponse(BaseModel):
    """Response payload for registered language profile enumeration.

    Note: ``meta`` is temporarily typed as :class:`VersionInfo` while the
    Phase 2 章 1 ``ResponseMeta`` rollout is pending. When ``ResponseMeta``
    lands, ``meta`` will be replaced and the OpenAPI artifact, tests, and CI
    expectations should be updated in the same PR. Clients should ignore
    unknown ``meta`` fields to remain compatible across this transition.
    """

    languages: list[LanguageInfo] = Field(
        description="Registered language profiles sorted by language_id."
    )
    meta: VersionInfo = Field(
        description=(
            "API and runtime version metadata. Temporarily VersionInfo; will "
            "be replaced by ResponseMeta in Phase 2 章 1."
        )
    )


# Ancient Greek headwords rarely exceed ~25 graphemes; 64 gives headroom for
# partial-form patterns with wildcards while bounding the Needleman-Wunsch
# quadratic cost for the /search endpoint.
_MAX_QUERY_LENGTH = 64

QueryForm = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


def _default_language_id() -> str:
    """Return the configured default language profile identifier."""
    return get_default_language_profile().language_id


class SearchRequest(BaseModel):
    """Client request for a phonological search query."""

    @model_validator(mode="before")
    @classmethod
    def _normalize_language_and_dialect(cls, data: Any) -> Any:
        """Normalize profile language and dialect, preserving legacy locale aliases."""
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        payload.pop("legacy_language_alias_used", None)
        raw_language = payload.get("language")
        if isinstance(raw_language, str):
            normalized = raw_language.strip().lower()
            if normalized in {"en", "ja"}:
                if "response_language" not in payload and "lang" not in payload:
                    payload["response_language"] = normalized
                payload["language"] = _default_language_id()
                payload["legacy_language_alias_used"] = True

        language_value = payload.get("language")
        if language_value is None:
            language = _default_language_id()
        elif isinstance(language_value, str):
            language = language_value.strip().lower() or _default_language_id()
        else:
            raise ValueError("language must be a string")

        try:
            profile = get_language_profile(language)
        except ValueError as exc:
            # Fallback for a transient registration race: a request for the
            # configured default language may arrive before that profile is
            # registered, so resolve it via get_default_language_profile().
            # Any other unknown id is a genuine error and re-raised below.
            default_profile = get_default_language_profile()
            if language == default_profile.language_id:
                profile = default_profile
            else:
                raise ValueError(
                    f"invalid language profile: {language!r}; "
                    "requested profile must be registered in phonology.core.ports.profiles"
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

        orthography_hint = cls._normalize_orthography_hint(
            payload.get("orthography_hint")
        )

        if orthography_hint is not None:
            allowed_hints = tuple(profile.deprecated_orthography_hints)
            if orthography_hint not in allowed_hints:
                allowed = ", ".join(repr(item) for item in allowed_hints)
                if not allowed:
                    allowed = "no deprecated orthography hints"
                raise ValueError(
                    f"orthography_hint must be one of ({allowed}) "
                    f"for language {language!r}"
                )

        payload["language"] = language
        payload["dialect_hint"] = dialect
        payload["orthography_hint"] = orthography_hint
        return payload

    query_form: QueryForm = Field(
        validation_alias=AliasChoices("query_form", "query"),
        description="Greek word to search for (Unicode, polytonic or monotonic).",
    )
    dialect_hint: str = Field(
        default_factory=lambda: get_default_language_profile().default_dialect,
        validation_alias=AliasChoices("dialect_hint", "dialect"),
        description=(
            "Dialect hint for IPA conversion. When omitted, the default is "
            "resolved from the selected language profile during validation."
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
        default_factory=_default_language_id,
        description=(
            "Language profile used for phonological search. When omitted, the "
            "configured default language profile is resolved during validation."
        ),
    )
    response_language: Literal["en", "ja"] = Field(
        default="en",
        validation_alias=AliasChoices("response_language", "lang"),
        description="Response language for generated prose text ('en' or 'ja').",
    )
    orthography_hint: str | None = Field(
        default=None,
        deprecated=True,
        description=(
            "Deprecated: hints are accepted for backward compatibility but no "
            "longer affect orthographic note generation. Notes are produced "
            "exclusively from curated runtime orthography data. Accepted values "
            "depend on the target language profile's deprecated orthography "
            "hints; any other value is rejected with a 422. This field will be "
            "removed in a future release; remove it from your requests."
        ),
    )

    legacy_language_alias_used: SkipJsonSchema[bool] = Field(
        default=False,
        exclude=True,
        repr=False,
        description="Internal marker for deprecated language=en|ja compatibility.",
    )

    @property
    def lang(self) -> Literal["en", "ja"]:
        """Backward-compatible access to the response prose language."""
        return self.response_language

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
            return get_default_language_profile().default_dialect
        if isinstance(value, str):
            normalized = value.strip().lower()
            if not normalized:
                return get_default_language_profile().default_dialect
            return normalized
        return value

    @field_validator("response_language", mode="before")
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
            return _default_language_id()
        if not isinstance(value, str):
            raise ValueError("language must be a string")
        normalized = value.strip().lower()
        return normalized or _default_language_id()

    @classmethod
    def _normalize_orthography_hint(cls, value: Any) -> Any:
        """Normalize the (deprecated) orthography hint string.

        Profile-independent string normalization only: ``None`` and blank
        strings collapse to ``None``; other strings are stripped and
        lowercased. Non-string values pass through unchanged so that the
        profile-aware allow-list check in ``_normalize_language_and_dialect``
        rejects them.
        """
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip().lower() or None
        return value


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


class OrthographicNote(BaseModel):
    """Candidate-level note about writing-system or spelling conventions."""

    # Language plugins validate their own note-kind vocabulary; the API returns
    # the plugin-provided machine-readable value without maintaining a central
    # allow-list.
    kind: str = Field(
        description="Machine-readable category for the orthographic note."
    )
    label: str = Field(description="Short display label for the note.")
    messages: list[str] = Field(
        description="Human-readable note messages for the candidate."
    )
    normalized_form: str | None = Field(
        default=None,
        description="Regularized or dictionary-facing form referenced by the note.",
    )
    romanization: str | None = Field(
        default=None,
        description="Romanized form referenced by the note.",
    )
    period_label: str | None = Field(
        default=None,
        description="Historical period or writing-system label for the note.",
    )
    references: list[str] = Field(
        default_factory=list,
        description="Short source or documentation references supporting the note.",
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description="Qualitative confidence assigned to the orthographic note.",
    )
    pre_reform_spelling: str | None = Field(
        default=None,
        description=(
            "Pre-403/2 BCE Attic inscriptional spelling paired with the normalized "
            "form, when applicable (e.g., παιδίο for παιδίου)."
        ),
    )
    pre_reform_romanization: str | None = Field(
        default=None,
        description=(
            "Romanization of the pre-reform spelling, using macron-ō for the long "
            "/oː/ written as single Ο before the orthographic reform."
        ),
    )

    @field_validator("kind", mode="after")
    @classmethod
    def _kind_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("orthographic note kind must not be blank")
        return value


class BuckReferenceMetadata(BaseModel):
    """Review metadata for Buck reference annotations."""

    status: str = Field(description="Data status for the Buck reference layer.")
    review_status: str = Field(description="Expert-review status for the data.")
    citation_ready: bool = Field(
        description=(
            "Whether the Buck reference data is ready for scholarly citation. "
            "False means clients must present it as provisional review metadata."
        )
    )
    review_note: str = Field(
        description="Short warning describing the review and citation boundary."
    )


class BuckReferenceAnnotation(BaseModel):
    """Metadata-only Buck reference linked to an applied phonological rule."""

    source_rule_id: str = Field(
        description="Applied Proteus rule id that produced this Buck annotation."
    )
    buck_rule_id: str = Field(description="Stable Buck-normalized rule identifier.")
    buck_section: str | None = Field(
        default=None,
        description="Canonical Buck section string, when available.",
    )
    category: str | None = Field(default=None, description="Buck rule category.")
    description: str | None = Field(
        default=None,
        description="Short normalized Buck rule summary; not a source-text quote.",
    )
    affected_dialects: list[str] = Field(
        default_factory=list,
        description="Dialect ids listed as affected by the Buck rule.",
    )
    status: str = Field(description="Data status inherited from Buck metadata.")
    review_status: str = Field(description="Review status inherited from Buck metadata.")
    citation_ready: bool = Field(
        description="Whether this Buck reference annotation is citation-ready."
    )
    review_note: str = Field(
        description="Short warning describing the review and citation boundary."
    )


class BuckMetadata(BaseModel):
    """Review metadata shared by Buck reference-data REST responses."""

    status: str = Field(description="Data status for the Buck reference layer.")
    review_status: str = Field(description="Expert-review status for the data.")
    citation_ready: bool = Field(
        description=(
            "Whether the Buck reference data is ready for scholarly citation. "
            "False means clients must present it as provisional review metadata."
        )
    )
    review_note: str = Field(
        description="Short warning describing the review and citation boundary."
    )


class BuckReferenceInfo(BaseModel):
    """Specific Buck section/page reference for a glossary entry."""

    section: str | None = Field(
        default=None,
        description="Canonical Buck section string, when available.",
    )
    page: int | None = Field(default=None, description="Buck page number, when stored.")


class BuckTransformation(BaseModel):
    """Normalized phonological transformation for a Buck rule."""

    # The Buck grammar-rules schema defines `transformation` with
    # additionalProperties: false, so only from/to/context can ever appear.
    # `extra="ignore"` keeps the public surface to this curated field set rather
    # than forwarding arbitrary keys. populate_by_name enables the `from` alias.
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    from_: str | None = Field(
        default=None,
        alias="from",
        description="Source phone/grapheme, when specified.",
    )
    to: str | None = Field(
        default=None,
        description="Resulting phone/grapheme, when specified.",
    )
    context: str | None = Field(
        default=None,
        description="Phonological context, when specified.",
    )


class BuckVariant(BaseModel):
    """Normalized dialectal variant example for a Buck rule."""

    # The Buck grammar-rules schema defines `variant` with
    # additionalProperties: true: variants are heterogeneous summaries, so
    # `extra="allow"` deliberately forwards their curated extra fields. Do not
    # tighten this without changing the data schema first.
    model_config = ConfigDict(extra="allow")

    form: str | None = Field(default=None, description="Variant form, when specified.")
    dialects: list[str] = Field(
        default_factory=list,
        description="Dialect ids exhibiting this variant.",
    )


class BuckRuleInfo(BaseModel):
    """Public REST representation of a Buck grammar-rule summary."""

    id: str = Field(description="Stable Buck-normalized rule identifier.")
    buck_section: str | None = Field(
        default=None,
        description="Canonical Buck section string, when available.",
    )
    category: str | None = Field(default=None, description="Rule category.")
    description: str | None = Field(
        default=None,
        description="Short normalized rule summary; not a long source-text quote.",
    )
    transformation: BuckTransformation | None = Field(
        default=None,
        description="Normalized transformation metadata, when the rule has one.",
    )
    affected_dialects: list[str] = Field(
        default_factory=list,
        description="Dialect ids listed as affected by this rule.",
    )
    variants: list[BuckVariant] = Field(
        default_factory=list,
        description="Short normalized variant examples; source text is not included.",
    )
    notes: str | None = Field(default=None, description="Short curation note.")
    status: str = Field(description="Data status inherited from Buck metadata.")
    review_status: str = Field(description="Review status inherited from Buck metadata.")
    citation_ready: bool = Field(
        description="Whether this Buck rule summary is citation-ready."
    )


class BuckDialectInfo(BaseModel):
    """Public REST representation of a Buck dialect catalog entry."""

    id: str = Field(description="Stable Buck-normalized dialect identifier.")
    name: str | None = Field(default=None, description="Human-readable dialect name.")
    kind: str | None = Field(default=None, description="Dialect catalog kind.")
    group: str | None = Field(default=None, description="Dialect group label.")
    parent: str | None = Field(default=None, description="Parent dialect/group id.")
    rules: list[str] = Field(
        default_factory=list,
        description="Rule ids directly listed for this dialect.",
    )
    status: str = Field(description="Data status inherited from Buck metadata.")
    review_status: str = Field(description="Review status inherited from Buck metadata.")
    citation_ready: bool = Field(
        description="Whether this Buck dialect catalog entry is citation-ready."
    )


class BuckGlossaryEntryInfo(BaseModel):
    """Public REST representation of a Buck glossary example."""

    word: str = Field(description="Dialectal or example form.")
    standard_form: str | None = Field(
        default=None,
        description="Regularized or dictionary-facing form.",
    )
    dialect: str = Field(description="Buck dialect id for the example.")
    rule_id: str | None = Field(
        default=None,
        description="Linked Buck rule id, when available.",
    )
    definition: str | None = Field(default=None, description="Short gloss.")
    inscription_no: str | list[int] | None = Field(
        default=None,
        description=(
            "Inscription identifier, when stored as a string or integer array."
        ),
    )
    buck_ref: BuckReferenceInfo | None = Field(
        default=None,
        description="Buck section/page metadata; source text is not included.",
    )
    notes: str | None = Field(default=None, description="Short curation note.")
    status: str = Field(description="Data status inherited from Buck metadata.")
    review_status: str = Field(description="Review status inherited from Buck metadata.")
    citation_ready: bool = Field(
        description="Whether this Buck glossary entry is citation-ready."
    )


class BuckRulesResponse(BaseModel):
    """Response payload for Buck rule searches."""

    rules: list[BuckRuleInfo] = Field(description="Matching Buck rule summaries.")
    count: int = Field(ge=0, description="Number of rules in the current page.")
    total: int = Field(
        ge=0,
        description="Total matching rules before pagination is applied.",
    )
    metadata: BuckMetadata = Field(description="Buck review metadata.")


class BuckDialectsResponse(BaseModel):
    """Response payload for Buck dialect enumeration."""

    dialects: list[BuckDialectInfo] = Field(
        description="Matching Buck dialect catalog entries."
    )
    count: int = Field(ge=0, description="Number of dialects in the current page.")
    total: int = Field(
        ge=0,
        description="Total matching dialects before pagination is applied.",
    )
    metadata: BuckMetadata = Field(description="Buck review metadata.")


class BuckGlossaryResponse(BaseModel):
    """Response payload for Buck glossary searches."""

    entries: list[BuckGlossaryEntryInfo] = Field(
        description="Matching Buck glossary entries."
    )
    count: int = Field(
        ge=0, description="Number of glossary entries in the current page."
    )
    total: int = Field(
        ge=0,
        description="Total matching glossary entries before pagination is applied.",
    )
    metadata: BuckMetadata = Field(description="Buck review metadata.")


class BuckRuleResponse(BaseModel):
    """Response payload for a single Buck rule lookup."""

    rule: BuckRuleInfo = Field(description="Matching Buck rule summary.")
    metadata: BuckMetadata = Field(description="Buck review metadata.")


class BuckDialectResponse(BaseModel):
    """Response payload for a single Buck dialect lookup."""

    dialect: BuckDialectInfo = Field(description="Matching Buck dialect.")
    rules: list[BuckRuleInfo] = Field(
        default_factory=list,
        description="Buck rule summaries linked from the dialect catalog.",
    )
    metadata: BuckMetadata = Field(description="Buck review metadata.")


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
    orthographic_notes: list[OrthographicNote] = Field(
        default_factory=list,
        description=(
            "Candidate-level notes about orthographic correspondences, historical "
            "spelling systems, or beginner-facing reading aids."
        ),
    )
    source_references: list[SourceReference] = Field(
        default_factory=list,
        description=(
            "External source metadata references for this candidate. Contains "
            "identifiers, short citations, links, and license notes only; source "
            "texts and excerpts are intentionally excluded."
        ),
    )
    buck_references: list[BuckReferenceAnnotation] = Field(
        default_factory=list,
        description=(
            "Metadata-only provisional Buck references linked from applied rule "
            "references. These annotations do not affect ranking or rule application."
        ),
    )
    explanation: str = Field(
        description="Human-readable prose summary of the derivation.",
    )


class SearchResponse(BaseModel):
    """Top-level response payload for a phonological search."""

    model_config = ConfigDict(extra="ignore")

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
        description=(
            "Version metadata for data sources used in this search. Mirrors "
            "meta.data_versions for backward compatibility."
        ),
    )
    buck_reference_metadata: BuckReferenceMetadata | None = Field(
        default=None,
        description=(
            "Review metadata for Buck reference annotations, when annotation was "
            "attempted for this response."
        ),
    )
    meta: ResponseMeta = Field(
        description="Version, request, and reproducibility metadata."
    )
