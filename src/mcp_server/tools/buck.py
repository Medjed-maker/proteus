"""MCP Buck reference tool definitions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field

from phonology.languages.ancient_greek.buck_service import (
    BuckDialect,
    BuckGlossaryEntry,
    BuckMetadata,
    BuckReference,
    BuckRule,
    build_buck_reference_index,
)

_SUPPORTED_BUCK_LANGUAGE = "ancient_greek"
_PROVISIONAL_REVIEW_NOTE = (
    "Buck reference data is provisional, not expert-reviewed, and must not be "
    "treated as citation-ready scholarly evidence."
)
_CITATION_READY_REVIEW_NOTE = (
    "Buck reference data is marked citation-ready; verify the specific context "
    "before scholarly citation."
)


class McpBuckMetadata(BaseModel):
    """Review metadata shared by Buck MCP responses."""

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


class McpBuckReferenceInfo(BaseModel):
    """Specific Buck section/page reference for a glossary entry."""

    section: str | None = Field(
        default=None,
        description="Canonical Buck section string, when available.",
    )
    page: int | None = Field(
        default=None,
        description="Buck page number, when available.",
    )


class McpBuckTransformation(BaseModel):
    """Normalized phonological transformation for a Buck rule.

    Fields are optional because Buck rules describe a range of phenomena; many
    rules carry no transformation at all. ``extra="allow"`` preserves any
    additional normalized keys without dropping data.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

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


class McpBuckVariant(BaseModel):
    """Normalized dialectal variant example for a Buck rule.

    ``extra="allow"`` preserves additional normalized keys without dropping data.
    """

    model_config = ConfigDict(extra="allow")

    form: str | None = Field(default=None, description="Variant form, when specified.")
    dialects: list[str] = Field(
        default_factory=list,
        description="Dialect ids exhibiting this variant.",
    )


class McpBuckRuleInfo(BaseModel):
    """Public MCP representation of a Buck grammar-rule summary."""

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
    transformation: McpBuckTransformation | None = Field(
        default=None,
        description="Normalized transformation metadata, when the rule has one.",
    )
    affected_dialects: list[str] = Field(
        default_factory=list,
        description="Dialect ids listed as affected by this rule.",
    )
    variants: list[McpBuckVariant] = Field(
        default_factory=list,
        description="Short normalized variant examples; source text is not included.",
    )
    notes: str | None = Field(
        default=None,
        description="Short curation note, when available.",
    )
    status: str = Field(description="Data status inherited from Buck metadata.")
    review_status: str = Field(description="Review status inherited from Buck metadata.")
    citation_ready: bool = Field(
        description="Whether this Buck rule summary is citation-ready."
    )


class McpBuckDialectInfo(BaseModel):
    """Public MCP representation of a Buck dialect catalog entry."""

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


class McpBuckGlossaryEntryInfo(BaseModel):
    """Public MCP representation of a Buck glossary example."""

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
    inscription_no: str | None = Field(
        default=None,
        description="Inscription identifier, when stored as a string.",
    )
    buck_ref: McpBuckReferenceInfo | None = Field(
        default=None,
        description="Buck section/page metadata; source text is not included.",
    )
    notes: str | None = Field(default=None, description="Short curation note.")
    status: str = Field(description="Data status inherited from Buck metadata.")
    review_status: str = Field(description="Review status inherited from Buck metadata.")
    citation_ready: bool = Field(
        description="Whether this Buck glossary entry is citation-ready."
    )


class McpBuckRuleSearchInput(BaseModel):
    """Validated input for Buck rule search."""

    model_config = ConfigDict(frozen=True)

    rule_id: str | None = Field(default=None, description="Exact Buck rule id.")
    category: str | None = Field(default=None, description="Exact Buck rule category.")
    dialect: str | None = Field(
        default=None,
        description="Dialect id matched against rule affected_dialects.",
    )
    section: str | int | float | None = Field(
        default=None,
        description="Buck section value; numeric and string values are canonicalized.",
    )
    source_language: str = Field(
        default=_SUPPORTED_BUCK_LANGUAGE,
        description="Language profile for Buck references. Currently ancient_greek only.",
    )
    max_results: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Maximum number of rules to return.",
    )


class McpBuckDialectInput(BaseModel):
    """Validated input for Buck dialect lookup."""

    model_config = ConfigDict(frozen=True)

    dialect_id: str = Field(description="Exact Buck dialect id.")
    include_rules: bool = Field(
        default=True,
        description="Whether to include linked rule details.",
    )
    include_inherited: bool = Field(
        default=True,
        description="Whether included rule details should walk the parent chain.",
    )
    source_language: str = Field(
        default=_SUPPORTED_BUCK_LANGUAGE,
        description="Language profile for Buck references. Currently ancient_greek only.",
    )


class McpBuckGlossarySearchInput(BaseModel):
    """Validated input for Buck glossary search."""

    model_config = ConfigDict(frozen=True)

    word: str | None = Field(
        default=None,
        description="Exact glossary word after service-level NFC normalization.",
    )
    standard_form: str | None = Field(
        default=None,
        description="Exact standard form after service-level NFC normalization.",
    )
    dialect: str | None = Field(default=None, description="Exact Buck dialect id.")
    rule_id: str | None = Field(default=None, description="Exact linked Buck rule id.")
    source_language: str = Field(
        default=_SUPPORTED_BUCK_LANGUAGE,
        description="Language profile for Buck references. Currently ancient_greek only.",
    )
    max_results: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Maximum number of glossary entries to return.",
    )


class McpBuckRulesOutput(BaseModel):
    """Structured MCP response for Buck rule search."""

    rules: list[McpBuckRuleInfo] = Field(description="Matching Buck rule summaries.")
    count: int = Field(ge=0, description="Number of returned rules.")
    metadata: McpBuckMetadata = Field(description="Buck review metadata.")


class McpBuckDialectOutput(BaseModel):
    """Structured MCP response for Buck dialect lookup."""

    dialect: McpBuckDialectInfo = Field(description="Matching Buck dialect.")
    rules: list[McpBuckRuleInfo] = Field(
        default_factory=list,
        description="Buck rule details linked from the dialect catalog.",
    )
    metadata: McpBuckMetadata = Field(description="Buck review metadata.")


class McpBuckGlossaryOutput(BaseModel):
    """Structured MCP response for Buck glossary search."""

    entries: list[McpBuckGlossaryEntryInfo] = Field(
        description="Matching Buck glossary entries."
    )
    count: int = Field(ge=0, description="Number of returned glossary entries.")
    metadata: McpBuckMetadata = Field(description="Buck review metadata.")


def search_buck_rules_for_mcp(
    request: McpBuckRuleSearchInput,
) -> McpBuckRulesOutput:
    """Search Buck-normalized rule summaries for MCP clients."""
    _validate_source_language(request.source_language)
    index = build_buck_reference_index()

    matched_rules = index.find_rules(
        rule_id=request.rule_id,
        section=request.section,
        category=request.category,
        dialect=request.dialect,
    )[: request.max_results]

    return McpBuckRulesOutput(
        rules=[_rule_info(rule) for rule in matched_rules],
        count=len(matched_rules),
        metadata=_metadata_info(index.metadata),
    )


def get_buck_dialect_for_mcp(
    request: McpBuckDialectInput,
) -> McpBuckDialectOutput:
    """Return a Buck dialect catalog entry for MCP clients."""
    _validate_source_language(request.source_language)
    index = build_buck_reference_index()
    dialect = index.get_dialect(request.dialect_id)
    if dialect is None:
        raise ValueError(f"Unknown Buck dialect: {request.dialect_id}")

    rules = (
        index.get_dialect_rules(
            request.dialect_id,
            include_inherited=request.include_inherited,
        )
        if request.include_rules
        else ()
    )
    return McpBuckDialectOutput(
        dialect=_dialect_info(dialect),
        rules=[_rule_info(rule) for rule in rules],
        metadata=_metadata_info(index.metadata),
    )


def search_buck_glossary_for_mcp(
    request: McpBuckGlossarySearchInput,
) -> McpBuckGlossaryOutput:
    """Search Buck-normalized glossary examples for MCP clients."""
    _validate_source_language(request.source_language)
    index = build_buck_reference_index()

    matched_entries = index.find_glossary_entries(
        word=request.word,
        standard_form=request.standard_form,
        dialect=request.dialect,
        rule_id=request.rule_id,
    )[: request.max_results]

    return McpBuckGlossaryOutput(
        entries=[_glossary_entry_info(entry) for entry in matched_entries],
        count=len(matched_entries),
        metadata=_metadata_info(index.metadata),
    )


def register_buck_reference_tools(app: FastMCP) -> None:
    """Register Buck reference tools on ``app``."""

    @app.tool("ancient_phonology.search_buck_rules")
    def ancient_phonology_search_buck_rules(
        request: McpBuckRuleSearchInput,
    ) -> dict[str, Any]:
        """Search provisional Buck rule summaries without returning source text."""
        output = search_buck_rules_for_mcp(request)
        return output.model_dump(mode="json", by_alias=True)

    @app.tool("ancient_phonology.get_buck_dialect")
    def ancient_phonology_get_buck_dialect(
        request: McpBuckDialectInput,
    ) -> dict[str, Any]:
        """Fetch a provisional Buck dialect entry and linked rule summaries."""
        output = get_buck_dialect_for_mcp(request)
        return output.model_dump(mode="json", by_alias=True)

    @app.tool("ancient_phonology.search_buck_glossary")
    def ancient_phonology_search_buck_glossary(
        request: McpBuckGlossarySearchInput,
    ) -> dict[str, Any]:
        """Search provisional Buck glossary examples without source-text quotes."""
        output = search_buck_glossary_for_mcp(request)
        return output.model_dump(mode="json", by_alias=True)


def _validate_source_language(source_language: str) -> None:
    if source_language != _SUPPORTED_BUCK_LANGUAGE:
        raise ValueError(
            "Unsupported Buck reference language: "
            f"{source_language!r}; expected {_SUPPORTED_BUCK_LANGUAGE!r}"
        )


def _metadata_info(metadata: BuckMetadata) -> McpBuckMetadata:
    return McpBuckMetadata(
        status=metadata.status,
        review_status=str(metadata.review_status),
        citation_ready=metadata.citation_ready,
        review_note=(
            _CITATION_READY_REVIEW_NOTE
            if metadata.citation_ready
            else _PROVISIONAL_REVIEW_NOTE
        ),
    )


def _rule_info(rule: BuckRule) -> McpBuckRuleInfo:
    return McpBuckRuleInfo(
        id=rule.id,
        buck_section=rule.buck_section,
        category=rule.category,
        description=rule.description,
        transformation=_transformation_info(rule.transformation),
        affected_dialects=list(rule.affected_dialects),
        variants=[
            McpBuckVariant.model_validate(_json_mapping(variant))
            for variant in rule.variants
        ],
        notes=rule.notes,
        status=rule.status,
        review_status=str(rule.review_status),
        citation_ready=rule.citation_ready,
    )


def _transformation_info(
    transformation: Mapping[str, Any],
) -> McpBuckTransformation | None:
    """Build a typed transformation, or ``None`` when the rule has no data."""
    if not transformation:
        return None
    return McpBuckTransformation.model_validate(_json_mapping(transformation))


def _dialect_info(dialect: BuckDialect) -> McpBuckDialectInfo:
    return McpBuckDialectInfo(
        id=dialect.id,
        name=dialect.name,
        kind=dialect.kind,
        group=dialect.group,
        parent=dialect.parent,
        rules=list(dialect.rules),
        status=dialect.status,
        review_status=str(dialect.review_status),
        citation_ready=dialect.citation_ready,
    )


def _glossary_entry_info(entry: BuckGlossaryEntry) -> McpBuckGlossaryEntryInfo:
    return McpBuckGlossaryEntryInfo(
        word=entry.word,
        standard_form=entry.standard_form,
        dialect=entry.dialect,
        rule_id=entry.rule_id,
        definition=entry.definition,
        inscription_no=entry.inscription_no,
        buck_ref=_reference_info(entry.buck_ref),
        notes=entry.notes,
        status=entry.status,
        review_status=str(entry.review_status),
        citation_ready=entry.citation_ready,
    )


def _reference_info(
    reference: BuckReference | None,
) -> McpBuckReferenceInfo | None:
    if reference is None:
        return None
    return McpBuckReferenceInfo(section=reference.section, page=reference.page)


def _json_mapping(raw_mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _json_value(value) for key, value in raw_mapping.items()}


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _json_mapping(value)
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


__all__ = [
    "McpBuckDialectInfo",
    "McpBuckDialectInput",
    "McpBuckDialectOutput",
    "McpBuckGlossaryEntryInfo",
    "McpBuckGlossaryOutput",
    "McpBuckGlossarySearchInput",
    "McpBuckMetadata",
    "McpBuckReferenceInfo",
    "McpBuckRuleInfo",
    "McpBuckRuleSearchInput",
    "McpBuckRulesOutput",
    "McpBuckTransformation",
    "McpBuckVariant",
    "get_buck_dialect_for_mcp",
    "register_buck_reference_tools",
    "search_buck_glossary_for_mcp",
    "search_buck_rules_for_mcp",
]
