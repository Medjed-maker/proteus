"""Buck reference-data REST endpoints.

Groups the read-only Buck rule/dialect/glossary routes behind an
``APIRouter`` so ``main.py`` stays focused on application assembly. Shared
review-note constants and pure dataclass-to-JSON conversions live in
``_buck_conversion``; this module only maps service dataclasses onto the public
Pydantic response models.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypeVar

from fastapi import APIRouter, HTTPException, Query

from phonology.languages.ancient_greek.buck_service import (
    BuckDialect as ServiceBuckDialect,
    BuckGlossaryEntry as ServiceBuckGlossaryEntry,
    BuckMetadata as ServiceBuckMetadata,
    BuckReference as ServiceBuckReference,
    BuckRule as ServiceBuckRule,
    build_buck_reference_index,
)

from ._buck_conversion import (
    SUPPORTED_BUCK_LANGUAGE,
    inscription_number_as_json,
    json_mapping,
    review_note_for,
)
from ._models import (
    BuckDialectInfo,
    BuckDialectResponse,
    BuckDialectsResponse,
    BuckGlossaryEntryInfo,
    BuckGlossaryResponse,
    BuckMetadata,
    BuckReferenceInfo,
    BuckRuleInfo,
    BuckRuleResponse,
    BuckRulesResponse,
    BuckTransformation,
    BuckVariant,
    ErrorResponse,
)

_T = TypeVar("_T")

_BUCK_NOT_FOUND_RESPONSE = {
    "model": ErrorResponse,
    "description": "Buck reference data is not available or the resource was not found.",
}
_BUCK_BAD_REQUEST_RESPONSE = {
    "model": ErrorResponse,
    "description": "Buck reference request parameters are invalid.",
}

router = APIRouter()


def _ensure_buck_language(language: str) -> None:
    """Reject languages that have no Buck reference-data integration."""
    if language.strip().lower() != SUPPORTED_BUCK_LANGUAGE:
        raise HTTPException(
            status_code=404,
            detail=f"Buck reference data is not available for language {language!r}",
        )


def _buck_metadata_info(metadata: ServiceBuckMetadata) -> BuckMetadata:
    """Convert service metadata to the API metadata response model."""
    return BuckMetadata(
        status=metadata.status,
        review_status=str(metadata.review_status),
        citation_ready=metadata.citation_ready,
        review_note=review_note_for(metadata.citation_ready),
    )


def _buck_rule_info(rule: ServiceBuckRule) -> BuckRuleInfo:
    """Convert a service Buck rule to the API rule response model."""
    return BuckRuleInfo(
        id=rule.id,
        buck_section=rule.buck_section,
        category=rule.category,
        description=rule.description,
        transformation=_buck_transformation_info(rule.transformation),
        affected_dialects=list(rule.affected_dialects),
        variants=[
            BuckVariant.model_validate(json_mapping(variant))
            for variant in rule.variants
        ],
        notes=rule.notes,
        status=rule.status,
        review_status=str(rule.review_status),
        citation_ready=rule.citation_ready,
    )


def _buck_transformation_info(
    transformation: Mapping[str, Any],
) -> BuckTransformation | None:
    """Convert transformation metadata to an API model, or ``None`` if empty."""
    if not transformation:
        return None
    return BuckTransformation.model_validate(json_mapping(transformation))


def _buck_dialect_info(dialect: ServiceBuckDialect) -> BuckDialectInfo:
    """Convert a service Buck dialect to the API dialect response model."""
    return BuckDialectInfo(
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


def _buck_glossary_entry_info(
    entry: ServiceBuckGlossaryEntry,
) -> BuckGlossaryEntryInfo:
    """Convert a service glossary entry to the API glossary response model."""
    return BuckGlossaryEntryInfo(
        word=entry.word,
        standard_form=entry.standard_form,
        dialect=entry.dialect,
        rule_id=entry.rule_id,
        definition=entry.definition,
        inscription_no=inscription_number_as_json(entry.inscription_no),
        buck_ref=_buck_reference_info(entry.buck_ref),
        notes=entry.notes,
        status=entry.status,
        review_status=str(entry.review_status),
        citation_ready=entry.citation_ready,
    )


def _buck_reference_info(
    reference: ServiceBuckReference | None,
) -> BuckReferenceInfo | None:
    """Convert an optional service reference to the API reference model."""
    if reference is None:
        return None
    return BuckReferenceInfo(section=reference.section, page=reference.page)


def _page_items(items: Sequence[_T], *, limit: int, offset: int) -> Sequence[_T]:
    """Return a paginated slice of ``items`` starting at ``offset``.

    Args:
        items: Sequence to paginate.
        limit: Maximum number of items to return.
        offset: Zero-based starting position.

    Returns:
        A sliced sequence containing at most ``limit`` items.
    """
    return items[offset : offset + limit]


@router.get(
    "/languages/{language}/buck/rules",
    response_model=BuckRulesResponse,
    responses={400: _BUCK_BAD_REQUEST_RESPONSE, 404: _BUCK_NOT_FOUND_RESPONSE},
)
async def buck_rules(
    language: str,
    category: str | None = None,
    dialect: str | None = None,
    section: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> BuckRulesResponse:
    """Return Buck-normalized rule summaries for a supported language."""
    _ensure_buck_language(language)
    index = build_buck_reference_index()
    try:
        rules = index.find_rules(category=category, dialect=dialect, section=section)
    except (TypeError, ValueError) as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    page = _page_items(rules, limit=limit, offset=offset)
    return BuckRulesResponse(
        rules=[_buck_rule_info(rule) for rule in page],
        count=len(page),
        total=len(rules),
        metadata=_buck_metadata_info(index.metadata),
    )


@router.get(
    "/languages/{language}/buck/rules/{rule_id}",
    response_model=BuckRuleResponse,
    responses={404: _BUCK_NOT_FOUND_RESPONSE},
)
async def buck_rule(language: str, rule_id: str) -> BuckRuleResponse:
    """Return one Buck-normalized rule summary by id."""
    _ensure_buck_language(language)
    index = build_buck_reference_index()
    rule = index.get_rule(rule_id)
    if rule is None:
        raise HTTPException(status_code=404, detail=f"Unknown Buck rule: {rule_id}")
    return BuckRuleResponse(
        rule=_buck_rule_info(rule),
        metadata=_buck_metadata_info(index.metadata),
    )


@router.get(
    "/languages/{language}/buck/dialects",
    response_model=BuckDialectsResponse,
    responses={404: _BUCK_NOT_FOUND_RESPONSE},
)
async def buck_dialects(
    language: str,
    kind: str | None = None,
    group: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> BuckDialectsResponse:
    """Return Buck dialect catalog entries for a supported language."""
    _ensure_buck_language(language)
    index = build_buck_reference_index()
    dialects = index.list_dialects(kind=kind, group=group)
    page = _page_items(dialects, limit=limit, offset=offset)
    return BuckDialectsResponse(
        dialects=[_buck_dialect_info(dialect) for dialect in page],
        count=len(page),
        total=len(dialects),
        metadata=_buck_metadata_info(index.metadata),
    )


@router.get(
    "/languages/{language}/buck/dialects/{dialect_id}",
    response_model=BuckDialectResponse,
    responses={404: _BUCK_NOT_FOUND_RESPONSE},
)
async def buck_dialect(
    language: str,
    dialect_id: str,
    include_rules: bool = True,
    include_inherited: bool = True,
) -> BuckDialectResponse:
    """Return one Buck dialect entry and optional linked rule summaries."""
    _ensure_buck_language(language)
    index = build_buck_reference_index()
    dialect = index.get_dialect(dialect_id)
    if dialect is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown Buck dialect: {dialect_id}",
        )
    rules = (
        index.get_dialect_rules(dialect_id, include_inherited=include_inherited)
        if include_rules
        else ()
    )
    return BuckDialectResponse(
        dialect=_buck_dialect_info(dialect),
        rules=[_buck_rule_info(rule) for rule in rules],
        metadata=_buck_metadata_info(index.metadata),
    )


@router.get(
    "/languages/{language}/buck/glossary",
    response_model=BuckGlossaryResponse,
    responses={400: _BUCK_BAD_REQUEST_RESPONSE, 404: _BUCK_NOT_FOUND_RESPONSE},
)
async def buck_glossary(
    language: str,
    word: str | None = None,
    standard_form: str | None = None,
    dialect: str | None = None,
    rule_id: str | None = None,
    accent_insensitive: bool = False,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> BuckGlossaryResponse:
    """Return Buck-normalized glossary examples for a supported language."""
    _ensure_buck_language(language)
    if accent_insensitive:
        raise HTTPException(
            status_code=400,
            detail=(
                "accent_insensitive glossary search is not supported yet; "
                "Buck glossary search uses NFC exact matching"
            ),
        )
    index = build_buck_reference_index()
    entries = index.find_glossary_entries(
        word=word,
        standard_form=standard_form,
        dialect=dialect,
        rule_id=rule_id,
    )
    page = _page_items(entries, limit=limit, offset=offset)
    return BuckGlossaryResponse(
        entries=[_buck_glossary_entry_info(entry) for entry in page],
        count=len(page),
        total=len(entries),
        metadata=_buck_metadata_info(index.metadata),
    )
