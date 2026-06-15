"""Buck reference annotation helpers for search responses."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Any

from phonology.languages.ancient_greek.buck_service import (
    BuckMetadata,
    BuckRule,
    build_buck_reference_index,
    canonicalize_buck_section,
)

from ._buck_conversion import review_note_for
from ._models import BuckReferenceAnnotation, BuckReferenceMetadata, RuleStep

# ``(?!\.?\d)`` rejects partial matches produced by backtracking: without it,
# a range such as "Buck §132-135" would match the section prefix "13".
_BUCK_SINGLE_SECTION_RE = re.compile(
    r"\bBuck\s+§(?!§)\s*(\d+(?:\.\d+)?)(?!\.?\d)(?!\s*[-–—])"
)


@dataclass(frozen=True)
class BuckAnnotationContext:
    """Precomputed Buck annotations keyed by applied Proteus rule id."""

    metadata: BuckReferenceMetadata | None
    references_by_rule_id: Mapping[str, tuple[BuckReferenceAnnotation, ...]]


EMPTY_BUCK_ANNOTATION_CONTEXT = BuckAnnotationContext(
    metadata=None,
    references_by_rule_id=MappingProxyType({}),
)


def build_buck_annotation_context(
    rules_registry: Mapping[str, Mapping[str, Any]],
    *,
    logger: logging.Logger,
) -> BuckAnnotationContext:
    """Build Buck annotations from rule metadata without affecting search ranking."""
    sections_by_rule_id = _buck_sections_by_source_rule(rules_registry)
    if not sections_by_rule_id:
        return EMPTY_BUCK_ANNOTATION_CONTEXT

    try:
        index = build_buck_reference_index()
    except Exception as exc:  # pragma: no cover - exact failure types vary by loader
        logger.warning(
            "Buck reference annotation disabled: %s",
            exc,
            exc_info=True,
        )
        return EMPTY_BUCK_ANNOTATION_CONTEXT

    annotations_by_rule_id: dict[str, tuple[BuckReferenceAnnotation, ...]] = {}
    for source_rule_id, sections in sections_by_rule_id.items():
        annotations = _annotations_for_source_rule(
            source_rule_id=source_rule_id,
            sections=sections,
            rules_by_section=index.get_rules_by_section,
        )
        if annotations:
            annotations_by_rule_id[source_rule_id] = annotations

    return BuckAnnotationContext(
        metadata=_metadata_info(index.metadata),
        references_by_rule_id=MappingProxyType(annotations_by_rule_id),
    )


def buck_references_for_rule_steps(
    context: BuckAnnotationContext,
    rule_steps: Sequence[RuleStep],
) -> list[BuckReferenceAnnotation]:
    """Return deduped Buck references for the given applied rule steps."""
    references: list[BuckReferenceAnnotation] = []
    seen_rule_ids: set[str] = set()
    seen_annotations: set[tuple[str, str]] = set()
    for step in rule_steps:
        if step.rule_id in seen_rule_ids:
            continue
        seen_rule_ids.add(step.rule_id)
        for annotation in context.references_by_rule_id.get(step.rule_id, ()):
            key = (annotation.source_rule_id, annotation.buck_rule_id)
            if key in seen_annotations:
                continue
            seen_annotations.add(key)
            references.append(annotation)
    return references


def _annotations_for_source_rule(
    *,
    source_rule_id: str,
    sections: Sequence[str],
    rules_by_section: Callable[[str], tuple[BuckRule, ...]],
) -> tuple[BuckReferenceAnnotation, ...]:
    annotations: list[BuckReferenceAnnotation] = []
    seen_buck_rule_ids: set[str] = set()
    for section in sections:
        for buck_rule in rules_by_section(section):
            if buck_rule.id in seen_buck_rule_ids:
                continue
            seen_buck_rule_ids.add(buck_rule.id)
            annotations.append(
                _annotation_info(
                    source_rule_id=source_rule_id,
                    rule=buck_rule,
                )
            )
    return tuple(annotations)


def _buck_sections_by_source_rule(
    rules_registry: Mapping[str, Mapping[str, Any]],
) -> dict[str, tuple[str, ...]]:
    sections_by_rule_id: dict[str, tuple[str, ...]] = {}
    for source_rule_id, rule in rules_registry.items():
        sections = _buck_sections_from_references(rule.get("references"))
        if sections:
            sections_by_rule_id[source_rule_id] = sections
    return sections_by_rule_id


def _buck_sections_from_references(raw_references: object) -> tuple[str, ...]:
    if not isinstance(raw_references, Iterable) or isinstance(
        raw_references,
        (str, bytes),
    ):
        return ()

    sections: list[str] = []
    seen: set[str] = set()
    for raw_reference in raw_references:
        if not isinstance(raw_reference, str):
            continue
        for match in _BUCK_SINGLE_SECTION_RE.finditer(raw_reference):
            section = canonicalize_buck_section(match.group(1))
            if section in seen:
                continue
            seen.add(section)
            sections.append(section)
    return tuple(sections)


def _metadata_info(metadata: BuckMetadata) -> BuckReferenceMetadata:
    return BuckReferenceMetadata(
        status=metadata.status,
        review_status=str(metadata.review_status),
        citation_ready=metadata.citation_ready,
        review_note=review_note_for(metadata.citation_ready),
    )


def _annotation_info(
    *,
    source_rule_id: str,
    rule: BuckRule,
) -> BuckReferenceAnnotation:
    return BuckReferenceAnnotation(
        source_rule_id=source_rule_id,
        buck_rule_id=rule.id,
        buck_section=rule.buck_section,
        category=rule.category,
        description=rule.description,
        affected_dialects=list(rule.affected_dialects),
        status=rule.status,
        review_status=str(rule.review_status),
        citation_ready=rule.citation_ready,
        review_note=review_note_for(rule.citation_ready),
    )


__all__ = [
    "BuckAnnotationContext",
    "EMPTY_BUCK_ANNOTATION_CONTEXT",
    "buck_references_for_rule_steps",
    "build_buck_annotation_context",
]
