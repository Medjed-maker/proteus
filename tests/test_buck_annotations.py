"""Tests for Buck reference annotations on search hits."""

from __future__ import annotations

import logging

import pytest

from api import _buck_annotations
from api._buck_annotations import (
    build_buck_annotation_context,
    buck_references_for_rule_steps,
)
from api._models import RuleStep


def test_build_context_maps_single_buck_sections_from_rule_references() -> None:
    context = build_buck_annotation_context(
        {
            "VSH-001": {
                "id": "VSH-001",
                "references": ["Buck §9", "Smyth §31"],
            }
        },
        logger=logging.getLogger(__name__),
    )

    annotations = buck_references_for_rule_steps(
        context,
        [
            RuleStep(
                rule_id="VSH-001",
                rule_name="Ionic long alpha to eta shift",
                from_phone="aː",
                to_phone="ɛː",
                position=0,
            )
        ],
    )

    assert context.metadata is not None
    assert context.metadata.review_status == "not_expert_reviewed"
    assert context.metadata.citation_ready is False
    assert [annotation.buck_rule_id for annotation in annotations] == ["grc_phon_9"]
    assert annotations[0].source_rule_id == "VSH-001"
    assert annotations[0].buck_section == "9"
    assert annotations[0].citation_ready is False


def test_build_context_canonicalizes_decimal_sections() -> None:
    context = build_buck_annotation_context(
        {"MPH-020": {"id": "MPH-020", "references": ["Buck §138.4"]}},
        logger=logging.getLogger(__name__),
    )

    annotations = context.references_by_rule_id["MPH-020"]

    assert [annotation.buck_section for annotation in annotations] == ["138.4"]
    assert [annotation.buck_rule_id for annotation in annotations] == [
        "grc_morph_138_4"
    ]


@pytest.mark.parametrize(
    ("raw_references", "expected_sections"),
    [
        (["Buck §9"], ("9",)),
        (["Buck §138.4"], ("138.4",)),
        (["Buck § 41"], ("41",)),
        (["see Buck §25 and Buck §32"], ("25", "32")),
        (["Buck §63-65"], ()),
        (["Buck §132-135"], ()),
        (["Buck §41-43"], ()),
        (["Buck §54.3-55"], ()),
        (["Buck §§63-65"], ()),
        (["Smyth §31"], ()),
    ],
)
def test_buck_sections_from_references_rejects_range_prefixes(
    raw_references: list[str],
    expected_sections: tuple[str, ...],
) -> None:
    """Range references must not leak partial section matches via backtracking."""
    sections = _buck_annotations._buck_sections_from_references(raw_references)

    assert sections == expected_sections


@pytest.mark.parametrize(
    "reference",
    [
        "Smyth §31",
        "Buck §§63-65",
        "Buck §63-65",
        # §13 exists in the Buck data, so a partial match on "Buck §132-135"
        # would wrongly attach the §13 rule. Guards against regex backtracking.
        "Buck §132-135",
    ],
)
def test_build_context_ignores_non_buck_and_broad_references(reference: str) -> None:
    context = build_buck_annotation_context(
        {"RULE-001": {"id": "RULE-001", "references": [reference]}},
        logger=logging.getLogger(__name__),
    )

    assert context.references_by_rule_id == {}


def test_buck_references_for_rule_steps_dedupes_rule_ids() -> None:
    context = build_buck_annotation_context(
        {"VSH-001": {"id": "VSH-001", "references": ["Buck §9"]}},
        logger=logging.getLogger(__name__),
    )
    steps = [
        RuleStep(
            rule_id="VSH-001",
            rule_name="first",
            from_phone="a",
            to_phone="e",
            position=0,
        ),
        RuleStep(
            rule_id="VSH-001",
            rule_name="second",
            from_phone="a",
            to_phone="e",
            position=1,
        ),
    ]

    annotations = buck_references_for_rule_steps(context, steps)

    assert [annotation.buck_rule_id for annotation in annotations] == ["grc_phon_9"]


def test_build_context_returns_empty_when_buck_service_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fail() -> object:
        raise OSError("buck data unavailable")

    monkeypatch.setattr(_buck_annotations, "build_buck_reference_index", fail)

    with caplog.at_level(logging.WARNING):
        context = build_buck_annotation_context(
            {"VSH-001": {"id": "VSH-001", "references": ["Buck §9"]}},
            logger=logging.getLogger(__name__),
        )

    assert context.metadata is None
    assert context.references_by_rule_id == {}
    assert "Buck reference annotation disabled" in caplog.text
