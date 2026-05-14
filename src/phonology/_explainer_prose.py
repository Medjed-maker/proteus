"""Prose rendering for structured phonological explanations."""

from __future__ import annotations

from dataclasses import dataclass
import math

from ._explainer_types import RuleApplication


@dataclass
class Explanation:
    """Full explanation for a source -> target derivation."""

    source: str
    target: str
    source_ipa: str
    target_ipa: str
    distance: float
    steps: list[RuleApplication]
    prose: str = ""


def to_prose(explanation: Explanation) -> str:
    """Generate canonical prose for a structured explanation.

    Args:
        explanation: Structured explanation object to render.

    Returns:
        A prose summary string containing the source/target alignment,
        normalized distance, and any applied rules.  When no rules were
        recorded, the summary includes a "No rule applications" note.
    """

    def format_with_optional_ipa(text: str, ipa: str) -> str:
        """Render text with IPA only when it differs from the surface form."""
        if text == ipa:
            return text
        return f"{text} /{ipa}/"

    # Dialect labels are omitted from prose, but step weights are retained.
    source_repr = format_with_optional_ipa(
        explanation.source,
        explanation.source_ipa,
    )
    target_repr = format_with_optional_ipa(
        explanation.target,
        explanation.target_ipa,
    )

    if explanation.steps:
        step_summary = "; ".join(
            (
                f"{step.rule_name} ({step.from_phone} -> {step.to_phone} "
                f"at position {step.position}, weight {step.weight:g})"
            )
            for step in explanation.steps
        )
        prose = (
            f"{source_repr} aligns to "
            f"{target_repr} with distance "
            f"{explanation.distance:.3f}. Applied rules: {step_summary}."
        )
    else:
        if math.isclose(explanation.distance, 0.0, abs_tol=1e-9):
            prose = f"{source_repr} is an exact match for {target_repr}."
        else:
            prose = (
                f"{source_repr} aligns to "
                f"{target_repr} with distance "
                f"{explanation.distance:.3f}. No rule applications were recorded."
            )

    return prose
