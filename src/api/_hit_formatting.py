"""Hit-formatting helpers that convert core search results to API response shapes."""

from __future__ import annotations

import logging
from typing import Any, Literal

from phonology.explainer import Explanation, RuleApplication, explain_alignment, to_prose
from phonology.ipa_converter import strip_ignored_ipa_combining_marks
from phonology import search as phonology_search
from phonology.search._constants import OBSERVED_PREFIX

from ._models import RuleStep, SearchHit

logger = logging.getLogger(__name__)

_OBSERVED_RULE_PREFIX = OBSERVED_PREFIX
# Gate for match_type classification: below this → "Low-confidence".
_LOW_CONFIDENCE_THRESHOLD = 0.55
_HIGH_SIMILARITY_THRESHOLD = 0.80
# Gate for uncertainty buckets in _build_uncertainty().
_MEDIUM_SIMILARITY_THRESHOLD = 0.70
# Gate for user-facing similarity message in _similarity_line().
# Same numeric value as _LOW_CONFIDENCE_THRESHOLD by design; kept
# separate because the two thresholds may diverge independently.
_MODERATE_SIMILARITY_THRESHOLD = 0.55

FallbackEditLabel = Literal["substitution", "deletion", "insertion"]


def _distance_from_confidence(confidence: float) -> float:
    """Convert a normalized confidence score into normalized distance."""
    return max(0.0, min(1.0, 1.0 - confidence))


def _same_ipa_ignoring_accents(source_ipa: str, query_ipa: str) -> bool:
    """Return whether two IPA strings match after removing ignored accent marks."""
    return strip_ignored_ipa_combining_marks(
        source_ipa.strip()
    ) == strip_ignored_ipa_combining_marks(query_ipa.strip())


def _is_observed_rule_step(step: RuleApplication) -> bool:
    """Return whether a rule application is an observed uncatalogued change."""
    return step.rule_id.startswith(_OBSERVED_RULE_PREFIX)


def _count_explicit_and_observed_steps(
    steps: list[RuleApplication],
) -> tuple[int, int]:
    """Return counts for explicit catalogued rules and observed changes."""
    explicit_count = sum(1 for step in steps if not _is_observed_rule_step(step))
    observed_count = len(steps) - explicit_count
    return explicit_count, observed_count


def _build_match_type(
    *,
    source_ipa: str,
    query_ipa: str,
    steps: list[RuleApplication],
    applied_rule_count: int,
    confidence: float,
) -> Literal["Exact", "Rule-based", "Distance-only", "Low-confidence"]:
    """Classify the search hit for UI display."""
    # Check ``steps`` (not ``applied_rule_count``) so that entries whose
    # IPA happens to match but still carry OBS-prefixed observed changes
    # are not mis-classified as "Exact".
    if _same_ipa_ignoring_accents(source_ipa, query_ipa) and not steps:
        return "Exact"
    if applied_rule_count > 0:
        return "Rule-based"
    if confidence < _LOW_CONFIDENCE_THRESHOLD:
        return "Low-confidence"
    return "Distance-only"


def _build_uncertainty(
    match_type: Literal["Exact", "Rule-based", "Distance-only", "Low-confidence"],
    *,
    applied_rule_count: int,
    confidence: float,
) -> Literal["Low", "Medium", "High"]:
    """Return an uncertainty bucket for a search hit."""
    if match_type == "Exact" or (
        applied_rule_count > 0 and confidence >= _HIGH_SIMILARITY_THRESHOLD
    ):
        return "Low"
    if confidence >= _MEDIUM_SIMILARITY_THRESHOLD:
        return "Medium"
    return "High"


def _build_candidate_bucket(
    match_type: Literal["Exact", "Rule-based", "Distance-only", "Low-confidence"],
    *,
    query_mode: Literal["Full-form", "Short-query", "Partial-form"],
    uncertainty: Literal["Low", "Medium", "High"],
) -> Literal["Supported", "Exploratory"]:
    """Return the default UI grouping bucket for a search hit."""
    if query_mode in {"Short-query", "Partial-form"}:
        if match_type in {"Exact", "Rule-based"}:
            return "Supported"
        return "Exploratory"
    if match_type in {"Exact", "Rule-based"} or uncertainty == "Low":
        return "Supported"
    return "Exploratory"


def _count_distinct_positions(steps: list[RuleApplication]) -> int | None:
    """Count distinct known alignment positions, or None if all are unknown."""
    if not steps:
        return 0
    known = {step.position for step in steps if step.position >= 0}
    if not known:
        return None
    return len(known)


def _fallback_edit_label(step: RuleApplication) -> FallbackEditLabel:
    """Return the edit type label for an observed uncatalogued step."""
    if not step.from_phone and not step.to_phone:
        # Should not occur for valid observed steps; default to substitution
        logger.warning(
            "RuleApplication %s has neither from_phone nor to_phone (from_phone=%r, to_phone=%r); "
            "defaulting to substitution",
            step.rule_id,
            step.from_phone,
            step.to_phone,
        )
        return "substitution"
    if step.from_phone and step.to_phone:
        return "substitution"
    if step.from_phone:
        return "deletion"
    return "insertion"


def _format_counted_noun(count: int, singular: str, plural: str | None = None) -> str:
    """Format a counted noun phrase with simple pluralization."""
    noun = singular if count == 1 else (plural or singular + "s")
    return f"{count} {noun}"


def _format_counted_phrase(
    count: int,
    singular: str,
    singular_verb: str,
    plural_verb: str,
    plural: str | None = None,
) -> tuple[str, str]:
    """Return a counted noun phrase and the verb agreeing with its count."""
    verb = singular_verb if count == 1 else plural_verb
    return _format_counted_noun(count, singular, plural), verb


def _format_position_summary(steps: list[RuleApplication]) -> str:
    """Describe whether the affected positions are known or unknown."""
    distinct_positions = _count_distinct_positions(steps)
    unknown_positions = sum(1 for step in steps if step.position < 0)
    if distinct_positions is None:
        if unknown_positions == 1:
            return "at an unknown position."
        return f"at {unknown_positions} unknown positions."
    if unknown_positions > 0:
        return (
            f"across {_format_counted_noun(distinct_positions, 'known position')} "
            f"and {_format_counted_noun(unknown_positions, 'unknown position')}."
        )
    return f"across {_format_counted_noun(distinct_positions, 'position')}."


def _build_alignment_summary(
    *,
    source_ipa: str,
    query_ipa: str,
    steps: list[RuleApplication],
) -> str:
    """Build a short alignment summary suitable for the result card body."""
    if _same_ipa_ignoring_accents(source_ipa, query_ipa) and not steps:
        return "No phonological difference."
    if not steps:
        return "Differences visible in full alignment."

    explicit_steps = [step for step in steps if not _is_observed_rule_step(step)]
    observed_steps = [step for step in steps if _is_observed_rule_step(step)]

    position_summary = _format_position_summary(steps)

    if explicit_steps and observed_steps:
        return (
            f"{_format_counted_noun(len(explicit_steps), 'matched rule')} and "
            f"{_format_counted_noun(len(observed_steps), 'fallback edit')} "
            f"{position_summary}"
        )

    if explicit_steps:
        return (
            f"{_format_counted_noun(len(explicit_steps), 'matched rule')} "
            f"{position_summary}"
        )

    if len(observed_steps) == 1:
        return f"1 fallback edit {position_summary}"

    operation_counts: dict[FallbackEditLabel, int] = {
        "deletion": 0,
        "insertion": 0,
        "substitution": 0,
    }
    for step in observed_steps:
        operation_counts[_fallback_edit_label(step)] += 1

    operation_summary = ", ".join(
        _format_counted_noun(operation_counts[label], label)
        for label in ("deletion", "insertion", "substitution")
        if operation_counts[label] > 0
    )
    return f"{operation_summary} {position_summary}"


def _similarity_line(confidence: float) -> str:
    """Return a short qualitative similarity statement."""
    if confidence >= _HIGH_SIMILARITY_THRESHOLD:
        return "High phonological similarity."
    if confidence >= _MODERATE_SIMILARITY_THRESHOLD:
        return "Moderate phonological similarity."
    return "Weak phonological similarity."


def _build_why_candidate(
    match_type: Literal["Exact", "Rule-based", "Distance-only", "Low-confidence"],
    *,
    applied_rule_count: int,
    observed_change_count: int,
    confidence: float,
) -> list[str]:
    """Build stable short bullet points describing why this candidate ranked."""
    if match_type == "Exact":
        first_line = "Exact phonological match."
    elif applied_rule_count > 0:
        phrase, verb = _format_counted_phrase(
            applied_rule_count,
            "explicit rule",
            "supports",
            "support",
        )
        first_line = f"{phrase} {verb} the match."
    else:
        first_line = "Ranked by phonological distance without explicit rule support."

    if match_type == "Exact":
        third_line = "No remaining unexplained differences."
    elif observed_change_count > 0:
        phrase, verb = _format_counted_phrase(
            observed_change_count,
            "fallback edit",
            "remains",
            "remain",
        )
        third_line = f"{phrase} {verb} uncatalogued."
    elif applied_rule_count > 0:
        third_line = "No fallback edits required."
    else:
        third_line = "See alignment for localized differences."

    return [
        first_line,
        _similarity_line(confidence),
        third_line,
    ]


def _build_search_hit(
    result: phonology_search.SearchResult,
    query_ipa: str,
    rules_registry: dict[str, dict[str, Any]],
    query_mode: Literal["Full-form", "Short-query", "Partial-form"],
) -> SearchHit:
    """Convert a core search result into the public API response shape."""
    source_ipa = result.ipa or ""
    # Both `distance` and `confidence` are included in the API response so
    # that consumers can choose whichever convention they prefer (lower-is-
    # better or higher-is-better).  They are strict inverses:
    # confidence = 1.0 - distance.
    distance = _distance_from_confidence(result.confidence)
    if result.rule_applications:
        explanation = Explanation(
            source=source_ipa,
            target=query_ipa,
            source_ipa=source_ipa,
            target_ipa=query_ipa,
            distance=distance,
            steps=list(result.rule_applications),
        )
    else:
        explanation = explain_alignment(
            source_ipa=source_ipa,
            target_ipa=query_ipa,
            rule_ids=result.applied_rules,
            all_rules=rules_registry,
            distance=distance,
        )
    steps = list(explanation.steps)
    applied_rule_count, observed_change_count = _count_explicit_and_observed_steps(steps)
    match_type = _build_match_type(
        source_ipa=source_ipa,
        query_ipa=query_ipa,
        steps=steps,
        applied_rule_count=applied_rule_count,
        confidence=result.confidence,
    )
    uncertainty = _build_uncertainty(
        match_type,
        applied_rule_count=applied_rule_count,
        confidence=result.confidence,
    )
    return SearchHit(
        headword=result.lemma,
        ipa=source_ipa,
        distance=distance,
        confidence=result.confidence,
        dialect_attribution=result.dialect_attribution or "",
        alignment_visualization=result.alignment_visualization or "",
        match_type=match_type,
        rule_support=applied_rule_count > 0,
        applied_rule_count=applied_rule_count,
        observed_change_count=observed_change_count,
        alignment_summary=_build_alignment_summary(
            source_ipa=source_ipa,
            query_ipa=query_ipa,
            steps=steps,
        ),
        why_candidate=_build_why_candidate(
            match_type,
            applied_rule_count=applied_rule_count,
            observed_change_count=observed_change_count,
            confidence=result.confidence,
        ),
        uncertainty=uncertainty,
        candidate_bucket=_build_candidate_bucket(
            match_type,
            query_mode=query_mode,
            uncertainty=uncertainty,
        ),
        rules_applied=[
            RuleStep(
                rule_id=step.rule_id,
                rule_name=step.rule_name,
                rule_name_en=step.rule_name_en,
                from_phone=step.from_phone,
                to_phone=step.to_phone,
                position=step.position,
            )
            for step in steps
        ],
        explanation=to_prose(explanation),
    )
