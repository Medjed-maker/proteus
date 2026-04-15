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

# ---------------------------------------------------------------------------
# Localization: prose string templates keyed by language code.
# ---------------------------------------------------------------------------
_PROSE: dict[str, dict[str, str]] = {
    "en": {
        # alignment_summary — exact match
        "no_diff": "No phonological difference.",
        "diff_visible": "Differences visible in full alignment.",
        # position summary suffixes
        "pos_unknown_one": "at an unknown position.",
        "pos_unknown_many": "at {n} unknown positions.",
        "pos_known_one": "across 1 position.",
        "pos_known_many": "across {n} positions.",
        "pos_mixed": "across {known} {known_noun} and {unknown} {unknown_noun}.",
        # counted noun singular/plural
        "rule_s": "matched rule",
        "rule_p": "matched rules",
        "edit_s": "fallback edit",
        "edit_p": "fallback edits",
        "deletion_s": "deletion",
        "deletion_p": "deletions",
        "insertion_s": "insertion",
        "insertion_p": "insertions",
        "substitution_s": "substitution",
        "substitution_p": "substitutions",
        "expl_rule_s": "explicit rule",
        "expl_rule_p": "explicit rules",
        "position_s": "position",
        "position_p": "positions",
        "known_position_s": "known position",
        "known_position_p": "known positions",
        "unknown_position_s": "unknown position",
        "unknown_position_p": "unknown positions",
        # verb agreement
        "verb_supports": "supports",
        "verb_support": "support",
        "verb_remains": "remains",
        "verb_remain": "remain",
        # alignment_summary combined phrase templates
        "summary_mixed": "{rules} and {edits} {pos}",
        "summary_rules_only": "{rules} {pos}",
        "summary_one_edit": "1 fallback edit {pos}",
        "summary_multi_edit": "{ops} {pos}",
        # similarity lines
        "sim_high": "High phonological similarity.",
        "sim_moderate": "Moderate phonological similarity.",
        "sim_weak": "Weak phonological similarity.",
        # why_candidate first line
        "why_exact": "Exact phonological match.",
        "why_rules": "{phrase} {verb} the match.",
        "why_distance": "Ranked by phonological distance without explicit rule support.",
        # why_candidate third line
        "why_no_diff": "No remaining unexplained differences.",
        "why_fallback": "{phrase} {verb} uncatalogued.",
        "why_no_fallback": "No fallback edits required.",
        "why_see_alignment": "See alignment for localized differences.",
    },
    "ja": {
        # alignment_summary — exact match
        "no_diff": "音韻的差異なし。",
        "diff_visible": "完全なアラインメントで差異を確認できます。",
        # position summary suffixes (Japanese: parenthetical style)
        "pos_unknown_one": "（位置不明）",
        "pos_unknown_many": "（{n}箇所・位置不明）",
        "pos_known_one": "（1箇所）",
        "pos_known_many": "（{n}箇所）",
        "pos_mixed": "（既知{known}箇所・不明{unknown}箇所）",
        # counted nouns (Japanese has no grammatical number — singular == plural)
        "rule_s": "マッチしたルール",
        "rule_p": "マッチしたルール",
        "edit_s": "フォールバック編集",
        "edit_p": "フォールバック編集",
        "deletion_s": "削除",
        "deletion_p": "削除",
        "insertion_s": "挿入",
        "insertion_p": "挿入",
        "substitution_s": "置換",
        "substitution_p": "置換",
        "expl_rule_s": "明示的ルール",
        "expl_rule_p": "明示的ルール",
        "position_s": "箇所",
        "position_p": "箇所",
        "known_position_s": "既知箇所",
        "known_position_p": "既知箇所",
        "unknown_position_s": "不明箇所",
        "unknown_position_p": "不明箇所",
        # verb / particle (in Japanese these are always the same form)
        "verb_supports": "が",
        "verb_support": "が",
        "verb_remains": "が",
        "verb_remain": "が",
        # alignment_summary combined phrase templates (Japanese word order differs)
        "summary_mixed": "{rules}と{edits}{pos}",
        "summary_rules_only": "{rules}{pos}",
        "summary_one_edit": "フォールバック編集1件{pos}",
        "summary_multi_edit": "{ops}{pos}",
        # similarity lines
        "sim_high": "音韻的類似度：高。",
        "sim_moderate": "音韻的類似度：中程度。",
        "sim_weak": "音韻的類似度：低。",
        # why_candidate first line
        "why_exact": "完全な音韻的一致。",
        "why_rules": "{phrase}によりマッチが支持されます。",
        "why_distance": "明示的なルールなしで音韻距離によりランキングされました。",
        # why_candidate third line
        "why_no_diff": "未説明の差異はありません。",
        "why_fallback": "{phrase}が未登録のまま残っています。",
        "why_no_fallback": "フォールバック編集は不要です。",
        "why_see_alignment": "局所的な差異はアラインメントを参照してください。",
    },
}


def _p(lang: Literal["en", "ja"], key: str) -> str:
    """Look up a prose template string for the given language, falling back to English."""
    return _PROSE.get(lang, _PROSE["en"]).get(key, _PROSE["en"].get(key, key))


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


def _format_counted_noun(
    count: int, singular: str, plural: str | None = None, *, lang: Literal["en", "ja"] = "en"
) -> str:
    """Format a counted noun phrase.

    For Japanese, no space and no pluralization: ``f"{count}{singular}"``.
    For English, standard pluralization rules apply.
    """
    if lang == "ja":
        return f"{count}{singular}"
    noun = singular if count == 1 else (plural or singular + "s")
    return f"{count} {noun}"


def _format_counted_phrase(
    count: int,
    singular: str,
    singular_verb: str,
    plural_verb: str,
    plural: str | None = None,
    *,
    lang: Literal["en", "ja"] = "en",
) -> tuple[str, str]:
    """Return a counted noun phrase and the verb agreeing with its count."""
    verb = singular_verb if count == 1 else plural_verb
    return _format_counted_noun(count, singular, plural, lang=lang), verb


def _format_position_summary(steps: list[RuleApplication], *, lang: Literal["en", "ja"] = "en") -> str:
    """Describe whether the affected positions are known or unknown."""
    distinct_positions = _count_distinct_positions(steps)
    unknown_positions = sum(1 for step in steps if step.position < 0)
    if distinct_positions is None:
        if unknown_positions == 1:
            return _p(lang, "pos_unknown_one")
        return _p(lang, "pos_unknown_many").format(n=unknown_positions)
    if unknown_positions > 0:
        template_text = _p(lang, "pos_mixed")
        # Only compute noun phrases if the template actually uses them
        needs_known_noun = "{known_noun}" in template_text
        needs_unknown_noun = "{unknown_noun}" in template_text

        format_args: dict[str, int | str] = {
            "known": distinct_positions,
            "unknown": unknown_positions,
        }

        if needs_known_noun:
            format_args["known_noun"] = _p(lang, "known_position_s" if distinct_positions == 1 else "known_position_p")

        if needs_unknown_noun:
            format_args["unknown_noun"] = _p(lang, "unknown_position_s" if unknown_positions == 1 else "unknown_position_p")

        return template_text.format(**format_args)
    if distinct_positions == 1:
        return _p(lang, "pos_known_one")
    return _p(lang, "pos_known_many").format(n=distinct_positions)


def _build_alignment_summary(
    *,
    source_ipa: str,
    query_ipa: str,
    steps: list[RuleApplication],
    lang: Literal["en", "ja"] = "en",
) -> str:
    """Build a short alignment summary suitable for the result card body."""
    if _same_ipa_ignoring_accents(source_ipa, query_ipa) and not steps:
        return _p(lang, "no_diff")
    if not steps:
        return _p(lang, "diff_visible")

    explicit_steps = [step for step in steps if not _is_observed_rule_step(step)]
    observed_steps = [step for step in steps if _is_observed_rule_step(step)]

    position_summary = _format_position_summary(steps, lang=lang)

    if lang == "ja":
        if explicit_steps and observed_steps:
            rules_phrase = f"{len(explicit_steps)}{_p(lang, 'rule_s')}"
            edits_phrase = f"{len(observed_steps)}{_p(lang, 'edit_s')}"
            return _p(lang, "summary_mixed").format(
                rules=rules_phrase, edits=edits_phrase, pos=position_summary
            )
        if explicit_steps:
            rules_phrase = f"{len(explicit_steps)}{_p(lang, 'rule_s')}"
            return _p(lang, "summary_rules_only").format(rules=rules_phrase, pos=position_summary)
        if len(observed_steps) == 1:
            return _p(lang, "summary_one_edit").format(pos=position_summary)
        operation_counts_ja: dict[FallbackEditLabel, int] = {
            "deletion": 0,
            "insertion": 0,
            "substitution": 0,
        }
        for step in observed_steps:
            operation_counts_ja[_fallback_edit_label(step)] += 1
        ops = "・".join(
            f"{operation_counts_ja[label]}{_p(lang, label + '_s')}"
            for label in ("deletion", "insertion", "substitution")
            if operation_counts_ja[label] > 0
        )
        return _p(lang, "summary_multi_edit").format(ops=ops, pos=position_summary)

    # English
    if explicit_steps and observed_steps:
        rules_phrase = _format_counted_noun(
            len(explicit_steps), _p(lang, "rule_s"), _p(lang, "rule_p"), lang=lang
        )
        edits_phrase = _format_counted_noun(
            len(observed_steps), _p(lang, "edit_s"), _p(lang, "edit_p"), lang=lang
        )
        return _p(lang, "summary_mixed").format(
            rules=rules_phrase, edits=edits_phrase, pos=position_summary
        )

    if explicit_steps:
        rules_phrase = _format_counted_noun(
            len(explicit_steps), _p(lang, "rule_s"), _p(lang, "rule_p"), lang=lang
        )
        return _p(lang, "summary_rules_only").format(rules=rules_phrase, pos=position_summary)

    if len(observed_steps) == 1:
        return _p(lang, "summary_one_edit").format(pos=position_summary)

    operation_counts: dict[FallbackEditLabel, int] = {
        "deletion": 0,
        "insertion": 0,
        "substitution": 0,
    }
    for step in observed_steps:
        operation_counts[_fallback_edit_label(step)] += 1

    operation_summary = ", ".join(
        _format_counted_noun(
            operation_counts[label], _p(lang, label + "_s"), _p(lang, label + "_p"), lang=lang
        )
        for label in ("deletion", "insertion", "substitution")
        if operation_counts[label] > 0
    )
    return _p(lang, "summary_multi_edit").format(ops=operation_summary, pos=position_summary)


def _similarity_line(confidence: float, *, lang: Literal["en", "ja"] = "en") -> str:
    """Return a short qualitative similarity statement."""
    if confidence >= _HIGH_SIMILARITY_THRESHOLD:
        return _p(lang, "sim_high")
    if confidence >= _MODERATE_SIMILARITY_THRESHOLD:
        return _p(lang, "sim_moderate")
    return _p(lang, "sim_weak")


def _build_why_candidate(
    match_type: Literal["Exact", "Rule-based", "Distance-only", "Low-confidence"],
    *,
    applied_rule_count: int,
    observed_change_count: int,
    confidence: float,
    lang: Literal["en", "ja"] = "en",
) -> list[str]:
    """Build stable short bullet points describing why this candidate ranked."""
    if match_type == "Exact":
        first_line = _p(lang, "why_exact")
    elif applied_rule_count > 0:
        phrase, verb = _format_counted_phrase(
            applied_rule_count,
            _p(lang, "expl_rule_s"),
            _p(lang, "verb_supports"),
            _p(lang, "verb_support"),
            _p(lang, "expl_rule_p"),
            lang=lang,
        )
        first_line = _p(lang, "why_rules").format(phrase=phrase, verb=verb)
    else:
        first_line = _p(lang, "why_distance")

    if match_type == "Exact":
        third_line = _p(lang, "why_no_diff")
    elif observed_change_count > 0:
        phrase, verb = _format_counted_phrase(
            observed_change_count,
            _p(lang, "edit_s"),
            _p(lang, "verb_remains"),
            _p(lang, "verb_remain"),
            _p(lang, "edit_p"),
            lang=lang,
        )
        third_line = _p(lang, "why_fallback").format(phrase=phrase, verb=verb)
    elif applied_rule_count > 0:
        third_line = _p(lang, "why_no_fallback")
    else:
        third_line = _p(lang, "why_see_alignment")

    return [
        first_line,
        _similarity_line(confidence, lang=lang),
        third_line,
    ]


def _build_search_hit(
    result: phonology_search.SearchResult,
    query_ipa: str,
    rules_registry: dict[str, dict[str, Any]],
    query_mode: Literal["Full-form", "Short-query", "Partial-form"],
    *,
    lang: Literal["en", "ja"] = "en",
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
            lang=lang,
        ),
        why_candidate=_build_why_candidate(
            match_type,
            applied_rule_count=applied_rule_count,
            observed_change_count=observed_change_count,
            confidence=result.confidence,
            lang=lang,
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
