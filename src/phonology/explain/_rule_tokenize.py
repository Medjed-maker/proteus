"""Rule pre-tokenization for longest-match-first scanning."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import re
from typing import Any, TypeAlias

from ..core.ipa import tokenize_ipa

Rule: TypeAlias = dict[str, Any]

_ALWAYS_MATCH_CONTEXTS = frozenset(
    {
        "",
        "all environments",
    }
)


@dataclass(frozen=True)
class TokenizedRule:
    """Rule metadata tokenized for mismatch-block matching."""

    rule: Rule
    input_tokens: tuple[str, ...]
    output_tokens: tuple[str, ...]
    order: int
    context_tail_tokens: tuple[str, ...] | None = None


def _tokenize_rule_side(
    raw_value: object,
    *,
    phone_inventory: tuple[str, ...],
) -> tuple[str, ...]:
    """Tokenize a YAML rule side into comparable IPA tokens."""
    if not isinstance(raw_value, str) or not raw_value:
        return ()
    return tuple(
        tokenize_ipa(
            raw_value,
            phone_inventory=phone_inventory,
        )
    )


def _resolve_phone_inventory(
    phone_inventory: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Return explicit phones, or generic character tokenization for ``None``."""
    if phone_inventory is not None:
        return tuple(phone_inventory)

    return ()


def _resolve_always_match_contexts(
    always_match_contexts: Iterable[str] | None = None,
) -> frozenset[str]:
    """Return generic and injected always-match context labels."""
    return _ALWAYS_MATCH_CONTEXTS | frozenset(
        context.strip().lower()
        for context in (always_match_contexts or ())
        if isinstance(context, str)
    )


def _tokenize_context_tail(
    context: object,
    *,
    phone_inventory: tuple[str, ...],
) -> tuple[str, ...] | None:
    """Tokenize `_...tail` context notation with the rule inventory."""
    if not isinstance(context, str):
        return None
    match = re.fullmatch(r"_\.\.\.(.+)", context.strip())
    if match is None:
        return None
    return _tokenize_rule_side(match.group(1), phone_inventory=phone_inventory)


def _rule_specificity(
    rule: Rule,
    *,
    always_match_contexts: frozenset[str],
) -> int:
    """Return a larger value for rules with narrower contextual applicability."""
    context = rule.get("context")
    if not isinstance(context, str):
        return 0
    return 0 if context.strip().lower() in always_match_contexts else 1


def _tokenize_rules(
    rules: list[Rule],
    *,
    phone_inventory: Iterable[str] | None = None,
    always_match_contexts: Iterable[str] | None = None,
) -> list[TokenizedRule]:
    """Pre-tokenize rules and sort them for longest-match-first scanning."""
    resolved_phone_inventory = _resolve_phone_inventory(phone_inventory)
    resolved_always_match_contexts = _resolve_always_match_contexts(
        always_match_contexts
    )
    tokenized: list[TokenizedRule] = []
    for order, rule in enumerate(rules):
        input_tokens = _tokenize_rule_side(
            rule.get("input"),
            phone_inventory=resolved_phone_inventory,
        )
        output_tokens = _tokenize_rule_side(
            rule.get("output"),
            phone_inventory=resolved_phone_inventory,
        )
        context_tail_tokens = _tokenize_context_tail(
            rule.get("context"),
            phone_inventory=resolved_phone_inventory,
        )
        if not input_tokens and not output_tokens:
            continue
        tokenized.append(
            TokenizedRule(
                rule=rule,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                order=order,
                context_tail_tokens=context_tail_tokens,
            )
        )

    return sorted(
        tokenized,
        key=lambda candidate: (
            -_rule_specificity(
                candidate.rule,
                always_match_contexts=resolved_always_match_contexts,
            ),
            -(len(candidate.input_tokens) + len(candidate.output_tokens)),
            -len(candidate.input_tokens),
            -len(candidate.output_tokens),
            candidate.order,
        ),
    )


def tokenize_rules_for_matching(
    rules: list[Rule],
    *,
    phone_inventory: Iterable[str] | None = None,
    always_match_contexts: Iterable[str] | None = None,
) -> list[TokenizedRule]:
    """Return reusable tokenized rule metadata for mismatch matching.

    Args:
        rules: A ``list[Rule]`` containing the phonological rules to pre-tokenize.
        phone_inventory: An optional ``Iterable[str]`` of phones used for
            tokenization and matching. ``None`` uses generic character
            tokenization.
        always_match_contexts: Optional language-specific context labels that
            should be treated as unconditional matches in addition to the
            generic contexts.

    Returns:
        A ``list[TokenizedRule]`` where each item stores the original rule,
        tokenized input/output phones, and a stable sort order for repeated
        longest-match-first scans.
    """
    return _tokenize_rules(
        rules,
        phone_inventory=phone_inventory,
        always_match_contexts=always_match_contexts,
    )
