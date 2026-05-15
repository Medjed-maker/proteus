"""Human-readable explanation of applied phonological rules.

Given aligned phoneme sequences and a rule inventory, detect which
phonological rules explain each mismatch block and generate structured
descriptions suitable for APIs and UI consumers.

This module is now a facade. The implementation has been split across:
    - ``_explainer_rule_paths``: trusted-directory registry and rule-path
      resolution.
    - ``_explainer_rule_loader``: YAML rule loading and version-metadata
      extraction (``load_rules``, ``get_rules_version``).
    - ``_explainer_rule_tokenize``: ``TokenizedRule`` plus tokenization /
      sorting (``tokenize_rules_for_matching``).
    - ``_explainer_rule_match``: ``Alignment`` / ``RuleApplication`` /
      mismatch-block matching state machine (``explain`` /
      ``explain_with_tokenized_rules``).
    - ``_explainer_prose``: ``Explanation`` plus ``to_prose``.

All private and public symbols that callers historically reached via
``phonology.explainer.<name>`` (including ``_MismatchBlock``,
``_find_matching_rule_candidate``, ``_resolve_and_validate_rules_dir``,
``_build_observed_application_for_column``, ``_advance_block_cursors``,
``_RuleMatchResult``, ``TokenizedRule``) remain importable from here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TypeAlias

from ._explainer_context import (
    _AFTER_E_I_R_PHONES,
    _NASAL_PHONES,
    _is_consonant,
    _is_vowel,
    _lookup_next_token,
    _lookup_prev_token,
    _matches_context,
    _matches_following_set,
    _matches_same_word_lookahead,
)
from ._explainer_rule_match import (
    _advance_block_cursors,
    _allows_empty_input,
    _block_column_index,
    _build_description,
    _build_observed_application_for_column,
    _collect_block_applications,
    _current_block_column,
    _display_phoneme,
    _extract_dialects,
    _find_matching_rule_candidate,
    _find_normal_block_match,
    _find_word_final_suffix_match,
    _has_crossing_gaps,
    _is_exact_match,
    _iter_mismatch_blocks,
    _matches_block_columns,
    _matches_lemma_constraints,
    _matches_word_final_suffix,
    _rule_name_en_for_description,
    _rule_name_for_description,
    explain,
    explain_with_tokenized_rules,
)
from ._explainer_types import (
    Alignment,
    POSITION_UNKNOWN,
    RuleApplication,
    RuleMetadata,
    _MismatchBlock,
    _RuleMatchResult,
    _WordFinalSuffixMatch,
)
from ._explainer_rule_loader import (
    _extract_rule_file_version,
    _extract_scalar_node_value,
    _is_valid_rule_version_value,
    get_rules_version,
    load_rules,
)
from ._explainer_rule_paths import (
    _RULES_BASE_DIR_OVERRIDE,
    _TRUSTED_EXTERNAL_RULES_DIRS,
    _TRUSTED_EXTERNAL_RULES_DIRS_LOCK,
    _get_rules_base_dir,
    _resolve_and_validate_rules_dir,
    _resolve_rules_dir,
    clear_trusted_external_rules_dirs,
    register_trusted_rules_dir,
)
from ._explainer_rule_tokenize import (
    Rule,
    TokenizedRule,
    _ALWAYS_MATCH_CONTEXTS,
    _rule_specificity,
    _tokenize_context_tail,
    _tokenize_rule_side,
    _tokenize_rules,
    tokenize_rules_for_matching,
)
from ._explainer_prose import Explanation, to_prose

logger = logging.getLogger("phonology.explainer")

__all__ = [
    "Rule",
    "Alignment",
    "TokenizedRule",
    "RuleApplication",
    "Explanation",
    "POSITION_UNKNOWN",
    "tokenize_rules_for_matching",
    "explain_with_tokenized_rules",
    "load_rules",
    "register_trusted_rules_dir",
    "clear_trusted_external_rules_dirs",
    "get_rules_version",
    "explain",
    "explain_alignment",
    "to_prose",
]

# Underscored alias kept for backward compatibility and re-exported for
# downstream callers that historically imported it from this facade.
_RuleMetadata: TypeAlias = RuleMetadata


def __getattr__(name: str) -> Any:
    """Expose lazily-resolved ``RULES_BASE_DIR`` to direct attribute access."""
    if name == "RULES_BASE_DIR":
        return _get_rules_base_dir()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose RULES_BASE_DIR in module directory."""
    return sorted(set(globals().keys()) | {"RULES_BASE_DIR"})


def explain_alignment(
    source_ipa: str,
    target_ipa: str,
    rule_ids: list[str],
    all_rules: dict[str, dict[str, Any]],
    distance: float = 0.0,
) -> Explanation:
    """Build a structured explanation for a phonological alignment.

    Since this function works with IPA strings rather than orthographic forms,
    the returned ``Explanation`` sets both ``source`` and ``source_ipa`` to
    *source_ipa*, and both ``target`` and ``target_ipa`` to *target_ipa*.
    Callers that display both fields should be aware of this intentional
    duplication.

    Args:
        source_ipa: Source word in IPA.
        target_ipa: Target word in IPA.
        rule_ids: Ordered list of rule ids applied.
        all_rules: Full rule registry from load_rules().
        distance: Normalized phonological distance for the alignment.

    Returns:
        Explanation object with step-by-step and prose description.
    """
    steps: list[RuleApplication] = []
    for rule_id in rule_ids:
        rule = all_rules.get(rule_id, {"id": rule_id})
        raw_input = rule.get("input")
        raw_output = rule.get("output")
        input_phoneme = raw_input if isinstance(raw_input, str) else ""
        output_phoneme = raw_output if isinstance(raw_output, str) else ""
        rule_name = _rule_name_for_description(rule)
        rule_name_en = _rule_name_en_for_description(rule)
        dialects = _extract_dialects(rule.get("dialects", []))
        steps.append(
            RuleApplication(
                rule_id=rule_id,
                description=_build_description(
                    rule,
                    rule_name,
                    input_phoneme,
                    output_phoneme,
                ),
                rule_name=rule_name,
                rule_name_en=rule_name_en,
                input_phoneme=input_phoneme,
                output_phoneme=output_phoneme,
                position=POSITION_UNKNOWN,
                dialects=dialects,
            )
        )

    return Explanation(
        source=source_ipa,
        target=target_ipa,
        source_ipa=source_ipa,
        target_ipa=target_ipa,
        distance=distance,
        steps=steps,
    )
