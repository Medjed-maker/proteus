"""Human-readable explanation of applied phonological rules.

Given a source word, a target word, and the sequence of rules that were
applied during alignment, generate structured and prose explanations
suitable for display in the API response or UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RuleApplication:
    """Record of a single rule applied during phonological alignment."""

    rule_id: str
    rule_name: str
    from_phone: str
    to_phone: str
    position: int
    dialect: str
    weight: float


@dataclass
class Explanation:
    """Full explanation for a source -> target derivation."""

    source: str
    target: str
    source_ipa: str
    target_ipa: str
    distance: float
    steps: list[RuleApplication]
    prose: str


def load_rules(rules_dir: Any) -> dict[str, dict]:
    """Load all YAML rule files from a directory.

    Args:
        rules_dir: Path to the rules directory (e.g. data/rules/ancient_greek/).

    Returns:
        Dict mapping rule_id -> rule dict.
    """
    raise NotImplementedError


def explain_alignment(
    source_ipa: str,
    target_ipa: str,
    rule_ids: list[str],
    all_rules: dict[str, dict],
) -> Explanation:
    """Build a structured explanation for a phonological alignment.

    Args:
        source_ipa: Source word in IPA.
        target_ipa: Target word in IPA.
        rule_ids: Ordered list of rule ids applied.
        all_rules: Full rule registry from load_rules().

    Returns:
        Explanation object with step-by-step and prose description.
    """
    raise NotImplementedError


def to_prose(explanation: Explanation) -> str:
    """Render an Explanation as a human-readable English paragraph.

    Args:
        explanation: Structured explanation object.

    Returns:
        Multi-sentence prose description of the phonological derivation.
    """
    raise NotImplementedError
