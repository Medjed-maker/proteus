"""Human-readable explanation of applied phonological rules.

Given a source word, a target word, and the sequence of rules that were
applied during alignment, generate structured and prose explanations
suitable for display in the API response or UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ._paths import resolve_repo_data_dir

__all__ = [
    "RuleApplication",
    "Explanation",
    "load_rules",
    "explain_alignment",
    "to_prose",
]

_RULES_BASE_DIR_OVERRIDE: Path | None = None


def _get_rules_base_dir() -> Path:
    """Lazily resolve the rules base directory.

    Tests can override resolution by setting ``_RULES_BASE_DIR_OVERRIDE``
    via ``monkeypatch.setattr``.
    """
    if _RULES_BASE_DIR_OVERRIDE is not None:
        return _RULES_BASE_DIR_OVERRIDE
    return resolve_repo_data_dir("rules")


def __getattr__(name: str) -> Path:
    if name == "RULES_BASE_DIR":
        return _get_rules_base_dir()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose RULES_BASE_DIR in module directory."""
    return sorted(set(globals().keys()) | {"RULES_BASE_DIR"})


def _resolve_rules_dir(rules_dir: Path | str, rules_base_dir: Path) -> Path:
    """Resolve rules directories relative to the packaged rules base."""
    candidate_rules_dir = Path(rules_dir)
    if candidate_rules_dir.is_absolute():
        return candidate_rules_dir

    parts = candidate_rules_dir.parts
    if len(parts) >= 2 and parts[:2] == ("data", "rules"):
        candidate_rules_dir = Path(*parts[2:])

    return rules_base_dir / candidate_rules_dir


@dataclass
class RuleApplication:
    """Record of a single rule applied during phonological alignment.

    Field mapping and derivation:
        rule_id    <- id
        rule_name  <- name_en (or name_ja for Japanese output)
        from_phone <- input
        to_phone   <- output
        position   <- derived from alignment step order/position, not stored in YAML
        dialects   <- dialects
        weight     <- numeric influence/priority for this recorded rule application;
                      downstream consumers can use higher values to score or order
                      stronger rule applications, while the default 1.0 is neutral
    """

    rule_id: str
    rule_name: str
    from_phone: str
    to_phone: str
    position: int
    dialects: list[str] = field(default_factory=list)
    weight: float = 1.0


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


def load_rules(rules_dir: Path | str) -> dict[str, dict]:
    """Load all YAML rule files from a directory.

    Args:
        rules_dir: Path to the rules directory. Relative inputs are resolved
            from the packaged rules base, so both ``"ancient_greek"`` and
            ``"data/rules/ancient_greek"`` resolve to the same runtime asset.

    Returns:
        Dict mapping rule_id -> rule dict.
    """
    rules_base_dir = _get_rules_base_dir()
    try:
        rules_base_dir = rules_base_dir.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(
            f"Configured rules base directory is missing: {exc}. "
            f"Create the {rules_base_dir} directory before calling load_rules()."
        ) from exc

    candidate_rules_dir = _resolve_rules_dir(rules_dir, rules_base_dir)
    try:
        resolved_rules_dir = candidate_rules_dir.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(
            f"load_rules could not find rules directory {candidate_rules_dir}. "
            f"Expected an existing directory within {rules_base_dir}."
        ) from exc
    if not resolved_rules_dir.is_relative_to(rules_base_dir):
        raise ValueError(
            f"load_rules path must stay within {rules_base_dir}, got {resolved_rules_dir}"
        )
    if not resolved_rules_dir.is_dir():
        raise ValueError(f"load_rules expected a directory, got {resolved_rules_dir}")

    rules: dict[str, dict] = {}
    rule_sources: dict[str, Path] = {}
    for rule_file in sorted(resolved_rules_dir.iterdir()):
        if not rule_file.is_file() or rule_file.suffix.lower() not in {".yaml", ".yml"}:
            continue

        document = yaml.safe_load(rule_file.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError(f"Rule file {rule_file} must contain a top-level mapping")

        raw_rules = document.get("rules")
        if not isinstance(raw_rules, list):
            raise ValueError(f"Rule file {rule_file} must define a list under 'rules'")

        for index, raw_rule in enumerate(raw_rules):
            if not isinstance(raw_rule, dict):
                raise ValueError(f"Rule entry {index} in {rule_file} must be a mapping")

            rule_id = raw_rule.get("id")
            if not isinstance(rule_id, str) or not rule_id.strip():
                raise ValueError(f"Rule entry {index} in {rule_file} must define a non-empty id")
            if rule_id in rules:
                first_defined_in = rule_sources[rule_id]
                raise ValueError(
                    f"Duplicate rule id {rule_id!r} found in {rule_file}; "
                    f"first defined in {first_defined_in}"
                )
            rules[rule_id] = raw_rule
            rule_sources[rule_id] = rule_file

    return rules


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
    """Generate canonical prose for a structured explanation.

    Args:
        explanation: Structured explanation object to render.
    """
    # Dialect labels are omitted from prose, but step weights are retained.
    if explanation.steps:
        step_summary = "; ".join(
            (
                f"{step.rule_name} ({step.from_phone} -> {step.to_phone} "
                f"at position {step.position}, weight {step.weight:g})"
            )
            for step in explanation.steps
        )
        prose = (
            f"{explanation.source} /{explanation.source_ipa}/ aligns to "
            f"{explanation.target} /{explanation.target_ipa}/ with distance "
            f"{explanation.distance:.3f}. Applied rules: {step_summary}."
        )
    else:
        prose = (
            f"{explanation.source} /{explanation.source_ipa}/ aligns to "
            f"{explanation.target} /{explanation.target_ipa}/ with distance "
            f"{explanation.distance:.3f}. No rule applications were recorded."
        )

    return prose
