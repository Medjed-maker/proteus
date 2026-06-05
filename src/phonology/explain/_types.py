"""Public and internal data types shared across the explainer modules.

Defining these in a separate module avoids cycles between the rule-matching
state machine, the prose renderer, and downstream consumers like
``phonology.search._types`` and ``phonology.search._scoring``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    from ._rule_tokenize import TokenizedRule


RuleMetadata: TypeAlias = Mapping[str, Any]
POSITION_UNKNOWN: int = -1


@dataclass(frozen=True)
class Alignment:
    """Aligned query/lemma token columns used for rule explanation."""

    aligned_query: tuple[str | None, ...]
    aligned_lemma: tuple[str | None, ...]


@dataclass(init=False)
class RuleApplication:
    """Record of a single rule applied during phonological alignment."""

    rule_id: str
    description: str
    input_phoneme: str
    output_phoneme: str
    position: int
    dialects: list[str] = field(init=False, repr=True)
    weight: float = 1.0
    _rule_name: str = field(init=False, repr=False)
    _rule_name_en: str = field(init=False, repr=False)

    def __init__(
        self,
        *,
        rule_id: str,
        description: str | None = None,
        input_phoneme: str | None = None,
        output_phoneme: str | None = None,
        position: int,
        dialects: list[str] | None = None,
        weight: float = 1.0,
        rule_name: str | None = None,
        rule_name_en: str | None = None,
        from_phone: str | None = None,
        to_phone: str | None = None,
    ) -> None:
        """Normalize canonical fields and backward-compatible aliases.

        ``rule_name`` and ``description`` are mutual fallbacks kept for
        backward compatibility. When both are provided, ``rule_name`` takes
        precedence for the internal display label, so ``_rule_name`` may differ
        from ``description``; for example, ``description="A", rule_name="B"``
        yields ``description == "A"`` and ``_rule_name == "B"``. When
        ``description`` is missing, it falls back to the resolved rule name.

        ``input_phoneme``/``from_phone`` and ``output_phoneme``/``to_phone``
        are alias pairs. The canonical ``input_phoneme`` and
        ``output_phoneme`` values take precedence over legacy
        ``from_phone``/``to_phone``. ``_rule_name`` and ``_rule_name_en`` store
        the canonical/legacy resolution used by the compatibility properties.
        """
        self.rule_id = rule_id
        # Backward-compatible fallback: rule_name and description are mutual
        # fallbacks (rule_name takes precedence) so callers can supply either.
        # Similarly, input_phoneme/from_phone and output_phoneme/to_phone are
        # aliases where the canonical field takes precedence over the legacy one.
        fallback = (
            rule_name
            if rule_name is not None
            else (description if description is not None else "")
        )
        self.description = description if description is not None else fallback
        self._rule_name = rule_name if rule_name is not None else fallback
        self._rule_name_en = rule_name_en if rule_name_en is not None else ""
        self.input_phoneme = (
            input_phoneme if input_phoneme is not None else (from_phone or "")
        )
        self.output_phoneme = (
            output_phoneme if output_phoneme is not None else (to_phone or "")
        )
        self.position = position
        self.dialects = list(dialects or [])
        self.weight = weight

    @property
    def rule_name(self) -> str:
        """Backward-compatible alias for the display label of this rule step."""
        return self._rule_name

    @property
    def rule_name_en(self) -> str:
        """English display name for the rule."""
        return self._rule_name_en

    @property
    def from_phone(self) -> str:
        """Backward-compatible alias for ``input_phoneme``."""
        return self.input_phoneme

    @property
    def to_phone(self) -> str:
        """Backward-compatible alias for ``output_phoneme``."""
        return self.output_phoneme


@dataclass(frozen=True)
class _MismatchBlock:
    """Continuous alignment block that contains no exact-match columns."""

    aligned_query: tuple[str | None, ...]
    aligned_lemma: tuple[str | None, ...]
    lemma_tokens: tuple[str, ...]
    query_tokens: tuple[str, ...]
    lemma_start_position: int
    query_start_position: int


@dataclass(frozen=True)
class _WordFinalSuffixMatch:
    """Metadata captured when a `_#` suffix rule matches."""

    lemma_start_position: int


@dataclass(frozen=True)
class _RuleMatchResult:
    """Matched rule metadata normalized for downstream application building."""

    matched_rule: TokenizedRule
    word_final_suffix_match: _WordFinalSuffixMatch | None
    consumed_lemma_tokens: int
    consumed_query_tokens: int
    application_position: int
