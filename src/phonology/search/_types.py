"""Shared dataclasses, NamedTuples, and type aliases for the search package.

This module has no internal dependencies on other search submodules; it
imports ``Alignment`` and ``RuleApplication`` directly from ``..explainer``
so that other submodules can depend on this one without triggering an
import cycle back through ``phonology.search``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple, TypeAlias

from ..distance import MatrixData
from ..explainer import Alignment, RuleApplication

DistanceMatrix: TypeAlias = MatrixData
KmerIndex: TypeAlias = dict[str, list[str]]
LexiconEntry: TypeAlias = dict[str, Any]
QueryMode: TypeAlias = Literal["Full-form", "Short-query", "Partial-form"]
PartialQueryShape: TypeAlias = Literal["prefix", "suffix", "infix"]


class LexiconRecord(NamedTuple):
    """A lexicon entry paired with its cached IPA tokens and count."""

    entry: LexiconEntry
    token_count: int
    ipa_tokens: tuple[str, ...] = ()


LexiconMap: TypeAlias = dict[str, LexiconRecord]
LexiconLookupValue: TypeAlias = LexiconEntry | LexiconRecord
LexiconLookup: TypeAlias = Mapping[str, LexiconLookupValue]


@dataclass
class SearchResult:
    """Single ranked hit returned by phonological search.

    ``dialect_attribution`` is always set to a descriptive string by
    ``extend_stage``, but defaults to ``None`` so that callers constructing
    instances outside the pipeline (e.g. tests, future stages) are not
    forced to supply a value. Consumers should handle ``None`` gracefully.
    ``entry_id`` identifies the source lexicon entry for this ``SearchResult``;
    pipeline stages or external constructors may populate it similarly to how
    ``extend_stage`` populates ``dialect_attribution``. Consumers should treat
    ``entry_id=None`` as "source entry unknown" and only rely on it when set.
    ``truncated`` indicates whether the search results were cut off due to size
    limits (True when results were limited). When True, clients may need to
    fetch additional pages and result counts may be partial.
    """

    lemma: str
    confidence: float
    dialect_attribution: str | None = None
    applied_rules: list[str] = field(default_factory=list)
    rule_applications: list[RuleApplication] = field(default_factory=list)
    alignment_visualization: str = ""
    ipa: str | None = None
    entry_id: str | None = None
    alignment: Alignment | None = field(default=None, repr=False, compare=False)
    truncated: bool = False


class PartialQueryPattern(NamedTuple):
    """Parsed user query containing a single wildcard marker."""

    shape: PartialQueryShape
    left_fragment: str
    right_fragment: str


class PartialQueryTokens(NamedTuple):
    """Wildcard query fragments converted into IPA token sequences."""

    shape: PartialQueryShape
    left_tokens: tuple[str, ...]
    right_tokens: tuple[str, ...]


class PartialMatchInfo(NamedTuple):
    """Fragment-aware match information for a partial-form candidate."""

    full_match: bool
    matched_fragments: int
    overlap_score: int
