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
    """A lexicon entry paired with token metadata.

    ``ipa_tokens=None`` means tokens have not been cached and should be
    resolved from the entry IPA on demand.
    """

    entry: LexiconEntry
    token_count: int
    ipa_tokens: tuple[str, ...] | None = None


LexiconMap: TypeAlias = dict[str, LexiconRecord]
LexiconLookupValue: TypeAlias = LexiconEntry | LexiconRecord
LexiconLookup: TypeAlias = Mapping[str, LexiconLookupValue]

# Candidate selection paths describe which ordered source produced the IDs
# sent to scoring. ``seed`` uses primary k=2 seeds for full/short queries;
# ``partial-seed`` uses those seeds plus bounded k=1 supplements before
# wildcard filtering. ``unigram-fallback`` and ``partial-unigram-fallback``
# are chosen when no primary seeds survive and k=1 matches are available.
# ``token-proximity-fallback`` and ``partial-token-proximity-fallback`` are the
# final similarity/token-count windows used when seed and unigram paths cannot
# provide candidates. Earlier paths take precedence over later fallbacks.
_CandidateSelectionPath: TypeAlias = Literal[
    "seed",
    "partial-seed",
    "unigram-fallback",
    "partial-unigram-fallback",
    "token-proximity-fallback",
    "partial-token-proximity-fallback",
]


@dataclass(slots=True)
class _CandidateSelectionResult:
    """Internal handoff from candidate selection into scoring/finalization.

    Attributes:
        candidate_ids: Ordered candidate entry IDs selected for scoring.
        lexicon_lookup: Lookup used by scoring and later result annotation.
        query_mode: Effective query mode after parsing and query preparation.
        query_tokens: Tokenized IPA query used for selection and fallback
            scoring context.
        selection_path: Source path that produced ``candidate_ids``; used by
            finalization to apply path-specific filtering and truncation.
        seed_candidate_count: Number of primary k=2 seed candidates observed
            before stage-2 windowing or partial wildcard filtering. Non-zero
            only on seeded paths.
        unigram_candidate_count: Number of k=1 candidates collected either as
            partial-query supplements or as the selected unigram fallback pool.
        fallback_limit: Explicit exploration cap used for unigram or
            token-proximity fallback paths. ``None`` on seeded paths; fallback
            finalization uses it to decide when candidate exploration may have
            been truncated.
    """

    candidate_ids: list[str]
    lexicon_lookup: LexiconLookup
    query_mode: QueryMode
    query_tokens: list[str]
    selection_path: _CandidateSelectionPath
    seed_candidate_count: int = 0
    unigram_candidate_count: int = 0
    fallback_limit: int | None = None


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
