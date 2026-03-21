"""BLAST-like three-stage phonological search.

Stage 1 — Seed:    find candidate words sharing a k-mer (phoneme n-gram).
Stage 2 — Extend:  compute phonological distance for candidate words.
Stage 3 — Filter:  rank by distance + apply threshold cutoff.

Inspired by NCBI BLAST's seed-and-extend heuristic, adapted for
phonological space rather than nucleotide/protein sequences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeAlias

from .distance import MatrixData

DistanceMatrix: TypeAlias = MatrixData


@dataclass
class SearchResult:
    """Single hit returned by phonological search.

    Attributes:
        headword: Matched lexicon entry in Greek script.
        ipa: Candidate IPA transcription from the lexicon.
        distance: Normalized phonological distance in the 0.0-1.0 range.
        rules_applied: Ordered rule identifiers used to explain the match; may be empty.
        score: Normalized ranking score derived from distance for ordering; higher is better.
    """

    headword: str
    ipa: str
    distance: float
    rules_applied: list[str] = field(default_factory=list)
    score: float = 0.0


@dataclass
class SearchConfig:
    """Tunable parameters for the three-stage search."""

    kmer_size: int = 2
    seed_threshold: float = 0.4
    extend_threshold: float = 0.6
    max_results: int = 20
    dialect: str = "attic"

    def __post_init__(self) -> None:
        """Validate configuration values used by the search pipeline."""
        if self.kmer_size <= 0:
            raise ValueError("kmer_size must be > 0")
        if not 0.0 <= self.seed_threshold <= 1.0:
            raise ValueError("seed_threshold must be between 0.0 and 1.0")
        if not 0.0 <= self.extend_threshold <= 1.0:
            raise ValueError("extend_threshold must be between 0.0 and 1.0")
        if self.max_results <= 0:
            raise ValueError("max_results must be > 0")
        if not isinstance(self.dialect, str) or not self.dialect.strip():
            raise ValueError("dialect must be a non-empty string")


def build_kmer_index(lexicon: list[dict[str, Any]], k: int = 2) -> dict[str, list[str]]:
    """Build a k-mer index over the lexicon for fast seed lookup.

    Args:
        lexicon: List of lemma dicts (as loaded from greek_lemmas.json).
        k: Phone n-gram size.

    Returns:
        Dict mapping k-mer string -> list of headword ids.
    """
    if k <= 0:
        raise ValueError(f"build_kmer_index requires k > 0 for k-mer size, got {k}")
    raise NotImplementedError


def seed_stage(query_ipa: str, index: dict[str, list[str]], k: int = 2) -> list[str]:
    """Stage 1: collect candidate headword ids sharing a k-mer with query.

    Args:
        query_ipa: Query word in IPA (space-separated phones).
        index: Pre-built k-mer index.
        k: Phone n-gram size.

    Returns:
        Deduplicated list of candidate headword ids.
    """
    raise NotImplementedError


def extend_stage(
    query_ipa: str,
    candidates: list[str],
    lexicon_map: dict[str, dict],
    matrix: DistanceMatrix,
    threshold: float,
) -> list[SearchResult]:
    """Stage 2: compute full phonological distance for candidates.

    Args:
        query_ipa: Query word in IPA.
        candidates: Candidate headword ids from seed stage.
        lexicon_map: Dict of headword id -> lemma dict.
        matrix: Loaded distance matrix.
        threshold: Drop candidates with normalized distance > threshold.

    Returns:
        List of SearchResult, unsorted.
    """
    raise NotImplementedError


def filter_stage(results: list[SearchResult], max_results: int) -> list[SearchResult]:
    """Stage 3: rank and truncate results.

    Args:
        results: Raw results from extend stage.
        max_results: Maximum hits to return. Must be a positive integer;
            callers should normally validate this via ``SearchConfig``.

    Returns:
        Sorted (ascending distance), truncated list.

    Raises:
        ValueError: If ``max_results`` is not positive.
    """
    if max_results <= 0:
        raise ValueError("max_results must be a positive integer")
    return sorted(results, key=lambda r: r.distance)[:max_results]


def search(
    query_greek: str,
    lexicon: list[dict[str, Any]],
    matrix: DistanceMatrix,
    config: SearchConfig | None = None,
) -> list[SearchResult]:
    """Run full three-stage search for a Greek query word.

    Args:
        query_greek: Query in Greek script (Unicode).
        lexicon: Full lemma list.
        matrix: Loaded distance matrix.
        config: Search configuration (defaults used if None).

    Returns:
        Ranked list of SearchResult hits.
    """
    raise NotImplementedError
