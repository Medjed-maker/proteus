"""BLAST-like three-stage phonological search.

Stage 1 — Seed:    find candidate words sharing a k-mer (phoneme n-gram).
Stage 2 — Extend:  compute phonological distance for candidate words.
Stage 3 — Filter:  rank by distance + apply threshold cutoff.

Inspired by NCBI BLAST's seed-and-extend heuristic, adapted for
phonological space rather than nucleotide/protein sequences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchResult:
    """Single hit returned by phonological search."""

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


def build_kmer_index(lexicon: list[dict[str, Any]], k: int = 2) -> dict[str, list[str]]:
    """Build a k-mer index over the lexicon for fast seed lookup.

    Args:
        lexicon: List of lemma dicts (as loaded from greek_lemmas.json).
        k: Phone n-gram size.

    Returns:
        Dict mapping k-mer string -> list of headword ids.
    """
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
    matrix: Any,
    threshold: float,
) -> list[SearchResult]:
    """Stage 2: compute full phonological distance for candidates.

    Args:
        query_ipa: Query word in IPA.
        candidates: Candidate headword ids from seed stage.
        lexicon_map: Dict of headword id -> lemma dict.
        matrix: Loaded distance matrix.
        threshold: Drop candidates with distance > threshold.

    Returns:
        List of SearchResult, unsorted.
    """
    raise NotImplementedError


def filter_stage(results: list[SearchResult], max_results: int) -> list[SearchResult]:
    """Stage 3: rank and truncate results.

    Args:
        results: Raw results from extend stage.
        max_results: Maximum hits to return.

    Returns:
        Sorted (ascending distance), truncated list.
    """
    return sorted(results, key=lambda r: r.distance)[:max_results]


def search(query_greek: str, lexicon: list[dict], matrix: Any, config: SearchConfig | None = None) -> list[SearchResult]:
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
