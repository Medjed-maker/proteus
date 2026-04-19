"""Read-only tuning constants and arithmetic helpers for the search package.

Constants that are monkeypatched by tests (``_GAP_PENALTY`` and
``_MATCH_SCORE``) stay in ``phonology.search.__init__`` — they cannot be
re-exported from here because callers in ``__init__`` read them via local
module lookup, which is what the monkeypatch relies on.
"""

from __future__ import annotations

_DEFAULT_KMER_SIZE = 2
_SEED_MULTIPLIER = 10
_MIN_STAGE2_CANDIDATES = 25
_MIN_PARTIAL_CANDIDATE_LIMIT = 10
_MIN_PARTIAL_STAGE2_CANDIDATES = 100
_PARTIAL_CANDIDATE_MULTIPLIER = 4
_SHORT_QUERY_CONFIDENCE_THRESHOLD = 0.65
_PARTIAL_QUERY_CONFIDENCE_THRESHOLD = 0.70
_PARTIAL_QUERY_MARKERS = frozenset(
    {
        "-",        # U+002D HYPHEN-MINUS
        "*",        # U+002A ASTERISK
        "~",        # U+007E TILDE
        "\u2010",  # HYPHEN
        "\u2011",  # NON-BREAKING HYPHEN
        "\u2012",  # FIGURE DASH
        "\u2013",  # EN DASH
        "\u2014",  # EM DASH
        "\u2015",  # HORIZONTAL BAR
        "\u2212",  # MINUS SIGN
        "\uFF0D",  # FULLWIDTH HYPHEN-MINUS
    }
)
_DEFAULT_FALLBACK_CANDIDATE_LIMIT = 2000
_SHORT_QUERY_MAX_ANNOTATION_BATCHES = 3
_LENGTH_PROXIMATE_LIMIT_MULTIPLIER = 2

# Prefix for synthetic observed-difference annotations generated during alignment.
# These pseudo-rules mark raw phoneme mismatches that were not explained by any
# phonological rule in the loaded rule set.
OBSERVED_PREFIX = "OBS-"


def _partial_candidate_limit(max_results: int) -> int:
    """Return the candidate cap used for partial-form filtering."""
    return max(_MIN_PARTIAL_CANDIDATE_LIMIT, max_results * _PARTIAL_CANDIDATE_MULTIPLIER)


def _annotation_candidate_limit(max_results: int) -> int:
    """Return the pre-filter annotation window for non-full-form queries."""
    return max(_MIN_PARTIAL_STAGE2_CANDIDATES, max_results * _SEED_MULTIPLIER)
