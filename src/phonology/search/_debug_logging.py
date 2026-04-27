"""Debug-only performance logging helpers for the phonological search pipeline.

All public functions in this module are no-ops when DEBUG logging is not
enabled on the calling logger, so they are safe to call unconditionally on
the hot path.
"""

from __future__ import annotations

import hashlib
import logging
import time
from logging import Logger


def summarize_query_ipa_for_logs(
    query_ipa: str,
    *,
    query_token_count: int,
    debug_enabled: bool = True,
) -> str:
    """Return a redacted search-query identifier safe for debug logs.

    When DEBUG logging is enabled, produces a 12-hex (48-bit) SHA-256 prefix.
    Prefix collisions are possible but rare enough to be acceptable for debug
    triage. Never use as a unique key.
    """
    if not debug_enabled:
        return (
            f"tokens={query_token_count} chars={len(query_ipa)} sha256=<debug-disabled>"
        )

    digest = hashlib.sha256(query_ipa.encode("utf-8")).hexdigest()[:12]
    return f"tokens={query_token_count} chars={len(query_ipa)} sha256={digest}"


def perf_counter_if_debug(logger: Logger) -> float:
    """Return ``time.perf_counter()`` only when DEBUG logging is active, else 0.0."""
    if not logger.isEnabledFor(logging.DEBUG):
        return 0.0
    return time.perf_counter()


def log_candidate_selection(
    logger: Logger,
    *,
    query_label: str,
    query_mode: str,
    selection_path: str,
    seed_candidate_count: int,
    unigram_candidate_count: int,
    selected_count: int,
    fallback_limit: int | None,
    elapsed_ms: float,
) -> None:
    """Emit a DEBUG log for the candidate-selection stage, if DEBUG is enabled."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    logger.debug(
        "Search candidate selection completed (%s): mode=%s path=%s seed_candidates=%d "
        "unigram_candidates=%d selected=%d fallback_limit=%s elapsed_ms=%.3f",
        query_label,
        query_mode,
        selection_path,
        seed_candidate_count,
        unigram_candidate_count,
        selected_count,
        fallback_limit,
        elapsed_ms,
    )


def log_scoring(
    logger: Logger,
    *,
    query_label: str,
    selected_count: int,
    scored_count: int,
    elapsed_ms: float,
) -> None:
    """Emit a DEBUG log for the scoring stage, if DEBUG is enabled."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    logger.debug(
        "Search scoring completed (%s): selected=%d scored=%d elapsed_ms=%.3f",
        query_label,
        selected_count,
        scored_count,
        elapsed_ms,
    )


def log_finalization(
    logger: Logger,
    *,
    query_label: str,
    query_mode: str,
    annotated_count: int,
    returned_count: int,
    elapsed_ms: float,
) -> None:
    """Emit a DEBUG log for the annotation/filtering stage, if DEBUG is enabled."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    logger.debug(
        "Search annotation/final filtering completed (%s): mode=%s annotated=%d returned=%d "
        "elapsed_ms=%.3f",
        query_label,
        query_mode,
        annotated_count,
        returned_count,
        elapsed_ms,
    )
