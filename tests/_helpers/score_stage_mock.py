"""Shared assertion helpers for ``_score_stage`` mocks.

Tests stub ``phonology.search._score_stage`` with closures that absorb
unknown keyword arguments via ``**_kwargs``. This module centralizes the
allowed keyword set so a new kwarg added to the real ``_score_stage``
signature is detected by every existing fake without requiring each test
to carry its own inline assertion.
"""

from __future__ import annotations

EXPECTED_SCORE_STAGE_KWARGS: frozenset[str] = frozenset({"phone_inventory"})
"""Allowed keyword argument names for ``_score_stage`` mocks.

Currently contains ``phone_inventory``. Callers and tests should only pass
keywords in this set to fake ``_score_stage`` implementations.
"""


def assert_only_expected_score_stage_kwargs(kwargs: dict[str, object]) -> None:
    """Raise AssertionError when ``kwargs`` contains unknown keyword names.

    Args:
        kwargs: The ``**_kwargs`` mapping captured by a fake ``_score_stage``.
            Compared against ``EXPECTED_SCORE_STAGE_KWARGS``.

    Raises:
        AssertionError: If any keyword outside the expected set is present.
            The message lists the unexpected names sorted for stability.
    """
    unexpected = set(kwargs) - EXPECTED_SCORE_STAGE_KWARGS
    if unexpected:
        raise AssertionError(
            f"Unexpected _score_stage kwargs: {sorted(unexpected)}"
        )
