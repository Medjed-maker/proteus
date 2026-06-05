"""Reusable fakes and installers for the search pipeline.

Tests should use these helpers instead of monkeypatching ``_seed_stage_core``
or ``_build_lexicon_map_for_inventory`` directly. The seam (which private
symbol the production pipeline calls) is encapsulated here, so renaming an
internal core function only requires updating this file rather than every
test.

Why this seam exists: the production pipeline (``search_execution`` →
``_execute_search``) calls ``_seed_stage_for_inventory`` and
``_build_lexicon_map_for_inventory``, both of which delegate to private cores.
Tests cannot intercept the pipeline by monkeypatching the public API
(``seed_stage`` / ``build_lexicon_map``) — those functions are only invoked
by external callers, not by ``search_execution``. So tests must monkeypatch
private symbols, but doing so directly hardcodes the symbol name into many
tests; this helper centralizes that knowledge.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping

from phonology import search as _search_module
from phonology.search import LexiconRecord


# --- Seed stage ---


def fake_seed_stage_returning(
    candidates: Iterable[str],
) -> Callable[..., list[str]]:
    """Return a fake seed-stage callable that always yields ``candidates``.

    Compatible with both the public ``seed_stage`` signature and the internal
    ``_seed_stage_core`` signature (uses ``**kwargs``).
    """
    materialized = list(candidates)

    def _stage(*_args: Any, **_kwargs: Any) -> list[str]:
        return list(materialized)

    return _stage


def install_seed_stage(
    monkeypatch: Any, fake: Callable[..., list[str]]
) -> None:
    """Install ``fake`` at the seed-stage seam in the production search pipeline.

    Equivalent to ``monkeypatch.setattr(_search_module, "_seed_stage_core", fake)``,
    but encapsulates the private symbol name so a future internal rename only
    requires updating this helper.
    """
    # Sanity check that the seam still exists in production code.
    getattr(_search_module, "_seed_stage_core")
    monkeypatch.setattr(_search_module, "_seed_stage_core", fake)


# --- Lexicon map ---


def fake_lexicon_map_returning(
    records: Mapping[str, LexiconRecord],
) -> Callable[..., dict[str, LexiconRecord]]:
    """Return a fake lexicon-map callable that always yields ``records``."""
    snapshot = dict(records)

    def _build(*_args: Any, **_kwargs: Any) -> dict[str, LexiconRecord]:
        return dict(snapshot)

    return _build


def fail_lexicon_map(
    message: str = "_build_lexicon_map_for_inventory should not be called",
) -> Callable[..., Any]:
    """Return a fake that raises if called — for tests asserting cache hits."""

    def _raise(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError(message)

    return _raise


def install_lexicon_map(monkeypatch: Any, fake: Callable[..., Any]) -> None:
    """Install ``fake`` at the lexicon-map seam in the production search pipeline.

    Equivalent to ``monkeypatch.setattr(_search_module,
    "_build_lexicon_map_for_inventory", fake)`` but encapsulates the private
    symbol name.
    """
    # Sanity check that the seam still exists in production code.
    getattr(_search_module, "_build_lexicon_map_for_inventory")
    monkeypatch.setattr(_search_module, "_build_lexicon_map_for_inventory", fake)


def install_test_language_profile(
    monkeypatch: Any,
    language_id: str = "test",
    *,
    converter: object | None = None,
    phone_inventory: tuple[str, ...] = (),
) -> None:
    """Register a lightweight profile seam for public non-default language tests."""
    original_get_language_profile = _search_module.get_language_profile
    profile = SimpleNamespace(
        language_id=language_id,
        default_dialect="test",
        converter=converter
        if converter is not None
        else (lambda text, *, dialect: text),
        phone_inventory=phone_inventory,
        vowel_phones=(),
        dialect_skeleton_builders=(),
    )

    def fake_get_language_profile(requested_language_id: str) -> object:
        if requested_language_id == language_id:
            return profile
        return original_get_language_profile(requested_language_id)

    monkeypatch.setattr(
        _search_module,
        "get_language_profile",
        fake_get_language_profile,
    )
