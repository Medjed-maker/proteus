"""Cached rules-registry helpers for the search pipeline.

This module owns the per-process rule loading caches:

- ``_load_rules_cached`` — LRU-cached YAML loader. Tests clear this via
  ``phonology.search._load_rules_cached.cache_clear()`` so the wrapper must be
  importable from the package top-level.
- ``get_rules_registry`` — public entry point. Looked up via the package
  namespace (e.g. ``phonology.search.get_rules_registry``) by callers; tests
  monkeypatch the matching attribute on ``phonology.search``.
- ``_get_tokenized_rules`` — tokenizer cache keyed on (language, phone
  inventory). ``tests/conftest.py`` clears its cache the same way as
  ``_load_rules_cached``.

Test-seam policy: ``get_rules_registry`` and ``_load_rules_cached`` resolve
``load_rules``, ``get_language_profile``, and ``get_default_language_profile``
via the ``phonology.search`` package namespace at call time so that
``monkeypatch.setattr(search_module, "load_rules", ...)`` and
``monkeypatch.setattr(search_module, "get_language_profile", ...)`` continue
to take effect after this split.

The constant ``_TOKENIZED_RULES_CACHE_MAXSIZE`` is re-exported from the package
``__init__`` for any test or instrumentation that still references it there.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from ..explainer import TokenizedRule
from ._types import PhoneInventory


_TOKENIZED_RULES_CACHE_MAXSIZE = 64


@lru_cache(maxsize=8)
def _load_rules_cached(rules_source: str | Path) -> dict[str, dict[str, Any]]:
    """Inner cached function that loads rules from a canonical source.

    Looks ``load_rules`` up on the ``phonology.search`` package so test
    monkeypatches that replace it there take effect on cache misses.
    """
    from . import load_rules  # type: ignore[attr-defined]

    return load_rules(rules_source)


def get_rules_registry(
    language: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Load the packaged rule registry for the specified language.

    Loads packaged phonological rules via ``load_rules`` and caches the result
    once per process. Subsequent calls with the same language return the
    cached registry.

    Args:
        language: The language identifier for which to load rules.
            ``None`` uses the default language profile's rules directory.

    Returns:
        A dictionary mapping rule IDs to rule definitions. Each rule definition
        is a dict[str, Any] containing the rule's metadata and patterns.

    Raises:
        OSError: Propagated from ``load_rules`` if file system errors occur.
        ValueError: Propagated from ``load_rules`` if rule validation fails,
            or raised if the registry cannot be loaded for the given language.
            Non-default language load failures include the original error
            information in the exception message.
        yaml.YAMLError: Propagated from ``load_rules`` if YAML parsing fails.
    """
    from . import (  # type: ignore[attr-defined]
        get_default_language_profile,
        get_language_profile,
    )

    rules_source = language
    if language is None:
        rules_source = get_default_language_profile().rules_dir
    elif isinstance(language, str):
        try:
            # Strict: an unregistered language id is a programmer error, not a
            # directory name. Pass Path(...) when a rules directory is intended.
            rules_source = get_language_profile(language).rules_dir
        except ValueError as err:
            default_profile = get_default_language_profile()
            if language.strip().lower() == default_profile.language_id:
                rules_source = default_profile.rules_dir
            else:
                raise ValueError(
                    f"get_rules_registry failed to load rules for language {language!r}: {err}"
                ) from err

    try:
        return _load_rules_cached(rules_source)
    except (OSError, ValueError, yaml.YAMLError) as err:
        raise ValueError(
            f"get_rules_registry failed to load rules for language {language!r}: {err}"
        ) from err


@lru_cache(maxsize=_TOKENIZED_RULES_CACHE_MAXSIZE)
def _get_tokenized_rules(
    language: str | Path | None = None,
    phone_inventory: PhoneInventory | None = None,
    always_match_contexts: tuple[str, ...] | None = None,
) -> tuple[TokenizedRule, ...]:
    """Get tokenized rules from the registry for matching.

    Uses normalized cache keys to avoid duplicate entries when callers pass
    non-canonical inventories. Consider increasing maxsize if many inventories
    are expected - monitor cache hit/miss patterns in production.

    Looks ``get_rules_registry`` and ``tokenize_rules_for_matching`` up on the
    ``phonology.search`` package so test monkeypatches that replace either of
    them there take effect on cache misses.
    """
    import sys

    from ..explainer import tokenize_rules_for_matching as _default_tokenize

    package = sys.modules.get("phonology.search")
    _get_rules_registry = getattr(package, "get_rules_registry", get_rules_registry)
    _tokenize_rules = getattr(package, "tokenize_rules_for_matching", _default_tokenize)
    rules_registry = _get_rules_registry(language)
    rules = list(rules_registry.values())
    tokenize_kwargs: dict[str, Any] = {}
    if always_match_contexts is not None:
        tokenize_kwargs["always_match_contexts"] = always_match_contexts
    if phone_inventory is not None:
        tokenize_kwargs["phone_inventory"] = phone_inventory
    return tuple(_tokenize_rules(rules, **tokenize_kwargs))
