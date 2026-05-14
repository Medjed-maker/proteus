"""Trusted-directory registry and rules-path resolution for the explainer.

Module-level state:
    _TRUSTED_EXTERNAL_RULES_DIRS:
        Set of absolute, symlink-resolved rule directories that may be loaded
        from outside the packaged rules base. Managed via
        :func:`register_trusted_rules_dir` and
        :func:`clear_trusted_external_rules_dirs`.
    _TRUSTED_EXTERNAL_RULES_DIRS_LOCK:
        ``threading.Lock`` guarding mutation of the trusted-directory set.
    _RULES_BASE_DIR_OVERRIDE:
        Test-only override consumed by :func:`_get_rules_base_dir`. Tests
        monkeypatch ``phonology.explainer._RULES_BASE_DIR_OVERRIDE``; the
        facade module re-exposes this attribute (see ``explainer.__getattr__``).

The logger uses the literal ``"phonology.explainer"`` name (not ``__name__``)
so that ``caplog.set_level(..., logger="phonology.explainer")`` in tests still
captures records emitted from this module.
"""

from __future__ import annotations

import logging
from pathlib import Path
import threading

from ._paths import (
    DEFAULT_LANGUAGE_ID,
    resolve_language_data_dir,
    resolve_repo_data_dir,
)
from ._trusted_paths import validate_no_symlinks_in_path

logger = logging.getLogger("phonology.explainer")

_RULES_BASE_DIR_OVERRIDE: Path | None = None
_TRUSTED_EXTERNAL_RULES_DIRS: set[Path] = set()
_TRUSTED_EXTERNAL_RULES_DIRS_LOCK = threading.Lock()


def _get_rules_base_dir() -> Path:
    """Lazily resolve the rules base directory.

    Tests can override resolution by monkeypatching the module-level
    ``_RULES_BASE_DIR_OVERRIDE`` attribute on ``phonology.explainer``; the
    facade looks the override up via ``__getattr__`` on this module.
    """
    from . import explainer

    override = getattr(explainer, "_RULES_BASE_DIR_OVERRIDE", _RULES_BASE_DIR_OVERRIDE)
    if override is not None:
        return override
    try:
        return resolve_language_data_dir(DEFAULT_LANGUAGE_ID, "rules")
    except FileNotFoundError:
        return resolve_repo_data_dir("rules")


def register_trusted_rules_dir(
    rules_dir: Path | str, *, allow_symlinks: bool = False
) -> None:
    """Trust a registered language profile's absolute rules directory.

    ``register_trusted_rules_dir`` stores ``Path(...).resolve(strict=True)`` in
    ``_TRUSTED_EXTERNAL_RULES_DIRS``. Callers are responsible for registering
    only explicitly validated absolute paths and avoiding unsafe symlink or
    unintended parent-directory targets; prefer stricter allowlists, symlink
    bans, or signed-rule validation where possible.

    Args:
        rules_dir: The rules directory to trust.
        allow_symlinks: If False (default), rejects paths containing symlinks.
                       Set to True to allow symlinked paths.

    Raises:
        ValueError: If symlinks are detected and allow_symlinks is False.
    """
    rules_path = Path(rules_dir).expanduser()
    if not rules_path.is_absolute():
        rules_path = Path.cwd() / rules_path

    if not allow_symlinks:
        validate_no_symlinks_in_path(
            rules_path,
            description="trusted rules directory",
        )

    resolved_rules_dir = rules_path.resolve(strict=True)
    with _TRUSTED_EXTERNAL_RULES_DIRS_LOCK:
        _TRUSTED_EXTERNAL_RULES_DIRS.add(resolved_rules_dir)


def clear_trusted_external_rules_dirs() -> None:
    """Test helper: clear externally-registered trusted rules directories."""
    with _TRUSTED_EXTERNAL_RULES_DIRS_LOCK:
        _TRUSTED_EXTERNAL_RULES_DIRS.clear()


def _resolve_rules_dir(rules_dir: Path | str, rules_base_dir: Path) -> Path:
    """Resolve rules directories relative to the packaged rules base."""
    candidate_rules_dir = Path(rules_dir)
    if candidate_rules_dir.is_absolute():
        return candidate_rules_dir

    parts = candidate_rules_dir.parts
    language_rules_prefix = ("data", "languages", DEFAULT_LANGUAGE_ID, "rules")
    if (
        len(parts) >= len(language_rules_prefix)
        and parts[: len(language_rules_prefix)] == language_rules_prefix
    ):
        candidate_rules_dir = Path(*parts[len(language_rules_prefix) :])
        return rules_base_dir / candidate_rules_dir

    if len(parts) >= 2 and parts[:2] == ("data", "rules"):
        candidate_rules_dir = Path(*parts[2:])

    # Legacy fallback for backward compatibility:
    # When candidate_rules_dir.parts == (DEFAULT_LANGUAGE_ID,), we have three fallback cases:
    # 1. Try legacy_language_dir (rules_base_dir / DEFAULT_LANGUAGE_ID) if it exists as a directory
    # 2. If rules_base_dir contains any YAML files (detected via glob("*.yaml")), use rules_base_dir directly
    # 3. Fall back to legacy_language_dir again (will fail gracefully if directory doesn't exist)
    # This handles old data layouts where rules were stored directly under language directories
    # or in the base rules directory without language subdirectories.
    if candidate_rules_dir.parts == (DEFAULT_LANGUAGE_ID,):
        legacy_language_dir = rules_base_dir / DEFAULT_LANGUAGE_ID
        if legacy_language_dir.is_dir():
            return legacy_language_dir
        if any(rules_base_dir.glob("*.yaml")):
            return rules_base_dir
        return legacy_language_dir

    return rules_base_dir / candidate_rules_dir


def _resolve_and_validate_rules_dir(rules_dir: Path | str, caller_name: str) -> Path:
    """Resolve and validate an internal rules directory path.

    This private helper resolves ``rules_dir`` relative to the packaged rules
    base directory returned by ``_get_rules_base_dir()``. Bare relative inputs
    such as ``"ancient_greek"`` and legacy repo-style paths such as
    ``"data/rules/ancient_greek"`` are both normalized under that base.

    The resolved path must exist, be a directory, and remain within the rules
    base. This prevents callers such as ``load_rules()`` and
    ``get_rules_version()`` from escaping the packaged rules tree via absolute
    paths, traversal segments, or symlink resolution.
    """
    rules_base_dir = _get_rules_base_dir()
    try:
        rules_base_dir = rules_base_dir.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(
            f"Configured rules base directory is missing: {exc}. "
            f"Create the {rules_base_dir} directory before calling {caller_name}()."
        ) from exc

    candidate_rules_dir = _resolve_rules_dir(rules_dir, rules_base_dir)
    try:
        resolved_rules_dir = candidate_rules_dir.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(
            f"{caller_name} could not find rules directory {candidate_rules_dir}. "
            f"Expected an existing directory within {rules_base_dir}."
        ) from exc
    with _TRUSTED_EXTERNAL_RULES_DIRS_LOCK:
        is_trusted_external_dir = resolved_rules_dir in _TRUSTED_EXTERNAL_RULES_DIRS
    if (
        not resolved_rules_dir.is_relative_to(rules_base_dir)
        and not is_trusted_external_dir
    ):
        raise ValueError(
            f"{caller_name} path must stay within {rules_base_dir}, got {resolved_rules_dir}"
        )
    if not resolved_rules_dir.is_dir():
        raise ValueError(
            f"{caller_name} expected a directory, got {resolved_rules_dir}"
        )

    return resolved_rules_dir
