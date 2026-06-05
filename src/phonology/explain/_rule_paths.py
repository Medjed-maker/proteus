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

from .._paths import resolve_repo_data_dir
from .._trusted_paths import validate_no_symlinks_in_path

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
    from .. import explainer

    override = getattr(explainer, "_RULES_BASE_DIR_OVERRIDE", _RULES_BASE_DIR_OVERRIDE)
    if override is not None:
        return override
    try:
        from ..core.ports.profiles import get_default_language_profile

        return get_default_language_profile().rules_dir
    except (FileNotFoundError, ValueError):
        return resolve_repo_data_dir("rules")


def register_trusted_rules_dir(
    rules_dir: Path | str, *, allow_symlinks: bool = False
) -> None:
    """Trust a registered language profile's absolute rules directory.

    The input is first passed through ``Path(...).expanduser()``. If the result
    is not absolute, it is joined to ``Path.cwd()`` before optional symlink
    validation. The final ``Path(...).resolve(strict=True)`` value is stored in
    ``_TRUSTED_EXTERNAL_RULES_DIRS``. Callers should pass absolute paths when
    they want to avoid cwd-based interpretation.

    Callers are responsible for registering only explicitly validated paths and
    avoiding unsafe symlink or unintended parent-directory targets; prefer
    stricter allowlists, symlink bans, or signed-rule validation where
    possible.

    Args:
        rules_dir: The rules directory to trust.
        allow_symlinks: If False (default), rejects paths containing symlinks
            before storing the resolved directory. Set to True to allow
            symlinked paths.

    Raises:
        ValueError: If symlinks are detected and allow_symlinks is False.
    """
    rules_path = Path(rules_dir).expanduser()
    if not rules_path.is_absolute():
        rules_path = Path.cwd() / rules_path
    logger.debug("Resolved rules directory: original=%s, resolved=%s", rules_dir, rules_path)

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
    """Resolve rules directories relative to the packaged rules base.
    
    Handles absolute paths, language-specific ``data/languages/<id>/rules``
    paths, and legacy ``data/rules/...`` mapping.
    
    Args:
        rules_dir: Input path (absolute or relative)
        rules_base_dir: Base directory for relative resolution
        
    Returns:
        Resolved Path object (may be relative to rules_base_dir)
    """
    candidate_rules_dir = Path(rules_dir)
    if candidate_rules_dir.is_absolute():
        return candidate_rules_dir

    parts = candidate_rules_dir.parts
    if len(parts) >= 4 and parts[:2] == ("data", "languages") and parts[3] == "rules":
        suffix = parts[4:]
        # An empty suffix means the input names the language rules directory
        # itself; Path(".") keeps rules_base_dir / candidate_rules_dir equal to
        # rules_base_dir.
        candidate_rules_dir = Path(*suffix) if suffix else Path(".")
        return rules_base_dir / candidate_rules_dir

    if len(parts) >= 2 and parts[:2] == ("data", "rules"):
        candidate_rules_dir = Path(*parts[2:])

    if len(candidate_rules_dir.parts) == 1:
        candidate_under_base = rules_base_dir / candidate_rules_dir
        try:
            from ..core.ports.profiles import get_default_language_profile, get_language_profile

            default_rules_dir = get_default_language_profile().rules_dir.resolve(
                strict=False
            )
            if rules_base_dir.resolve(strict=False) != default_rules_dir:
                return candidate_under_base

            return get_language_profile(candidate_rules_dir.parts[0]).rules_dir
        except (FileNotFoundError, ValueError):
            pass

    return rules_base_dir / candidate_rules_dir


def _resolve_and_validate_rules_dir(rules_dir: Path | str, caller_name: str) -> Path:
    """Resolve and validate an internal rules directory path.

    This private helper resolves ``rules_dir`` relative to the packaged rules
    base directory returned by ``_get_rules_base_dir()``. Relative and
    repo-style inputs are both normalized under that base.

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
    # is_relative_to is a pure path comparison; compute it outside the lock to
    # keep the critical section limited to the shared-set membership check.
    is_relative = resolved_rules_dir.is_relative_to(rules_base_dir)
    if (
        not is_relative
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
