"""Trusted-directory registry for language-profile matrix assets."""

from __future__ import annotations

import os
from pathlib import Path
import threading

from ..._trusted_paths import validate_no_symlinks_in_path

_TRUSTED_EXTERNAL_MATRIX_DIRS: set[Path] = set()
_TRUSTED_EXTERNAL_MATRIX_DIRS_LOCK = threading.Lock()


def register_trusted_matrices_dir(
    matrices_dir: Path | str,
    *,
    allow_symlinks: bool = False,
) -> None:
    """Register an additional directory from which matrix JSON files may be loaded.

    The directory is resolved with ``strict=True`` and stored as a trusted
    matrix root. Symlinked directories are stored by their absolute lexical
    path only when explicitly allowed, matching the distance loader's lexical
    containment checks.
    """
    matrices_path = Path(matrices_dir).expanduser()
    if not matrices_path.is_absolute():
        matrices_path = Path.cwd() / matrices_path

    if not allow_symlinks:
        validate_no_symlinks_in_path(
            matrices_path,
            description="trusted matrices directory",
        )

    resolved = matrices_path.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"matrices_dir must be a directory: {resolved}")
    trusted_path = Path(os.path.abspath(matrices_path)) if allow_symlinks else resolved
    with _TRUSTED_EXTERNAL_MATRIX_DIRS_LOCK:
        _TRUSTED_EXTERNAL_MATRIX_DIRS.add(trusted_path)


def clear_trusted_external_matrix_dirs() -> None:
    """Clear externally registered trusted matrix directories."""
    with _TRUSTED_EXTERNAL_MATRIX_DIRS_LOCK:
        _TRUSTED_EXTERNAL_MATRIX_DIRS.clear()


def list_trusted_external_matrix_dirs() -> tuple[Path, ...]:
    """Return a sorted snapshot of externally registered trusted matrix dirs."""
    with _TRUSTED_EXTERNAL_MATRIX_DIRS_LOCK:
        return tuple(sorted(_TRUSTED_EXTERNAL_MATRIX_DIRS))


__all__ = [
    "clear_trusted_external_matrix_dirs",
    "list_trusted_external_matrix_dirs",
    "register_trusted_matrices_dir",
]
