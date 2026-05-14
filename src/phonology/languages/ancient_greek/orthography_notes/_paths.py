"""Filesystem path resolver for packaged orthographic correspondence data."""

from __future__ import annotations

from pathlib import Path

from phonology._paths import DEFAULT_LANGUAGE_ID, resolve_language_data_dir

from .schema import _ORTHOGRAPHY_FILENAME


def _orthography_data_path() -> Path:
    """Return the packaged orthographic correspondence YAML path."""
    return (
        resolve_language_data_dir(DEFAULT_LANGUAGE_ID, "orthography")
        / _ORTHOGRAPHY_FILENAME
    )
