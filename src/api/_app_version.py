"""Application version resolution decoupled from the FastAPI app module."""

from __future__ import annotations

import logging
import os
import tomllib
from importlib import metadata
from pathlib import Path

_APP_VERSION_ENV_VAR = "PROTEUS_APP_VERSION"

logger = logging.getLogger(__name__)


def _load_app_version() -> str:
    """Return the application version from env, package metadata, or pyproject."""
    # Strip only one leading ``v``: ``lstrip("v")`` collapses doubled ``v``s
    # like ``"vv1.0"`` to ``"1.0"`` instead of the intended ``"v1.0"``.
    env_version = os.environ.get(_APP_VERSION_ENV_VAR, "").strip().removeprefix("v")
    if env_version:
        return env_version

    try:
        return metadata.version("proteus")
    except metadata.PackageNotFoundError:
        pass

    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        with pyproject_path.open("rb") as pyproject_file:
            pyproject = tomllib.load(pyproject_file)
    except (OSError, tomllib.TOMLDecodeError):
        logger.exception("Failed to read application version from %s", pyproject_path)
        return "unknown"

    project = pyproject.get("project", {})
    if isinstance(project, dict):
        version = project.get("version")
        if isinstance(version, str) and version.strip():
            return version.strip()

    logger.error("Application version is missing from %s", pyproject_path)
    return "unknown"


__all__ = ["_APP_VERSION_ENV_VAR", "_load_app_version"]
