"""Runtime version metadata for REST and MCP surfaces."""

from __future__ import annotations

import importlib.resources as resources
from functools import lru_cache
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any

from ._app_version import (
    _APP_VERSION_ENV_VAR as _APP_VERSION_ENV_VAR,
    _load_app_version,
)
from ._constants import API_VERSION, SCHEMA_VERSION
from ._models import VersionInfo

logger = logging.getLogger(__name__)

_BUILD_TIMESTAMP_ENV_VAR = "PROTEUS_BUILD_TIMESTAMP"
_GIT_SHA_ENV_VAR = "PROTEUS_GIT_SHA"
# Limit publicly exposed git SHA to 12 hex chars. Long enough to be unique in
# practical git repos, short enough to limit deployment fingerprinting from
# the unauthenticated /version endpoint.
_GIT_SHA_MAX_LENGTH = 12

_APP_VERSION = _load_app_version()
APP_VERSION: str = _APP_VERSION


@lru_cache
def _load_rule_schema_version() -> str:
    """Return the rule-file JSON schema identifier, or an empty string."""
    schema_candidates: list[Any] = []
    try:
        schema_candidates.append(
            resources.files("phonology").joinpath(
                "data",
                "schemas",
                "phonology_rule_file.schema.json",
            )
        )
    except (ModuleNotFoundError, FileNotFoundError):
        pass

    schema_candidates.append(
        Path(__file__).resolve().parents[2]
        / "data"
        / "schemas"
        / "phonology_rule_file.schema.json"
    )

    for schema_path in schema_candidates:
        # Restrict to Path-like candidates; future Traversable variants without
        # filesystem semantics (e.g., MultiplexedPath under zipped packages) are
        # skipped explicitly so an AttributeError does not mask unrelated bugs.
        if not isinstance(schema_path, (str, os.PathLike)):
            logger.debug(
                "Skipping non-path-like schema candidate of type %s",
                type(schema_path).__name__,
            )
            continue
        try:
            with open(schema_path, "rb") as schema_file:
                schema = json.load(schema_file)
        except (OSError, json.JSONDecodeError):
            continue
        schema_id = schema.get("$id") if isinstance(schema, dict) else None
        if isinstance(schema_id, str):
            return schema_id

    return ""


def build_version_info() -> VersionInfo:
    """Return API and runtime version metadata shared by versioned endpoints."""
    python_version = ".".join(str(item) for item in sys.version_info[:3])
    return VersionInfo(
        engine_version=_APP_VERSION,
        api_version=API_VERSION,
        schema_version=SCHEMA_VERSION,
        rule_schema_version=_load_rule_schema_version(),
        build_timestamp=os.environ.get(_BUILD_TIMESTAMP_ENV_VAR, "").strip(),
        git_sha=os.environ.get(_GIT_SHA_ENV_VAR, "").strip()[:_GIT_SHA_MAX_LENGTH],
        python_version=python_version,
        mcp_server_version=_APP_VERSION,
    )


# Backward-compatible private name used by existing tests.
_build_version_info = build_version_info


__all__ = [
    "APP_VERSION",
    "_APP_VERSION",
    "_APP_VERSION_ENV_VAR",
    "_BUILD_TIMESTAMP_ENV_VAR",
    "_GIT_SHA_ENV_VAR",
    "_GIT_SHA_MAX_LENGTH",
    "_build_version_info",
    "_load_app_version",
    "_load_rule_schema_version",
    "build_version_info",
]
