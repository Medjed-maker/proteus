"""Proteus MCP server package."""

from __future__ import annotations

# Use the runtime metadata module's already-resolved version so MCP and REST
# surfaces share a single source of truth without importing the REST app.
from api._runtime_metadata import APP_VERSION as __version__

__all__ = ["__version__"]
