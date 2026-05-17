"""Proteus MCP server package."""

from __future__ import annotations

# Use the API module's already-resolved version so MCP and REST surfaces share
# a single source of truth and never drift if PROTEUS_APP_VERSION is set after
# api.main is imported.
from api.main import APP_VERSION as __version__

__all__ = ["__version__"]
