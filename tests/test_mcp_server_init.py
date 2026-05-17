"""Tests for Proteus MCP server package initialization."""

from __future__ import annotations

from importlib import metadata

from api import main as api_main


def test_mcp_server_module_importable() -> None:
    """Importing the MCP server module should register tools without side effects."""
    import mcp_server.server as server

    assert server.app is not None


def test_mcp_server_entrypoint_script_resolves() -> None:
    """Editable installs should expose the proteus-mcp console script."""
    scripts = metadata.entry_points(group="console_scripts")
    entrypoint = next((item for item in scripts if item.name == "proteus-mcp"), None)

    assert entrypoint is not None
    assert entrypoint.value == "mcp_server.server:main"


def test_mcp_server_version_matches_app_version() -> None:
    """The MCP server version tracks the application engine version."""
    import mcp_server

    assert mcp_server.__version__ == api_main._APP_VERSION
