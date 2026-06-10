"""MCP server entry point for Proteus."""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP
from phonology.core.ports.profiles import register_default_profiles

from .tools import register_buck_reference_tools, register_search_tool

logger = logging.getLogger("proteus.mcp")

app = FastMCP("proteus")
register_search_tool(app)
register_buck_reference_tools(app)


def main() -> None:
    """Run the Proteus MCP server over stdio."""
    logging.basicConfig(level=logging.INFO)
    register_default_profiles()
    logger.info("Starting Proteus MCP server over stdio")
    app.run(transport="stdio")


if __name__ == "__main__":
    main()
