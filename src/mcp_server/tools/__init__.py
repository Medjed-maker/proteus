"""MCP tool registration helpers."""

from __future__ import annotations

from .buck import register_buck_reference_tools
from .search import register_search_tool

__all__ = ["register_buck_reference_tools", "register_search_tool"]
