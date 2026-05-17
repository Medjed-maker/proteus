"""Export Proteus MCP tool schemas."""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
import sys
from typing import Any, Callable

import anyio

try:
    from scripts._cli_utils import positive_int
except ModuleNotFoundError:  # pragma: no cover - exercised by direct script usage
    from _cli_utils import positive_int

from mcp_server.server import app
from mcp_server.tools.search import McpSearchOutput

DEFAULT_OUTPUT = Path("docs/mcp/tools.json")

_OUTPUT_SCHEMAS: dict[str, Callable[[], dict[str, Any]]] = {
    "ancient_phonology.search": McpSearchOutput.model_json_schema,
}


async def _build_schema() -> dict[str, Any]:
    """Return the current MCP tool schema payload.

    Lists tools from ``app.list_tools()``, converts each tool into a
    JSON-serializable payload, and injects output schemas from the explicit
    tool-name mapping above. Returns a ``dict[str, Any]`` containing a
    name-sorted ``tools`` list. This reads the MCP app's registered tool state
    but does not write files.
    """
    tools = await app.list_tools()
    tool_payloads: list[dict[str, Any]] = []
    for tool in tools:
        payload = tool.model_dump(mode="json")
        output_schema = _OUTPUT_SCHEMAS.get(str(payload["name"]))
        if output_schema is not None:
            payload["outputSchema"] = output_schema()
        tool_payloads.append(payload)
    return {"tools": sorted(tool_payloads, key=lambda item: str(item["name"]))}


def _json_bytes(payload: dict[str, Any], *, indent: int) -> bytes:
    """Serialize a payload to pretty-printed UTF-8 JSON bytes.

    Args:
        payload: JSON-serializable mapping to serialize.
        indent: JSON indentation width.

    Returns:
        UTF-8 encoded bytes with a trailing newline, using ``ensure_ascii=False``
        and ``sort_keys=True`` for deterministic readable output.
    """
    text = json.dumps(payload, ensure_ascii=False, indent=indent, sort_keys=True)
    return f"{text}\n".encode("utf-8")


def main() -> int:
    """Export MCP tool schemas or verify that the committed artifact is current."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--indent", type=positive_int, default=2)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    payload = anyio.run(_build_schema)
    rendered = _json_bytes(payload, indent=args.indent)

    if args.check:
        try:
            current = args.output.read_bytes()
        except FileNotFoundError:
            print(f"{args.output} does not exist", file=sys.stderr)
            return 1
        if current != rendered:
            current_text = current.decode("utf-8")
            rendered_text = rendered.decode("utf-8")
            diff = difflib.unified_diff(
                current_text.splitlines(keepends=True),
                rendered_text.splitlines(keepends=True),
                fromfile=str(args.output),
                tofile="generated",
                lineterm="",
            )
            sys.stderr.writelines(diff)
            return 1
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
