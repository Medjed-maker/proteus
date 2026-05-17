"""Export the FastAPI OpenAPI schema as a stable JSON artifact."""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path
from typing import Any

try:
    from scripts._cli_utils import positive_int
except ModuleNotFoundError:  # pragma: no cover - exercised by direct script usage
    from _cli_utils import positive_int

DEFAULT_OUTPUT = Path("docs/api/openapi.json")


def _schema_text(*, indent: int) -> str:
    """Return the current OpenAPI schema serialized deterministically."""
    from api.main import app

    schema: dict[str, Any] = app.openapi()
    return json.dumps(schema, indent=indent, sort_keys=True) + "\n"


def _check_artifact(output: Path, generated: str) -> int:
    """Return 0 when the committed artifact matches the generated schema."""
    try:
        existing = output.read_text(encoding="utf-8")
    except FileNotFoundError:
        existing = ""

    if existing == generated:
        return 0

    diff = difflib.unified_diff(
        existing.splitlines(keepends=True),
        generated.splitlines(keepends=True),
        fromfile=str(output),
        tofile=f"{output} (generated)",
    )
    sys.stderr.writelines(diff)
    print(
        "\nOpenAPI artifact is out of date. Regenerate it with:\n"
        f"  uv run python scripts/export_openapi.py --output {output}",
        file=sys.stderr,
    )
    return 1


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the OpenAPI export script."""
    parser = argparse.ArgumentParser(
        description="Export the Proteus FastAPI OpenAPI schema."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"schema artifact path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--indent",
        type=positive_int,
        default=2,
        help="JSON indentation width (default: 2)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the committed artifact differs from the generated schema",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    output: Path = args.output
    generated = _schema_text(indent=args.indent)

    if args.check:
        return _check_artifact(output, generated)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(generated, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
