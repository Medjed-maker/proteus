#!/usr/bin/env bash
# Extract Greek lemma data from the Perseus LSJ XML files.
#
# Downloads the PerseusDL/lexica repository (sparse checkout of LSJ only)
# and invokes the Python extraction module.
#
# Usage:
#   bash scripts/extract-lsj.sh [--limit N] [--dry-run] [-v]
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Check for required command
command -v uv >/dev/null 2>&1 || { echo "Error: uv is required but not installed." >&2; exit 1; }

cd "$PROJECT_ROOT"
uv run --extra extract python -m phonology.build_lexicon "$@"
