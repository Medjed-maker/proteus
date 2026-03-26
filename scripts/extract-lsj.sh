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

# Check for required commands
command -v git >/dev/null 2>&1 || { echo "Error: git is required but not installed." >&2; exit 1; }
command -v uv >/dev/null 2>&1 || { echo "Error: uv is required but not installed." >&2; exit 1; }

LSJ_REPO_DIR="${PROJECT_ROOT}/data/external/lsj"
XML_DIR="${LSJ_REPO_DIR}/CTS_XML_TEI/perseus/pdllex/grc/lsj"

# Clone the Perseus lexica repo (sparse checkout, LSJ only) if not present
if [ ! -d "$XML_DIR" ]; then
    echo "==> Cloning Perseus LSJ repository (sparse checkout)..."
    # Clean up any partial clone from previous failed attempt
    rm -rf "${LSJ_REPO_DIR:?}"
    git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/PerseusDL/lexica.git "$LSJ_REPO_DIR"
    (
        cd "$LSJ_REPO_DIR"
        git sparse-checkout set CTS_XML_TEI/perseus/pdllex/grc/lsj
    )
    echo "==> LSJ XML files downloaded to ${XML_DIR}"
else
    echo "==> LSJ XML directory already exists: ${XML_DIR}"
fi

# Run the extraction
echo "==> Running LSJ extraction..."
uv run --extra extract python -m phonology.lsj_extractor --xml-dir "$XML_DIR" "$@"
