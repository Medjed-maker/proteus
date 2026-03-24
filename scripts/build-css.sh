#!/usr/bin/env bash
set -euo pipefail

TAILWIND_VERSION="v3.4.17"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN_DIR="${PROJECT_ROOT}/.tailwind"
INPUT="${PROJECT_ROOT}/src/proteus/web/input.css"
OUTPUT="${PROJECT_ROOT}/src/proteus/web/static/styles.css"

# Pinned SHA-256 checksums for tailwindcss v3.4.17 binaries.
# To update: download sha256sums.txt from the release page and replace these values.
# https://github.com/tailwindlabs/tailwindcss/releases/tag/v3.4.17
CHECKSUM_linux_arm64="69b1378b8133192d7d2feb12a116fa12d035594f58db3eff215879e4ad8cf39b"
CHECKSUM_linux_x64="7d24f7fa191d2193b78cd5f5a42a6093e14409521908529f42d80b11fde1f1d4"
CHECKSUM_macos_arm64="a1d0c7985759accca0bf12e51ac1dcbf0f6cf2fffb62e6e0f62d091c477a10a3"
CHECKSUM_macos_x64="6cbdad74be776c087ffa5e9a057512c54898f9fe8828d3362212dfe32fc933a3"

# Detect platform
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
case "${OS}" in
  darwin) PLATFORM="macos" ;;
  linux)  PLATFORM="linux" ;;
  *)      echo "Unsupported OS: ${OS}" >&2; exit 1 ;;
esac
case "${ARCH}" in
  x86_64)       SUFFIX="${PLATFORM}-x64" ;;
  arm64|aarch64) SUFFIX="${PLATFORM}-arm64" ;;
  *)            echo "Unsupported arch: ${ARCH}" >&2; exit 1 ;;
esac

BINARY="${BIN_DIR}/tailwindcss-${TAILWIND_VERSION}-${SUFFIX}"

# Download if not present
if [ ! -x "${BINARY}" ]; then
  mkdir -p "${BIN_DIR}"
  URL="https://github.com/tailwindlabs/tailwindcss/releases/download/${TAILWIND_VERSION}/tailwindcss-${SUFFIX}"

  TMP_BINARY="$(mktemp "${BIN_DIR}/tailwindcss.XXXXXX")"
  cleanup() {
    rm -f "${TMP_BINARY}"
  }
  trap cleanup EXIT

  echo "Downloading tailwindcss ${TAILWIND_VERSION} for ${SUFFIX}..."
  curl -sSL --fail "${URL}" -o "${TMP_BINARY}"

  # Verify SHA-256 integrity against pinned checksum
  echo "Verifying checksum..."
  CHECKSUM_VAR="CHECKSUM_${SUFFIX//-/_}"
  EXPECTED="${!CHECKSUM_VAR}"
  if [ -z "${EXPECTED}" ]; then
    echo "ERROR: No pinned checksum for tailwindcss-${SUFFIX}." >&2
    exit 1
  fi

  if command -v sha256sum &>/dev/null; then
    ACTUAL=$(sha256sum "${TMP_BINARY}" | awk '{print $1}')
  elif command -v shasum &>/dev/null; then
    ACTUAL=$(shasum -a 256 "${TMP_BINARY}" | awk '{print $1}')
  else
    echo "ERROR: sha256sum or shasum is required but neither was found." >&2
    exit 1
  fi

  if [ "${ACTUAL}" != "${EXPECTED}" ]; then
    echo "ERROR: Checksum mismatch for tailwindcss-${SUFFIX}" >&2
    echo "  Expected: ${EXPECTED}" >&2
    echo "  Actual:   ${ACTUAL}" >&2
    exit 1
  fi
  echo "Checksum verified: ${ACTUAL}"

  chmod +x "${TMP_BINARY}"
  if mv "${TMP_BINARY}" "${BINARY}"; then
    trap - EXIT
  else
    echo "ERROR: Failed to install tailwindcss binary to ${BINARY}" >&2
    exit 1
  fi
  echo "Downloaded: ${BINARY}"
fi

# Build minified CSS
mkdir -p "$(dirname "${OUTPUT}")"
"${BINARY}" -c "${PROJECT_ROOT}/tailwind.config.js" -i "${INPUT}" -o "${OUTPUT}" --minify
echo "Built: ${OUTPUT} ($(wc -c < "${OUTPUT}" | tr -d ' ') bytes)"
