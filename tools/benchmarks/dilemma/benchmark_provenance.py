"""Portable provenance records for claim-bearing benchmark artefacts."""

from __future__ import annotations

import hashlib
import json
from importlib import metadata
from pathlib import Path

DILEMMA_VERSION = "1.2.0"
DILEMMA_COMMIT = "f82f15a62ddce5d55c19b299c34a6c89476af5ce"
DILEMMA_DATA_FILES = (
    "lookup.db",
    "spell_index.db",
    "corpus_freq.json",
    "lemma_attestation.json",
)
BENCHMARK_DIR = Path(__file__).resolve().parent


def _update_digest(digest, path: Path) -> None:
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)


def _digest_paths(paths: list[Path], root: Path) -> str:
    """Hash relative path names and contents in deterministic order."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        name = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        _update_digest(digest, path)
    return digest.hexdigest()


def file_identity(path: Path, *, name: str | None = None) -> dict:
    """Return a portable identity for one required input file."""
    if not path.is_file():
        raise FileNotFoundError(f"provenance input not found: {path}")
    return {
        "name": name or path.name,
        "bytes": path.stat().st_size,
        "sha256": _file_digest(path),
    }


def directory_identity(path: Path, pattern: str = "*.txt") -> dict:
    """Return a content identity for a flat corpus directory."""
    root = path.resolve()
    files = sorted(root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no {pattern} provenance inputs under {path}")
    return {
        "name": root.name,
        "files": len(files),
        "bytes": sum(item.stat().st_size for item in files),
        "sha256": _digest_paths(files, root),
    }


def benchmark_code_identity() -> dict:
    """Return a content identity for the executable benchmark code.

    A repository commit cannot identify an uncommitted benchmark run. Hashing
    the runtime scripts themselves records the code that actually executed and
    remains portable across checkout locations. Tests are excluded because
    they do not participate in result generation.
    """
    files = sorted(
        path for path in BENCHMARK_DIR.glob("*.py")
        if not path.name.startswith("test_")
    )
    if not files:  # pragma: no cover - broken source distribution
        raise FileNotFoundError(f"no benchmark scripts under {BENCHMARK_DIR}")
    return {
        "name": "dilemma-benchmark-code",
        "files": len(files),
        "bytes": sum(item.stat().st_size for item in files),
        "sha256": _digest_paths(files, BENCHMARK_DIR),
    }


def _installed_vcs_commit(distribution: metadata.Distribution) -> str | None:
    """Read the installed VCS commit from PEP 610 direct-url metadata."""
    raw = distribution.read_text("direct_url.json")
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    vcs_info = payload.get("vcs_info")
    if not isinstance(vcs_info, dict):
        return None
    commit = vcs_info.get("commit_id")
    return commit if isinstance(commit, str) and commit else None


def dilemma_package_identity(*, require_expected: bool = False) -> dict:
    """Identify the installed Dilemma package and enforce the benchmark pin."""
    distribution = metadata.distribution("dilemma-nlp")
    actual_version = distribution.version
    actual_commit = _installed_vcs_commit(distribution)
    if require_expected:
        problems = []
        if actual_version != DILEMMA_VERSION:
            problems.append(
                f"version {DILEMMA_VERSION} required, found {actual_version}")
        if actual_commit != DILEMMA_COMMIT:
            found = actual_commit or "unverifiable (no PEP 610 VCS commit)"
            problems.append(
                f"commit {DILEMMA_COMMIT} required, found {found}")
        if problems:
            raise RuntimeError("Dilemma benchmark pin mismatch: " + "; ".join(problems))
    return {
        "name": "dilemma-nlp",
        "version": actual_version,
        "expected_version": DILEMMA_VERSION,
        "source_commit": actual_commit,
        "expected_source_commit": DILEMMA_COMMIT,
    }


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    _update_digest(digest, path)
    return digest.hexdigest()


def dilemma_data_identity(data_dir: Path) -> dict:
    """Hash the Dilemma files that define lookup and ranking behaviour."""
    root = data_dir.resolve()
    return {
        "name": root.name,
        "source_version": DILEMMA_VERSION,
        "expected_source_commit": DILEMMA_COMMIT,
        "files": [file_identity(root / name) for name in DILEMMA_DATA_FILES],
    }
