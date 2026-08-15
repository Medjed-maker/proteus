"""Licence-manifest tests for the Dilemma / PapyGreek benchmark artefacts.

Section 5.1 of ``DATA_LICENSE.md`` declares which committed benchmark files are
CC BY-SA 4.0 derivatives of the PapyGreek Treebanks. That declaration is a
licence notice to third parties, so it must not drift from what the repository
actually ships -- in either direction. Under-declaring omits a share-alike
notice someone is entitled to; over-declaring would encumber the repository's
own separately licensed code.

Two markers identify a derivative, and only two:

* a run of three or more Greek letters, and
* one of the EpiDoc document filenames PapyGreek itself enumerates in
  ``papygreek_ids.json``.

A bare Trismegistos number is deliberately not a marker. TM ids are 4-6 digit
integers and collide with the counts and percentages that fill the numeric
result summaries, so using them would flag files carrying nothing from
PapyGreek.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_LICENSE_PATH = ROOT_DIR / "DATA_LICENSE.md"
BENCHMARK_DIR = ROOT_DIR / "tools" / "benchmarks" / "dilemma"
PAPYGREEK_IDS_PATH = BENCHMARK_DIR / "papygreek_ids.json"
RESULTS_STATUS_PATH = BENCHMARK_DIR / "RESULTS_STATUS.md"

SCANNED_DIRS = ("tools/benchmarks/dilemma", "docs/benchmarks")

#: Greek and Greek Extended. Matches the character class quoted in DATA_LICENSE.
GREEK_RUN = re.compile(r"[Ͱ-Ͽἀ-῿]{3,}")

#: The only files the declaration exempts. Benchmark scripts are the
#: repository's own code (separately licensed, see LICENSE) and quote Greek
#: illustratively -- phone inventories, generic example words such as
#: "anthropos", and short forms cited in a comment to justify a parsing
#: decision. `test_benchmark_scripts_only_quote_greek_illustratively` holds that
#: assumption to account rather than leaving it unchecked.
#:
#: Everything else committed under the scanned directories is declarable,
#: whatever its suffix. An allowlist of the suffixes committed today (.json,
#: .md) would be narrower than the rule DATA_LICENSE.md section 5.1 actually
#: states -- a new .jsonl or .txt carrying PapyGreek forms would be a
#: derivative that no test asked about. Between the two tests, every committed
#: file is therefore accounted for exactly once: code here, data and prose
#: against the manifest.
CODE_SUFFIXES = frozenset({".py"})

#: Above this many distinct Greek forms, a script is no longer quoting
#: illustratively -- it is carrying a dataset, and belongs in the declaration.
#: The current maximum is 29 (test_clean.py).
MAX_ILLUSTRATIVE_GREEK_FORMS_PER_SCRIPT = 60

MANIFEST_START = "the CC BY-SA 4.0 derivatives committed here are:"
MANIFEST_END = "The remaining committed results_*.json"

BACKTICKED = re.compile(r"`([^`]+)`")
RESULTS_STATUS_ROW = re.compile(
    r"\|\s*`(results_[^`]+)`\s*\|\s*`([0-9a-f]{64})`\s*\|\s*"
    r"(Historical|Current|Unclassified)\s*\|\s*([^|\n]+)\|"
)
SHA256 = re.compile(r"[0-9a-f]{64}")


def _committed_paths() -> tuple[Path, ...]:
    """Return every benchmark file git would include in a commit.

    This is the tracked files plus the untracked-but-not-ignored ones, which is
    exactly the set a ``git add`` of these directories would stage.
    """
    try:
        completed = subprocess.run(
            [
                "git",
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
                *SCANNED_DIRS,
            ],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:  # pragma: no cover - git absent
        pytest.skip("git is not available")
    if completed.returncode != 0:  # pragma: no cover - not a work tree
        pytest.skip("not a git work tree")

    return tuple(
        ROOT_DIR / line
        for line in completed.stdout.split("\0")
        if line
    )


def test_committed_paths_preserves_spaces(monkeypatch: pytest.MonkeyPatch) -> None:
    """Git paths must be parsed without treating spaces as separators."""
    calls: list[list[str]] = []

    def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(
            args, 0, "docs/benchmarks/a file.md\0", "")

    monkeypatch.setattr(subprocess, "run", run)

    assert _committed_paths() == (ROOT_DIR / "docs/benchmarks/a file.md",)
    assert "-z" in calls[0]


def _declared_paths() -> frozenset[str]:
    """Return the repo-relative paths declared in DATA_LICENSE.md section 5.1."""
    text = DATA_LICENSE_PATH.read_text(encoding="utf-8")

    start = text.find(MANIFEST_START)
    assert start != -1, (
        f"DATA_LICENSE.md no longer contains the manifest marker {MANIFEST_START!r}. "
        "If section 5.1 was restructured, update this test alongside it."
    )
    end = text.find(MANIFEST_END, start)
    assert end != -1, (
        f"DATA_LICENSE.md no longer contains the manifest terminator {MANIFEST_END!r}."
    )

    return frozenset(BACKTICKED.findall(text[start:end]))


def _papygreek_document_filenames() -> frozenset[str]:
    """Return the EpiDoc filenames of the evaluated PapyGreek documents."""
    payload = json.loads(PAPYGREEK_IDS_PATH.read_text(encoding="utf-8"))
    return frozenset(document["file"] for document in payload["docs"])


def _greek_forms(text: str) -> frozenset[str]:
    """Return the distinct Greek runs in ``text``."""
    return frozenset(GREEK_RUN.findall(text))


def _carries_papygreek(path: Path, document_filenames: frozenset[str]) -> bool:
    """Report whether ``path`` carries either PapyGreek marker."""
    text = path.read_text(encoding="utf-8", errors="replace")
    if _greek_forms(text):
        return True
    return any(filename in text for filename in document_filenames)


def test_declared_manifest_covers_every_papygreek_derivative() -> None:
    """Every committed artefact carrying a PapyGreek marker should be declared."""
    declared = _declared_paths()
    document_filenames = _papygreek_document_filenames()

    undeclared = sorted(
        path.relative_to(ROOT_DIR).as_posix()
        for path in _committed_paths()
        if path.suffix not in CODE_SUFFIXES
        and path.exists()
        and _carries_papygreek(path, document_filenames)
        and path.relative_to(ROOT_DIR).as_posix() not in declared
    )

    assert not undeclared, (
        "These committed files carry PapyGreek surface forms or document "
        "filenames but are missing from the CC BY-SA 4.0 declaration in "
        "DATA_LICENSE.md section 5.1:\n  " + "\n  ".join(undeclared)
    )


def test_declared_manifest_has_no_stale_entries() -> None:
    """Every declared path should still exist on disk and still be committed.

    Both halves matter. A path git still tracks but that no longer exists in
    the working tree is a staged deletion, and the declaration would outlive
    the file it describes.
    """
    committed = {
        path.relative_to(ROOT_DIR).as_posix()
        for path in _committed_paths()
        if path.exists()
    }

    stale = sorted(entry for entry in _declared_paths() if entry not in committed)

    assert not stale, (
        "DATA_LICENSE.md section 5.1 declares files that are no longer "
        "committed (deleted, renamed, or newly gitignored):\n  " + "\n  ".join(stale)
    )


def test_benchmark_scripts_only_quote_greek_illustratively() -> None:
    """Scripts may cite Greek forms, but must not carry a PapyGreek dataset.

    The declaration deliberately covers data and prose rather than code. This
    keeps that exemption honest: a script that names an evaluated document, or
    that accumulates Greek at dataset scale, has stopped being illustrative.
    """
    document_filenames = _papygreek_document_filenames()

    offenders: list[str] = []
    for path in _committed_paths():
        if path.suffix not in CODE_SUFFIXES or not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        relative = path.relative_to(ROOT_DIR).as_posix()

        named = sorted(name for name in document_filenames if name in text)
        if named:
            offenders.append(f"{relative}: names PapyGreek documents {named[:3]}")

        forms = _greek_forms(text)
        if len(forms) > MAX_ILLUSTRATIVE_GREEK_FORMS_PER_SCRIPT:
            offenders.append(f"{relative}: carries {len(forms)} distinct Greek forms")

    assert not offenders, (
        "These benchmark scripts have outgrown the 'code quotes Greek "
        "illustratively' exemption in DATA_LICENSE.md section 5.1. Either move "
        "the data out of the script, or declare the file:\n  "
        + "\n  ".join(offenders)
    )


def _results_status_rows() -> list[tuple[str, str, str, str]]:
    """Parse result status rows from the Markdown manifest."""
    return RESULTS_STATUS_ROW.findall(
        RESULTS_STATUS_PATH.read_text(encoding="utf-8"))


def _current_provenance_errors(name: str, payload: object) -> list[str]:
    """Return missing-provenance errors for a Current result payload."""
    records = payload if isinstance(payload, list) else [payload]
    if not records or any(not isinstance(record, dict) for record in records):
        return [f"{name}: result payload has no object records"]

    errors = []
    for index, record in enumerate(records):
        provenance = record.get("provenance")
        label = f"{name}[{index}]" if isinstance(payload, list) else name
        if not isinstance(provenance, dict):
            errors.append(f"{label}: missing provenance")
            continue
        code = provenance.get("benchmark_code")
        inputs = provenance.get("inputs")
        if not isinstance(code, dict) or not SHA256.fullmatch(
                str(code.get("sha256", ""))):
            errors.append(f"{label}: missing benchmark_code SHA-256")
        if not isinstance(inputs, dict) or not inputs:
            errors.append(f"{label}: missing input identities")
        else:
            for input_name, identity in inputs.items():
                if not _has_content_identity(identity):
                    errors.append(
                        f"{label}: input {input_name} has no content SHA-256")
    return errors


def _has_content_identity(identity: object) -> bool:
    """Report whether an input identity is content-addressed."""
    if not isinstance(identity, dict):
        return False
    if SHA256.fullmatch(str(identity.get("sha256", ""))):
        return True
    files = identity.get("files")
    return (
        isinstance(files, list)
        and bool(files)
        and all(
            isinstance(item, dict)
            and SHA256.fullmatch(str(item.get("sha256", "")))
            for item in files
        )
    )


def test_results_status_covers_and_validates_committed_files() -> None:
    """Every committed result should have one evidence-backed status row.

    RESULTS_STATUS.md decides citability by filename *and* digest, and requires
    a regenerated file to have its status re-established in the same change.
    A silent regeneration would otherwise leave the manifest asserting a
    provenance the bytes no longer have.
    """
    rows = _results_status_rows()
    assert rows, "RESULTS_STATUS.md declares no result files."

    names = [name for name, _, _, _ in rows]
    duplicates = sorted(name for name in set(names) if names.count(name) > 1)
    committed = {
        path.name
        for path in _committed_paths()
        if path.parent == BENCHMARK_DIR
        and path.name.startswith("results_")
        and path.suffix == ".json"
        and path.exists()
    }
    declared = set(names)
    inventory_errors = []
    if duplicates:
        inventory_errors.append("duplicate rows: " + ", ".join(duplicates))
    if committed - declared:
        inventory_errors.append(
            "unlisted committed results: " + ", ".join(sorted(committed - declared)))
    if declared - committed:
        inventory_errors.append(
            "listed non-committed results: " + ", ".join(sorted(declared - committed)))

    mismatches: list[str] = inventory_errors
    for name, expected_digest, status, reason in rows:
        path = BENCHMARK_DIR / name
        if not path.exists():
            mismatches.append(f"{name}: listed but missing")
            continue
        actual_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_digest != expected_digest:
            mismatches.append(f"{name}: {expected_digest[:12]} -> {actual_digest[:12]}")
        if not reason.strip():
            mismatches.append(f"{name}: status reason is empty")
        if status == "Current":
            payload = json.loads(path.read_text(encoding="utf-8"))
            mismatches.extend(_current_provenance_errors(name, payload))

    assert not mismatches, (
        "RESULTS_STATUS.md is out of date. A regenerated result is not citable "
        "until its row records the new digest and provenance:\n  "
        + "\n  ".join(mismatches)
    )


def test_current_provenance_requires_code_and_inputs() -> None:
    """A digest alone must never be sufficient for Current status."""
    assert _current_provenance_errors("result.json", {}) == [
        "result.json: missing provenance"]

    incomplete = {"provenance": {"benchmark_code": {"sha256": "0" * 64}}}
    assert _current_provenance_errors("result.json", incomplete) == [
        "result.json: missing input identities"]

    unhashed = {
        "provenance": {
            "benchmark_code": {"sha256": "0" * 64},
            "inputs": {"dataset": {}},
        }
    }
    assert _current_provenance_errors("result.json", unhashed) == [
        "result.json: input dataset has no content SHA-256"]

    complete = {
        "provenance": {
            "benchmark_code": {"sha256": "0" * 64},
            "inputs": {"dataset": {"sha256": "1" * 64}},
        }
    }
    assert _current_provenance_errors("result.json", complete) == []
