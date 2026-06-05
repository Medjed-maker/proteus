"""Regression gate for language-specific terms in core phonology modules."""

from __future__ import annotations

from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "src" / "phonology"
FORBIDDEN_LANGUAGE_TERMS = re.compile(
    r"attic|doric|ionic|koine|ancient_greek",
    re.IGNORECASE,
)

# Strict mode: the core is language-independent, so no allowlist is permitted.
# Any Ancient Greek vocabulary appearing in a core module (anything under
# ``src/phonology/`` except ``languages/``) fails this gate. Language-specific
# terms belong exclusively in ``src/phonology/languages/<language>/``.
ALLOWLIST: set[tuple[str, int, str]] = set()


def _scan_core_language_terms() -> set[tuple[str, int, str]]:
    hits: set[tuple[str, int, str]] = set()
    for path in sorted(CORE_ROOT.rglob("*.py")):
        relative_to_core = path.relative_to(CORE_ROOT)
        if "languages" in relative_to_core.parts:
            continue
        relative_path = path.relative_to(REPO_ROOT).as_posix()
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            for match in FORBIDDEN_LANGUAGE_TERMS.finditer(line):
                hits.add((relative_path, line_number, match.group(0).lower()))
    return hits


def _format_hits(hits: set[tuple[str, int, str]]) -> str:
    return "\n".join(f"{path}:{line}: {term}" for path, line, term in sorted(hits))


def test_core_language_terms_match_migration_allowlist() -> None:
    """Fail on new core language coupling and stale migration allowlist entries."""
    actual_hits = _scan_core_language_terms()
    unexpected_hits = actual_hits - ALLOWLIST
    stale_allowlist_entries = ALLOWLIST - actual_hits

    assert not unexpected_hits, (
        "Unexpected language-specific terms appeared in core phonology files:\n"
        f"{_format_hits(unexpected_hits)}"
    )
    assert not stale_allowlist_entries, (
        "Core language-independence allowlist contains stale entries:\n"
        f"{_format_hits(stale_allowlist_entries)}"
    )
