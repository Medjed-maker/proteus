"""Alignment visualization and dialect attribution helpers.

``_apply_rule_markers`` stays in ``phonology.search.__init__`` because it
calls ``tokenize_ipa`` by bare name, which tests monkeypatch at
``search_module`` level. The helpers here are pure string/list
manipulation and do not touch any monkeypatched surface.
"""

from __future__ import annotations

from ..explainer import RuleApplication
from ._constants import OBSERVED_PREFIX
from ._types import LexiconEntry


def _build_alignment_markers(
    aligned_query: list[str | None],
    aligned_lemma: list[str | None],
) -> list[str]:
    """Build baseline visualization markers for an aligned token sequence."""
    markers: list[str] = []
    for query_token, lemma_token in zip(aligned_query, aligned_lemma, strict=True):
        if query_token is not None and lemma_token is not None and query_token == lemma_token:
            markers.append("|")
        elif query_token is None or lemma_token is None:
            markers.append(" ")
        else:
            markers.append(".")
    return markers


def _is_observed_application(application: RuleApplication) -> bool:
    """Return True for synthetic observed-difference annotations."""
    rule_id = application.rule_id
    if not isinstance(rule_id, str) or not rule_id:
        return False
    return rule_id.startswith(OBSERVED_PREFIX)


def _collect_application_dialects(applications: list[RuleApplication]) -> list[str]:
    """Return dialect labels from rule applications in first-seen order."""
    matched_dialects: list[str] = []
    seen_dialects: set[str] = set()
    for application in applications:
        for dialect in application.dialects:
            if dialect in seen_dialects:
                continue
            seen_dialects.add(dialect)
            matched_dialects.append(dialect)
    return matched_dialects


def _format_alignment_visualization(
    aligned_query: list[str | None],
    aligned_lemma: list[str | None],
    markers: list[str],
) -> str:
    """Render a fixed three-line ASCII alignment visualization."""
    query_label = "query: "
    lemma_label = "lemma: "
    marker_prefix = " " * len(query_label)
    query_cells = [token if token is not None else "-" for token in aligned_query]
    lemma_cells = [token if token is not None else "-" for token in aligned_lemma]

    if not query_cells and not lemma_cells:
        return f"{query_label.rstrip()}\n{marker_prefix.rstrip()}\n{lemma_label.rstrip()}"

    widths = [
        max(len(query_cell), len(lemma_cell), 1)
        for query_cell, lemma_cell in zip(query_cells, lemma_cells, strict=True)
    ]
    query_line = query_label + " ".join(
        f"{cell:<{width}}" for cell, width in zip(query_cells, widths, strict=True)
    )
    marker_line = marker_prefix + " ".join(
        f"{marker:<{width}}" for marker, width in zip(markers, widths, strict=True)
    )
    lemma_line = lemma_label + " ".join(
        f"{cell:<{width}}" for cell, width in zip(lemma_cells, widths, strict=True)
    )
    return "\n".join([query_line.rstrip(), marker_line.rstrip(), lemma_line.rstrip()])


def _candidate_dialect(entry: LexiconEntry) -> str:
    """Return a normalized display label for a candidate's lemma dialect."""
    dialect = entry.get("dialect")
    if dialect is None or str(dialect).strip() == "":
        return "unknown"
    return str(dialect)


def _build_dialect_attribution(
    candidate_dialect: str,
    matched_dialects: list[str] | None = None,
) -> str:
    """Return the public dialect attribution string for a search hit."""
    if matched_dialects:
        return (
            f"lemma dialect: {candidate_dialect}; "
            f"query-compatible dialects: {', '.join(matched_dialects)}"
        )
    return f"lemma dialect: {candidate_dialect}"
