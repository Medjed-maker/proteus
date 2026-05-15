"""LSJ extractor helper module.

Logger uses the literal ``phonology.lsj_extractor`` name so existing
``caplog.set_level(logger="phonology.lsj_extractor")`` blocks in
``tests/test_lsj_extractor.py`` keep capturing diagnostics from this module
after the split.
"""

from __future__ import annotations

import logging
from typing import Any

from ._constants import (
    _ATTIC_INLINE_RE,
    _INLINE_POS_PATTERNS,
    _POST_GLOSS_PRIMARY_POS_RE,
)
from ._normalize import _normalize_intro_text
from ._xml import _elem_text, _local_name

logger = logging.getLogger("phonology.lsj_extractor")


def _append_intro_child_text(
    parts: list[str], child: Any, *, skip_mood_text: bool = False
) -> None:
    """Append heading or sense-intro text from ``child`` to ``parts``.

    When ``skip_mood_text`` is true, the body of ``<mood>`` elements is ignored
    while their tail text is preserved. This keeps inline POS scans from
    misreading ``<mood>part.</mood>`` as the POS abbreviation ``Part.``.
    """
    local = _local_name(child)
    if local in {"cit", "quote", "bibl"}:
        parts.append(child.tail or "")
        return
    if skip_mood_text and local == "mood":
        body = _elem_text(child).strip().rstrip(".")
        if body.lower() == "part":
            # "part." would match the "particle" regex — skip it.
            # Unambiguous forms like "Participle" / "Partic." fall through.
            parts.append(child.tail or "")
            return
    parts.append(_elem_text(child))
    parts.append(child.tail or "")


def _entry_intro_text(entry: Any, *, skip_mood_text: bool = False) -> str:
    """Return the text that appears before the first top-level ``<sense>``."""
    parts: list[str] = [entry.text or ""]
    for child in entry:
        local = _local_name(child)
        if local == "sense":
            break
        _append_intro_child_text(parts, child, skip_mood_text=skip_mood_text)
    return _normalize_intro_text(parts)


def _sense_intro_texts(
    entry: Any, *, limit: int = 3, skip_mood_text: bool = False
) -> list[str]:
    """Return introductory text for the first few top-level ``<sense>`` blocks.

    Only the first *limit* senses are scanned because entry-level POS labels
    almost always appear in the opening senses; deeply nested sub-senses
    rarely contribute useful part-of-speech information.
    """
    texts: list[str] = []
    for child in entry:
        if _local_name(child) != "sense":
            continue
        if child.get("level") not in {None, "1"}:
            continue
        parts: list[str] = [child.text or ""]
        for sub in child:
            sub_local = _local_name(sub)
            if sub_local in {"tr", "sense"}:
                break
            _append_intro_child_text(parts, sub, skip_mood_text=skip_mood_text)
        text = _normalize_intro_text(parts)
        if text:
            texts.append(text)
        if len(texts) >= limit:
            break
    return texts


def _sense_post_gloss_pos_texts(entry: Any, *, limit: int = 3) -> list[str]:
    """Return post-gloss prose candidates for fallback POS inference only."""
    texts: list[str] = []
    for child in entry:
        if _local_name(child) != "sense":
            continue
        if child.get("level") not in {None, "1"}:
            continue
        seen_gloss = False
        parts: list[str] = []
        for sub in child:
            sub_local = _local_name(sub)
            if sub_local == "sense":
                break
            if not seen_gloss:
                if sub_local == "tr":
                    seen_gloss = True
                    parts.append(sub.tail or "")
                continue
            if sub_local == "tr":
                break
            if sub_local in {"cit", "quote", "bibl"}:
                parts.append(sub.tail or "")
                continue
            parts.append(_elem_text(sub))
            parts.append(sub.tail or "")
        text = _normalize_intro_text(parts)
        if text and _POST_GLOSS_PRIMARY_POS_RE.match(text):
            texts.append(text)
        if len(texts) >= limit:
            break
    return texts


def _inline_pos_candidates(text: str) -> list[tuple[bool, int, str]]:
    """Return inline POS candidates as ``(attic_priority, index, pos)`` tuples."""
    candidates: list[tuple[bool, int, str]] = []
    for pos, pattern in _INLINE_POS_PATTERNS:
        for match in pattern.finditer(text):
            attic_priority = _has_attic_inline_context(text, match.start())
            candidates.append((attic_priority, match.start(), pos))
    return sorted(candidates, key=lambda item: item[1])


def _has_attic_inline_context(text: str, match_start: int) -> bool:
    """Return True when the match belongs to a clause marked as Attic."""
    clause_start = 0
    for delimiter in (";", ":"):
        clause_start = max(clause_start, text.rfind(delimiter, 0, match_start) + 1)
    return bool(_ATTIC_INLINE_RE.search(text[clause_start:match_start]))

