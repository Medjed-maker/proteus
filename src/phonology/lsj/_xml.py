"""LSJ extractor helper module — XML traversal helpers.

Logger uses the literal ``phonology.lsj_extractor`` name so existing
``caplog.set_level(logger="phonology.lsj_extractor")`` blocks in
``tests/test_lsj_extractor.py`` keep capturing diagnostics from this module
after the split.
"""

from __future__ import annotations

import logging
from typing import Any, cast

logger = logging.getLogger("phonology.lsj_extractor")


def _elem_text(element: Any) -> str:
    """Return the full text content of an element (including tail of children)."""
    return "".join(element.itertext()).strip() if element is not None else ""


def _find_text(parent: Any, tag: str, **attribs: str) -> str:
    """Find the first child matching tag and optional attributes, return text."""
    for child in parent:
        local = _local_name(child)
        if local != tag:
            continue
        if all(child.get(k) == v for k, v in attribs.items()):
            return _elem_text(child)
    return ""


def _find_text_deep(parent: Any, tag: str, **attribs: str) -> str:
    """Search heading-area descendants for matching tag (fallback for nested elements).

    Unlike ``_find_text`` which only searches direct children, this searches
    descendant subtrees of direct children that are part of the heading prose,
    while skipping citation/bookkeeping blocks that may mention other forms.
    This prevents matching ``<gen>`` elements inside citations that do not
    describe the headword itself.

    Used as a fallback when direct-child search fails due to malformed XML
    nesting (e.g. ``<gen>`` inside ``<foreign>``).
    """
    for child in parent:
        if _local_name(child) in {"sense", "cit", "quote", "bibl"}:
            continue
        for descendant in child.iter():
            if _local_name(descendant) != tag:
                continue
            if all(descendant.get(k) == v for k, v in attribs.items()):
                return _elem_text(descendant)
    return ""


def _find_gen_text(entry: Any) -> str:
    """Find gender text from ``<gen>`` element, with fallback for nested tags.

    Some LSJ entries (e.g. βίος / n19972) have ``<gen>`` nested inside
    ``<foreign>`` elements.  Try direct children first, then fall back to
    a deep descendant search.
    """
    text = _find_text(entry, "gen", lang="greek")
    if text:
        return text
    text = _find_text(entry, "gen")
    if text:
        return text
    text = _find_text_deep(entry, "gen", lang="greek")
    if text:
        return text
    return _find_text_deep(entry, "gen")


def _find_texts(parent: Any, tag: str, **attribs: str) -> list[str]:
    """Find all direct children matching tag and optional attributes, return texts."""
    texts: list[str] = []
    for child in parent:
        local = _local_name(child)
        if local != tag:
            continue
        if all(child.get(k) == v for k, v in attribs.items()):
            text = _elem_text(child)
            if text:
                texts.append(text)
    return texts


def _local_name(element: Any) -> str:
    """Return the local XML tag name for ``element``."""
    tag = cast(str, element.tag)
    return tag.split("}")[-1] if "}" in tag else tag
