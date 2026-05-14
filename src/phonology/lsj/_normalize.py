"""LSJ extractor helper module.

Logger uses the literal ``phonology.lsj_extractor`` name so existing
``caplog.set_level(logger="phonology.lsj_extractor")`` blocks in
``tests/test_lsj_extractor.py`` keep capturing diagnostics from this module
after the split.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("phonology.lsj_extractor")

from typing import Any

from ._constants import _BETA_REMOVE_RE, _KEY_HOMONYM_SUFFIX_RE, _INTRO_WHITESPACE_RE
from ._xml import _elem_text, _local_name


def _normalize_beta_token(text: str) -> str:
    """Normalize a short Beta Code token used in iType/gender inference."""
    token = text.strip(" \t\n\r,;:.()[]").lower()
    return _BETA_REMOVE_RE.sub("", token)


def _normalize_headword_key(key: str) -> str:
    """Strip LSJ homonym numbering from a key before Beta Code conversion."""
    return _KEY_HOMONYM_SUFFIX_RE.sub("", key)


def _headword_itypes(entry: Any) -> list[str]:
    """Return iTypes attached to the first direct Greek headword block only."""
    itypes: list[str] = []
    in_headword_block = False
    for child in entry:
        local = _local_name(child)
        if not in_headword_block:
            if local == "orth" and child.get("lang") == "greek":
                in_headword_block = True
            continue
        if local == "sense":
            break
        if local == "orth" and child.get("lang") == "greek":
            break
        if local == "itype":
            text = _elem_text(child)
            if text:
                itypes.append(text)
    return itypes


def _has_descendant(entry: Any, tags: set[str]) -> bool:
    """Return True when any descendant tag matches one of ``tags``."""
    for descendant in entry.iter():
        local = _local_name(descendant)
        if local in tags:
            return True
    return False


def _looks_like_verb_key(key: str) -> bool:
    """Heuristic for LSJ headwords that are dictionary verb forms."""
    cleaned = _normalize_headword_key(key).rstrip(":")
    # The -w ending alone would over-match (e.g. some adverbs); callers
    # guard this with _has_descendant(entry, {"tns", "mood"}) to require
    # morphological evidence of verbal inflection.
    return cleaned.endswith("w") or cleaned.endswith("mai") or cleaned.endswith("mi")


def _looks_like_adjective_itypes(itypes: list[str], key: str = "") -> bool:
    """Infer adjective POS from ``itypes`` normalized by ``_normalize_beta_token``.

    Detects Greek adjective patterns such as ``-ος/-η/-ον`` or ``-ος/-α/-ον``
    via ``on`` with ``h`` or ``a``, two-termination ``-ος/-ον`` adjectives via
    ``on`` on an ``-ος`` headword, and third-declension ``-ης/-ες`` via ``es``.
    """
    normalized = {_normalize_beta_token(value) for value in itypes if value.strip()}
    normalized_key = _normalize_beta_token(_normalize_headword_key(key))
    has_two_termination_pattern = "on" in normalized and normalized_key.endswith("os")
    return (
        ("on" in normalized and ("a" in normalized or "h" in normalized))
        or has_two_termination_pattern
        or "es" in normalized
    )


def _looks_like_known_numeral_key(key: str) -> bool:
    """Return True for common Attic cardinal numeral headwords.

    Resolves ``_load_pos_overrides`` lazily via ``phonology.lsj_extractor`` so
    test monkeypatches on the shim's ``resolve_repo_data_dir`` and
    ``_pos_overrides`` attributes take effect.
    """
    from ..lsj_extractor import _load_pos_overrides

    normalized = _normalize_headword_key(key).replace("^", "").replace("_", "")
    return normalized in _load_pos_overrides()["numeral_keys"]


def _normalize_intro_text(parts: list[str]) -> str:
    """Collapse whitespace and trim helper text gathered from XML intro blocks."""
    return _INTRO_WHITESPACE_RE.sub(" ", "".join(parts)).strip()


