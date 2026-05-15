"""LSJ extractor helper module.

Logger uses the literal ``phonology.lsj_extractor`` name so existing
``caplog.set_level(logger="phonology.lsj_extractor")`` blocks in
``tests/test_lsj_extractor.py`` keep capturing diagnostics from this module
after the split.
"""

from __future__ import annotations

import logging
from typing import Any

from ._constants import _GENDER_REQUIRED_POS
from ._fields import (
    _extract_dialect,
    _extract_gender,
    _extract_gloss,
    _extract_headword,
    _extract_pos,
)
from ._heading import _has_multiple_intro_greek_forms
from ._normalize import _normalize_headword_key

logger = logging.getLogger("phonology.lsj_extractor")


def _validate_entry(entry_elem: Any) -> tuple[str, str, str, str, str] | None:
    """Return validated entry core fields, or None when unusable."""
    entry_type = entry_elem.get("type", "")
    if entry_type and entry_type != "main":
        return None

    xml_id = entry_elem.get("id", "")
    if not xml_id:
        return None
    numeric = xml_id.lstrip("n")
    if not numeric.isdigit():
        return None
    lsj_id = f"LSJ-{int(numeric):06d}"
    key = entry_elem.get("key", "")

    headword = _extract_headword(entry_elem)
    if not headword:
        return None

    gloss = _extract_gloss(entry_elem)
    if not gloss:
        return None

    pos = _extract_pos(entry_elem)
    if pos is None:
        return None

    return lsj_id, key, headword, gloss, pos


def _extract_attributes(
    entry_elem: Any, key: str, pos: str, lsj_id: str, headword: str
) -> tuple[str, str | None] | None:
    """Return dialect and gender attributes, or None for filtered entries."""
    from .. import lsj_extractor as _shim

    _load_pos_overrides = _shim._load_pos_overrides
    dialect = _extract_dialect(entry_elem)
    if dialect != "attic":
        logger.info(
            "Skipping non-Attic entry %s (%s): dialect=%s",
            lsj_id,
            headword,
            dialect,
        )
        return None

    gender = _extract_gender(entry_elem)
    has_multiple_intro_forms = _has_multiple_intro_greek_forms(entry_elem)
    common_gender_keys = _load_pos_overrides().get("common_gender_keys", frozenset())
    if pos in {"pronoun", "article", "numeral"} and has_multiple_intro_forms:
        gender = "common"
    elif pos == "noun" and _normalize_headword_key(key) in common_gender_keys:
        gender = "common"
    elif pos in _GENDER_REQUIRED_POS and gender is None:
        gender = "common"  # default for undetermined gender

    return dialect, gender


def _convert_forms(
    headword: str, dialect: str, lsj_id: str
) -> tuple[str, str] | None:
    """Return transliteration and IPA forms, or None when conversion fails."""
    from .. import lsj_extractor as _shim

    # These are re-exported from the shim so tests can monkeypatch them.
    to_ipa = _shim.to_ipa
    transliterate = _shim.transliterate

    translit = transliterate(headword)
    if not translit:
        return None

    try:
        ipa = to_ipa(headword, dialect=dialect)
    except (ValueError, KeyError, NotImplementedError) as e:
        logger.info(
            "IPA conversion failed for %s (%s): %s: %s",
            lsj_id,
            headword,
            type(e).__name__,
            e,
        )
        return None
    if not ipa:
        return None
    return translit, ipa


def extract_entry(entry_elem: Any) -> dict[str, Any] | None:
    """Extract a single lemma dictionary from an LSJ ``<entryFree>`` element.

    Args:
        entry_elem: XML element-like object for an LSJ ``entryFree`` node. It
            must expose ``get()``, iteration over child elements, and descendant
            traversal used by the field extractors.

    Side effects:
        Resolves ``_load_pos_overrides``, ``transliterate``, and ``to_ipa`` via
        ``phonology.lsj_extractor`` at call time so shim monkeypatches apply.
        Logs skipped non-Attic entries and IPA conversion failures.

    Returns:
        A ``dict[str, Any]`` lemma, or ``None`` when the entry is not usable:
        non-main type, invalid LSJ id, missing headword or gloss, undetermined
        POS, too-short open-class headword, non-Attic dialect, empty
        transliteration, or unavailable IPA.

    Raises:
        Propagates unexpected exceptions from XML traversal or field extraction.
        Known IPA conversion failures are caught and return ``None``.
    """
    validated = _validate_entry(entry_elem)
    if validated is None:
        return None
    lsj_id, key, headword, gloss, pos = validated

    if len(headword.strip()) <= 1 and pos not in {
        "article",
        "pronoun",
        "interjection",
    }:
        return None

    attributes = _extract_attributes(entry_elem, key, pos, lsj_id, headword)
    if attributes is None:
        return None
    dialect, gender = attributes

    converted = _convert_forms(headword, dialect, lsj_id)
    if converted is None:
        return None
    translit, ipa = converted

    result: dict[str, Any] = {
        "id": lsj_id,
        "headword": headword,
        "transliteration": translit,
        "ipa": ipa,
        "pos": pos,
        "gloss": gloss,
        "dialect": dialect,
    }
    if gender is not None:
        result["gender"] = gender

    return result
