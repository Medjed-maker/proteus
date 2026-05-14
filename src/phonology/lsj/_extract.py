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


def extract_entry(entry_elem: Any) -> dict[str, Any] | None:
    """Extract a single lemma dict from an <entryFree> element.

    Returns None if the entry cannot produce a valid lemma (missing
    headword, gloss, or undetermined POS).

    Resolves ``_load_pos_overrides``, ``transliterate``, and ``to_ipa`` via
    ``phonology.lsj_extractor`` at call time so that test monkeypatches on the
    shim still take effect.
    """
    from .. import lsj_extractor as _shim

    _load_pos_overrides = _shim._load_pos_overrides
    # ``to_ipa`` and ``transliterate`` are re-exported from the shim
    # specifically so tests can monkeypatch them. mypy strict cannot tell those
    # are intentional re-exports because they come in via ``from X import Y``
    # without an explicit ``as Y`` in the shim's source.
    to_ipa = _shim.to_ipa  # type: ignore[attr-defined]
    transliterate = _shim.transliterate  # type: ignore[attr-defined]
    # Only process main entries
    entry_type = entry_elem.get("type", "")
    if entry_type and entry_type != "main":
        return None

    # Extract ID
    xml_id = entry_elem.get("id", "")
    if not xml_id:
        return None
    numeric = xml_id.lstrip("n")
    if not numeric.isdigit():
        return None
    lsj_id = f"LSJ-{int(numeric):06d}"
    key = entry_elem.get("key", "")

    # Extract headword
    headword = _extract_headword(entry_elem)
    if not headword:
        return None

    stripped = headword.strip()

    # Extract gloss
    gloss = _extract_gloss(entry_elem)
    if not gloss:
        return None

    # Extract POS
    pos = _extract_pos(entry_elem)
    if pos is None:
        return None

    # Skip entries that are purely numeric, punctuation, or isolated single letters,
    # except for genuine closed-class headwords such as ὁ.
    if len(stripped) <= 1 and pos not in {"article", "pronoun", "interjection"}:
        return None

    # Extract dialect before IPA generation so the output stays Attic-only.
    dialect = _extract_dialect(entry_elem)
    if dialect != "attic":
        logger.info(
            "Skipping non-Attic entry %s (%s): dialect=%s",
            lsj_id,
            headword,
            dialect,
        )
        return None

    # Extract gender
    gender = _extract_gender(entry_elem)
    has_multiple_intro_forms = _has_multiple_intro_greek_forms(entry_elem)
    if pos in {"pronoun", "article", "numeral"} and has_multiple_intro_forms:
        gender = "common"
    elif (
        pos == "noun"
        and _normalize_headword_key(key) in _load_pos_overrides()["common_gender_keys"]
    ):
        gender = "common"
    elif pos in _GENDER_REQUIRED_POS and gender is None:
        gender = "common"  # default for undetermined gender

    # Generate transliteration
    translit = transliterate(headword)
    if not translit:
        return None

    # Generate IPA
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


# ---------------------------------------------------------------------------
# XML iteration
# ---------------------------------------------------------------------------


