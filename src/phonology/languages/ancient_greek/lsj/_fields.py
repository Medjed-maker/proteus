"""LSJ extractor helper module.

Logger uses the literal ``phonology.languages.ancient_greek.lsj_extractor`` name so existing
``caplog.set_level(logger="phonology.languages.ancient_greek.lsj_extractor")`` blocks in
``tests/test_lsj_extractor.py`` keep capturing diagnostics from this module
after the split.
"""

from __future__ import annotations

import logging
import unicodedata
from typing import Any, NamedTuple

from ..betacode import beta_to_unicode
from ._constants import (
    _GENDER_MAP,
    _PARTICIPLE_OF_RE,
    _POS_MAP,
)
from ._heading import (
    _has_plural_neuter_article,
    _leading_dialect_labels,
)
from ._intro import (
    _entry_intro_text,
    _inline_pos_candidates,
    _sense_intro_texts,
    _sense_post_gloss_pos_texts,
)
from ._normalize import (
    _has_descendant,
    _headword_itypes,
    _looks_like_adjective_itypes,
    _looks_like_known_numeral_key,
    _looks_like_verb_key,
    _normalize_headword_key,
)
from ._xml import _elem_text, _find_gen_text, _find_text, _local_name

logger = logging.getLogger("phonology.languages.ancient_greek.lsj_extractor")

_MAX_GLOSS_LENGTH = 200
_ELLIPSIS = "..."
_DIALECT_PRIORITY = ("attic", "ionic", "doric", "aeolic")


def _extract_headword(entry: Any) -> str:
    """Extract the headword from the entry, trying multiple sources.

    Attempts conversion in order: normalized entry key,
    <orth extent='full' lang='greek'>, then any <orth lang='greek'>. Returns empty
    string if all conversions fail.
    """
    candidates = [
        _normalize_headword_key(entry.get("key", "")),
        _find_text(entry, "orth", extent="full", lang="greek"),
        _find_text(entry, "orth", lang="greek"),
    ]
    seen: set[str] = set()
    for beta in candidates:
        if not beta or beta in seen:
            continue
        seen.add(beta)
        # Strip quantity markers (^ for breve, _ for macron) before conversion
        cleaned = beta.replace("^", "").replace("_", "")
        try:
            return beta_to_unicode(cleaned)
        except ValueError as exc:
            logger.warning("Beta Code conversion failed for %r: %s", cleaned, exc)
    return ""


class _PosInferenceContext(NamedTuple):
    """Pre-computed inputs shared by ordered POS inference rules."""

    entry: Any
    key: str
    normalized_key: str
    # Raw iType morphology for the entry. Currently consumed only via
    # _looks_like_adjective_itypes() to derive ``adjective_itypes`` below;
    # retained on the context so future POS inference rules can inspect it
    # directly without re-extracting it from the entry element.
    itypes: list[str]
    adjective_itypes: bool
    participial_candidate: bool


def _infer_explicit_pos(entry: Any) -> str | None:
    """Infer POS from a direct explicit ``<pos>`` tag."""
    pos_text = _find_text(entry, "pos")
    if not pos_text:
        return None
    pos_text = pos_text.strip().rstrip(".")
    # Try exact match first, then with trailing period.
    for candidate in (pos_text, pos_text + "."):
        if candidate in _POS_MAP:
            return _POS_MAP[candidate]
    return None


def _infer_participle_intro_marker(entry: Any) -> bool:
    """Return True when heading prose marks a participial candidate."""
    return bool(_PARTICIPLE_OF_RE.search(_entry_intro_text(entry)))


def _infer_inline_prose_pos(entry: Any) -> str | None:
    """Infer POS from inline heading and early sense prose labels."""
    inline_candidates: list[tuple[int, int, int, str]] = []
    inline_texts = [
        _entry_intro_text(entry, skip_mood_text=True),
        *_sense_intro_texts(entry, skip_mood_text=True),
    ]
    for source_index, text in enumerate(inline_texts):
        for attic_priority, position, pos in _inline_pos_candidates(text):
            # 0 = Attic-priority (sorts first), 1 = non-Attic.
            sort_priority = 0 if attic_priority else 1
            inline_candidates.append((sort_priority, source_index, position, pos))
    if not inline_candidates:
        return None
    inline_candidates.sort()
    return inline_candidates[0][3]


def _infer_known_numeral_pos(context: _PosInferenceContext) -> str | None:
    """Infer common cardinal numeral headwords omitted by LSJ POS markup."""
    if context.key and _looks_like_known_numeral_key(context.key):
        return "numeral"
    return None


def _infer_gender_based_pos(context: _PosInferenceContext) -> str | None:
    """Infer noun or participle from heading gender morphology."""
    gen_text = _find_gen_text(context.entry)
    if not gen_text:
        return None
    if context.participial_candidate and _has_plural_neuter_article(context.entry):
        return "participle"
    return "noun"


def _infer_adverb_ending_pos(context: _PosInferenceContext) -> str | None:
    """Infer adverb POS from common headword endings."""
    if (
        context.normalized_key.endswith("ws")
        or context.normalized_key.endswith("w=s")
        or context.normalized_key.endswith("ei/")
    ):
        return "adverb"
    return None


def _infer_adjective_itype_pos(context: _PosInferenceContext) -> str | None:
    """Infer adjective POS from direct headword iType morphology."""
    if context.adjective_itypes:
        return "adjective"
    return None


def _infer_verb_indicator_pos(context: _PosInferenceContext) -> str | None:
    """Infer verb POS from verb-like key plus verbal morphology evidence."""
    if (
        context.key
        and _looks_like_verb_key(context.key)
        and _has_descendant(context.entry, {"tns", "mood"})
    ):
        return "verb"
    return None


def _infer_post_gloss_pos(entry: Any) -> str | None:
    """Infer POS from post-gloss prose as a last-resort fallback."""
    post_gloss_candidates: list[tuple[int, int, int, str]] = []
    for source_index, text in enumerate(_sense_post_gloss_pos_texts(entry)):
        for attic_priority, position, pos in _inline_pos_candidates(text):
            sort_priority = 0 if attic_priority else 1
            post_gloss_candidates.append((sort_priority, source_index, position, pos))
    if not post_gloss_candidates:
        return None
    post_gloss_candidates.sort()
    return post_gloss_candidates[0][3]


def _infer_final_participle_pos(context: _PosInferenceContext) -> str | None:
    """Infer participle only after stronger POS signals fail."""
    if context.participial_candidate:
        return "participle"
    return None


def _extract_pos(entry: Any) -> str | None:
    """Infer the part of speech from the entry. Returns None if undetermined."""
    explicit_pos = _infer_explicit_pos(entry)
    if explicit_pos is not None:
        return explicit_pos

    itypes = _headword_itypes(entry)
    key = entry.get("key", "")
    context = _PosInferenceContext(
        entry=entry,
        key=key,
        normalized_key=_normalize_headword_key(key),
        itypes=itypes,
        adjective_itypes=_looks_like_adjective_itypes(itypes, key),
        participial_candidate=_infer_participle_intro_marker(entry),
    )

    pos = _infer_inline_prose_pos(entry)
    if pos is not None:
        return pos

    for infer_rule in (
        _infer_known_numeral_pos,
        _infer_gender_based_pos,
        _infer_adverb_ending_pos,
        _infer_adjective_itype_pos,
        _infer_verb_indicator_pos,
    ):
        pos = infer_rule(context)
        if pos is not None:
            return pos

    pos = _infer_post_gloss_pos(entry)
    if pos is not None:
        return pos

    return _infer_final_participle_pos(context)


def _extract_gender(entry: Any) -> str | None:
    """Extract gender from <gen> element."""
    gen_text = _find_gen_text(entry)
    if gen_text:
        gen_stripped = unicodedata.normalize("NFC", gen_text.strip())
        for code, gender in _GENDER_MAP.items():
            if gen_stripped.startswith(code):
                return gender
        # Fallback: _GENDER_MAP uses Beta Code prefixes (startswith), but some
        # LSJ entries already contain pre-converted Unicode articles.
        if gen_stripped == "ὁ":
            return "masculine"
        if gen_stripped == "ἡ":
            return "feminine"
        if gen_stripped in ("τό", "τὸ"):
            return "neuter"
    return None


def _extract_gloss(entry: Any) -> str:
    """Extract the English gloss from <tr> elements in the first sense."""
    glosses: list[str] = []

    # Try to find <tr> elements within the first <sense>
    for child in entry:
        local = _local_name(child)
        if local == "sense":
            for sub in child.iter():
                sub_local = _local_name(sub)
                if sub_local == "tr":
                    text = _elem_text(sub).strip().strip(",").strip()
                    if text:
                        glosses.append(text)
            if glosses:
                break  # Only take glosses from the first sense

    # Fallback: <tr> directly under entry (some entries skip <sense>)
    if not glosses:
        for child in entry:
            local = _local_name(child)
            if local == "tr":
                text = _elem_text(child).strip().strip(",").strip()
                if text:
                    glosses.append(text)

    combined = ", ".join(glosses)
    if len(combined) > _MAX_GLOSS_LENGTH:
        truncated = combined[: _MAX_GLOSS_LENGTH - len(_ELLIPSIS)]
        # Avoid orphaning a combining mark from its base character onto the ellipsis.
        while truncated and unicodedata.category(truncated[-1]).startswith("M"):
            truncated = truncated[:-1]
        combined = _ELLIPSIS if not truncated else truncated + _ELLIPSIS
    return combined


def _extract_dialect(entry: Any) -> str:
    """Extract entry-level dialect from the heading before the first sense."""
    labels = set(_leading_dialect_labels(entry))
    for dialect in _DIALECT_PRIORITY:
        if dialect in labels:
            return dialect
    if labels:
        return sorted(labels)[0]
    return "attic"


# ---------------------------------------------------------------------------
# Entry processing
# ---------------------------------------------------------------------------
