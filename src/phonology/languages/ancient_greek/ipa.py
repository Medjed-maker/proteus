"""Greek script to IPA conversion.

Converts Ancient Greek Unicode text (polytonic or monotonic) to
scholarly IPA transcription using Attic pronunciation as default.
"""

from __future__ import annotations

import logging
import unicodedata
from typing import NamedTuple

from ..._phones import VOWEL_PHONES
from ...core.ipa import (
    ACCENT_MARKS,
    strip_ignored_ipa_combining_marks as _strip_ignored_ipa_combining_marks,
)
from ...core.ipa import tokenize_ipa as _tokenize_ipa_with_inventory

# Mapping: Greek letter (NFC) -> IPA phone(s)
# Covers basic alphabet; diacritics handled separately.
_LETTER_MAP: dict[str, str] = {
    "α": "a",
    "β": "b",
    "γ": "ɡ",
    "δ": "d",
    "ε": "e",
    "ζ": "zd",
    "η": "ɛː",
    "θ": "tʰ",
    "ι": "i",
    "κ": "k",
    "λ": "l",
    "μ": "m",
    "ν": "n",
    "ξ": "ks",
    "ο": "o",
    "π": "p",
    "ρ": "r",
    "σ": "s",
    "ς": "s",
    "τ": "t",
    "υ": "y",
    "φ": "pʰ",
    "χ": "kʰ",
    "ψ": "ps",
    "ω": "ɔː",
    "ϝ": "w",
}

_DIPHTHONG_MAP: dict[str, str] = {
    "αι": "ai",
    "ει": "eː",
    "οι": "oi",
    "αυ": "au",
    "ευ": "eu",
    "ηυ": "ɛːy",
    "ου": "oː",
}

_DIAERESIS = "\u0308"
_ROUGH_BREATHING = "\u0314"
_YPOGEGRAMMENI = "\u0345"
_LONG_ALPHA_MARKS = frozenset({"\u0304", "\u0342"})  # macron, circumflex
_COMBINING_ACUTE = "\u0301"
_DIPHTHONG_BOUNDARY = "|"
_EXTRA_IPA_PHONES: frozenset[str] = frozenset(
    {
        # tokenize_ipa() must also accept IPA phones that come from runtime data
        # rather than direct _LETTER_MAP/_DIPHTHONG_MAP output in greek_to_ipa():
        # "aː"/"yː" appear in committed rule/matrix/lexicon IPA data, while "u"
        # can arrive in caller-provided IPA after
        # _normalize_ipa_for_tokenization() strips marks.
        "f",
        "aː",
        "ð",
        "h",
        "u",
        "yː",
        "x",
        "ɣ",
        "θ",
    }
)
_IPA_PHONE_INVENTORY = tuple(
    sorted(
        set(_LETTER_MAP.values()) | set(_DIPHTHONG_MAP.values()) | _EXTRA_IPA_PHONES,
        key=len,
        reverse=True,
    )
)
logger = logging.getLogger("phonology.ipa_converter")
_KOINE_INTERVOCALIC_MAP: dict[str, str] = {"ɡ": "ɣ", "d": "ð"}
_KOINE_DIRECT_MAP: dict[str, str] = {"pʰ": "f", "tʰ": "θ", "kʰ": "x"}


class NormalizationResult(NamedTuple):
    """Greek normalization data needed for IPA conversion."""

    text: str
    accent_positions: set[int]
    rough_h_positions: set[int]
    long_alpha_positions: set[int]


def strip_ignored_ipa_combining_marks(ipa_text: str) -> str:
    """Remove accent/stress combining marks while preserving other IPA diacritics.

    Public wrapper around the internal ``_strip_ignored_ipa_combining_marks``
    helper so that sibling modules can import a stable, public symbol.
    """
    return _strip_ignored_ipa_combining_marks(ipa_text)


def _should_keep_rough_breathed_diphthong(
    normalized: list[str], base: str, marks: set[str]
) -> bool:
    """Return True when rough breathing belongs to a surviving diphthong.

    In polytonic Greek, initial rough breathing on the second element of a
    diphthong is written over ``υ``. For the current compact IPA inventory we
    keep the existing diphthong token instead of inserting an extra ``h`` that
    would split ``αυ``/``ευ``/``ηυ`` into separate phones.
    """
    return (
        base == "υ"
        and _ROUGH_BREATHING in marks
        and _DIAERESIS not in marks
        and bool(normalized)
        and normalized[-1] in {"α", "ε", "η"}
    )


def _normalize_greek_for_ipa(
    greek_text: str,
) -> NormalizationResult:
    """Normalize Greek text for phone conversion without losing boundaries.

    Args:
        greek_text: NFC-encoded Greek text to normalize before IPA conversion.

    Returns:
        Normalized text and position metadata for IPA conversion.
        ``accent_positions`` contains the indices of accented Greek letters in
        the normalized text, while ``rough_h_positions`` records only the
        inserted ``h`` markers that originate from Greek rough breathing.
        ``long_alpha_positions`` records Greek alpha letters marked long by
        macron or circumflex.
    """
    nfd = unicodedata.normalize("NFD", greek_text)
    normalized: list[str] = []
    accent_positions: set[int] = set()
    rough_h_positions: set[int] = set()
    long_alpha_positions: set[int] = set()
    index = 0

    while index < len(nfd):
        base = nfd[index]
        if unicodedata.category(base) == "Mn":
            index += 1
            continue

        index += 1
        marks: list[str] = []
        while index < len(nfd) and unicodedata.category(nfd[index]) == "Mn":
            marks.append(nfd[index])
            index += 1
        marks_set = set(marks)

        has_accent = bool(ACCENT_MARKS & marks_set)

        if _DIAERESIS in marks_set:
            normalized.append(_DIPHTHONG_BOUNDARY)

        if _ROUGH_BREATHING in marks_set and not _should_keep_rough_breathed_diphthong(
            normalized,
            base.lower(),
            marks_set,
        ):
            rough_h_positions.add(len(normalized))
            normalized.append("h")

        accent_pos = len(normalized)
        normalized.append(base.lower())
        if has_accent:
            accent_positions.add(accent_pos)
        if base.lower() == "α" and _LONG_ALPHA_MARKS & marks_set:
            long_alpha_positions.add(accent_pos)

        if _YPOGEGRAMMENI in marks_set:
            normalized.append("ι")

    return NormalizationResult(
        text="".join(normalized),
        accent_positions=accent_positions,
        rough_h_positions=rough_h_positions,
        long_alpha_positions=long_alpha_positions,
    )


def _apply_accent(phone: str) -> str:
    """Insert combining acute after the first character of an IPA phone.

    The result is NFC-normalized so that characters with precomposed
    accented forms (e.g. ``o`` + acute → ``ó`` U+00F3) are composed,
    while characters without precomposed forms (e.g. ``ɛ`` + acute)
    remain as decomposed sequences.
    """
    if not phone:
        return phone
    accented = phone[0] + _COMBINING_ACUTE + phone[1:]
    return unicodedata.normalize("NFC", accented)


def _reapply_ipa_accents(mapped_phone: str, original_phone: str) -> str:
    """Copy accent/stress marks from ``original_phone`` onto ``mapped_phone``.

    Args:
        mapped_phone: The IPA phone after consonant shift mapping.
        original_phone: The original IPA phone containing accent marks.

    Returns:
        The mapped phone with accent marks reattached after its first character,
        NFC-normalized.
    """
    accents = "".join(
        char
        for char in unicodedata.normalize("NFD", original_phone)
        if char in ACCENT_MARKS
    )
    if not accents or not mapped_phone:
        return mapped_phone
    return unicodedata.normalize("NFC", mapped_phone[0] + accents + mapped_phone[1:])


def _is_vowel_phone(phone: str) -> bool:
    """Return True when a phone is a vowel after accent normalization."""
    return strip_ignored_ipa_combining_marks(phone) in VOWEL_PHONES


def apply_koine_consonant_shifts(phones: list[str]) -> list[str]:
    """Convert an Attic IPA phone sequence into the supported Koine subset.

    This intentionally implements only the minimal query-side Koine
    consonant changes that correspond to the committed Koine consonant rules.
    """
    normalized_phones = [strip_ignored_ipa_combining_marks(phone) for phone in phones]
    shifted: list[str] = []

    for index, phone in enumerate(normalized_phones):
        if phone in _KOINE_DIRECT_MAP:
            shifted.append(
                _reapply_ipa_accents(_KOINE_DIRECT_MAP[phone], phones[index])
            )
            continue

        prev_phone = normalized_phones[index - 1] if index > 0 else None
        next_phone = (
            normalized_phones[index + 1] if index + 1 < len(normalized_phones) else None
        )
        if (
            phone in _KOINE_INTERVOCALIC_MAP
            and prev_phone is not None
            and next_phone is not None
            and _is_vowel_phone(prev_phone)
            and _is_vowel_phone(next_phone)
        ):
            shifted.append(
                _reapply_ipa_accents(_KOINE_INTERVOCALIC_MAP[phone], phones[index])
            )
            continue

        shifted.append(phones[index])

    return shifted


def strip_diacritics(greek_text: str) -> str:
    """Remove all diacritics, returning bare Greek letters.

    Args:
        greek_text: NFC-encoded polytonic Greek text.

    Returns:
        Greek text with accents, breathings, and subscripts removed.
    """
    nfd = unicodedata.normalize("NFD", greek_text)
    stripped = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", stripped)


def greek_to_ipa(text: str) -> list[str]:
    """Convert Greek text to a list of IPA phones.

    Uses greedy left-to-right scan: diphthongs checked before single letters.
    Characters not matched by ``_DIPHTHONG_MAP`` or ``_LETTER_MAP`` are
    silently skipped after diacritics are stripped.

    Args:
        text: Greek Unicode string (polytonic or monotonic).

    Returns:
        List of IPA phone strings.
    """
    normalization = _normalize_greek_for_ipa(text)
    result: list[str] = []
    i = 0
    while i < len(normalization.text):
        if normalization.text[i] == _DIPHTHONG_BOUNDARY:
            i += 1
            continue
        if normalization.text[i] == "h" and i in normalization.rough_h_positions:
            phone = "h"
            if i in normalization.accent_positions:
                phone = _apply_accent(phone)
            result.append(phone)
            i += 1
            continue
        pair = normalization.text[i : i + 2]
        if (
            len(pair) == 2
            and pair in _DIPHTHONG_MAP
            and i not in normalization.long_alpha_positions
        ):
            phone = _DIPHTHONG_MAP[pair]
            if (
                i in normalization.accent_positions
                or (i + 1) in normalization.accent_positions
            ):
                phone = _apply_accent(phone)
            result.append(phone)
            i += 2
        elif normalization.text[i] in _LETTER_MAP:
            if normalization.text[i] == "α" and i in normalization.long_alpha_positions:
                phone = "aː"
            else:
                phone = _LETTER_MAP[normalization.text[i]]
            if i in normalization.accent_positions:
                phone = _apply_accent(phone)
            result.append(phone)
            i += 1
        else:
            logger.debug(
                "Skipping unknown Greek character %r (ord=%s) at index %s",
                normalization.text[i],
                ord(normalization.text[i]),
                i,
            )
            i += 1
    return result


def tokenize_ipa(ipa_text: str) -> list[str]:
    """Tokenize compact or space-separated IPA into comparable phone units.

    Uses greedy left-to-right matching against the known phone inventory.
    Unknown segments are emitted as single-character literals with any
    trailing combining marks.

    Args:
        ipa_text: IPA string (compact or space-separated).

    Returns:
        List of IPA phone tokens, with stress marks stripped for comparison.
    """
    return _tokenize_ipa_with_inventory(
        ipa_text,
        phone_inventory=_IPA_PHONE_INVENTORY,
    )


def to_ipa(text: str, *, dialect: str = "attic") -> str:
    """Convert a Greek string to IPA.

    Args:
        text: NFC-encoded Greek Unicode string.
        dialect: Pronunciation variety. Supports ``"attic"`` and a
            query-side ``"koine"`` normalization that applies the committed
            Koine consonant shifts after Attic conversion.

    Returns:
        Compact IPA transcription string.

    Raises:
        NotImplementedError: If ``dialect`` is unsupported.
    """
    if dialect not in {"attic", "koine"}:
        raise NotImplementedError(f"Dialect {dialect!r} not yet supported")
    phones = greek_to_ipa(text)
    if dialect == "koine":
        phones = apply_koine_consonant_shifts(phones)
    return "".join(phones)


def get_known_phones() -> tuple[str, ...]:
    """Return a tuple of all known IPA phone strings.

    Returns:
        All supported IPA phones, sorted by length (descending).
    """
    return _IPA_PHONE_INVENTORY
