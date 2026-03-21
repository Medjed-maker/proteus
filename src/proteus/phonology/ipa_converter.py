"""Greek script to IPA conversion.

Converts Ancient Greek Unicode text (polytonic or monotonic) to
scholarly IPA transcription using Attic pronunciation as default.
"""

from __future__ import annotations

import logging
import unicodedata

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
_DIPHTHONG_BOUNDARY = "|"
_EXTRA_IPA_PHONES: frozenset[str] = frozenset(
    {
        # tokenize_ipa() must also accept IPA phones that come from runtime data
        # rather than direct _LETTER_MAP/_DIPHTHONG_MAP output in greek_to_ipa():
        # "aː"/"yː" appear in committed rule/matrix/lexicon IPA data, while "u"
        # can arrive in caller-provided IPA after
        # _normalize_ipa_for_tokenization() strips marks.
        "aː",
        "h",
        "u",
        "yː",
    }
)
_IPA_PHONE_INVENTORY = tuple(
    sorted(
        set(_LETTER_MAP.values()) | set(_DIPHTHONG_MAP.values()) | _EXTRA_IPA_PHONES,
        key=len,
        reverse=True,
    )
)
logger = logging.getLogger(__name__)


def _should_keep_rough_breathed_diphthong(
    normalized: list[str], base: str, marks: list[str]
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


def _normalize_greek_for_ipa(greek_text: str) -> str:
    """Normalize Greek text for phone conversion without losing boundaries."""
    nfd = unicodedata.normalize("NFD", greek_text)
    normalized: list[str] = []
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

        if _DIAERESIS in marks:
            normalized.append(_DIPHTHONG_BOUNDARY)

        if _ROUGH_BREATHING in marks and not _should_keep_rough_breathed_diphthong(
            normalized,
            base.lower(),
            marks,
        ):
            normalized.append("h")

        normalized.append(base.lower())

        if _YPOGEGRAMMENI in marks:
            normalized.append("ι")

    return "".join(normalized)


def _normalize_ipa_for_tokenization(ipa_text: str) -> str:
    """Remove stress marks so compact and lexicon IPA compare consistently."""
    nfd = unicodedata.normalize("NFD", ipa_text)
    return "".join(char for char in nfd if unicodedata.category(char) != "Mn")


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
    text = _normalize_greek_for_ipa(text)
    result: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == _DIPHTHONG_BOUNDARY:
            i += 1
            continue
        if text[i] == "h":
            result.append("h")
            i += 1
            continue
        pair = text[i : i + 2]
        if len(pair) == 2 and pair in _DIPHTHONG_MAP:
            result.append(_DIPHTHONG_MAP[pair])
            i += 2
        elif text[i] in _LETTER_MAP:
            result.append(_LETTER_MAP[text[i]])
            i += 1
        else:
            logger.debug(
                "Skipping unknown Greek character %r (ord=%s) at index %s",
                text[i],
                ord(text[i]),
                i,
            )
            i += 1
    return result


def tokenize_ipa(ipa_text: str) -> list[str]:
    """Tokenize compact or space-separated IPA into comparable phone units."""
    text = _normalize_ipa_for_tokenization(ipa_text)
    tokens: list[str] = []
    index = 0

    while index < len(text):
        if text[index].isspace():
            index += 1
            continue

        for phone in _IPA_PHONE_INVENTORY:
            if text.startswith(phone, index):
                tokens.append(phone)
                index += len(phone)
                break
        else:
            logger.debug(
                "Treating unknown IPA token %r at index %s as a literal",
                text[index],
                index,
            )
            tokens.append(text[index])
            index += 1

    return tokens


def to_ipa(greek_text: str, dialect: str = "attic") -> str:
    """Convert a Greek string to IPA.

    Args:
        greek_text: NFC-encoded Greek Unicode string.
        dialect: Pronunciation variety ('attic', 'ionic', 'doric', 'koine').

    Returns:
        Compact IPA transcription string.

    Raises:
        NotImplementedError: If the dialect is not yet supported.
    """
    if dialect != "attic":
        raise NotImplementedError(f"Dialect {dialect!r} not yet supported")
    phones = greek_to_ipa(greek_text)
    return "".join(phones)


def get_known_phones() -> tuple[str, ...]:
    """Return a tuple of all known IPA phone strings.

    Returns:
        All supported IPA phones, sorted by length (descending).
    """
    return _IPA_PHONE_INVENTORY
