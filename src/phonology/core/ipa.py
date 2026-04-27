"""Language-independent IPA normalization and tokenization helpers."""

from __future__ import annotations

import logging
import unicodedata
from collections.abc import Iterable

logger = logging.getLogger(__name__)

ACCENT_MARKS = frozenset({"\u0301", "\u0300", "\u0342"})  # acute, grave, circumflex


def strip_ignored_ipa_combining_marks(ipa_text: str, *, recompose: bool = True) -> str:
    """Remove accent/stress combining marks while preserving other IPA diacritics."""
    nfd = unicodedata.normalize("NFD", ipa_text)
    stripped = "".join(char for char in nfd if char not in ACCENT_MARKS)
    if recompose:
        return unicodedata.normalize("NFC", stripped)
    return stripped


def normalize_ipa_for_tokenization(ipa_text: str) -> str:
    """Return NFD IPA text with ignored accent marks removed."""
    return strip_ignored_ipa_combining_marks(ipa_text, recompose=False)


def _consume_trailing_combining_marks(text: str, start: int) -> tuple[str, int]:
    """Return any combining marks immediately following ``start`` in ``text``."""
    end = start
    while end < len(text) and unicodedata.category(text[end]) == "Mn":
        end += 1
    return text[start:end], end


def sorted_phone_inventory(phone_inventory: Iterable[str]) -> tuple[str, ...]:
    """Return phone inventory sorted for greedy longest-match tokenization."""
    return tuple(sorted(set(phone_inventory), key=len, reverse=True))


def tokenize_ipa(
    ipa_text: str,
    *,
    phone_inventory: Iterable[str],
) -> list[str]:
    """Tokenize compact or space-separated IPA into comparable phone units.

    Unknown segments are emitted as single-character literals with any
    trailing combining marks, so callers can still compare toy or incomplete
    inventories without failing conversion.
    """
    inventory = sorted_phone_inventory(phone_inventory)
    text = normalize_ipa_for_tokenization(ipa_text)
    tokens: list[str] = []
    index = 0

    while index < len(text):
        if text[index].isspace():
            index += 1
            continue

        for phone in inventory:
            if text.startswith(phone, index):
                token_end = index + len(phone)
                trailing_marks, token_end = _consume_trailing_combining_marks(
                    text, token_end
                )
                tokens.append(phone + trailing_marks)
                index = token_end
                break
        else:
            literal_end = index + 1
            trailing_marks, literal_end = _consume_trailing_combining_marks(
                text, literal_end
            )
            literal = text[index] + trailing_marks
            logger.debug(
                "Treating unknown IPA token %r at index %s as a literal",
                literal,
                index,
            )
            tokens.append(literal)
            index = literal_end

    return tokens
