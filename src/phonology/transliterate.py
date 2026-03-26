"""Greek Unicode to scholarly Latin transliteration.

Produces romanised forms following the conventions used in the Proteus
lexicon (e.g. ``θ`` → ``th``, ``χ`` → ``kh``, ``η`` → ``ē``).
"""

from __future__ import annotations

import logging
import unicodedata

# Diphthongs checked before single letters (greedy left-to-right).
_DIPHTHONG_MAP: dict[str, str] = {
    "αι": "ai",
    "ει": "ei",
    "οι": "oi",
    "υι": "yi",
    "αυ": "au",
    "ευ": "eu",
    "ηυ": "ēy",
    "ου": "ou",
}

# Single letter mapping (base lowercase Greek → Latin).
_LETTER_MAP: dict[str, str] = {
    "α": "a",
    "β": "b",
    "γ": "g",
    "δ": "d",
    "ε": "e",
    "ζ": "z",
    "η": "ē",
    "θ": "th",
    "ι": "i",
    "κ": "k",
    "λ": "l",
    "μ": "m",
    "ν": "n",
    "ξ": "x",
    "ο": "o",
    "π": "p",
    "ρ": "r",
    "σ": "s",
    "ς": "s",
    "τ": "t",
    "υ": "y",
    "φ": "ph",
    "χ": "kh",
    "ψ": "ps",
    "ω": "ō",
    "ϝ": "w",
}

_ROUGH_BREATHING = "\u0314"  # combining reversed comma above
_DIAERESIS = "\u0308"
_IOTA_SUBSCRIPT = "\u0345"
# γ before velar or related consonants (including ξ as /ks/) is nasalised.
_NASAL_ASSIMILATION_FOLLOWERS = frozenset({"γ", "κ", "χ", "ξ"})

logger = logging.getLogger(__name__)


def transliterate(greek: str) -> str:
    """Transliterate a Greek Unicode string to scholarly Latin.

    Rough breathing is rendered as a leading ``h``.  Accents and other
    diacritics are stripped.

    Args:
        greek: NFC-normalised polytonic Greek string.

    Returns:
        Lowercase Latin transliteration.
    """
    if not greek:
        return ""

    # NFD decompose to separate base characters from diacritics
    nfd = unicodedata.normalize("NFD", greek)

    # Detect rough breathing positions (before stripping diacritics).
    # In NFD, combining marks follow their base character, so we track
    # the index of the most recently seen base character.
    rough_positions: set[int] = set()
    diaeresis_positions: set[int] = set()
    bases_chars: list[str] = []
    for ch in nfd:
        cat = unicodedata.category(ch)
        if not cat.startswith("M"):
            bases_chars.append(ch.lower())
        elif ch == _ROUGH_BREATHING:
            rough_positions.add(len(bases_chars) - 1)
        elif ch == _DIAERESIS:
            diaeresis_positions.add(len(bases_chars) - 1)
        elif ch == _IOTA_SUBSCRIPT:
            bases_chars.append("ι")

    # Strip combining marks while expanding iota subscript into an explicit iota.
    bases = "".join(bases_chars)

    # Greedy left-to-right transliteration
    result: list[str] = []
    i = 0

    while i < len(bases):
        rough_here = i in rough_positions

        # Rough rho is transliterated as rh-, not hr-.
        if rough_here and bases[i] == "ρ":
            result.append("rh")
            i += 1
            continue

        # γ before a velar is transliterated as a nasal + the following velar.
        if i + 1 < len(bases):
            if bases[i] == "γ" and bases[i + 1] in _NASAL_ASSIMILATION_FOLLOWERS:
                result.append("n")
                result.append(_LETTER_MAP[bases[i + 1]])
                i += 2
                continue

        # Try diphthong match first
        if i + 1 < len(bases):
            digraph = bases[i : i + 2]
            if (
                digraph in _DIPHTHONG_MAP
                and i not in diaeresis_positions
                and (i + 1) not in diaeresis_positions
            ):
                if rough_here or (i + 1) in rough_positions:
                    result.append("h")
                result.append(_DIPHTHONG_MAP[digraph])
                i += 2
                continue

        if rough_here:
            result.append("h")
        ch = bases[i]
        if ch in _LETTER_MAP:
            result.append(_LETTER_MAP[ch])
        # Non-Greek characters (spaces, punctuation) pass through
        elif ch.isascii():
            result.append(ch)
        else:
            logger.warning("Unsupported non-ASCII character %r at base index %d", ch, i)
            result.append(ch)
        i += 1

    return "".join(result)
