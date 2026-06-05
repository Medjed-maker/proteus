"""Beta Code to Unicode Greek converter.

Converts TLG/Perseus Beta Code encoding (ASCII representation of
polytonic Ancient Greek) to Unicode Greek with combining diacritics.
"""

from __future__ import annotations

import unicodedata

# Beta Code base letter → lowercase Greek
_BETA_LETTERS: dict[str, str] = {
    "a": "\u03b1",  # α
    "b": "\u03b2",  # β
    "g": "\u03b3",  # γ
    "d": "\u03b4",  # δ
    "e": "\u03b5",  # ε
    "z": "\u03b6",  # ζ
    "h": "\u03b7",  # η
    "q": "\u03b8",  # θ
    "i": "\u03b9",  # ι
    "k": "\u03ba",  # κ
    "l": "\u03bb",  # λ
    "m": "\u03bc",  # μ
    "n": "\u03bd",  # ν
    "c": "\u03be",  # ξ
    "o": "\u03bf",  # ο
    "p": "\u03c0",  # π
    "r": "\u03c1",  # ρ
    "s": "\u03c3",  # σ (final sigma handled separately)
    "t": "\u03c4",  # τ
    "u": "\u03c5",  # υ
    "f": "\u03c6",  # φ
    "x": "\u03c7",  # χ
    "y": "\u03c8",  # ψ
    "w": "\u03c9",  # ω
    "v": "\u03dd",  # ϝ (digamma)
}

# Beta Code diacritic → Unicode combining character
_BETA_DIACRITICS: dict[str, str] = {
    ")": "\u0313",  # combining comma above (smooth breathing)
    "(": "\u0314",  # combining reversed comma above (rough breathing)
    "/": "\u0301",  # combining acute accent
    "\\": "\u0300",  # combining grave accent
    "=": "\u0342",  # combining Greek perispomeni (circumflex)
    "+": "\u0308",  # combining diaeresis
    "|": "\u0345",  # combining Greek ypogegrammeni (iota subscript)
}

# Characters that signal end-of-word for final sigma detection
_WORD_BOUNDARY = frozenset(
    "".join(
        (
            " \t\n\r,.;:!?",  # standard punctuation/whitespace
            "\u0387",  # Greek ano teleia (·)
            "\u037e",  # Greek question mark (U+037E)
            "()[]{}\"'-",
        )
    )
)


def beta_to_unicode(text: str) -> str:
    """Convert an ASCII Beta Code string to Unicode Greek.

    Handles lowercase and uppercase letters (``*`` prefix), diacritics,
    and final sigma. Returns NFC-normalised output.

    Characters that are not recognised as Beta Code letters or diacritics
    (including pre-existing Unicode Greek) are passed through unchanged.

    Args:
        text: ASCII Beta Code string (e.g. ``"a)/nqrwpos"``).

    Returns:
        Unicode Greek string (e.g. ``"ἄνθρωπος"``).
    """
    if not text:
        return ""

    result: list[str] = []
    i = 0
    length = len(text)

    while i < length:
        ch = text[i]

        # Uppercase marker: * followed by optional diacritics then a letter
        if ch == "*":
            marker_index = i
            i += 1
            # Collect diacritics that appear between * and the letter
            combined_diacritics: list[str] = []
            beta_diacritics: list[str] = []
            while i < length and text[i] in _BETA_DIACRITICS:
                beta_diacritics.append(text[i])
                combined_diacritics.append(_BETA_DIACRITICS[text[i]])
                i += 1
            if i >= length or text[i].lower() not in _BETA_LETTERS:
                unexpected = "<end>" if i >= length else text[i]
                raise ValueError(
                    "Uppercase marker '*' at index "
                    f"{marker_index} must be followed by a Beta Code letter; "
                    f"got {unexpected!r} after diacritics {''.join(beta_diacritics)!r}"
                )
            base = _BETA_LETTERS[text[i].lower()].upper()
            # Collect any post-letter diacritics as well
            i += 1
            while i < length and text[i] in _BETA_DIACRITICS:
                combined_diacritics.append(_BETA_DIACRITICS[text[i]])
                i += 1
            result.append(base + "".join(combined_diacritics))
            continue

        # Lowercase Greek letter
        lower = ch.lower()
        if lower in _BETA_LETTERS:
            base = _BETA_LETTERS[lower]
            i += 1
            # Collect trailing diacritics
            diacritics: list[str] = []
            while i < length and text[i] in _BETA_DIACRITICS:
                diacritics.append(_BETA_DIACRITICS[text[i]])
                i += 1
            result.append(base + "".join(diacritics))
            continue

        # Non-Greek character: pass through (punctuation, spaces, etc.)
        result.append(ch)
        i += 1

    # Apply final sigma: replace σ at word boundaries with ς
    output = list("".join(result))
    for idx in range(len(output)):
        if output[idx] == "σ":
            # Check if next non-combining character is a word boundary or end
            next_idx = idx + 1
            while next_idx < len(output) and unicodedata.category(
                output[next_idx]
            ).startswith("M"):
                next_idx += 1
            if next_idx >= len(output) or output[next_idx] in _WORD_BOUNDARY:
                output[idx] = "ς"

    return unicodedata.normalize("NFC", "".join(output))
