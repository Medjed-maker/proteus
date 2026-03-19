"""Greek script to IPA conversion.

Converts Ancient Greek Unicode text (polytonic or monotonic) to
scholarly IPA transcription using Attic pronunciation as default.
"""

from __future__ import annotations

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


def to_ipa(greek_text: str, dialect: str = "attic") -> str:
    """Convert a Greek string to IPA.

    Args:
        greek_text: NFC-encoded Greek Unicode string.
        dialect: Pronunciation variety ('attic', 'ionic', 'doric', 'koine').

    Returns:
        IPA transcription string.

    Raises:
        NotImplementedError: If the dialect is not yet supported.
    """
    raise NotImplementedError("IPA conversion not yet implemented")


def strip_diacritics(greek_text: str) -> str:
    """Remove all diacritics, returning bare Greek letters.

    Args:
        greek_text: NFC-encoded polytonic Greek text.

    Returns:
        Greek text with accents, breathings, and subscripts removed.
    """
    raise NotImplementedError
