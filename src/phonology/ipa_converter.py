"""Backward-compatible Ancient Greek IPA conversion exports.

Language-independent IPA tokenization lives in :mod:`phonology.core.ipa`.
Ancient Greek Unicode conversion lives in
:mod:`phonology.languages.ancient_greek.ipa`. This module preserves the
original public import path used by existing callers.
"""

from __future__ import annotations

from .languages.ancient_greek.ipa import (
    apply_koine_consonant_shifts,
    get_known_phones,
    greek_to_ipa,
    strip_diacritics,
    strip_ignored_ipa_combining_marks,
    tokenize_ipa,
)


def to_ipa(text: str, *, dialect: str = "attic") -> str:
    """Convert an Ancient Greek string to IPA using the legacy import seam."""
    if dialect not in {"attic", "koine"}:
        raise NotImplementedError(f"Dialect {dialect!r} not yet supported")
    phones = greek_to_ipa(text)
    if dialect == "koine":
        phones = apply_koine_consonant_shifts(phones)
    return "".join(phones)


__all__ = [
    "apply_koine_consonant_shifts",
    "get_known_phones",
    "greek_to_ipa",
    "strip_diacritics",
    "strip_ignored_ipa_combining_marks",
    "to_ipa",
    "tokenize_ipa",
]
