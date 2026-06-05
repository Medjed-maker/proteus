"""IPA phone inventory for the Ancient Greek plugin.

This module is plugin-specific to ``phonology.languages.ancient_greek`` — it is
not a package-wide shared constant module. The vowel phones represent the
scholarly Ancient Greek IPA inventory used by Proteus, with an Attic-oriented
default pronunciation as documented in README and the grapheme→IPA converter in
``ipa.py`` of this package. Includes both monophthongs (short and long) and
diphthongs.
"""

from collections.abc import Sequence

VOWEL_PHONES: frozenset[str] = frozenset(
    {
        "a",
        "ai",
        "au",
        "aː",
        "e",
        "eu",
        "eː",
        "i",
        "o",
        "oi",
        "oː",
        "u",
        "y",
        "yː",
        "ɔː",
        "ɛː",
        "ɛːy",
    }
)

# Greek dichronous (δίχρονα) vowels: alpha, iota, and upsilon do not mark
# vowel length in the script (IPA: a, i, y, u), so a written form can surface
# as either the short or long IPA phone (e.g. unmarked alpha tokenizes as ``a``
# while the macron form ᾱ tokenizes as ``aː``). The mid vowels are excluded on purpose: epsilon
# ``e`` / eta ``ɛː`` and omicron ``o`` / omega ``ɔː`` are distinct letters with
# distinct quality, and the spurious diphthongs ει → ``eː`` / ου → ``oː`` keep
# ``e``/``eː`` and ``o``/``oː`` phonemically separate. Collapsing those would
# mis-identify rules, so only the genuinely length-ambiguous bases appear here.
_LENGTH_MARK = "ː"
DICHRONOUS_VOWEL_BASES: frozenset[str] = frozenset({"a", "i", "y", "u"})


def _dichronous_unmarked_base(phone: str) -> str | None:
    """Return the dichronous base when ``phone`` has no length mark."""
    if phone.endswith(_LENGTH_MARK):
        return None
    return phone if phone in DICHRONOUS_VOWEL_BASES else None


def _dichronous_long_base(phone: str) -> str | None:
    """Return the dichronous base when ``phone`` has an explicit length mark."""
    if not phone.endswith(_LENGTH_MARK):
        return None
    base = phone[: -len(_LENGTH_MARK)]
    return base if base in DICHRONOUS_VOWEL_BASES else None


def phones_match(actual: str, expected: str) -> bool:
    """Compare a form phone to a rule phone with directional length tolerance.

    Returns True on exact equality, or when the form phone is an unmarked
    dichronous vowel and the rule phone is the corresponding explicitly long
    vowel. All other phones — including the reverse long-to-short direction and
    the length-contrasting mid vowels — must match exactly.
    """
    if actual == expected:
        return True
    actual_base = _dichronous_unmarked_base(actual)
    return actual_base is not None and actual_base == _dichronous_long_base(expected)


def token_seq_matches(actual: Sequence[str], expected: Sequence[str]) -> bool:
    """Elementwise :func:`phones_match` over two token sequences of equal length."""
    return len(actual) == len(expected) and all(
        phones_match(actual_token, expected_token)
        for actual_token, expected_token in zip(actual, expected)
    )
