"""Shared IPA phone constants for the phonology package."""

# Vowel phones for the scholarly Ancient Greek IPA inventory used by Proteus
# (Attic-oriented default pronunciation, as documented in README and ipa_converter.py).
# Includes monophthongs (short and long) and diphthongs
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
