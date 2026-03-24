"""Shared IPA phone constants for the phonology package.

This module defines phonetic inventories for Ancient Greek.
The vowel phones represent the scholarly Ancient Greek IPA inventory used by Proteus,
with an Attic-oriented default pronunciation as documented in README and ipa_converter.py.
Includes both monophthongs (short and long) and diphthongs.
"""
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
