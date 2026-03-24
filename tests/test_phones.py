"""Tests for phonology._phones."""

import pytest

from phonology._phones import VOWEL_PHONES


def test_vowel_phones_contains_expected_tokens() -> None:
    assert {"a", "ai", "oː"}.issubset(VOWEL_PHONES)


def test_vowel_phones_is_not_empty() -> None:
    assert VOWEL_PHONES


def test_vowel_phones_elements_are_strings() -> None:
    assert all(isinstance(phone, str) for phone in VOWEL_PHONES)


def test_vowel_phones_is_frozenset() -> None:
    assert type(VOWEL_PHONES) is frozenset


def test_vowel_phones_has_no_duplicates() -> None:
    # Complements test_vowel_phones_is_frozenset: verifies that the data
    # source was normalized into a duplicate-free collection.
    assert len(VOWEL_PHONES) == len(set(VOWEL_PHONES))


def test_vowel_phones_cannot_be_modified() -> None:
    with pytest.raises(AttributeError):
        VOWEL_PHONES.add("ea")  # type: ignore[attr-defined]

    with pytest.raises(AttributeError):
        VOWEL_PHONES.update({"ea"})  # type: ignore[attr-defined]
