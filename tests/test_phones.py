"""Tests for phonology._phones."""

import pytest

from phonology._phones import VOWEL_PHONES, phones_match, token_seq_matches


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


@pytest.mark.parametrize("phone", ["a", "aː", "ɛː", "e", "t", "oː"])
def test_phones_match_is_reflexive(phone: str) -> None:
    assert phones_match(phone, phone)


@pytest.mark.parametrize(
    ("short", "long"),
    [("a", "aː"), ("i", "iː"), ("y", "yː"), ("u", "uː")],
)
def test_phones_match_allows_unmarked_dichronous_for_explicit_long(
    short: str, long: str
) -> None:
    # Alpha, iota, and upsilon are orthographically length-ambiguous, so their
    # unmarked short realizations may satisfy explicitly long rule phones.
    assert phones_match(short, long)


@pytest.mark.parametrize(
    ("long", "short"),
    [("aː", "a"), ("iː", "i"), ("yː", "y"), ("uː", "u")],
)
def test_phones_match_rejects_reverse_dichronous_length_direction(
    long: str, short: str
) -> None:
    assert not phones_match(long, short)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("e", "ɛː"),  # epsilon vs eta: distinct quality
        ("o", "ɔː"),  # omicron vs omega: distinct quality
        ("e", "eː"),  # epsilon vs spurious-diphthong ει: must stay separate
        ("o", "oː"),  # omicron vs spurious-diphthong ου: must stay separate
        ("a", "ɛː"),  # unrelated vowels
        ("a", "e"),
        ("t", "d"),  # consonants never collapse
    ],
)
def test_phones_match_keeps_non_dichronous_contrasts(first: str, second: str) -> None:
    assert not phones_match(first, second)
    assert not phones_match(second, first)


def test_token_seq_matches_elementwise_with_dichronous_tolerance() -> None:
    assert token_seq_matches(["m", "a", "t"], ["m", "aː", "t"])


def test_token_seq_matches_rejects_length_mismatch() -> None:
    assert not token_seq_matches(["m", "a"], ["m", "a", "t"])


def test_token_seq_matches_rejects_distinct_mid_vowel_length() -> None:
    assert not token_seq_matches(["o"], ["oː"])
