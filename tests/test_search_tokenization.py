"""Tests for search tokenization helpers."""

from __future__ import annotations

from collections.abc import Iterable

import pytest

import phonology.search._tokenization as tokenization


def test_tokenize_for_inventory_delegates_to_shim_without_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """None inventory uses the backward-compatible search tokenizer."""
    calls: list[str] = []

    def fake_tokenize_ipa(ipa_text: str) -> list[str]:
        calls.append(ipa_text)
        return ["delegated"]

    monkeypatch.setattr(tokenization, "tokenize_ipa", fake_tokenize_ipa)

    assert tokenization.tokenize_for_inventory("kʰa", phone_inventory=None) == [
        "delegated"
    ]
    assert calls == ["kʰa"]


def test_tokenize_for_inventory_empty_inventory_matches_core_tokenizer() -> None:
    """An empty inventory keeps the same literal fallback behavior as core IPA."""
    ipa_text = "kʰ a"

    assert tokenization.tokenize_for_inventory(ipa_text, phone_inventory=()) == (
        tokenization.tokenize_ipa_with_inventory(ipa_text, phone_inventory=())
    )


def test_tokenize_for_inventory_uses_greedy_longest_match() -> None:
    """Inventory tokenization prefers multi-character phones over prefixes."""
    inventory = ("t", "t͡s", "s", "k", "kʰ", "a")

    assert tokenization.tokenize_for_inventory("t͡skʰa", inventory) == [
        "t͡s",
        "kʰ",
        "a",
    ]


def test_tokenize_for_inventory_materializes_single_use_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generator inventories are materialized before reaching the core tokenizer."""
    observed: list[tuple[list[str], list[str]]] = []

    def fake_tokenize_ipa_with_inventory(
        ipa_text: str,
        *,
        phone_inventory: Iterable[str],
    ) -> list[str]:
        observed.append((list(phone_inventory), list(phone_inventory)))
        return [ipa_text]

    def phone_generator() -> Iterable[str]:
        yield from ("kʰ", "k", "a")

    monkeypatch.setattr(
        tokenization,
        "tokenize_ipa_with_inventory",
        fake_tokenize_ipa_with_inventory,
    )

    assert tokenization.tokenize_for_inventory("kʰa", phone_generator()) == ["kʰa"]
    assert observed == [(["kʰ", "k", "a"], ["kʰ", "k", "a"])]
