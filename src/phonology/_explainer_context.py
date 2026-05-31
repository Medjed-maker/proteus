"""Rule-context predicates and token-lookup helpers for the explainer."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import re

from ._phones import VOWEL_PHONES
from ._explainer_rule_tokenize import _ALWAYS_MATCH_CONTEXTS
from .ipa_converter import tokenize_ipa

_NASAL_PHONES = frozenset({"m", "n"})
_AFTER_E_I_R_PHONES = frozenset({"e", "i", "r"})

_ContextHandler = Callable[
    [
        Sequence[str],
        Sequence[str],
        int,
        int,
        int,
        int,
        tuple[str, ...] | None,
    ],
    bool,
]


def _is_vowel(token: str | None) -> bool:
    """Return True when *token* is one of the known vowel phones."""
    return token in VOWEL_PHONES


def _is_consonant(token: str | None) -> bool:
    """Return True when *token* is present and not classified as a vowel."""
    return token is not None and token not in VOWEL_PHONES


def _lookup_prev_token(
    lemma_tokens: Sequence[str],
    query_tokens: Sequence[str],
    *,
    lemma_start: int,
    query_start: int,
) -> str | None:
    """Return the nearest preceding token, preferring the lemma side."""
    if lemma_start > 0:
        return lemma_tokens[lemma_start - 1]
    if query_start > 0:
        return query_tokens[query_start - 1]
    return None


def _lookup_next_token(
    lemma_tokens: Sequence[str],
    query_tokens: Sequence[str],
    *,
    lemma_end: int,
    query_end: int,
    offset: int = 0,
) -> str | None:
    """Return a following token from the remaining lemma-then-query sequence.

    The function treats the unconsumed tokens as one concatenated sequence:
    ``lemma_tokens[lemma_end:]`` followed by ``query_tokens[query_end:]``.
    ``offset=0`` returns the immediate next token on the lemma side when one
    remains. Once ``offset`` moves past the remaining lemma tokens, lookup
    continues on the query side using ``offset - remaining_lemma``.
    """
    remaining_lemma = max(0, len(lemma_tokens) - lemma_end)
    if offset < remaining_lemma:
        return lemma_tokens[lemma_end + offset]

    query_index = query_end + max(0, offset - remaining_lemma)
    if query_index < len(query_tokens):
        return query_tokens[query_index]
    return None


def _matches_following_set(
    context: str,
    lemma_tokens: Sequence[str],
    query_tokens: Sequence[str],
    *,
    lemma_end: int,
    query_end: int,
) -> bool:
    """Return True when the next token is contained in the context brace set."""
    match = re.fullmatch(r"_\{([^}]+)\}", context)
    if match is None:
        return False
    allowed = {item.strip() for item in match.group(1).split(",") if item.strip()}
    next_token = _lookup_next_token(
        lemma_tokens,
        query_tokens,
        lemma_end=lemma_end,
        query_end=query_end,
    )
    return next_token in allowed


def _matches_same_word_lookahead(
    context: str,
    lemma_tokens: Sequence[str],
    *,
    lemma_end: int,
    context_tail_tokens: tuple[str, ...] | None = None,
) -> bool:
    """Return True when the remaining lemma sequence contains the required tail."""
    match = re.fullmatch(r"_\.\.\.(.+)", context)
    if match is None:
        return False
    tail_tokens = context_tail_tokens
    if tail_tokens is None:
        tail_tokens = tuple(tokenize_ipa(match.group(1)))
    if not tail_tokens:
        return False
    remaining = lemma_tokens[lemma_end:]
    target_length = len(tail_tokens)
    return any(
        tuple(remaining[index : index + target_length]) == tail_tokens
        for index in range(len(remaining) - target_length + 1)
    )


def _matches_word_boundary(
    lemma_tokens: Sequence[str],
    query_tokens: Sequence[str],
    lemma_start: int,
    lemma_end: int,
    query_start: int,
    query_end: int,
    context_tail_tokens: tuple[str, ...] | None,
) -> bool:
    """Return True when the rule span reaches the end of the word."""
    del lemma_start, query_start, context_tail_tokens
    next_token = _lookup_next_token(
        lemma_tokens,
        query_tokens,
        lemma_end=lemma_end,
        query_end=query_end,
    )
    return next_token is None


def _matches_word_initial_before_vowel(
    lemma_tokens: Sequence[str],
    query_tokens: Sequence[str],
    lemma_start: int,
    lemma_end: int,
    query_start: int,
    query_end: int,
    context_tail_tokens: tuple[str, ...] | None,
) -> bool:
    """Return True when the rule span starts the word and is followed by a vowel."""
    del context_tail_tokens
    if lemma_start != 0 or query_start != 0:
        return False
    next_token = _lookup_next_token(
        lemma_tokens,
        query_tokens,
        lemma_end=lemma_end,
        query_end=query_end,
    )
    return _is_vowel(next_token)


def _matches_intervocalic(
    lemma_tokens: Sequence[str],
    query_tokens: Sequence[str],
    lemma_start: int,
    lemma_end: int,
    query_start: int,
    query_end: int,
    context_tail_tokens: tuple[str, ...] | None,
) -> bool:
    """Return True when the rule span is between vowels."""
    del context_tail_tokens
    prev_token = _lookup_prev_token(
        lemma_tokens,
        query_tokens,
        lemma_start=lemma_start,
        query_start=query_start,
    )
    next_token = _lookup_next_token(
        lemma_tokens,
        query_tokens,
        lemma_end=lemma_end,
        query_end=query_end,
    )
    return _is_vowel(prev_token) and _is_vowel(next_token)


def _matches_nasal_consonant(
    lemma_tokens: Sequence[str],
    query_tokens: Sequence[str],
    lemma_start: int,
    lemma_end: int,
    query_start: int,
    query_end: int,
    context_tail_tokens: tuple[str, ...] | None,
) -> bool:
    """Return True when the rule span is followed by nasal + consonant."""
    del lemma_start, query_start, context_tail_tokens
    next_token = _lookup_next_token(
        lemma_tokens,
        query_tokens,
        lemma_end=lemma_end,
        query_end=query_end,
    )
    following_token = _lookup_next_token(
        lemma_tokens,
        query_tokens,
        lemma_end=lemma_end,
        query_end=query_end,
        offset=1,
    )
    return next_token in _NASAL_PHONES and _is_consonant(following_token)


def _matches_after_eir(
    lemma_tokens: Sequence[str],
    query_tokens: Sequence[str],
    lemma_start: int,
    lemma_end: int,
    query_start: int,
    query_end: int,
    context_tail_tokens: tuple[str, ...] | None,
) -> bool:
    """Return True when the previous phone is e, i, or r."""
    del lemma_end, query_end, context_tail_tokens
    prev_token = _lookup_prev_token(
        lemma_tokens,
        query_tokens,
        lemma_start=lemma_start,
        query_start=query_start,
    )
    return prev_token in _AFTER_E_I_R_PHONES


def _matches_except_after_eir(
    lemma_tokens: Sequence[str],
    query_tokens: Sequence[str],
    lemma_start: int,
    lemma_end: int,
    query_start: int,
    query_end: int,
    context_tail_tokens: tuple[str, ...] | None,
) -> bool:
    """Return True when the previous phone is not e, i, or r."""
    del lemma_end, query_end, context_tail_tokens
    prev_token = _lookup_prev_token(
        lemma_tokens,
        query_tokens,
        lemma_start=lemma_start,
        query_start=query_start,
    )
    # Word-initial spans have no previous phone, so they belong to the
    # elsewhere environment rather than the "after e/i/r" environment.
    if prev_token is None:
        return True
    return prev_token not in _AFTER_E_I_R_PHONES


_CONTEXT_HANDLERS: dict[str, _ContextHandler] = {
    "_#": _matches_word_boundary,
    "#_v": _matches_word_initial_before_vowel,
    "v_v": _matches_intervocalic,
    "_nc": _matches_nasal_consonant,
    "after e, i, or r": _matches_after_eir,
    "all environments except after e, i, or r": _matches_except_after_eir,
}


def _matches_context(
    context: object,
    lemma_tokens: Sequence[str],
    query_tokens: Sequence[str],
    *,
    lemma_start: int,
    lemma_end: int,
    query_start: int,
    query_end: int,
    context_tail_tokens: tuple[str, ...] | None = None,
) -> bool:
    """Return True when a rule context is satisfied at the matched span."""
    if context is None:
        return True
    if not isinstance(context, str):
        return False

    normalized = context.strip().lower()
    if normalized in _ALWAYS_MATCH_CONTEXTS:
        return True
    handler = _CONTEXT_HANDLERS.get(normalized)
    if handler is not None:
        return handler(
            lemma_tokens,
            query_tokens,
            lemma_start,
            lemma_end,
            query_start,
            query_end,
            context_tail_tokens,
        )
    if _matches_following_set(
        normalized,
        lemma_tokens,
        query_tokens,
        lemma_end=lemma_end,
        query_end=query_end,
    ):
        return True
    if _matches_same_word_lookahead(
        normalized,
        lemma_tokens,
        lemma_end=lemma_end,
        context_tail_tokens=context_tail_tokens,
    ):
        return True
    return False
