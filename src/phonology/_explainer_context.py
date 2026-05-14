"""Rule-context predicates and token-lookup helpers for the explainer."""

from __future__ import annotations

from collections.abc import Sequence
import re

from ._phones import VOWEL_PHONES
from ._explainer_rule_tokenize import _ALWAYS_MATCH_CONTEXTS
from .ipa_converter import tokenize_ipa

_NASAL_PHONES = frozenset({"m", "n"})
_AFTER_E_I_R_PHONES = frozenset({"e", "i", "r"})


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
    if normalized == "_#":
        next_token = _lookup_next_token(
            lemma_tokens,
            query_tokens,
            lemma_end=lemma_end,
            query_end=query_end,
        )
        return next_token is None
    if normalized == "v_v":
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
    if normalized == "_nc":
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
    if normalized == "after e, i, or r":
        prev_token = _lookup_prev_token(
            lemma_tokens,
            query_tokens,
            lemma_start=lemma_start,
            query_start=query_start,
        )
        return prev_token in _AFTER_E_I_R_PHONES
    if normalized == "all environments except after e, i, or r":
        prev_token = _lookup_prev_token(
            lemma_tokens,
            query_tokens,
            lemma_start=lemma_start,
            query_start=query_start,
        )
        return prev_token not in _AFTER_E_I_R_PHONES
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
