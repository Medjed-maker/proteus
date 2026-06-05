"""Mismatch-block matching, observed-fallback applications, and explain() entry.

This module owns the rule-matching state machine. It walks an :class:`Alignment`,
splits mismatched columns into :class:`_MismatchBlock` instances, and runs each
candidate :class:`TokenizedRule` against every block cursor position. Suffix and
context predicates live here as well.

The logger is bound to ``"phonology.explainer"`` so ``caplog`` tests keep
working after the split.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
import logging

from ._context import (
    _matches_context,
)
from ._rule_tokenize import (
    Rule,
    TokenizedRule,
    tokenize_rules_for_matching,
)
from ._types import (
    Alignment,
    RuleApplication,
    RuleMetadata,
    _MismatchBlock,
    _RuleMatchResult,
    _WordFinalSuffixMatch,
)

logger = logging.getLogger("phonology.explainer")


PhoneMatcher = Callable[[str, str], bool]


def _resolve_matching_phone_inventory(
    query_tokens: Sequence[str],
    lemma_tokens: Sequence[str],
    phone_inventory: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Return explicit phones or infer tokenization from already-tokenized forms."""
    if phone_inventory is not None:
        return tuple(phone_inventory)
    return tuple(
        sorted(
            {token for token in (*query_tokens, *lemma_tokens) if token},
            key=lambda token: (-len(token), token),
        )
    )


def _default_phone_matcher(actual: str, expected: str) -> bool:
    """Return the default profile's phone-match result, falling back to equality."""
    from ..core.ports.profiles import get_default_language_profile

    matcher = get_default_language_profile().phone_matcher
    if matcher is None:
        return actual == expected
    return matcher(actual, expected)


def _phone_equal(
    actual: str,
    expected: str,
    *,
    strict: bool,
    phone_matcher: PhoneMatcher | None = None,
) -> bool:
    """Compare a form phone to a rule phone, exactly or with length tolerance.

    The strict pass is used first so explicitly length-marked forms keep their
    exact rule; directional dichronous length tolerance is only applied as a
    fallback (see :func:`_find_matching_rule_candidate`).
    """
    if strict:
        return actual == expected
    matcher = phone_matcher or _default_phone_matcher
    return matcher(actual, expected)


def _phone_seq_equal(
    actual: Sequence[str],
    expected: Sequence[str],
    *,
    strict: bool,
    phone_matcher: PhoneMatcher | None = None,
) -> bool:
    """Sequence form of :func:`_phone_equal`."""
    if strict:
        return tuple(actual) == tuple(expected)
    return len(actual) == len(expected) and all(
        _phone_equal(
            actual_token,
            expected_token,
            strict=False,
            phone_matcher=phone_matcher,
        )
        for actual_token, expected_token in zip(actual, expected)
    )


def _matches_lemma_constraints(
    rule: Rule,
    lemma_metadata: RuleMetadata | None,
) -> bool:
    """Return whether lemma metadata satisfies optional rule constraints."""
    constraints = rule.get("lemma_constraints")
    if constraints is None:
        return True
    if not isinstance(constraints, Mapping) or lemma_metadata is None:
        return False

    for key, expected in constraints.items():
        if not isinstance(key, str):
            return False
        actual = lemma_metadata.get(key)
        if isinstance(expected, list):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def _is_exact_match(query_token: str | None, lemma_token: str | None) -> bool:
    """Return True when the aligned column is an exact, non-gap match."""
    return (
        query_token is not None
        and lemma_token is not None
        and query_token == lemma_token
    )


def _iter_mismatch_blocks(alignment: Alignment) -> list[_MismatchBlock]:
    """Split an alignment into continuous mismatch blocks."""
    if len(alignment.aligned_query) != len(alignment.aligned_lemma):
        raise ValueError("Alignment columns must have identical lengths")

    blocks: list[_MismatchBlock] = []
    lemma_position = 0
    query_position = 0
    index = 0

    while index < len(alignment.aligned_query):
        query_token = alignment.aligned_query[index]
        lemma_token = alignment.aligned_lemma[index]
        if _is_exact_match(query_token, lemma_token):
            lemma_position += 1
            query_position += 1
            index += 1
            continue

        block_lemma_start = lemma_position
        block_query_start = query_position
        block_aligned_query: list[str | None] = []
        block_aligned_lemma: list[str | None] = []
        block_lemma_tokens: list[str] = []
        block_query_tokens: list[str] = []

        while index < len(alignment.aligned_query) and not _is_exact_match(
            alignment.aligned_query[index],
            alignment.aligned_lemma[index],
        ):
            current_query = alignment.aligned_query[index]
            current_lemma = alignment.aligned_lemma[index]
            block_aligned_query.append(current_query)
            block_aligned_lemma.append(current_lemma)
            if current_lemma is not None:
                block_lemma_tokens.append(current_lemma)
                lemma_position += 1
            if current_query is not None:
                block_query_tokens.append(current_query)
                query_position += 1
            index += 1

        blocks.append(
            _MismatchBlock(
                aligned_query=tuple(block_aligned_query),
                aligned_lemma=tuple(block_aligned_lemma),
                lemma_tokens=tuple(block_lemma_tokens),
                query_tokens=tuple(block_query_tokens),
                lemma_start_position=block_lemma_start,
                query_start_position=block_query_start,
            )
        )

    return blocks


def _rule_name_for_description(rule: Rule) -> str:
    """Return the preferred display label for a rule."""
    for key in ("name_ja", "name_en", "id"):
        value = rule.get(key)
        if isinstance(value, str) and value:
            return value
    return "unknown rule"


def _rule_name_en_for_description(rule: Rule) -> str:
    """Return the English display label for a rule, if available."""
    value = rule.get("name_en")
    if isinstance(value, str) and value:
        return value
    return ""


def _display_phoneme(phoneme: str) -> str:
    """Render phonemes in prose, using ∅ for empty sides."""
    return phoneme if phoneme else "∅"


def _extract_dialects(raw_dialects: object) -> list[str]:
    """Return unique string dialect labels while preserving their first-seen order."""
    if not isinstance(raw_dialects, list):
        return []

    dialects: list[str] = []
    seen: set[str] = set()
    for dialect in raw_dialects:
        if not isinstance(dialect, str) or dialect in seen:
            continue
        seen.add(dialect)
        dialects.append(dialect)
    return dialects


def _build_description(
    rule: Rule,
    rule_name: str,
    input_phoneme: str,
    output_phoneme: str,
) -> str:
    """Build a concise Japanese-first explanation string for a rule application."""
    description = (
        f"{rule_name}: /{_display_phoneme(input_phoneme)}/ → "
        f"/{_display_phoneme(output_phoneme)}/"
    )
    context = rule.get("context")
    if isinstance(context, str) and context.strip():
        description = f"{description} （環境: {context.strip()}）"
    return description


def _block_column_index(
    block: _MismatchBlock,
    *,
    lemma_index: int,
    query_index: int,
) -> int | None:
    """Map token offsets within a mismatch block to the local aligned column index."""
    consumed_lemma = 0
    consumed_query = 0
    for column_index, (query_token, lemma_token) in enumerate(
        zip(block.aligned_query, block.aligned_lemma)
    ):
        if consumed_lemma == lemma_index and consumed_query == query_index:
            return column_index
        if lemma_token is not None:
            consumed_lemma += 1
        if query_token is not None:
            consumed_query += 1

    if consumed_lemma == lemma_index and consumed_query == query_index:
        return len(block.aligned_query)
    return None


def _current_block_column(
    block: _MismatchBlock,
    *,
    lemma_index: int,
    query_index: int,
) -> tuple[int, str | None, str | None]:
    """Return the current aligned column for the given mismatch-block cursor."""
    column_index = _block_column_index(
        block,
        lemma_index=lemma_index,
        query_index=query_index,
    )
    if column_index is None or column_index >= len(block.aligned_query):
        raise RuntimeError(
            "Mismatch block cursor advanced beyond aligned columns before block consumption finished"
        )
    return (
        column_index,
        block.aligned_query[column_index],
        block.aligned_lemma[column_index],
    )


def _has_crossing_gaps(
    aligned_query: Sequence[str | None],
    aligned_lemma: Sequence[str | None],
    start_column: int,
    end_column: int | None = None,
) -> bool:
    """Return True when both a lemma gap and a query gap appear in [start_column, end_column)."""
    q_span = aligned_query[start_column:end_column]
    l_span = aligned_lemma[start_column:end_column]
    return None in q_span and None in l_span


def _allows_empty_input(rule: Rule) -> bool:
    """Return True when a rule explicitly opts into empty-input insertion matches."""
    return bool(rule.get("is_insertion") or rule.get("allows_empty_input"))


def _matches_block_columns(
    block: _MismatchBlock,
    candidate: TokenizedRule,
    *,
    lemma_index: int,
    query_index: int,
    strict: bool,
    phone_matcher: PhoneMatcher | None = None,
) -> bool:
    """Return True when a candidate rule matches the aligned columns at this block offset."""
    column_index = _block_column_index(
        block,
        lemma_index=lemma_index,
        query_index=query_index,
    )
    if column_index is None or column_index >= len(block.aligned_query):
        return False

    input_length = len(candidate.input_tokens)
    output_length = len(candidate.output_tokens)
    input_index = 0
    output_index = 0
    start_match_column = column_index

    while column_index < len(block.aligned_query):
        if input_index == input_length and output_index == output_length:
            break

        query_token = block.aligned_query[column_index]
        lemma_token = block.aligned_lemma[column_index]

        if lemma_token is not None:
            if input_index >= input_length or not _phone_equal(
                lemma_token,
                candidate.input_tokens[input_index],
                strict=strict,
                phone_matcher=phone_matcher,
            ):
                return False
            input_index += 1

        if query_token is not None:
            if output_index >= output_length or not _phone_equal(
                query_token,
                candidate.output_tokens[output_index],
                strict=strict,
                phone_matcher=phone_matcher,
            ):
                return False
            output_index += 1

        column_index += 1

    if input_index != input_length or output_index != output_length:
        return False

    if _has_crossing_gaps(
        block.aligned_query, block.aligned_lemma, start_match_column, column_index
    ):
        return False

    if input_length == 0:
        saw_lemma_gap = None in block.aligned_lemma[start_match_column:column_index]
        if not _allows_empty_input(candidate.rule) or not saw_lemma_gap:
            return False
    if output_length == 0:
        saw_query_gap = None in block.aligned_query[start_match_column:column_index]
        if not saw_query_gap:
            return False
    return True


def _matches_word_final_suffix(
    candidate: TokenizedRule,
    block: _MismatchBlock,
    *,
    lemma_tokens: Sequence[str],
    query_tokens: Sequence[str],
    lemma_index: int,
    query_index: int,
    strict: bool,
    phone_matcher: PhoneMatcher | None = None,
) -> _WordFinalSuffixMatch | None:
    """Return match metadata when a `_#` rule matches the word-final suffix."""
    context = candidate.rule.get("context")
    if not isinstance(context, str) or context.strip().lower() != "_#":
        return None

    lemma_candidate_start = len(lemma_tokens) - len(candidate.input_tokens)
    query_candidate_start = len(query_tokens) - len(candidate.output_tokens)
    if lemma_candidate_start < 0 or query_candidate_start < 0:
        return None
    if not _phone_seq_equal(
        lemma_tokens[lemma_candidate_start:],
        candidate.input_tokens,
        strict=strict,
        phone_matcher=phone_matcher,
    ):
        return None
    if not _phone_seq_equal(
        query_tokens[query_candidate_start:],
        candidate.output_tokens,
        strict=strict,
        phone_matcher=phone_matcher,
    ):
        return None

    global_lemma_start = block.lemma_start_position + lemma_index
    global_query_start = block.query_start_position + query_index
    if lemma_candidate_start > global_lemma_start:
        return None
    if query_candidate_start > global_query_start:
        return None

    input_offset = global_lemma_start - lemma_candidate_start
    output_offset = global_query_start - query_candidate_start
    if candidate.input_tokens[:input_offset] != candidate.output_tokens[:output_offset]:
        return None

    block_input_length = len(block.lemma_tokens) - lemma_index
    block_output_length = len(block.query_tokens) - query_index
    if block_input_length < 0 or block_output_length < 0:
        return None
    if not _phone_seq_equal(
        block.lemma_tokens[lemma_index:],
        candidate.input_tokens[input_offset : input_offset + block_input_length],
        strict=strict,
        phone_matcher=phone_matcher,
    ) or not _phone_seq_equal(
        block.query_tokens[query_index:],
        candidate.output_tokens[output_offset : output_offset + block_output_length],
        strict=strict,
        phone_matcher=phone_matcher,
    ):
        return None

    column_index = _block_column_index(
        block,
        lemma_index=lemma_index,
        query_index=query_index,
    )
    if column_index is None:
        return None

    if _has_crossing_gaps(block.aligned_query, block.aligned_lemma, column_index):
        return None

    if (
        candidate.input_tokens[input_offset + block_input_length :]
        == candidate.output_tokens[output_offset + block_output_length :]
    ):
        return _WordFinalSuffixMatch(lemma_start_position=lemma_candidate_start)
    return None


def _find_word_final_suffix_match(
    block: _MismatchBlock,
    query_tokens: Sequence[str],
    lemma_tokens: Sequence[str],
    candidate: TokenizedRule,
    *,
    lemma_index: int,
    query_index: int,
    strict: bool,
    phone_matcher: PhoneMatcher | None = None,
) -> _RuleMatchResult | None:
    """Return a normalized match result for a `_#` suffix rule candidate."""
    suffix_match = _matches_word_final_suffix(
        candidate,
        block,
        lemma_tokens=lemma_tokens,
        query_tokens=query_tokens,
        lemma_index=lemma_index,
        query_index=query_index,
        strict=strict,
        phone_matcher=phone_matcher,
    )
    if suffix_match is None:
        return None
    return _RuleMatchResult(
        matched_rule=candidate,
        word_final_suffix_match=suffix_match,
        consumed_lemma_tokens=len(block.lemma_tokens) - lemma_index,
        consumed_query_tokens=len(block.query_tokens) - query_index,
        application_position=suffix_match.lemma_start_position,
    )


def _find_normal_block_match(
    block: _MismatchBlock,
    query_tokens: Sequence[str],
    lemma_tokens: Sequence[str],
    candidate: TokenizedRule,
    *,
    lemma_index: int,
    query_index: int,
    strict: bool,
    phone_matcher: PhoneMatcher | None = None,
    phone_inventory: Iterable[str] | None = None,
    vowel_phones: frozenset[str] | Sequence[str] | None = None,
    always_match_contexts: Iterable[str] | None = None,
) -> _RuleMatchResult | None:
    """Return a normalized match result for a non-suffix block candidate."""
    input_length = len(candidate.input_tokens)
    output_length = len(candidate.output_tokens)
    if not _phone_seq_equal(
        block.lemma_tokens[lemma_index : lemma_index + input_length],
        candidate.input_tokens,
        strict=strict,
        phone_matcher=phone_matcher,
    ):
        return None
    if not _phone_seq_equal(
        block.query_tokens[query_index : query_index + output_length],
        candidate.output_tokens,
        strict=strict,
        phone_matcher=phone_matcher,
    ):
        return None
    if not _matches_block_columns(
        block,
        candidate,
        lemma_index=lemma_index,
        query_index=query_index,
        strict=strict,
        phone_matcher=phone_matcher,
    ):
        return None

    global_lemma_start = block.lemma_start_position + lemma_index
    global_query_start = block.query_start_position + query_index
    if not _matches_context(
        candidate.rule.get("context"),
        lemma_tokens,
        query_tokens,
        lemma_start=global_lemma_start,
        lemma_end=global_lemma_start + input_length,
        query_start=global_query_start,
        query_end=global_query_start + output_length,
        context_tail_tokens=candidate.context_tail_tokens,
        phone_inventory=phone_inventory,
        vowel_phones=vowel_phones,
        always_match_contexts=always_match_contexts,
    ):
        return None

    return _RuleMatchResult(
        matched_rule=candidate,
        word_final_suffix_match=None,
        consumed_lemma_tokens=input_length,
        consumed_query_tokens=output_length,
        application_position=global_lemma_start,
    )


def _find_matching_rule_candidate(
    block: _MismatchBlock,
    query_tokens: Sequence[str],
    lemma_tokens: Sequence[str],
    tokenized_rules: Sequence[TokenizedRule],
    *,
    lemma_index: int,
    query_index: int,
    lemma_metadata: RuleMetadata | None = None,
    phone_matcher: PhoneMatcher | None = None,
    phone_inventory: Iterable[str] | None = None,
    vowel_phones: frozenset[str] | Sequence[str] | None = None,
    always_match_contexts: Iterable[str] | None = None,
) -> _RuleMatchResult | None:
    """Return the first matching rule candidate at the current block cursor.

    Rules are tried in two passes. The strict pass requires exact phone
    equality and reproduces the historical matching behavior. Only when no rule
    matches exactly does the tolerant pass run, which lets unmarked dichronous
    vowels (alpha, iota, upsilon) satisfy explicitly long rule phones so that
    length-ambiguous forms such as unmarked ``μάτηρ`` still resolve to their
    catalogued rule. This ordering ensures an explicitly length-marked form
    keeps its exact rule rather than a looser dichronous fallback.
    """
    for strict in (True, False):
        for candidate in tokenized_rules:
            if not _matches_lemma_constraints(candidate.rule, lemma_metadata):
                continue

            suffix_match = _find_word_final_suffix_match(
                block,
                query_tokens,
                lemma_tokens,
                candidate,
                lemma_index=lemma_index,
                query_index=query_index,
                strict=strict,
                phone_matcher=phone_matcher,
            )
            if suffix_match is not None:
                return suffix_match

            normal_match = _find_normal_block_match(
                block,
                query_tokens,
                lemma_tokens,
                candidate,
                lemma_index=lemma_index,
                query_index=query_index,
                strict=strict,
                phone_matcher=phone_matcher,
                phone_inventory=phone_inventory,
                vowel_phones=vowel_phones,
                always_match_contexts=always_match_contexts,
            )
            if normal_match is not None:
                return normal_match
    return None


def _build_observed_application_for_column(
    block: _MismatchBlock,
    *,
    lemma_index: int,
    query_index: int,
) -> RuleApplication:
    """Build an observed fallback application for the current mismatch column."""
    column_index, query_token, lemma_token = _current_block_column(
        block,
        lemma_index=lemma_index,
        query_index=query_index,
    )
    if lemma_token is None and query_token is None:
        raise RuntimeError(
            "Encountered an invalid mismatch block column with lemma_token=None "
            f"and query_token=None at column_index={column_index}, "
            f"lemma_index={lemma_index}, query_index={query_index}; "
            "the cursor cannot advance from a double-gap column."
        )

    lemma_phone = lemma_token or ""
    query_phone = query_token or ""
    if lemma_phone and query_phone:
        obs_id, obs_ja, obs_en = "OBS-SUB", "観測された置換", "Observed substitution"
    elif lemma_phone:
        obs_id, obs_ja, obs_en = "OBS-DEL", "観測された脱落", "Observed deletion"
    else:
        obs_id, obs_ja, obs_en = "OBS-INS", "観測された挿入", "Observed insertion"

    return RuleApplication(
        rule_id=obs_id,
        description=(
            f"{obs_ja} / {obs_en}:"
            f" /{_display_phoneme(lemma_phone)}/"
            f" → /{_display_phoneme(query_phone)}/"
        ),
        rule_name=obs_ja,
        rule_name_en=obs_en,
        input_phoneme=lemma_phone,
        output_phoneme=query_phone,
        position=block.lemma_start_position + lemma_index,
    )


def _advance_block_cursors(
    block: _MismatchBlock,
    *,
    lemma_index: int,
    query_index: int,
    match_result: _RuleMatchResult | None = None,
) -> tuple[int, int]:
    """Advance mismatch-block cursors and enforce forward progress."""
    next_lemma_index = lemma_index
    next_query_index = query_index

    if match_result is not None:
        next_lemma_index += match_result.consumed_lemma_tokens
        next_query_index += match_result.consumed_query_tokens
    else:
        _, query_token, lemma_token = _current_block_column(
            block,
            lemma_index=lemma_index,
            query_index=query_index,
        )
        if lemma_token is not None:
            next_lemma_index += 1
        if query_token is not None:
            next_query_index += 1

    if next_lemma_index == lemma_index and next_query_index == query_index:
        raise RuntimeError(
            "Mismatch block cursor failed to advance; each iteration must consume "
            "at least one lemma or query token."
        )
    return next_lemma_index, next_query_index


def _collect_block_applications(
    block: _MismatchBlock,
    query_tokens: Sequence[str],
    lemma_tokens: Sequence[str],
    tokenized_rules: Sequence[TokenizedRule],
    lemma_metadata: RuleMetadata | None = None,
    phone_matcher: PhoneMatcher | None = None,
    phone_inventory: Iterable[str] | None = None,
    vowel_phones: frozenset[str] | Sequence[str] | None = None,
    always_match_contexts: Iterable[str] | None = None,
) -> list[RuleApplication]:
    """Match longest rules against a mismatch block."""
    applications: list[RuleApplication] = []
    lemma_index = 0
    query_index = 0

    while lemma_index < len(block.lemma_tokens) or query_index < len(
        block.query_tokens
    ):
        match_result = _find_matching_rule_candidate(
            block,
            query_tokens,
            lemma_tokens,
            tokenized_rules,
            lemma_index=lemma_index,
            query_index=query_index,
            lemma_metadata=lemma_metadata,
            phone_matcher=phone_matcher,
            phone_inventory=phone_inventory,
            vowel_phones=vowel_phones,
            always_match_contexts=always_match_contexts,
        )

        if match_result is None:
            applications.append(
                _build_observed_application_for_column(
                    block,
                    lemma_index=lemma_index,
                    query_index=query_index,
                )
            )
            lemma_index, query_index = _advance_block_cursors(
                block,
                lemma_index=lemma_index,
                query_index=query_index,
            )
            continue

        matched_rule = match_result.matched_rule
        input_phoneme = "".join(matched_rule.input_tokens)
        output_phoneme = "".join(matched_rule.output_tokens)
        rule_name = _rule_name_for_description(matched_rule.rule)
        rule_name_en = _rule_name_en_for_description(matched_rule.rule)
        dialects = _extract_dialects(matched_rule.rule.get("dialects", []))
        applications.append(
            RuleApplication(
                rule_id=str(matched_rule.rule.get("id", "unknown-rule")),
                description=_build_description(
                    matched_rule.rule,
                    rule_name,
                    input_phoneme,
                    output_phoneme,
                ),
                rule_name=rule_name,
                rule_name_en=rule_name_en,
                input_phoneme=input_phoneme,
                output_phoneme=output_phoneme,
                position=match_result.application_position,
                dialects=dialects,
            )
        )
        lemma_index, query_index = _advance_block_cursors(
            block,
            lemma_index=lemma_index,
            query_index=query_index,
            match_result=match_result,
        )

    return applications


def explain_with_tokenized_rules(
    query_tokens: Sequence[str],
    lemma_tokens: Sequence[str],
    alignment: Alignment,
    tokenized_rules: Sequence[TokenizedRule],
    lemma_metadata: RuleMetadata | None = None,
    phone_matcher: PhoneMatcher | None = None,
    phone_inventory: Iterable[str] | None = None,
    vowel_phones: frozenset[str] | Sequence[str] | None = None,
    always_match_contexts: Iterable[str] | None = None,
) -> list[RuleApplication]:
    """Explain aligned IPA mismatches using pre-tokenized phonological rules.

    Args:
        query_tokens: Tokenized IPA phones for the query form.
        lemma_tokens: Tokenized IPA phones for the lemma form.
        alignment: Alignment of aligned query/lemma token columns.
        tokenized_rules: Pre-tokenized phonological rules to match against
            mismatches.
        lemma_metadata: Optional lexicon metadata used by constrained rules.

    Returns:
        A list[RuleApplication] describing which rules account for aligned
        query/lemma mismatches.
    """
    resolved_phone_inventory = _resolve_matching_phone_inventory(
        query_tokens,
        lemma_tokens,
        phone_inventory,
    )
    applications: list[RuleApplication] = []
    for block in _iter_mismatch_blocks(alignment):
        applications.extend(
            _collect_block_applications(
                block,
                query_tokens=query_tokens,
                lemma_tokens=lemma_tokens,
                tokenized_rules=tokenized_rules,
                lemma_metadata=lemma_metadata,
                phone_matcher=phone_matcher,
                phone_inventory=resolved_phone_inventory,
                vowel_phones=vowel_phones,
                always_match_contexts=always_match_contexts,
            )
        )
    return applications


def explain(
    query_tokens: Sequence[str],
    lemma_tokens: Sequence[str],
    alignment: Alignment,
    rules: list[Rule],
    lemma_metadata: RuleMetadata | None = None,
    phone_matcher: PhoneMatcher | None = None,
    phone_inventory: Iterable[str] | None = None,
    vowel_phones: frozenset[str] | Sequence[str] | None = None,
    always_match_contexts: Iterable[str] | None = None,
) -> list[RuleApplication]:
    """Explain which phonological rules account for aligned IPA mismatches.

    Args:
        query_tokens: Tokenized IPA phones for the query form.
        lemma_tokens: Tokenized IPA phones for the lemma form.
        alignment: Alignment of aligned query/lemma token columns.
        rules: Phonological rules to match against mismatches.
        lemma_metadata: Optional lexicon metadata used by constrained rules.

    Returns:
        A list[RuleApplication] describing which rules account for aligned
        query/lemma mismatches.
    """
    resolved_phone_inventory = _resolve_matching_phone_inventory(
        query_tokens,
        lemma_tokens,
        phone_inventory,
    )
    return explain_with_tokenized_rules(
        query_tokens=query_tokens,
        lemma_tokens=lemma_tokens,
        alignment=alignment,
        tokenized_rules=tokenize_rules_for_matching(
            rules,
            phone_inventory=resolved_phone_inventory,
            always_match_contexts=always_match_contexts,
        ),
        lemma_metadata=lemma_metadata,
        phone_matcher=phone_matcher,
        phone_inventory=resolved_phone_inventory,
        vowel_phones=vowel_phones,
        always_match_contexts=always_match_contexts,
    )
