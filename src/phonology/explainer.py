"""Human-readable explanation of applied phonological rules.

Given aligned phoneme sequences and a rule inventory, detect which
phonological rules explain each mismatch block and generate structured
descriptions suitable for APIs and UI consumers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import re
from typing import Any, Sequence, TypeAlias

import yaml  # type: ignore[import-untyped]

from ._paths import resolve_repo_data_dir
from ._phones import VOWEL_PHONES
from .ipa_converter import tokenize_ipa

__all__ = [
    "Rule",
    "Alignment",
    "RuleApplication",
    "Explanation",
    "POSITION_UNKNOWN",
    "load_rules",
    "explain",
    "explain_alignment",
    "to_prose",
]

Rule: TypeAlias = dict[str, Any]
_RULES_BASE_DIR_OVERRIDE: Path | None = None
_NASAL_PHONES = frozenset({"m", "n"})
_AFTER_E_I_R_PHONES = frozenset({"e", "i", "r"})
_ALWAYS_MATCH_CONTEXTS = frozenset(
    {
        "",
        "all environments",
        "vowel contraction across hiatus",
        "quantitative metathesis environments",
    }
)
POSITION_UNKNOWN: int = -1


@dataclass(frozen=True)
class Alignment:
    """Aligned query/lemma token columns used for rule explanation."""

    aligned_query: tuple[str | None, ...]
    aligned_lemma: tuple[str | None, ...]


def _get_rules_base_dir() -> Path:
    """Lazily resolve the rules base directory.

    Tests can override resolution by setting ``_RULES_BASE_DIR_OVERRIDE``
    via ``monkeypatch.setattr``.
    """
    if _RULES_BASE_DIR_OVERRIDE is not None:
        return _RULES_BASE_DIR_OVERRIDE
    return resolve_repo_data_dir("rules")


def __getattr__(name: str) -> Path:
    if name == "RULES_BASE_DIR":
        return _get_rules_base_dir()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose RULES_BASE_DIR in module directory."""
    return sorted(set(globals().keys()) | {"RULES_BASE_DIR"})


def _resolve_rules_dir(rules_dir: Path | str, rules_base_dir: Path) -> Path:
    """Resolve rules directories relative to the packaged rules base."""
    candidate_rules_dir = Path(rules_dir)
    if candidate_rules_dir.is_absolute():
        return candidate_rules_dir

    parts = candidate_rules_dir.parts
    if len(parts) >= 2 and parts[:2] == ("data", "rules"):
        candidate_rules_dir = Path(*parts[2:])

    return rules_base_dir / candidate_rules_dir


@dataclass
class RuleApplication:
    """Record of a single rule applied during phonological alignment."""

    rule_id: str
    description: str
    input_phoneme: str
    output_phoneme: str
    position: int
    dialects: list[str] = field(default_factory=list)
    weight: float = 1.0
    _rule_name: str = field(init=False, repr=False)
    _rule_name_en: str = field(init=False, repr=False)

    def __init__(
        self,
        *,
        rule_id: str,
        description: str | None = None,
        input_phoneme: str | None = None,
        output_phoneme: str | None = None,
        position: int,
        dialects: list[str] | None = None,
        weight: float = 1.0,
        rule_name: str | None = None,
        rule_name_en: str | None = None,
        from_phone: str | None = None,
        to_phone: str | None = None,
    ) -> None:
        self.rule_id = rule_id
        # Backward-compatible fallback: rule_name and description are mutual
        # fallbacks (rule_name takes precedence) so callers can supply either.
        # Similarly, input_phoneme/from_phone and output_phoneme/to_phone are
        # aliases where the canonical field takes precedence over the legacy one.
        fallback = (
            rule_name if rule_name is not None else (description if description is not None else "")
        )
        self.description = description if description is not None else fallback
        self._rule_name = rule_name if rule_name is not None else fallback
        self._rule_name_en = rule_name_en if rule_name_en is not None else ""
        self.input_phoneme = input_phoneme if input_phoneme is not None else (from_phone or "")
        self.output_phoneme = (
            output_phoneme if output_phoneme is not None else (to_phone or "")
        )
        self.position = position
        self.dialects = list(dialects or [])
        self.weight = weight

    @property
    def rule_name(self) -> str:
        """Backward-compatible alias for the display label of this rule step."""
        return self._rule_name

    @property
    def rule_name_en(self) -> str:
        """English display name for the rule."""
        return self._rule_name_en

    @property
    def from_phone(self) -> str:
        """Backward-compatible alias for ``input_phoneme``."""
        return self.input_phoneme

    @property
    def to_phone(self) -> str:
        """Backward-compatible alias for ``output_phoneme``."""
        return self.output_phoneme


@dataclass
class Explanation:
    """Full explanation for a source -> target derivation."""

    source: str
    target: str
    source_ipa: str
    target_ipa: str
    distance: float
    steps: list[RuleApplication]
    prose: str = ""


def load_rules(rules_dir: Path | str) -> dict[str, dict]:
    """Load all YAML rule files from a directory.

    Args:
        rules_dir: Path to the rules directory. Relative inputs are resolved
            from the packaged rules base, so both ``"ancient_greek"`` and
            ``"data/rules/ancient_greek"`` resolve to the same runtime asset.

    Returns:
        Dict mapping rule_id -> rule dict.
    """
    rules_base_dir = _get_rules_base_dir()
    try:
        rules_base_dir = rules_base_dir.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(
            f"Configured rules base directory is missing: {exc}. "
            f"Create the {rules_base_dir} directory before calling load_rules()."
        ) from exc

    candidate_rules_dir = _resolve_rules_dir(rules_dir, rules_base_dir)
    try:
        resolved_rules_dir = candidate_rules_dir.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(
            f"load_rules could not find rules directory {candidate_rules_dir}. "
            f"Expected an existing directory within {rules_base_dir}."
        ) from exc
    if not resolved_rules_dir.is_relative_to(rules_base_dir):
        raise ValueError(
            f"load_rules path must stay within {rules_base_dir}, got {resolved_rules_dir}"
        )
    if not resolved_rules_dir.is_dir():
        raise ValueError(f"load_rules expected a directory, got {resolved_rules_dir}")

    rules: dict[str, dict] = {}
    rule_sources: dict[str, Path] = {}
    for rule_file in sorted(resolved_rules_dir.iterdir()):
        if not rule_file.is_file() or rule_file.suffix.lower() not in {".yaml", ".yml"}:
            continue

        document = yaml.safe_load(rule_file.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError(f"Rule file {rule_file} must contain a top-level mapping")

        raw_rules = document.get("rules")
        if not isinstance(raw_rules, list):
            raise ValueError(f"Rule file {rule_file} must define a list under 'rules'")

        for index, raw_rule in enumerate(raw_rules):
            if not isinstance(raw_rule, dict):
                raise ValueError(f"Rule entry {index} in {rule_file} must be a mapping")

            rule_id = raw_rule.get("id")
            if not isinstance(rule_id, str) or not rule_id.strip():
                raise ValueError(f"Rule entry {index} in {rule_file} must define a non-empty id")
            if rule_id in rules:
                first_defined_in = rule_sources[rule_id]
                raise ValueError(
                    f"Duplicate rule id {rule_id!r} found in {rule_file}; "
                    f"first defined in {first_defined_in}"
                )
            rules[rule_id] = raw_rule
            rule_sources[rule_id] = rule_file

    return rules


@dataclass(frozen=True)
class _TokenizedRule:
    """Rule metadata tokenized for mismatch-block matching."""

    rule: Rule
    input_tokens: tuple[str, ...]
    output_tokens: tuple[str, ...]
    order: int


@dataclass(frozen=True)
class _MismatchBlock:
    """Continuous alignment block that contains no exact-match columns."""

    aligned_query: tuple[str | None, ...]
    aligned_lemma: tuple[str | None, ...]
    lemma_tokens: tuple[str, ...]
    query_tokens: tuple[str, ...]
    lemma_start_position: int
    query_start_position: int


def _tokenize_rule_side(raw_value: object) -> tuple[str, ...]:
    """Tokenize a YAML rule side into comparable IPA tokens."""
    if not isinstance(raw_value, str) or not raw_value:
        return ()
    return tuple(tokenize_ipa(raw_value))


def _tokenize_rules(rules: list[Rule]) -> list[_TokenizedRule]:
    """Pre-tokenize rules and sort them for longest-match-first scanning."""
    tokenized: list[_TokenizedRule] = []
    for order, rule in enumerate(rules):
        input_tokens = _tokenize_rule_side(rule.get("input"))
        output_tokens = _tokenize_rule_side(rule.get("output"))
        if not input_tokens and not output_tokens:
            continue
        tokenized.append(
            _TokenizedRule(
                rule=rule,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                order=order,
            )
        )

    return sorted(
        tokenized,
        key=lambda candidate: (
            -_rule_specificity(candidate.rule),
            -(len(candidate.input_tokens) + len(candidate.output_tokens)),
            -len(candidate.input_tokens),
            -len(candidate.output_tokens),
            candidate.order,
        ),
    )


def _rule_specificity(rule: Rule) -> int:
    """Return a larger value for rules with narrower contextual applicability."""
    context = rule.get("context")
    if not isinstance(context, str):
        return 0
    return 0 if context.strip().lower() in _ALWAYS_MATCH_CONTEXTS else 1


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


def _is_vowel(token: str | None) -> bool:
    """Return True when *token* is one of the known vowel phones."""
    return token in VOWEL_PHONES


def _is_consonant(token: str | None) -> bool:
    """Return True when *token* is present and not classified as a vowel."""
    return token is not None and token not in VOWEL_PHONES


def _lookup_prev_token(
    lemma_ipa: list[str],
    query_ipa: list[str],
    *,
    lemma_start: int,
    query_start: int,
) -> str | None:
    """Return the nearest preceding token, preferring the lemma side."""
    if lemma_start > 0:
        return lemma_ipa[lemma_start - 1]
    if query_start > 0:
        return query_ipa[query_start - 1]
    return None


def _lookup_next_token(
    lemma_ipa: list[str],
    query_ipa: list[str],
    *,
    lemma_end: int,
    query_end: int,
    offset: int = 0,
) -> str | None:
    """Return a following token from the remaining lemma-then-query sequence.

    The function treats the unconsumed tokens as one concatenated sequence:
    ``lemma_ipa[lemma_end:]`` followed by ``query_ipa[query_end:]``.
    ``offset=0`` returns the immediate next token on the lemma side when one
    remains. Once ``offset`` moves past the remaining lemma tokens, lookup
    continues on the query side using ``offset - remaining_lemma``.

    Returns:
        ``str | None``: The token at the requested offset, or ``None`` if no
        token remains in either sequence.

    Examples:
        If ``lemma_ipa=['l', 'e', 'm']``, ``query_ipa=['q', 'u']``,
        ``lemma_end=1``, and ``query_end=0``, then ``offset=0`` returns ``'e'``
        and ``offset=2`` returns ``'q'``.
        If ``lemma_ipa=['l']``, ``query_ipa=['q']``, ``lemma_end=1``,
        ``query_end=1``, and ``offset=0``, the function returns ``None``.
    """
    remaining_lemma = max(0, len(lemma_ipa) - lemma_end)
    if offset < remaining_lemma:
        return lemma_ipa[lemma_end + offset]

    query_index = query_end + max(0, offset - remaining_lemma)
    if query_index < len(query_ipa):
        return query_ipa[query_index]
    return None


def _matches_following_set(
    context: str,
    lemma_ipa: list[str],
    query_ipa: list[str],
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
        lemma_ipa,
        query_ipa,
        lemma_end=lemma_end,
        query_end=query_end,
    )
    return next_token in allowed


def _matches_same_word_lookahead(
    context: str,
    lemma_ipa: list[str],
    *,
    lemma_end: int,
) -> bool:
    """Return True when the remaining lemma sequence contains the required tail."""
    match = re.fullmatch(r"_\.\.\.(.+)", context)
    if match is None:
        return False
    tail_tokens = tuple(tokenize_ipa(match.group(1)))
    if not tail_tokens:
        return False
    remaining = lemma_ipa[lemma_end:]
    target_length = len(tail_tokens)
    return any(
        tuple(remaining[index : index + target_length]) == tail_tokens
        for index in range(len(remaining) - target_length + 1)
    )


def _matches_context(
    context: object,
    lemma_ipa: list[str],
    query_ipa: list[str],
    *,
    lemma_start: int,
    lemma_end: int,
    query_start: int,
    query_end: int,
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
            lemma_ipa,
            query_ipa,
            lemma_end=lemma_end,
            query_end=query_end,
        )
        return next_token is None
    if normalized == "v_v":
        prev_token = _lookup_prev_token(
            lemma_ipa,
            query_ipa,
            lemma_start=lemma_start,
            query_start=query_start,
        )
        next_token = _lookup_next_token(
            lemma_ipa,
            query_ipa,
            lemma_end=lemma_end,
            query_end=query_end,
        )
        return _is_vowel(prev_token) and _is_vowel(next_token)
    if normalized == "_nc":
        next_token = _lookup_next_token(
            lemma_ipa,
            query_ipa,
            lemma_end=lemma_end,
            query_end=query_end,
        )
        following_token = _lookup_next_token(
            lemma_ipa,
            query_ipa,
            lemma_end=lemma_end,
            query_end=query_end,
            offset=1,
        )
        return next_token in _NASAL_PHONES and _is_consonant(following_token)
    if normalized == "after e, i, or r":
        prev_token = _lookup_prev_token(
            lemma_ipa,
            query_ipa,
            lemma_start=lemma_start,
            query_start=query_start,
        )
        return prev_token in _AFTER_E_I_R_PHONES
    if normalized == "all environments except after e, i, or r":
        prev_token = _lookup_prev_token(
            lemma_ipa,
            query_ipa,
            lemma_start=lemma_start,
            query_start=query_start,
        )
        return prev_token not in _AFTER_E_I_R_PHONES
    if _matches_following_set(
        normalized,
        lemma_ipa,
        query_ipa,
        lemma_end=lemma_end,
        query_end=query_end,
    ):
        return True
    if _matches_same_word_lookahead(normalized, lemma_ipa, lemma_end=lemma_end):
        return True
    return False


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
    candidate: _TokenizedRule,
    *,
    lemma_index: int,
    query_index: int,
) -> bool:
    """Return True when a candidate rule matches the aligned columns at this block offset.

    The function verifies that the rule's input tokens (lemma side) and output
    tokens (query side) appear in the expected alignment columns, respecting
    gap structure.  It walks the alignment column-by-column starting from the
    position determined by ``_block_column_index(block, lemma_index, query_index)``.

    Column-walking logic:
      * For each alignment column, the lemma token is checked against the next
        unconsumed ``candidate.input_tokens`` element, and the query token
        against the next ``candidate.output_tokens`` element.
      * Walking stops when all input and output tokens have been consumed.
      * If both lemma-side and query-side gaps are encountered while matching
        the same candidate, the match is rejected as a crossing-gap alignment.

    Post-walk validation:
      * All input and output tokens must be fully consumed
        (``input_index == input_length`` and ``output_index == output_length``).
      * Empty-input rules (insertions) match only when the rule is explicitly
        marked via ``_allows_empty_input(candidate.rule)`` AND at least one
        lemma-gap column was encountered.
      * Empty-output rules (deletions) match only when at least one query-gap
        column was encountered.

    Args:
        block: The mismatch block containing aligned query/lemma columns.
        candidate: Pre-tokenized rule to test against the block.
        lemma_index: Offset into ``block.lemma_tokens`` (gap-free) where the
            candidate's input side should start matching.
        query_index: Offset into ``block.query_tokens`` (gap-free) where the
            candidate's output side should start matching.

    Returns:
        True if the candidate's input/output tokens align with the block's
        columns at the given offsets; False otherwise.
    """
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
            if input_index >= input_length or lemma_token != candidate.input_tokens[input_index]:
                return False
            input_index += 1

        if query_token is not None:
            if (
                output_index >= output_length
                or query_token != candidate.output_tokens[output_index]
            ):
                return False
            output_index += 1

        column_index += 1

    if input_index != input_length or output_index != output_length:
        return False

    # Check for crossing gaps in the matched span
    if _has_crossing_gaps(block.aligned_query, block.aligned_lemma, start_match_column, column_index):
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
    candidate: _TokenizedRule,
    block: _MismatchBlock,
    *,
    lemma_index: int,
    query_index: int,
    lemma_suffix_tokens: tuple[str, ...],
    query_suffix_tokens: tuple[str, ...],
) -> bool:
    """Return True when a `_#` rule matches the remaining word-final suffixes.

    Args:
        candidate: Pre-tokenized rule candidate whose ``input_tokens`` and
            ``output_tokens`` are compared against the remaining word-final
            suffix slices.
        block: Current mismatch block. The `_#` shortcut only applies when the
            remaining mismatched portion of the suffix is fully contained in
            this block; any suffix tail after the block must therefore be an
            exact token-for-token match on both sides.
        lemma_index: Offset into ``block.lemma_tokens`` where the candidate's
            lemma-side suffix should start.
        query_index: Offset into ``block.query_tokens`` where the candidate's
            query-side suffix should start.
        lemma_suffix_tokens: Pre-computed ``tuple(lemma_ipa[lemma_start:])``
            for the current outer-loop position, cached by the caller to
            avoid redundant tuple conversions across candidate rules.
        query_suffix_tokens: Pre-computed ``tuple(query_ipa[query_start:])``
            for the current outer-loop position.

    Returns:
        ``bool``: ``True`` only when the candidate rule's context is ``"_#"``,
        the remaining lemma/query suffixes match the candidate exactly, the
        portion still inside the current mismatch block matches the block's
        remaining token sequences, the remaining aligned block columns do not
        contain crossing gaps, and any candidate tail beyond this block is
        identical on both sides (so no later mismatch block remains).
        Otherwise returns ``False``.
    """
    context = candidate.rule.get("context")
    if not isinstance(context, str) or context.strip().lower() != "_#":
        return False
    if (
        lemma_suffix_tokens != candidate.input_tokens
        or query_suffix_tokens != candidate.output_tokens
    ):
        return False

    block_input_length = len(block.lemma_tokens) - lemma_index
    block_output_length = len(block.query_tokens) - query_index
    if block_input_length < 0 or block_output_length < 0:
        return False
    if (
        candidate.input_tokens[:block_input_length] != block.lemma_tokens[lemma_index:]
        or candidate.output_tokens[:block_output_length] != block.query_tokens[query_index:]
    ):
        return False

    column_index = _block_column_index(
        block,
        lemma_index=lemma_index,
        query_index=query_index,
    )
    if column_index is None:
        return False

    if _has_crossing_gaps(block.aligned_query, block.aligned_lemma, column_index):
        return False

    return (
        candidate.input_tokens[block_input_length:]
        == candidate.output_tokens[block_output_length:]
    )


def _collect_block_applications(
    block: _MismatchBlock,
    query_ipa: list[str],
    lemma_ipa: list[str],
    tokenized_rules: list[_TokenizedRule],
) -> list[RuleApplication]:
    """Match longest rules against a mismatch block."""
    applications: list[RuleApplication] = []
    lemma_index = 0
    query_index = 0

    while lemma_index < len(block.lemma_tokens) or query_index < len(block.query_tokens):
        matched_rule: _TokenizedRule | None = None
        matched_word_final_suffix = False
        # Word-final suffix rules (_# context) are checked before normal block
        # matching so that longer suffix-level rules take priority over
        # individual-segment rules that would otherwise consume the first
        # mismatched token.  tokenized_rules is pre-sorted by
        # _tokenize_rules() (specificity desc, total token length desc), so
        # the first candidate that passes _matches_word_final_suffix is the
        # longest available _# rule.
        # Constant within the inner candidate loop; computed here to share
        # with both _matches_word_final_suffix and _matches_context.
        global_lemma_start = block.lemma_start_position + lemma_index
        global_query_start = block.query_start_position + query_index
        # Cache suffix tuples once per outer-loop position so that the inner
        # candidate loop avoids redundant list→tuple conversions.
        lemma_suffix_tokens = tuple(lemma_ipa[global_lemma_start:])
        query_suffix_tokens = tuple(query_ipa[global_query_start:])
        for candidate in tokenized_rules:
            input_length = len(candidate.input_tokens)
            output_length = len(candidate.output_tokens)

            if _matches_word_final_suffix(
                candidate,
                block,
                lemma_index=lemma_index,
                query_index=query_index,
                lemma_suffix_tokens=lemma_suffix_tokens,
                query_suffix_tokens=query_suffix_tokens,
            ):
                matched_rule = candidate
                matched_word_final_suffix = True
                break

            if block.lemma_tokens[lemma_index : lemma_index + input_length] != candidate.input_tokens:
                continue
            if block.query_tokens[query_index : query_index + output_length] != candidate.output_tokens:
                continue
            if not _matches_block_columns(
                block,
                candidate,
                lemma_index=lemma_index,
                query_index=query_index,
            ):
                continue
            if not _matches_context(
                candidate.rule.get("context"),
                lemma_ipa,
                query_ipa,
                lemma_start=global_lemma_start,
                lemma_end=global_lemma_start + input_length,
                query_start=global_query_start,
                query_end=global_query_start + output_length,
            ):
                continue
            matched_rule = candidate
            break

        if matched_rule is None:
            # Generate an observed-difference annotation for the unmatched
            # mismatch so the user still sees what changed even without a
            # catalogued phonological rule.
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
            applications.append(
                RuleApplication(
                    rule_id=obs_id,
                    description=(
                        f"{obs_ja} / {obs_en}:"
                        f" /{_display_phoneme(lemma_phone)}/"
                        f" \u2192 /{_display_phoneme(query_phone)}/"
                    ),
                    rule_name=obs_ja,
                    rule_name_en=obs_en,
                    input_phoneme=lemma_phone,
                    output_phoneme=query_phone,
                    position=block.lemma_start_position + lemma_index,
                )
            )
            if lemma_token is not None:
                lemma_index += 1
            if query_token is not None:
                query_index += 1
            continue

        input_length = len(matched_rule.input_tokens)
        output_length = len(matched_rule.output_tokens)
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
                position=block.lemma_start_position + lemma_index,
                dialects=dialects,
            )
        )
        if matched_word_final_suffix:
            lemma_index = len(block.lemma_tokens)
            query_index = len(block.query_tokens)
        else:
            lemma_index += input_length
            query_index += output_length

    return applications


def explain(
    query_ipa: list[str],
    lemma_ipa: list[str],
    alignment: Alignment,
    rules: list[Rule],
) -> list[RuleApplication]:
    """Explain which phonological rules account for aligned IPA mismatches.

    Args:
        query_ipa: Tokenized IPA phones for the query form.
        lemma_ipa: Tokenized IPA phones for the lemma form.
        alignment: Alignment of aligned query/lemma token columns.
        rules: A list[Rule] of phonological rules to match against mismatches.

    Returns:
        A list[RuleApplication] describing which rules account for aligned
        query/lemma mismatches.
    """
    tokenized_rules = _tokenize_rules(rules)
    applications: list[RuleApplication] = []
    for block in _iter_mismatch_blocks(alignment):
        applications.extend(
            _collect_block_applications(
                block,
                query_ipa=query_ipa,
                lemma_ipa=lemma_ipa,
                tokenized_rules=tokenized_rules,
            )
        )
    return applications


def explain_alignment(
    source_ipa: str,
    target_ipa: str,
    rule_ids: list[str],
    all_rules: dict[str, dict],
    distance: float = 0.0,
) -> Explanation:
    """Build a structured explanation for a phonological alignment.

    Since this function works with IPA strings rather than orthographic forms,
    the returned ``Explanation`` sets both ``source`` and ``source_ipa`` to
    *source_ipa*, and both ``target`` and ``target_ipa`` to *target_ipa*.
    Callers that display both fields should be aware of this intentional
    duplication.

    Args:
        source_ipa: Source word in IPA.
        target_ipa: Target word in IPA.
        rule_ids: Ordered list of rule ids applied.
        all_rules: Full rule registry from load_rules().
        distance: Normalized phonological distance for the alignment.

    Returns:
        Explanation object with step-by-step and prose description.
    """
    steps: list[RuleApplication] = []
    for rule_id in rule_ids:
        rule = all_rules.get(rule_id, {"id": rule_id})
        raw_input = rule.get("input")
        raw_output = rule.get("output")
        input_phoneme = raw_input if isinstance(raw_input, str) else ""
        output_phoneme = raw_output if isinstance(raw_output, str) else ""
        rule_name = _rule_name_for_description(rule)
        rule_name_en = _rule_name_en_for_description(rule)
        dialects = _extract_dialects(rule.get("dialects", []))
        steps.append(
            RuleApplication(
                rule_id=rule_id,
                description=_build_description(
                    rule,
                    rule_name,
                    input_phoneme,
                    output_phoneme,
                ),
                rule_name=rule_name,
                rule_name_en=rule_name_en,
                input_phoneme=input_phoneme,
                output_phoneme=output_phoneme,
                position=POSITION_UNKNOWN,
                dialects=dialects,
            )
        )

    return Explanation(
        source=source_ipa,
        target=target_ipa,
        source_ipa=source_ipa,
        target_ipa=target_ipa,
        distance=distance,
        steps=steps,
    )


def to_prose(explanation: Explanation) -> str:
    """Generate canonical prose for a structured explanation.

    Args:
        explanation: Structured explanation object to render.

    Returns:
        A prose summary string containing the source/target alignment,
        normalized distance, and any applied rules.  When no rules were
        recorded, the summary includes a "No rule applications" note.
    """
    def format_with_optional_ipa(text: str, ipa: str) -> str:
        """Render text with IPA only when it differs from the surface form."""
        if text == ipa:
            return text
        return f"{text} /{ipa}/"

    # Dialect labels are omitted from prose, but step weights are retained.
    source_repr = format_with_optional_ipa(
        explanation.source,
        explanation.source_ipa,
    )
    target_repr = format_with_optional_ipa(
        explanation.target,
        explanation.target_ipa,
    )

    if explanation.steps:
        step_summary = "; ".join(
            (
                f"{step.rule_name} ({step.from_phone} -> {step.to_phone} "
                f"at position {step.position}, weight {step.weight:g})"
            )
            for step in explanation.steps
        )
        prose = (
            f"{source_repr} aligns to "
            f"{target_repr} with distance "
            f"{explanation.distance:.3f}. Applied rules: {step_summary}."
        )
    else:
        if math.isclose(explanation.distance, 0.0, abs_tol=1e-9):
            prose = f"{source_repr} is an exact match for {target_repr}."
        else:
            prose = (
                f"{source_repr} aligns to "
                f"{target_repr} with distance "
                f"{explanation.distance:.3f}. No rule applications were recorded."
            )

    return prose
