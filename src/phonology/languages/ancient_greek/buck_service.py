"""Read-only query service for Buck-normalized Ancient Greek reference data."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import lru_cache
import math
from types import MappingProxyType
from typing import Any, Callable, Mapping, NewType
import unicodedata

from .buck import BuckData, load_buck_data

__all__ = [
    "BuckDialect",
    "BuckGlossaryEntry",
    "BuckMetadata",
    "BuckReference",
    "BuckReferenceIndex",
    "BuckReviewStatus",
    "BuckRule",
    "build_buck_reference_index",
    "canonicalize_buck_section",
    "clear_buck_reference_index_cache",
]


_ReadOnlyMapping = Mapping[str, Any]
_InscriptionNumber = str | tuple[int, ...]
BuckReviewStatus = NewType("BuckReviewStatus", str)


@dataclass(frozen=True)
class BuckMetadata:
    """Review and citation metadata for Buck reference data."""

    status: str
    review_status: BuckReviewStatus
    citation_ready: bool


@dataclass(frozen=True)
class BuckReference:
    """Specific Buck section/page reference for an example entry."""

    section: str | None
    page: int | None


@dataclass(frozen=True)
class BuckRule:
    """Read-only Buck grammar rule summary."""

    id: str
    buck_section: str | None
    category: str | None
    description: str | None
    transformation: _ReadOnlyMapping
    affected_dialects: tuple[str, ...]
    variants: tuple[_ReadOnlyMapping, ...]
    notes: str | None
    status: str
    review_status: BuckReviewStatus
    citation_ready: bool


@dataclass(frozen=True)
class BuckDialect:
    """Read-only Buck dialect catalog entry."""

    id: str
    name: str | None
    kind: str | None
    group: str | None
    parent: str | None
    rules: tuple[str, ...]
    status: str
    review_status: BuckReviewStatus
    citation_ready: bool


@dataclass(frozen=True)
class BuckGlossaryEntry:
    """Read-only Buck glossary example entry."""

    word: str
    standard_form: str | None
    dialect: str
    rule_id: str | None
    definition: str | None
    inscription_no: _InscriptionNumber | None
    buck_ref: BuckReference | None
    notes: str | None
    status: str
    review_status: BuckReviewStatus
    citation_ready: bool


@dataclass(frozen=True)
class BuckReferenceIndex:
    """Read-only index over validated Buck reference data."""

    metadata: BuckMetadata
    rules: tuple[BuckRule, ...]
    dialects: tuple[BuckDialect, ...]
    glossary_entries: tuple[BuckGlossaryEntry, ...]
    _rules_by_id: Mapping[str, BuckRule]
    _rules_by_section: Mapping[str, tuple[BuckRule, ...]]
    _dialects_by_id: Mapping[str, BuckDialect]
    _glossary_by_word: Mapping[str, tuple[BuckGlossaryEntry, ...]]
    _glossary_by_standard_form: Mapping[str, tuple[BuckGlossaryEntry, ...]]

    def get_rule(self, rule_id: str) -> BuckRule | None:
        """Return the rule with *rule_id*, if present."""
        return self._rules_by_id.get(rule_id)

    def list_rules(
        self,
        *,
        category: str | None = None,
        dialect: str | None = None,
    ) -> tuple[BuckRule, ...]:
        """Return rules, optionally filtered by category and affected dialect.

        The ``dialect`` filter matches against each rule's ``affected_dialects``
        field, *not* the ``rules`` list in the dialect catalog. Use
        :meth:`get_dialect_rules` for the catalog-driven relationship.
        """
        return tuple(
            rule
            for rule in self.rules
            if _rule_matches_filters(rule, category=category, dialect=dialect)
        )

    def get_rules_by_section(self, section: str | int | float) -> tuple[BuckRule, ...]:
        """Return rules whose Buck section matches *section* after canonicalization."""
        return self._rules_by_section.get(canonicalize_buck_section(section), ())

    def find_rules(
        self,
        *,
        rule_id: str | None = None,
        section: str | int | float | None = None,
        category: str | None = None,
        dialect: str | None = None,
    ) -> tuple[BuckRule, ...]:
        """Return rules matching the given constraints in deterministic id order.

        Candidate selection prefers the most specific key available: an exact
        ``rule_id`` first, then ``section`` (canonicalized), otherwise the full
        rule set. The ``category`` and ``dialect`` filters then apply on top,
        using the same matching semantics as :meth:`list_rules` (``dialect``
        matches a rule's ``affected_dialects``). Results are sorted by ``id`` so
        callers that truncate are not exposed to source-file ordering.
        """
        canonical_section = (
            canonicalize_buck_section(section) if section is not None else None
        )

        if rule_id is not None:
            rule = self.get_rule(rule_id)
            candidates: tuple[BuckRule, ...] = () if rule is None else (rule,)
        elif canonical_section is not None:
            candidates = self._rules_by_section.get(canonical_section, ())
        else:
            candidates = self.rules

        matched = (
            rule
            for rule in candidates
            if (canonical_section is None or rule.buck_section == canonical_section)
            and _rule_matches_filters(rule, category=category, dialect=dialect)
        )
        return tuple(sorted(matched, key=lambda rule: rule.id))

    def get_dialect(self, dialect_id: str) -> BuckDialect | None:
        """Return the dialect with *dialect_id*, if present."""
        return self._dialects_by_id.get(dialect_id)

    def list_dialects(
        self, *, kind: str | None = None, group: str | None = None
    ) -> tuple[BuckDialect, ...]:
        """Return dialects, optionally filtered by dialect kind and/or group."""
        return tuple(
            dialect
            for dialect in self.dialects
            if (kind is None or dialect.kind == kind)
            and (group is None or dialect.group == group)
        )

    def get_dialect_rules(
        self,
        dialect_id: str,
        *,
        include_inherited: bool = True,
    ) -> tuple[BuckRule, ...]:
        """Return rules assigned to *dialect_id*, optionally walking parents.

        This follows the dialect catalog's ``rules`` list (and, when
        ``include_inherited`` is true, the parent chain). It is an independent
        relationship from a rule's ``affected_dialects`` field used by
        :meth:`list_rules`; the two are not guaranteed to agree.
        """
        dialect = self.get_dialect(dialect_id)
        if dialect is None:
            return ()

        rule_ids: list[str] = []
        current: BuckDialect | None = dialect
        seen_dialects: set[str] = set()
        while current is not None and current.id not in seen_dialects:
            seen_dialects.add(current.id)
            rule_ids.extend(current.rules)
            if not include_inherited or current.parent is None:
                break
            current = self.get_dialect(current.parent)

        seen: set[str] = set()
        resolved: list[BuckRule] = []
        for rule_id in rule_ids:
            if rule_id in seen:
                continue
            seen.add(rule_id)
            rule = self.get_rule(rule_id)
            if rule is not None:
                resolved.append(rule)
        return tuple(resolved)

    def list_glossary_entries(
        self,
        *,
        dialect: str | None = None,
        rule_id: str | None = None,
    ) -> tuple[BuckGlossaryEntry, ...]:
        """Return glossary entries, optionally filtered by dialect and rule id."""
        return tuple(
            entry
            for entry in self.glossary_entries
            if _glossary_matches_filters(entry, dialect=dialect, rule_id=rule_id)
        )

    def find_glossary_entries(
        self,
        *,
        word: str | None = None,
        standard_form: str | None = None,
        dialect: str | None = None,
        rule_id: str | None = None,
    ) -> tuple[BuckGlossaryEntry, ...]:
        """Return glossary entries matching the given constraints, deterministically.

        Candidate selection uses the exact ``word`` index when provided, else the
        exact ``standard_form`` index, else every entry. When both ``word`` and
        ``standard_form`` are given, the candidate set is intersected with the
        ``standard_form`` matches so all exact constraints hold simultaneously.
        The ``dialect`` and ``rule_id`` filters then apply with the same
        semantics as :meth:`list_glossary_entries`. Results are sorted by
        ``(word, dialect, rule_id)`` so truncating callers see a stable order.
        """
        if word is not None:
            candidates: tuple[BuckGlossaryEntry, ...] = self.find_glossary_by_word(word)
            if standard_form is not None:
                allowed = set(self.find_glossary_by_standard_form(standard_form))
                candidates = tuple(
                    entry for entry in candidates if entry in allowed
                )
        elif standard_form is not None:
            candidates = self.find_glossary_by_standard_form(standard_form)
        else:
            candidates = self.glossary_entries

        matched = (
            entry
            for entry in candidates
            if _glossary_matches_filters(entry, dialect=dialect, rule_id=rule_id)
        )
        return tuple(
            sorted(
                matched,
                key=lambda entry: (entry.word, entry.dialect, entry.rule_id or ""),
            )
        )

    def find_glossary_by_word(self, word: str) -> tuple[BuckGlossaryEntry, ...]:
        """Return glossary entries whose word matches *word* after NFC normalization."""
        return self._glossary_by_word.get(_normalize_lookup_text(word), ())

    def find_glossary_by_standard_form(
        self,
        standard_form: str,
    ) -> tuple[BuckGlossaryEntry, ...]:
        """Return glossary entries matching *standard_form* after NFC normalization."""
        return self._glossary_by_standard_form.get(
            _normalize_lookup_text(standard_form),
            (),
        )


def _rule_matches_filters(
    rule: BuckRule,
    *,
    category: str | None,
    dialect: str | None,
) -> bool:
    """Return whether *rule* satisfies the category and affected-dialect filters.

    The ``dialect`` filter matches a rule's ``affected_dialects`` field, not the
    dialect catalog's ``rules`` list. This is the single definition of rule
    filtering semantics shared by :meth:`BuckReferenceIndex.list_rules` and
    :meth:`BuckReferenceIndex.find_rules`.
    """
    return (category is None or rule.category == category) and (
        dialect is None or dialect in rule.affected_dialects
    )


def _glossary_matches_filters(
    entry: BuckGlossaryEntry,
    *,
    dialect: str | None,
    rule_id: str | None,
) -> bool:
    """Return whether *entry* satisfies the dialect and rule-id filters.

    This is the single definition of glossary filtering semantics shared by
    :meth:`BuckReferenceIndex.list_glossary_entries` and
    :meth:`BuckReferenceIndex.find_glossary_entries`.
    """
    return (dialect is None or entry.dialect == dialect) and (
        rule_id is None or entry.rule_id == rule_id
    )


@lru_cache(maxsize=1)
def _build_buck_reference_index_cached() -> BuckReferenceIndex:
    return _build_index(load_buck_data())


def build_buck_reference_index() -> BuckReferenceIndex:
    """Return the read-only Buck reference index built from validated loader data.

    The index is deeply immutable (frozen dataclasses, tuples, and read-only
    mappings), so a single cached instance is shared across callers. Use
    :func:`clear_buck_reference_index_cache` after changing the underlying
    data location (e.g. trusted-directory overrides in tests).
    """
    return _build_buck_reference_index_cached()


def clear_buck_reference_index_cache() -> None:
    """Clear the cached Buck reference index.

    Tests and trusted-directory override callers should use this public helper
    (together with :func:`phonology.languages.ancient_greek.buck.clear_buck_data_cache`)
    instead of reaching into the private ``_build_buck_reference_index_cached``
    wrapper.
    """
    _build_buck_reference_index_cached.cache_clear()


def canonicalize_buck_section(section: str | int | float) -> str:
    """Return the canonical string representation for a Buck section value."""
    if isinstance(section, str):
        stripped = section.strip()
        if not stripped:
            raise ValueError("Buck section must not be empty")
        return _canonicalize_decimal_string(stripped)
    if isinstance(section, int):
        return str(section)
    if isinstance(section, float):
        if not math.isfinite(section):
            raise ValueError("Buck section must be a finite number")
        return _canonicalize_decimal_string(str(section))
    raise TypeError(f"Unsupported Buck section type: {type(section).__name__}")


def _build_index(data: BuckData) -> BuckReferenceIndex:
    metadata = _parse_metadata(data["grammar_rules"].get("meta", {}))
    rules = tuple(
        _parse_rule(raw_rule, metadata)
        for raw_rule in data["grammar_rules"].get("rules", [])
        if isinstance(raw_rule, dict)
    )
    dialects = tuple(
        _parse_dialect(raw_dialect, metadata)
        for raw_dialect in data["dialects"].get("dialects", [])
        if isinstance(raw_dialect, dict)
    )
    glossary_entries = tuple(
        _parse_glossary_entry(raw_entry, metadata)
        for raw_entry in data["glossary"].get("words", [])
        if isinstance(raw_entry, dict)
    )

    return BuckReferenceIndex(
        metadata=metadata,
        rules=rules,
        dialects=dialects,
        glossary_entries=glossary_entries,
        _rules_by_id=MappingProxyType({rule.id: rule for rule in rules}),
        _rules_by_section=_index_rules_by_section(rules),
        _dialects_by_id=MappingProxyType({dialect.id: dialect for dialect in dialects}),
        _glossary_by_word=_index_glossary_by_key(
            glossary_entries,
            lambda entry: entry.word,
        ),
        _glossary_by_standard_form=_index_glossary_by_key(
            glossary_entries,
            lambda entry: entry.standard_form,
        ),
    )


def _parse_metadata(raw_meta: object) -> BuckMetadata:
    if not isinstance(raw_meta, dict):
        raw_meta = {}
    raw_citation_ready = raw_meta.get("citation_ready", False)
    return BuckMetadata(
        status=_string_or_default(raw_meta.get("status"), "provisional"),
        review_status=BuckReviewStatus(
            _string_or_default(
                raw_meta.get("review_status"),
                "not_expert_reviewed",
            )
        ),
        citation_ready=(
            raw_citation_ready if isinstance(raw_citation_ready, bool) else False
        ),
    )


def _parse_rule(raw_rule: dict[str, Any], metadata: BuckMetadata) -> BuckRule:
    raw_section = raw_rule.get("buck_section")
    raw_transformation = raw_rule.get("transformation")
    raw_variants = raw_rule.get("variants") or []
    return BuckRule(
        id=str(raw_rule["id"]),
        buck_section=(
            canonicalize_buck_section(raw_section) if raw_section is not None else None
        ),
        category=_optional_string(raw_rule.get("category")),
        description=_optional_string(raw_rule.get("description")),
        transformation=(
            _freeze_mapping(raw_transformation)
            if isinstance(raw_transformation, dict)
            else MappingProxyType({})
        ),
        affected_dialects=_string_tuple(raw_rule.get("affected_dialects") or []),
        variants=tuple(
            _freeze_mapping(variant)
            for variant in raw_variants
            if isinstance(variant, dict)
        ),
        notes=_optional_string(raw_rule.get("notes")),
        status=metadata.status,
        review_status=metadata.review_status,
        citation_ready=metadata.citation_ready,
    )


def _parse_dialect(raw_dialect: dict[str, Any], metadata: BuckMetadata) -> BuckDialect:
    return BuckDialect(
        id=str(raw_dialect["id"]),
        name=_optional_string(raw_dialect.get("name")),
        kind=_optional_string(raw_dialect.get("kind")),
        group=_optional_string(raw_dialect.get("group")),
        parent=_optional_string(raw_dialect.get("parent")),
        rules=_string_tuple(raw_dialect.get("rules") or []),
        status=metadata.status,
        review_status=metadata.review_status,
        citation_ready=metadata.citation_ready,
    )


def _parse_glossary_entry(
    raw_entry: dict[str, Any],
    metadata: BuckMetadata,
) -> BuckGlossaryEntry:
    return BuckGlossaryEntry(
        word=str(raw_entry["word"]),
        standard_form=_optional_string(raw_entry.get("standard_form")),
        dialect=str(raw_entry["dialect"]),
        rule_id=_optional_string(raw_entry.get("rule_id")),
        definition=_optional_string(raw_entry.get("definition")),
        inscription_no=_inscription_number(raw_entry.get("inscription_no")),
        buck_ref=_parse_reference(raw_entry.get("buck_ref")),
        notes=_optional_string(raw_entry.get("notes")),
        status=metadata.status,
        review_status=metadata.review_status,
        citation_ready=metadata.citation_ready,
    )


def _parse_reference(raw_reference: object) -> BuckReference | None:
    if not isinstance(raw_reference, dict):
        return None
    raw_section = raw_reference.get("section")
    raw_page = raw_reference.get("page")
    return BuckReference(
        section=(
            canonicalize_buck_section(raw_section) if raw_section is not None else None
        ),
        page=raw_page if isinstance(raw_page, int) else None,
    )


def _index_rules_by_section(
    rules: tuple[BuckRule, ...],
) -> Mapping[str, tuple[BuckRule, ...]]:
    grouped: dict[str, list[BuckRule]] = {}
    for rule in rules:
        if rule.buck_section is None:
            continue
        grouped.setdefault(rule.buck_section, []).append(rule)
    return MappingProxyType(
        {section: tuple(section_rules) for section, section_rules in grouped.items()}
    )


def _index_glossary_by_key(
    entries: tuple[BuckGlossaryEntry, ...],
    key_fn: Callable[[BuckGlossaryEntry], str | None],
) -> Mapping[str, tuple[BuckGlossaryEntry, ...]]:
    grouped: dict[str, list[BuckGlossaryEntry]] = {}
    for entry in entries:
        raw_key = key_fn(entry)
        if raw_key is None:
            continue
        grouped.setdefault(_normalize_lookup_text(raw_key), []).append(entry)
    return MappingProxyType(
        {key: tuple(key_entries) for key, key_entries in grouped.items()}
    )


def _freeze_mapping(raw_mapping: Mapping[str, Any]) -> _ReadOnlyMapping:
    return MappingProxyType(
        {str(key): _freeze_value(value) for key, value in raw_mapping.items()}
    )


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, list | tuple):
        return tuple(_freeze_value(item) for item in value)
    return value


def _string_tuple(raw_values: object) -> tuple[str, ...]:
    if not isinstance(raw_values, list | tuple):
        return ()
    return tuple(value for value in raw_values if isinstance(value, str))


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _inscription_number(value: object) -> _InscriptionNumber | None:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        return tuple(value)
    return None


def _string_or_default(value: object, default: str) -> str:
    return value if isinstance(value, str) and value else default


def _normalize_lookup_text(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def _canonicalize_decimal_string(value: str) -> str:
    try:
        decimal = Decimal(value)
    except InvalidOperation:
        return value
    normalized = decimal.normalize()
    if normalized == normalized.to_integral():
        return str(normalized.quantize(Decimal(1)))
    return format(normalized, "f")
