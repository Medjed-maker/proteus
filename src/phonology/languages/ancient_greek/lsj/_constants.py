"""Static maps, regex patterns, and frozensets shared across LSJ helpers."""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# POS mapping: LSJ abbreviation → schema enum
# ---------------------------------------------------------------------------

_POS_MAP: dict[str, str] = {
    "Adj.": "adjective",
    "adj.": "adjective",
    "Adv.": "adverb",
    "adv.": "adverb",
    "Art.": "article",
    "art.": "article",
    "Article": "article",
    "article": "article",
    "Subst.": "noun",
    "subst.": "noun",
    "Prep.": "preposition",
    "prep.": "preposition",
    "Preposition": "preposition",
    "preposition": "preposition",
    "Conj.": "conjunction",
    "conj.": "conjunction",
    "Conjunction": "conjunction",
    "conjunction": "conjunction",
    "Part.": "particle",
    "part.": "particle",
    "Particle": "particle",
    "particle": "particle",
    "Interj.": "interjection",
    "interj.": "interjection",
    "Interjection": "interjection",
    "interjection": "interjection",
    "Num.": "numeral",
    "num.": "numeral",
    "Numeral": "numeral",
    "numeral": "numeral",
    "Pron.": "pronoun",
    "pron.": "pronoun",
    "Pronoun": "pronoun",
    "pronoun": "pronoun",
    "Partic.": "participle",
    "partic.": "participle",
    "Participle": "participle",
    "participle": "participle",
    "Verb": "verb",
    "v.": "verb",
}

# Gender article Beta Code → enum
_GENDER_MAP: dict[str, str] = {
    "o(": "masculine",
    "h(": "feminine",
    "to/": "neuter",
}


# Dialect abbreviation → Proteus dialect label
_DIALECT_MAP: dict[str, str] = {
    "Att.": "attic",
    "Ion.": "ionic",
    "Dor.": "doric",
    "Aeol.": "aeolic",
    "Ep.": "ionic",  # Epic Greek is Ionic-based and pre-Attic; kept as ionic so the
    # Attic-only filter rejects these entries rather than silently including them.
    "Lacon.": "doric",  # Laconian is a Doric sub-dialect
}

# POS values that require gender per the schema
_GENDER_REQUIRED_POS = frozenset(
    {"noun", "adjective", "pronoun", "article", "numeral", "participle"}
)


_BETA_REMOVE_RE = re.compile(r"[()/=\\+|'*]")
_KEY_HOMONYM_SUFFIX_RE = re.compile(r"\d+$")
_INTRO_WHITESPACE_RE = re.compile(r"\s+")
# ORDER MATTERS: patterns are iterated in tuple order.  Although the consumer
# (_inline_pos_candidates) sorts by match position, tuple order acts as a
# tiebreaker when two patterns could match at the same offset.  In particular,
# "particle" (Part.) must precede "participle" (Partic.) -- the negative
# lookahead Part\.(?!\w) is the primary guard, but tuple order provides a
# secondary safety net.
_INLINE_POS_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("article", re.compile(r"\b(?:Art\.|Article)(?=$|[.,;:]|\s+of\b)", re.IGNORECASE)),
    ("pronoun", re.compile(r"\b(?:Pron\.|Pronoun)(?=$|[.,;:]|\s+of\b)", re.IGNORECASE)),
    (
        "preposition",
        re.compile(
            r"\b(?:Prep\.|Preposition)(?=$|[.,;:]|\s+(?:with|gen\.|dat\.|acc\.|c\.)\b)",
            re.IGNORECASE,
        ),
    ),
    ("conjunction", re.compile(r"\b(?:Conj\.|Conjunction)(?=$|[.,;:])", re.IGNORECASE)),
    ("particle", re.compile(r"\b(?:Part\.(?!\w)|Particle)(?=$|[.,;:])", re.IGNORECASE)),
    (
        "interjection",
        re.compile(r"\b(?:Interj\.|Interjection)(?=$|[.,;:])", re.IGNORECASE),
    ),
    ("numeral", re.compile(r"\b(?:Num\.|Numeral)(?=$|[.,;:])", re.IGNORECASE)),
    (
        "participle",
        re.compile(r"\b(?:Partic\.|Participle)(?=$|[.,;:\s])", re.IGNORECASE),
    ),
)
_ATTIC_INLINE_RE = re.compile(r"\bAtt\.|\bAttic\b", re.IGNORECASE)
_HEADING_NEARBY_VARIANT_CONTEXT_RE = re.compile(
    r"(?:\b(?:so in|mostly|rarely|before vowels|before consonants)\b|\bstrengthd\.)",
    re.IGNORECASE,
)
_HEADING_ENTRY_LEVEL_CONSTRAINT_RE = re.compile(
    r"\b(?:form of|only in)\b",
    re.IGNORECASE,
)
_HEADING_LEXICOGRAPHIC_CONTEXT_RE = re.compile(
    r"\bolder\s+form\b",
    re.IGNORECASE,
)
_POST_GLOSS_PRIMARY_POS_RE = re.compile(
    r"^\s*[:\-—]*\s*(?:Art\.|Article|Pron\.|Pronoun|Partic\.|Participle)\s+of\b",
    re.IGNORECASE,
)
_PARTICIPLE_OF_RE = re.compile(
    r"\b(?:part\.|partic\.|participle)\s+of\b",
    re.IGNORECASE,
)
_HEADING_GREEK_FORM_TAGS = frozenset({"orth", "foreign", "pron", "gen"})
_HEADING_SURFACE_FORM_TAGS = frozenset({"orth", "foreign", "pron"})
_HEADING_HEADWORD_CONTEXT_TAGS = frozenset({"pos"})
_HEADING_NONPROSE_CONTEXT_TAGS = frozenset(
    {
        "author",
        "bibl",
        "biblScope",
        "cit",
        "foreign",
        "gen",
        "gramGrp",
        "itype",
        "mood",
        "orth",
        "pron",
        "quote",
        "tns",
    }
)
