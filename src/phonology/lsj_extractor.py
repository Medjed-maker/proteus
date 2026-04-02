"""Extract Greek lemma data from Perseus LSJ XML files.

Parses the TEI P4 XML files from the PerseusDL/lexica repository,
extracts headwords, parts of speech, glosses, and other metadata,
then writes a JSON lexicon file conforming to the Proteus schema.

Usage::

    python -m phonology.lsj_extractor --xml-dir data/external/lsj/CTS_XML_TEI/...
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._paths import resolve_repo_data_dir
from .betacode import beta_to_unicode
from .ipa_converter import to_ipa
from .transliterate import transliterate

logger = logging.getLogger(__name__)

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

# Cached POS override data, loaded lazily from data/lexicon/pos_overrides.yaml.
_pos_overrides: dict[str, frozenset[str]] | None = None


def _empty_pos_overrides() -> dict[str, frozenset[str]]:
    """Return the canonical empty POS override mapping."""
    return {
        "common_gender_keys": frozenset(),
        "numeral_keys": frozenset(),
    }


def _load_pos_overrides(*, cli_mode: bool = False) -> dict[str, frozenset[str]]:
    """Load POS override keys from ``data/lexicon/pos_overrides.yaml`` (cached).

    Args:
        cli_mode: When True, exceptions from missing or malformed override files are
            re-raised after logging; when False (default), errors are logged and the
            function returns empty overrides. In CLI mode this ensures extraction fails
            fast on configuration problems.

    Returns:
        A dictionary mapping lemma keys to frozensets of POS tags.

    Side Effects:
        Caches the result in the global ``_pos_overrides`` variable.
    """
    global _pos_overrides  # noqa: PLW0603
    if _pos_overrides is not None:
        return _pos_overrides
    import yaml  # type: ignore[import-untyped]

    raw: dict[str, object] = {}
    overrides_path: Path | str = "<unresolved>"
    try:
        overrides_path = resolve_repo_data_dir("lexicon") / "pos_overrides.yaml"
        loaded = yaml.safe_load(overrides_path.read_text(encoding="utf-8"))
    except FileNotFoundError as err:
        logger.error("POS overrides file not found or lexicon data dir missing: %s", err)
        if cli_mode:
            raise
    except (OSError, UnicodeError) as err:
        logger.error(
            "Failed to read POS overrides YAML at %s; using empty overrides: %s",
            overrides_path,
            err,
        )
        if cli_mode:
            raise
    except yaml.YAMLError as err:
        logger.error(
            "Failed to parse POS overrides YAML at %s; using empty overrides: %s",
            overrides_path,
            err,
        )
        if cli_mode:
            raise
    else:
        if isinstance(loaded, dict):
            raw = loaded

    def _ensure_list(val: object) -> list[str]:
        if isinstance(val, list):
            non_strings = [v for v in val if not isinstance(v, str)]
            if non_strings:
                logger.warning(
                    "Ignoring non-string values in POS overrides list: %s",
                    non_strings,
                )
            return [str(v) for v in val if isinstance(v, str)]
        return []

    _pos_overrides = _empty_pos_overrides()
    if raw:
        _pos_overrides = {
            "common_gender_keys": frozenset(_ensure_list(raw.get("common_gender_keys", []))),
            "numeral_keys": frozenset(_ensure_list(raw.get("numeral_keys", []))),
        }
    return _pos_overrides

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


# ---------------------------------------------------------------------------
# XML element text helpers
# ---------------------------------------------------------------------------

def _elem_text(element: Any) -> str:
    """Return the full text content of an element (including tail of children)."""
    return "".join(element.itertext()).strip() if element is not None else ""


def _find_text(parent: Any, tag: str, **attribs: str) -> str:
    """Find the first child matching tag and optional attributes, return text."""
    for child in parent:
        local = _local_name(child)
        if local != tag:
            continue
        if all(child.get(k) == v for k, v in attribs.items()):
            return _elem_text(child)
    return ""


def _find_texts(parent: Any, tag: str, **attribs: str) -> list[str]:
    """Find all direct children matching tag and optional attributes, return texts."""
    texts: list[str] = []
    for child in parent:
        local = _local_name(child)
        if local != tag:
            continue
        if all(child.get(k) == v for k, v in attribs.items()):
            text = _elem_text(child)
            if text:
                texts.append(text)
    return texts


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
    ("interjection", re.compile(r"\b(?:Interj\.|Interjection)(?=$|[.,;:])", re.IGNORECASE)),
    ("numeral", re.compile(r"\b(?:Num\.|Numeral)(?=$|[.,;:])", re.IGNORECASE)),
    ("participle", re.compile(r"\b(?:Partic\.|Participle)(?=$|[.,;:]|\s+of\b)", re.IGNORECASE)),
)
_ATTIC_INLINE_RE = re.compile(r"\bAtt\.|\bAttic\b", re.IGNORECASE)
_PARTICIPLE_OF_RE = re.compile(r"\b(?:part|partic)\.\s+of\b", re.IGNORECASE)
_HEADING_GREEK_FORM_TAGS = frozenset({"orth", "foreign", "pron", "gen"})


def _normalize_beta_token(text: str) -> str:
    """Normalize a short Beta Code token used in iType/gender inference."""
    token = text.strip(" \t\n\r,;:.()[]").lower()
    return _BETA_REMOVE_RE.sub("", token)


def _normalize_headword_key(key: str) -> str:
    """Strip LSJ homonym numbering from a key before Beta Code conversion."""
    return _KEY_HOMONYM_SUFFIX_RE.sub("", key)


def _headword_itypes(entry: Any) -> list[str]:
    """Return iTypes attached to the first direct Greek headword block only."""
    itypes: list[str] = []
    in_headword_block = False
    for child in entry:
        local = _local_name(child)
        if not in_headword_block:
            if local == "orth" and child.get("lang") == "greek":
                in_headword_block = True
            continue
        if local == "sense":
            break
        if local == "orth" and child.get("lang") == "greek":
            break
        if local == "itype":
            text = _elem_text(child)
            if text:
                itypes.append(text)
    return itypes


def _has_descendant(entry: Any, tags: set[str]) -> bool:
    """Return True when any descendant tag matches one of ``tags``."""
    for descendant in entry.iter():
        local = _local_name(descendant)
        if local in tags:
            return True
    return False


def _looks_like_verb_key(key: str) -> bool:
    """Heuristic for LSJ headwords that are dictionary verb forms."""
    cleaned = _normalize_headword_key(key).rstrip(":")
    # The -w ending alone would over-match (e.g. some adverbs); callers
    # guard this with _has_descendant(entry, {"tns", "mood"}) to require
    # morphological evidence of verbal inflection.
    return cleaned.endswith("w") or cleaned.endswith("mai") or cleaned.endswith("mi")


def _looks_like_adjective_itypes(itypes: list[str], key: str = "") -> bool:
    """Infer adjective POS from ``itypes`` normalized by ``_normalize_beta_token``.

    Detects Greek adjective patterns such as ``-ος/-η/-ον`` or ``-ος/-α/-ον``
    via ``on`` with ``h`` or ``a``, two-termination ``-ος/-ον`` adjectives via
    ``on`` on an ``-ος`` headword, and third-declension ``-ης/-ες`` via ``es``.
    """
    normalized = {_normalize_beta_token(value) for value in itypes if value.strip()}
    normalized_key = _normalize_beta_token(_normalize_headword_key(key))
    has_two_termination_pattern = "on" in normalized and normalized_key.endswith("os")
    return (
        ("on" in normalized and ("a" in normalized or "h" in normalized))
        or has_two_termination_pattern
        or "es" in normalized
    )


def _looks_like_known_numeral_key(key: str) -> bool:
    """Return True for common Attic cardinal numeral headwords."""
    normalized = _normalize_headword_key(key).replace("^", "").replace("_", "")
    return normalized in _load_pos_overrides()["numeral_keys"]


def _local_name(element: Any) -> str:
    """Return the local XML tag name for ``element``."""
    return element.tag.split("}")[-1] if "}" in element.tag else element.tag


def _normalize_intro_text(parts: list[str]) -> str:
    """Collapse whitespace and trim helper text gathered from XML intro blocks."""
    return _INTRO_WHITESPACE_RE.sub(" ", "".join(parts)).strip()


def _entry_intro_text(entry: Any) -> str:
    """Return the text that appears before the first top-level ``<sense>``."""
    parts: list[str] = [entry.text or ""]
    for child in entry:
        local = _local_name(child)
        if local == "sense":
            break
        if local in {"cit", "quote", "bibl"}:
            parts.append(child.tail or "")
            continue
        parts.append(_elem_text(child))
        parts.append(child.tail or "")
    return _normalize_intro_text(parts)


def _sense_intro_texts(entry: Any, *, limit: int = 3) -> list[str]:
    """Return introductory text for the first few top-level ``<sense>`` blocks.

    Only the first *limit* senses are scanned because entry-level POS labels
    almost always appear in the opening senses; deeply nested sub-senses
    rarely contribute useful part-of-speech information.
    """
    texts: list[str] = []
    for child in entry:
        if _local_name(child) != "sense":
            continue
        if child.get("level") not in {None, "1"}:
            continue
        parts: list[str] = [child.text or ""]
        for sub in child:
            sub_local = _local_name(sub)
            if sub_local in {"tr", "sense"}:
                break
            if sub_local in {"cit", "quote", "bibl"}:
                parts.append(sub.tail or "")
                continue
            parts.append(_elem_text(sub))
            parts.append(sub.tail or "")
        text = _normalize_intro_text(parts)
        if text:
            texts.append(text)
        if len(texts) >= limit:
            break
    return texts


def _inline_pos_candidates(text: str) -> list[tuple[bool, int, str]]:
    """Return inline POS candidates as ``(attic_priority, index, pos)`` tuples."""
    candidates: list[tuple[bool, int, str]] = []
    for pos, pattern in _INLINE_POS_PATTERNS:
        for match in pattern.finditer(text):
            attic_priority = _has_attic_inline_context(text, match.start())
            candidates.append((attic_priority, match.start(), pos))
    return sorted(candidates, key=lambda item: item[1])


def _has_attic_inline_context(text: str, match_start: int) -> bool:
    """Return True when the match belongs to a clause marked as Attic."""
    clause_start = 0
    for delimiter in (";", ":"):
        clause_start = max(clause_start, text.rfind(delimiter, 0, match_start) + 1)
    return bool(_ATTIC_INLINE_RE.search(text[clause_start:match_start]))


def _is_heading_greek_form(child: Any) -> bool:
    """Return True when a heading child presents an alternate Greek form."""
    return (
        _local_name(child) in _HEADING_GREEK_FORM_TAGS
        and child.get("lang") == "greek"
        and bool(_elem_text(child))
    )


def _has_following_heading_greek_form(children: list[Any], start_index: int) -> bool:
    """Return True when a heading dialect label is followed by an alternate form."""
    for child in children[start_index + 1 :]:
        local = _local_name(child)
        if local == "sense":
            return False
        if local == "gramGrp":
            return False
        if _is_heading_greek_form(child):
            return True
    return False


def _leading_dialect_labels(entry: Any) -> list[str]:
    """Return dialect labels attached before the first top-level ``<sense>``."""
    labels: list[str] = []
    children = list(entry)
    for index, child in enumerate(children):
        local = _local_name(child)
        if local == "sense":
            break
        if local == "gramGrp":
            variant_only = _has_following_heading_greek_form(children, index)
            for descendant in child.iter():
                if _local_name(descendant) != "gram":
                    continue
                if descendant.get("type", "") != "dialect":
                    continue
                if variant_only:
                    continue
                text = _elem_text(descendant).strip()
                if text in _DIALECT_MAP:
                    labels.append(_DIALECT_MAP[text])
    return labels


def _has_plural_neuter_article(entry: Any) -> bool:
    """Return True when the entry heading marks a neuter plural article."""
    gen_text = _find_text(entry, "gen", lang="greek")
    if not gen_text:
        gen_text = _find_text(entry, "gen")
    normalized = unicodedata.normalize("NFC", gen_text.strip())
    return normalized in {"ta/", "τὰ", "τά"}


def _has_multiple_intro_greek_forms(entry: Any) -> bool:
    """Return True when the entry intro contains multiple Greek form markers."""
    count = 0
    for child in entry:
        local = _local_name(child)
        if local == "sense":
            break
        if child.get("lang") == "greek" and local in _HEADING_GREEK_FORM_TAGS:
            if _elem_text(child):
                count += 1
        if count > 1:
            return True
    return False


# ---------------------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------------------

def _extract_headword(entry: Any) -> str:
    """Extract the headword from the entry, trying multiple sources.

    Attempts conversion in order: normalized entry key,
    <orth extent='full' lang='greek'>, then any <orth lang='greek'>. Returns empty
    string if all conversions fail.
    """
    candidates = [
        _normalize_headword_key(entry.get("key", "")),
        _find_text(entry, "orth", extent="full", lang="greek"),
        _find_text(entry, "orth", lang="greek"),
    ]
    seen: set[str] = set()
    for beta in candidates:
        if not beta or beta in seen:
            continue
        seen.add(beta)
        # Strip quantity markers (^ for breve, _ for macron) before conversion
        cleaned = beta.replace("^", "").replace("_", "")
        try:
            return beta_to_unicode(cleaned)
        except ValueError as exc:
            logger.warning("Beta Code conversion failed for %r: %s", cleaned, exc)
    return ""


def _extract_pos(entry: Any) -> str | None:
    """Infer the part of speech from the entry. Returns None if undetermined."""
    itypes = _headword_itypes(entry)
    key = entry.get("key", "")
    adjective_itypes = _looks_like_adjective_itypes(itypes, key)
    normalized_key = _normalize_headword_key(key)
    participial_candidate = False

    # 1. Explicit <pos> tag
    pos_text = _find_text(entry, "pos")
    if pos_text:
        pos_text = pos_text.strip().rstrip(".")
        # Try exact match first, then with trailing period
        for candidate in (pos_text, pos_text + "."):
            if candidate in _POS_MAP:
                return _POS_MAP[candidate]

    # 2. Participial headwords are often introduced as "part. of <verb>" via <mood>.
    entry_intro = _entry_intro_text(entry)
    if _PARTICIPLE_OF_RE.search(entry_intro):
        participial_candidate = True

    # 3. Inline LSJ prose labels in the entry intro and first few sense headings.
    inline_candidates: list[tuple[int, int, int, str]] = []
    for source_index, text in enumerate([entry_intro, *_sense_intro_texts(entry)]):
        for attic_priority, position, pos in _inline_pos_candidates(text):
            # 0 = Attic-priority (sorts first), 1 = non-Attic
            sort_priority = 0 if attic_priority else 1
            inline_candidates.append((sort_priority, source_index, position, pos))
    if inline_candidates:
        # Sort by priority, then source order, then position in text
        inline_candidates.sort()
        return inline_candidates[0][3]

    # 4. Common cardinal numeral headwords are unlabeled in LSJ headings.
    if key and _looks_like_known_numeral_key(key):
        return "numeral"

    # 5. Gender-based inference: presence of <gen> → noun
    gen_text = _find_text(entry, "gen", lang="greek")
    if not gen_text:
        gen_text = _find_text(entry, "gen")
    if gen_text:
        if participial_candidate and _has_plural_neuter_article(entry):
            return "participle"
        return "noun"

    # 6. Headword ending heuristics
    if (
        normalized_key.endswith("ws")
        or normalized_key.endswith("w=s")
        or normalized_key.endswith("ei/")
    ):
        return "adverb"

    # 7. Multi-form iTypes like "a, on" usually mark adjectives.
    if adjective_itypes:
        return "adjective"

    # 8. Verb indicators only help when the headword itself looks verbal.
    if key and _looks_like_verb_key(key) and _has_descendant(entry, {"tns", "mood"}):
        return "verb"

    # 9. Fall back to participial classification only when stronger signals failed.
    if participial_candidate:
        return "participle"

    return None


def _extract_gender(entry: Any) -> str | None:
    """Extract gender from <gen> element."""
    gen_text = _find_text(entry, "gen", lang="greek")
    if not gen_text:
        # Try without lang attribute
        gen_text = _find_text(entry, "gen")
    if gen_text:
        gen_stripped = unicodedata.normalize("NFC", gen_text.strip())
        for code, gender in _GENDER_MAP.items():
            if gen_stripped.startswith(code):
                return gender
        # Fallback: _GENDER_MAP uses Beta Code prefixes (startswith), but some
        # LSJ entries already contain pre-converted Unicode articles.
        if gen_stripped == "ὁ":
            return "masculine"
        if gen_stripped == "ἡ":
            return "feminine"
        if gen_stripped in ("τό", "τὸ"):
            return "neuter"
    return None


def _extract_gloss(entry: Any) -> str:
    """Extract the English gloss from <tr> elements in the first sense."""
    glosses: list[str] = []

    # Try to find <tr> elements within the first <sense>
    for child in entry:
        local = _local_name(child)
        if local == "sense":
            for sub in child.iter():
                sub_local = _local_name(sub)
                if sub_local == "tr":
                    text = _elem_text(sub).strip().strip(",").strip()
                    if text:
                        glosses.append(text)
            if glosses:
                break  # Only take glosses from the first sense

    # Fallback: <tr> directly under entry (some entries skip <sense>)
    if not glosses:
        for child in entry:
            local = _local_name(child)
            if local == "tr":
                text = _elem_text(child).strip().strip(",").strip()
                if text:
                    glosses.append(text)

    combined = ", ".join(glosses)
    if len(combined) > 200:
        combined = combined[:197] + "..."
    return combined


def _extract_dialect(entry: Any) -> str:
    """Extract entry-level dialect from the heading before the first sense."""
    labels = set(_leading_dialect_labels(entry))
    if "attic" in labels:
        return "attic"
    if labels:
        return next(iter(labels))
    return "attic"


# ---------------------------------------------------------------------------
# Entry processing
# ---------------------------------------------------------------------------

def extract_entry(entry_elem: Any) -> dict[str, Any] | None:
    """Extract a single lemma dict from an <entryFree> element.

    Returns None if the entry cannot produce a valid lemma (missing
    headword, gloss, or undetermined POS).
    """
    # Only process main entries
    entry_type = entry_elem.get("type", "")
    if entry_type and entry_type != "main":
        return None

    # Extract ID
    xml_id = entry_elem.get("id", "")
    if not xml_id:
        return None
    numeric = xml_id.lstrip("n")
    if not numeric.isdigit():
        return None
    lsj_id = f"LSJ-{int(numeric):06d}"
    key = entry_elem.get("key", "")

    # Extract headword
    headword = _extract_headword(entry_elem)
    if not headword:
        return None

    stripped = headword.strip()

    # Extract gloss
    gloss = _extract_gloss(entry_elem)
    if not gloss:
        return None

    # Extract POS
    pos = _extract_pos(entry_elem)
    if pos is None:
        return None

    # Skip entries that are purely numeric, punctuation, or isolated single letters,
    # except for genuine closed-class headwords such as ὁ.
    if len(stripped) <= 1 and pos not in {"article", "pronoun", "interjection"}:
        return None

    # Extract dialect before IPA generation so the output stays Attic-only.
    dialect = _extract_dialect(entry_elem)
    if dialect != "attic":
        logger.info(
            "Skipping non-Attic entry %s (%s): dialect=%s",
            lsj_id,
            headword,
            dialect,
        )
        return None

    # Extract gender
    gender = _extract_gender(entry_elem)
    has_multiple_intro_forms = _has_multiple_intro_greek_forms(entry_elem)
    if pos in {"pronoun", "article", "numeral"} and has_multiple_intro_forms:
        gender = "common"
    elif pos == "noun" and _normalize_headword_key(key) in _load_pos_overrides()["common_gender_keys"]:
        gender = "common"
    elif pos in _GENDER_REQUIRED_POS and gender is None:
        gender = "common"  # default for undetermined gender

    # Generate transliteration
    translit = transliterate(headword)
    if not translit:
        return None

    # Generate IPA
    try:
        ipa = to_ipa(headword, dialect=dialect)
    except (ValueError, KeyError, NotImplementedError) as e:
        logger.info(
            "IPA conversion failed for %s (%s): %s: %s",
            lsj_id,
            headword,
            type(e).__name__,
            e,
        )
        return None
    if not ipa:
        return None

    result: dict[str, Any] = {
        "id": lsj_id,
        "headword": headword,
        "transliteration": translit,
        "ipa": ipa,
        "pos": pos,
        "gloss": gloss,
        "dialect": dialect,
    }
    if gender is not None:
        result["gender"] = gender

    return result


# ---------------------------------------------------------------------------
# XML iteration
# ---------------------------------------------------------------------------

def iter_xml_entries(xml_path: Path) -> Iterator[dict[str, Any]]:
    """Yield lemma dicts from a single LSJ XML file using streaming parse."""
    from lxml import etree  # type: ignore[import-untyped]

    context = etree.iterparse(
        str(xml_path), events=("end",), tag="entryFree", recover=True
    )
    for _event, element in context:
        entry = extract_entry(element)
        if entry is not None:
            yield entry
        # Free memory: remove all previously processed siblings.
        # The current element stays in the tree until the next iteration.
        element.clear()
        parent = element.getparent()
        while parent is not None and element.getprevious() is not None:
            del parent[0]


def find_xml_files(xml_dir: Path) -> list[Path]:
    """Find and sort all LSJ XML files in the given directory."""
    files = sorted(
        xml_dir.glob("grc.lsj.perseus-eng*.xml"),
        key=lambda p: int("".join(filter(str.isdigit, p.stem.split("eng")[-1])) or "0"),
    )
    if not files:
        raise FileNotFoundError(
            f"No LSJ XML files found in {xml_dir} "
            f"(expected grc.lsj.perseus-eng*.xml)"
        )
    return files


def extract_all(
    xml_dir: Path, *, limit: int | None = None
) -> Iterator[dict[str, Any]]:
    """Yield lemma dicts from all LSJ XML files in order."""
    files = find_xml_files(xml_dir)
    count = 0
    seen_ids: set[str] = set()

    for xml_file in files:
        logger.info("Processing %s", xml_file.name)
        for entry in iter_xml_entries(xml_file):
            # Deduplicate by ID
            if entry["id"] in seen_ids:
                continue
            seen_ids.add(entry["id"])

            yield entry
            count += 1
            if limit is not None and count >= limit:
                return


# ---------------------------------------------------------------------------
# Document building
# ---------------------------------------------------------------------------

def _document_dialect(entries: list[dict[str, Any]]) -> str:
    """Return the single dialect represented in the extracted output."""
    dialects = {
        str(entry["dialect"]).strip()
        for entry in entries
        if isinstance(entry.get("dialect"), str) and entry["dialect"].strip()
    }
    if not dialects:
        return "attic"
    if len(dialects) == 1:
        return next(iter(dialects))
    raise ValueError(
        "build_lexicon_document expected a single output dialect, "
        f"got {sorted(dialects)!r}"
    )


def _document_dialect_label(dialect: str) -> str:
    """Return a human-readable dialect label for metadata text."""
    return dialect.capitalize()


def build_lexicon_document(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the complete lexicon JSON document with metadata."""
    dialect = _document_dialect(entries)
    dialect_label = _document_dialect_label(dialect)
    return {
        "schema_version": "2.0.0",
        "_meta": {
            "source": "LSJ (Liddell-Scott-Jones, A Greek-English Lexicon, 9th ed.)",
            "encoding": "Unicode NFC",
            "ipa_system": "scholarly Ancient Greek IPA",
            "dialect": dialect,
            "version": "2.0.0",
            "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "license": "CC-BY-SA 4.0",
            "contributors": [
                "Perseus Digital Library, Tufts University",
                "Proteus maintainers",
            ],
            "data_schema_ref": "data/lexicon/greek_lemmas.schema.json",
            "description": (
                "Ancient Greek lemma dataset extracted from the Perseus Digital "
                f"Library LSJ XML, filtered to {dialect_label} entries with "
                f"{dialect_label} IPA and scholarly transliterations."
            ),
            "note": (
                "Extracted via scripts/extract-lsj.sh from PerseusDL/lexica "
                f"CTS_XML_TEI; output dialect is {dialect}"
            ),
        },
        "lemmas": entries,
    }


def validate_document(document: dict[str, Any], schema_path: Path | None = None) -> None:
    """Validate the lexicon document against the JSON schema."""
    import jsonschema

    if schema_path is None:
        schema_path = resolve_repo_data_dir("lexicon") / "greek_lemmas.schema.json"

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator_cls = jsonschema.Draft202012Validator
    validator = validator_cls(schema, format_checker=jsonschema.FormatChecker())

    errors = list(validator.iter_errors(document))
    if errors:
        for err in errors[:10]:
            logger.error("Schema validation error: %s at %s", err.message, list(err.absolute_path))
        raise ValueError(f"Schema validation failed with {len(errors)} error(s)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(
    xml_dir: Path | None = None,
    output_path: Path | None = None,
    *,
    limit: int | None = None,
    dry_run: bool = False,
) -> int:
    """Extract LSJ entries and write the Proteus lexicon JSON.

    Returns 0 on success, 1 on failure.
    """
    if output_path is None:
        output_path = resolve_repo_data_dir("lexicon") / "greek_lemmas.json"

    if xml_dir is None:
        raise ValueError(
            "xml_dir is required. Pass --xml-dir or set it programmatically."
        )

    logger.info("Extracting from %s", xml_dir)
    entries = list(extract_all(xml_dir, limit=limit))
    logger.info("Extracted %d entries", len(entries))

    if not entries:
        logger.error("No entries extracted — aborting")
        return 1

    # Count stats
    pos_counts: dict[str, int] = {}
    for entry in entries:
        pos = entry["pos"]
        pos_counts[pos] = pos_counts.get(pos, 0) + 1

    document = build_lexicon_document(entries)

    try:
        validate_document(document)
        logger.info("Schema validation passed")
    except ValueError as exc:
        logger.error("Validation failed: %s", exc)
        return 1

    if dry_run:
        print(f"Dry run: {len(entries)} entries would be written to {output_path}")
        for pos, count in sorted(pos_counts.items()):
            print(f"  {pos}: {count}")
        return 0

    output_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Lexicon written: {len(entries)} entries to {output_path}")
    for pos, count in sorted(pos_counts.items()):
        print(f"  {pos}: {count}")
    return 0


def run_cli() -> int:
    """Parse arguments and run extraction."""
    parser = argparse.ArgumentParser(
        description="Extract Greek lemma data from Perseus LSJ XML files.",
    )
    parser.add_argument(
        "--xml-dir",
        type=Path,
        required=True,
        help="Directory containing grc.lsj.perseus-eng*.xml files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: data/lexicon/greek_lemmas.json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N entries (for development)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate without writing output",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        _load_pos_overrides(cli_mode=True)
        return main(
            xml_dir=args.xml_dir,
            output_path=args.output,
            limit=args.limit,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        logger.error("Extraction failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(run_cli())
