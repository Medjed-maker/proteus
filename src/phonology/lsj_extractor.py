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
    "Subst.": "noun",
    "subst.": "noun",
    "Prep.": "preposition",
    "prep.": "preposition",
    "Conj.": "conjunction",
    "conj.": "conjunction",
    "Part.": "particle",
    "part.": "particle",
    "Interj.": "interjection",
    "interj.": "interjection",
    "Num.": "numeral",
    "num.": "numeral",
    "Pron.": "pronoun",
    "pron.": "pronoun",
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


# ---------------------------------------------------------------------------
# XML element text helpers
# ---------------------------------------------------------------------------

def _elem_text(element: Any) -> str:
    """Return the full text content of an element (including tail of children)."""
    return "".join(element.itertext()).strip() if element is not None else ""


def _find_text(parent: Any, tag: str, **attribs: str) -> str:
    """Find the first child matching tag and optional attributes, return text."""
    for child in parent:
        local = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if local != tag:
            continue
        if all(child.get(k) == v for k, v in attribs.items()):
            return _elem_text(child)
    return ""


# ---------------------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------------------

def _extract_headword(entry: Any) -> str:
    """Extract the headword from the first <orth extent='full' lang='greek'>."""
    beta = _find_text(entry, "orth", extent="full", lang="greek")
    if not beta:
        # Fallback: try any <orth> with lang="greek"
        beta = _find_text(entry, "orth", lang="greek")
    if not beta:
        # Last resort: use the key attribute
        beta = entry.get("key", "")
    if not beta:
        return ""
    # Strip quantity markers (^ for breve, _ for macron) before conversion
    cleaned = beta.replace("^", "").replace("_", "")
    return beta_to_unicode(cleaned)


def _extract_pos(entry: Any) -> str | None:
    """Infer the part of speech from the entry. Returns None if undetermined."""
    # 1. Explicit <pos> tag
    pos_text = _find_text(entry, "pos")
    if pos_text:
        pos_text = pos_text.strip().rstrip(".")
        # Try exact match first, then with trailing period
        for candidate in (pos_text, pos_text + "."):
            if candidate in _POS_MAP:
                return _POS_MAP[candidate]

    # 2. Check for verb indicators (tense/mood elements within entry)
    for descendant in entry.iter():
        local = descendant.tag.split("}")[-1] if "}" in descendant.tag else descendant.tag
        if local in ("tns", "mood"):
            return "verb"

    # 3. Gender-based inference: presence of <gen> → noun
    gen_text = _find_text(entry, "gen", lang="greek")
    if gen_text:
        return "noun"

    # 4. Headword ending heuristics
    key = entry.get("key", "")
    if key:
        if key.endswith("ws") or key.endswith("w=s"):
            return "adverb"

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
        local = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if local == "sense":
            for sub in child.iter():
                sub_local = sub.tag.split("}")[-1] if "}" in sub.tag else sub.tag
                if sub_local == "tr":
                    text = _elem_text(sub).strip()
                    if text:
                        glosses.append(text)
            if glosses:
                break  # Only take glosses from the first sense

    # Fallback: <tr> directly under entry (some entries skip <sense>)
    if not glosses:
        for child in entry:
            local = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if local == "tr":
                text = _elem_text(child).strip()
                if text:
                    glosses.append(text)

    combined = ", ".join(glosses)
    if len(combined) > 200:
        combined = combined[:197] + "..."
    return combined


def _extract_dialect(entry: Any) -> str:
    """Extract dialect or default to attic."""
    for descendant in entry.iter():
        local = descendant.tag.split("}")[-1] if "}" in descendant.tag else descendant.tag
        if local == "gram":
            gram_type = descendant.get("type", "")
            if gram_type == "dialect":
                text = _elem_text(descendant).strip()
                if text in _DIALECT_MAP:
                    return _DIALECT_MAP[text]
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

    # Extract headword
    headword = _extract_headword(entry_elem)
    if not headword:
        return None

    # Skip entries that are purely numeric, punctuation, or single letters
    stripped = headword.strip()
    if len(stripped) <= 1:
        return None

    # Extract gloss
    gloss = _extract_gloss(entry_elem)
    if not gloss:
        return None

    # Extract POS
    pos = _extract_pos(entry_elem)
    if pos is None:
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
    if pos in _GENDER_REQUIRED_POS and gender is None:
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
