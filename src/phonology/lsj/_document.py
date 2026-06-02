"""LSJ extractor helper module.

Logger uses the literal ``phonology.lsj_extractor`` name so existing
``caplog.set_level(logger="phonology.lsj_extractor")`` blocks in
``tests/test_lsj_extractor.py`` keep capturing diagnostics from this module
after the split.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any

from .._paths import resolve_repo_data_dir

logger: logging.Logger = logging.getLogger("phonology.lsj_extractor")


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


def build_lexicon_document(
    entries: list[dict[str, Any]], *, supplemental_sources: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Build a complete LSJ lexicon document from extracted lemma entries.

    Args:
        entries: Extracted lemma dictionaries. Each entry is expected to include
            keys such as ``id`` (``str``), ``headword`` (``str``),
            ``transliteration`` (``str``), ``ipa`` (``str``), ``pos`` (``str``),
            ``gloss`` (``str``), ``dialect`` (``str``), and optionally
            ``gender`` (``str``).
        supplemental_sources: Optional provenance labels for curated supplemental
            entries merged into the generated document.

    Returns:
        A ``dict[str, Any]`` with ``schema_version``, ``_meta`` metadata, and
        ``lemmas`` containing the original ``entries`` list.

    Examples:
        >>> doc = build_lexicon_document([
        ...     {"id": "LSJ-000001", "headword": "λόγος", "dialect": "attic"}
        ... ])
        >>> doc["schema_version"]
        '2.0.0'
        >>> doc["lemmas"][0]["id"]
        'LSJ-000001'
    """
    dialect = _document_dialect(entries)
    dialect_label = _document_dialect_label(dialect)
    has_supplemental_sources = bool(supplemental_sources)
    source = "LSJ (Liddell-Scott-Jones, A Greek-English Lexicon, 9th ed.)"
    description = (
        "Ancient Greek lemma dataset extracted from the Perseus Digital "
        f"Library LSJ XML, filtered to {dialect_label} entries with "
        f"{dialect_label} IPA and scholarly transliterations."
    )
    note = (
        "Extracted via scripts/extract-lsj.sh from PerseusDL/lexica "
        f"CTS_XML_TEI; output dialect is {dialect}"
    )
    license_text = "CC-BY-SA-4.0"
    if has_supplemental_sources:
        source = (
            "LSJ extraction plus Proteus curated supplemental entries "
            f"({source})"
        )
        description = (
            "Ancient Greek lemma dataset from LSJ extraction plus Proteus "
            "curated supplemental entries, filtered to "
            f"{dialect_label} entries with {dialect_label} IPA and scholarly "
            "transliterations."
        )
        note = (
            f"{note}; supplemental sources: {', '.join(supplemental_sources)}"
        )
        license_text = (
            "CC-BY-SA-4.0; supplemental entries curated by Proteus maintainers"
        )
    return {
        "schema_version": "2.0.0",
        "_meta": {
            "source": source,
            "encoding": "Unicode NFC",
            "ipa_system": "scholarly Ancient Greek IPA",
            "dialect": dialect,
            "version": "2.0.0",
            "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "license": license_text,
            "contributors": [
                "Perseus Digital Library, Tufts University",
                "Proteus maintainers",
            ],
            "data_schema_ref": "data/languages/ancient_greek/lexicon/greek_lemmas.schema.json",
            "description": description,
            "note": note,
        },
        "lemmas": entries,
    }


def validate_document(
    document: dict[str, Any], schema_path: Path | None = None
) -> None:
    """Validate a lexicon document against the JSON schema.

    Args:
        document: The ``dict[str, Any]`` lexicon document to validate.
        schema_path: Optional path to the JSON schema. When omitted, the
            packaged Ancient Greek lemma schema is used.

    Returns:
        ``None``. The function logs up to ten schema validation errors as a
        side effect before raising.

    Raises:
        ValueError: If schema validation fails.
        OSError: If the schema file cannot be read.
        json.JSONDecodeError: If the schema file is invalid JSON.
        jsonschema.SchemaError: If the loaded schema itself is invalid.
    """
    import jsonschema

    if schema_path is None:
        schema_path = resolve_repo_data_dir("lexicon") / "greek_lemmas.schema.json"

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator_cls = jsonschema.Draft202012Validator
    validator = validator_cls(schema, format_checker=jsonschema.FormatChecker())

    errors = list(validator.iter_errors(document))
    if errors:
        for err in errors[:10]:
            logger.error(
                "Schema validation error: %s at %s",
                err.message,
                list(err.absolute_path),
            )
        raise ValueError(f"Schema validation failed with {len(errors)} error(s)")


def validate_supplemental_lemma_entries(
    entries: list[dict[str, Any]], schema_path: Path | None = None
) -> None:
    """Validate curated supplemental lemma entries against the lemma subschema.

    Each entry is checked against the ``lemma`` definition in the packaged
    ``greek_lemmas.schema.json`` (reusing the canonical required-field and enum
    rules instead of duplicating them). This surfaces a clear error early, at
    load time, rather than letting a malformed entry fail later with an opaque
    ``KeyError`` during downstream processing.

    Args:
        entries: Supplemental lemma dictionaries to validate.
        schema_path: Optional path to the JSON schema. When omitted, the
            packaged Ancient Greek lemma schema is used.

    Raises:
        ValueError: If any entry fails schema validation.
        OSError: If the schema file cannot be read.
        json.JSONDecodeError: If the schema file is invalid JSON.
        jsonschema.SchemaError: If the loaded schema itself is invalid.
    """
    if not entries:
        return

    import jsonschema

    if schema_path is None:
        schema_path = resolve_repo_data_dir("lexicon") / "greek_lemmas.schema.json"

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    lemma_schema = {"$defs": schema["$defs"], "$ref": "#/$defs/lemma"}
    validator = jsonschema.Draft202012Validator(
        lemma_schema, format_checker=jsonschema.FormatChecker()
    )

    for index, entry in enumerate(entries):
        errors = sorted(validator.iter_errors(entry), key=lambda err: list(err.path))
        if errors:
            messages = "; ".join(err.message for err in errors[:5])
            raise ValueError(
                "Supplemental lexicon entry at index "
                f"{index} failed schema validation: {messages}"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
