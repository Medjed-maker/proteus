"""Extract Greek lemma data from Perseus LSJ XML files.

This module is now a compatibility shim. The bulk of the implementation lives
in the ``phonology.lsj`` sub-package. The shim keeps test-seam state local
(``_pos_overrides`` cache, ``resolve_repo_data_dir``, ``transliterate``,
``to_ipa``) so that:

- ``monkeypatch.setattr(phonology.lsj_extractor, "_pos_overrides", None)``
  resets the cache;
- ``monkeypatch.setattr(phonology.lsj_extractor, "resolve_repo_data_dir", ...)``
  redirects the YAML lookup;
- ``monkeypatch.setattr(phonology.lsj_extractor, "transliterate", ...)``
  injects fake transliteration; and
- ``monkeypatch.setattr(phonology.lsj_extractor, "to_ipa", ...)`` injects fake
  IPA conversion for ``extract_entry``.

Public/private symbols that callers historically reached via
``phonology.lsj_extractor.<name>`` continue to work because every member is
imported (or re-defined) at module scope below.

Logger name remains ``"phonology.lsj_extractor"`` so existing
``caplog.set_level(..., logger="phonology.lsj_extractor")`` blocks continue
to capture diagnostics.

Usage from a checkout::

    python -m phonology.lsj_extractor --xml-dir data/external/lsj/CTS_XML_TEI/...
"""

from __future__ import annotations

import logging
from pathlib import Path

from ._paths import resolve_repo_data_dir  # noqa: F401  (test seam: monkeypatched)
from .ipa_converter import to_ipa  # noqa: F401  (test seam: monkeypatched)
from .transliterate import transliterate  # noqa: F401  (test seam: monkeypatched)

logger = logging.getLogger("phonology.lsj_extractor")

# Re-export the sub-package surface.
from .lsj import (
    build_lexicon_document,
    extract_all,
    extract_entry,
    find_xml_files,
    iter_xml_entries,
    validate_document,
)
from .lsj._constants import (  # noqa: F401
    _DIALECT_MAP,
    _GENDER_MAP,
    _GENDER_REQUIRED_POS,
    _POS_MAP,
)
from .lsj._document import (  # noqa: F401
    _document_dialect,
    _document_dialect_label,
)
from .lsj._fields import (  # noqa: F401
    _PosInferenceContext,
    _extract_dialect,
    _extract_gender,
    _extract_gloss,
    _extract_headword,
    _extract_pos,
    _infer_adjective_itype_pos,
    _infer_adverb_ending_pos,
    _infer_explicit_pos,
    _infer_final_participle_pos,
    _infer_gender_based_pos,
    _infer_inline_prose_pos,
    _infer_known_numeral_pos,
    _infer_participle_intro_marker,
    _infer_post_gloss_pos,
    _infer_verb_indicator_pos,
)
from .lsj._heading import (  # noqa: F401
    _DialectDecisionContext,
    _DialectLabelDecision,
    _HeadingContext,
    _backward_scan_heading_context,
    _build_dialect_decision_context,
    _decide_dialect_label,
    _has_attic_dialect_label,
    _has_dialect_variant_chain,
    _has_distinct_following_heading_surface_form,
    _has_distinct_nominal_surface_variant,
    _has_following_heading_headword_context,
    _has_heading_prose_headword_context,
    _has_multiple_intro_greek_forms,
    _has_nearby_variant_context,
    _has_nominal_morphology_continuation,
    _has_plural_neuter_article,
    _has_prior_heading_headword_context,
    _heading_spelling_form,
    _is_attic_without_prior,
    _is_heading_dialect_gramgrp,
    _is_heading_gen_marker,
    _is_heading_itype_marker,
    _is_heading_surface_form,
    _is_single_dialect_surface_variant,
    _is_variant_only,
    _leading_dialect_labels,
    _mapped_heading_dialect,
    _qualifies_by_context,
    _qualifies_by_gen_marker,
    _qualifies_by_nearby_variant_note,
    _scan_heading_context,
    _should_keep_heading_dialect_label,
)
from .lsj._intro import (  # noqa: F401
    _append_intro_child_text,
    _entry_intro_text,
    _has_attic_inline_context,
    _inline_pos_candidates,
    _sense_intro_texts,
    _sense_post_gloss_pos_texts,
)
from .lsj._normalize import (  # noqa: F401
    _has_descendant,
    _headword_itypes,
    _looks_like_adjective_itypes,
    _looks_like_verb_key,
    _normalize_beta_token,
    _normalize_headword_key,
    _normalize_intro_text,
)
from .lsj._xml import (  # noqa: F401
    _elem_text,
    _find_gen_text,
    _find_text,
    _find_text_deep,
    _find_texts,
    _local_name,
)


# ---------------------------------------------------------------------------
# POS overrides cache (kept here so tests can reset `_pos_overrides` and
# monkeypatch `resolve_repo_data_dir` on this shim's namespace).
# ---------------------------------------------------------------------------

_pos_overrides: dict[str, frozenset[str]] | None = None


def _empty_pos_overrides() -> dict[str, frozenset[str]]:
    """Return the canonical empty POS override mapping."""
    return {
        "common_gender_keys": frozenset(),
        "numeral_keys": frozenset(),
    }


def _load_pos_overrides(*, cli_mode: bool = False) -> dict[str, frozenset[str]]:
    """Load POS override keys from the Ancient Greek lexicon directory (cached).

    Args:
        cli_mode: When True, exceptions from missing or malformed override files
            are re-raised after logging; when False (default), errors are logged
            and the function returns empty overrides. In CLI mode this ensures
            extraction fails fast on configuration problems.

    Side effects:
        Caches the result in the module-level ``_pos_overrides`` variable.
    """
    global _pos_overrides  # noqa: PLW0603
    if _pos_overrides is not None:
        return _pos_overrides
    import yaml

    raw: dict[str, object] = {}
    overrides_path: Path | str = "<unresolved>"
    try:
        overrides_path = resolve_repo_data_dir("lexicon") / "pos_overrides.yaml"
        loaded = yaml.safe_load(overrides_path.read_text(encoding="utf-8"))
    except FileNotFoundError as err:
        logger.error(
            "POS overrides file not found or lexicon data dir missing: %s", err
        )
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
            "common_gender_keys": frozenset(
                _ensure_list(raw.get("common_gender_keys", []))
            ),
            "numeral_keys": frozenset(_ensure_list(raw.get("numeral_keys", []))),
        }
    return _pos_overrides


def _looks_like_known_numeral_key(key: str) -> bool:
    """Return True for common Attic cardinal numeral headwords."""
    from .lsj._normalize import _normalize_headword_key as _norm

    normalized = _norm(key).replace("^", "").replace("_", "")
    return normalized in _load_pos_overrides()["numeral_keys"]


def main(
    xml_dir: Path | None = None,
    output_path: Path | None = None,
    *,
    limit: int | None = None,
    dry_run: bool = False,
) -> int:
    """Extract LSJ entries and write the Proteus lexicon JSON.

    Lives in the shim so test monkeypatches on
    ``lsj_extractor_module.extract_all`` / ``build_lexicon_document`` /
    ``validate_document`` take effect when this function is called through
    ``phonology.lsj_extractor.main``.

    Returns 0 on success, 1 on failure.
    """
    import json

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
    """Parse arguments and run extraction.

    Lives in the shim so test monkeypatches on
    ``lsj_extractor_module.main`` and ``lsj_extractor_module._load_pos_overrides``
    are honoured by the same module's ``run_cli``.
    """
    import argparse

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
        help="Output JSON path (default: data/languages/ancient_greek/lexicon/greek_lemmas.json)",
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


__all__ = [
    "extract_all",
    "extract_entry",
    "find_xml_files",
    "iter_xml_entries",
    "build_lexicon_document",
    "validate_document",
    "main",
    "run_cli",
]


if __name__ == "__main__":
    raise SystemExit(run_cli())
