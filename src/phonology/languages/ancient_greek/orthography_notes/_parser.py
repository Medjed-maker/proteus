"""YAML loading and per-entry parsing for orthographic correspondence data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

from phonology.core.ports.orthography_notes import (
    OrthographicNoteConfidence,
)
from ..transliterate import transliterate

from .schema import (
    _ALLOWED_CONFIDENCE,
    _ALLOWED_KINDS,
    AncientGreekNoteKind,
    _CorrespondenceEntry,
    _nfc,
)
from ._validators import (
    _optional_candidate_headwords,
    _optional_pre_reform_spelling,
    _optional_str,
    _require_str,
    _require_str_list,
    _validate_review_metadata,
)


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load and validate a YAML mapping from a path.

    Args:
        path: YAML file path to read.

    Returns:
        Parsed top-level YAML mapping.

    Raises:
        ValueError: If the file cannot be read, ``yaml.safe_load`` cannot parse
        the file, or the top-level document is not a mapping.
    """
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, PermissionError, UnicodeDecodeError, OSError) as exc:
        raise ValueError(f"Failed to read orthography YAML file {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(
            f"Failed to parse orthography YAML file {path}: {exc}"
        ) from exc
    if not isinstance(document, dict):
        raise ValueError(
            f"Orthography data file {path} must contain a top-level mapping"
        )
    return document


def _parse_entry(raw_entry: Any, *, path: Path, index: int) -> _CorrespondenceEntry:
    """Parse, validate, and normalize one YAML entry.

    Args:
        raw_entry: Candidate YAML entry, expected to be a mapping.
        path: Source YAML path used in validation errors.
        index: Zero-based entry index used in validation errors.

    Returns:
        _CorrespondenceEntry with normalized string fields and review metadata.

    Raises:
        ValueError: If raw_entry is not a dict, required fields fail _require_str,
        romanization has an invalid type, or domain values are unsupported.
    """
    if not isinstance(raw_entry, dict):
        raise ValueError(f"Orthographic entry {index} in {path} must be a mapping")

    original = _require_str(raw_entry, "original", path=path, index=index)
    normalized = _require_str(raw_entry, "normalized", path=path, index=index)
    raw_romanization = raw_entry.get("romanization")
    if raw_romanization is None:
        romanization = transliterate(normalized)
    elif isinstance(raw_romanization, str):
        stripped = raw_romanization.strip()
        romanization = stripped or transliterate(normalized)
    else:
        raise ValueError(
            f"Orthographic entry {index} in {path} must define 'romanization' as a string"
        )

    kind = _require_str(raw_entry, "kind", path=path, index=index)
    if kind not in _ALLOWED_KINDS:
        raise ValueError(
            f"Orthographic entry {index} in {path} has unsupported kind {kind!r}"
        )
    confidence = _require_str(raw_entry, "confidence", path=path, index=index)
    if confidence not in _ALLOWED_CONFIDENCE:
        raise ValueError(
            f"Orthographic entry {index} in {path} has unsupported confidence {confidence!r}"
        )
    tags = _require_str_list(raw_entry, "tags", path=path, index=index)
    references = _require_str_list(raw_entry, "references", path=path, index=index)
    pre_reform_spelling = _optional_pre_reform_spelling(
        raw_entry, path=path, index=index
    )
    pre_reform_romanization = _nfc(
        _optional_str(raw_entry, "pre_reform_romanization", path=path, index=index)
    )
    review_metadata = _validate_review_metadata(
        raw_entry,
        path=path,
        index=index,
        kind=kind,
        tags=tags,
        references=references,
    )

    return _CorrespondenceEntry(
        original=original,
        normalized=normalized,
        candidate_headwords=_optional_candidate_headwords(
            raw_entry,
            normalized=normalized,
            path=path,
            index=index,
        ),
        romanization=romanization,
        kind=cast(AncientGreekNoteKind, kind),
        tags=tags,
        confidence=cast(OrthographicNoteConfidence, confidence),
        references=references,
        review_status=review_metadata["review_status"],
        citation_ready=review_metadata["citation_ready"],
        source_type=review_metadata["source_type"],
        source_ids=review_metadata["source_ids"],
        reference_urls=review_metadata["reference_urls"],
        review_notes=review_metadata["review_notes"],
        reviewed_by=review_metadata["reviewed_by"],
        reviewed_at=review_metadata["reviewed_at"],
        pre_reform_spelling=pre_reform_spelling,
        pre_reform_romanization=pre_reform_romanization,
    )
