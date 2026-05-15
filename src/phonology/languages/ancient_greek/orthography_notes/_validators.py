"""Field-level validators for YAML orthographic-note entries."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

from .schema import (
    _ALLOWED_REVIEW_STATUS,
    _ALLOWED_SOURCE_TYPES,
    _REQUIRED_REVIEW_METADATA_KEYS,
    ReviewStatus,
    SourceType,
    _ReviewMetadata,
    _nfc,
)


def _require_str(raw: dict[str, Any], key: str, *, path: Path, index: int) -> str:
    """Require a non-empty NFC-normalized string field.

    Args:
        raw: Raw YAML entry mapping.
        key: Field name to read.
        path: YAML file path for error messages.
        index: Entry index for error messages.

    Returns:
        Stripped and NFC-normalized string.

    Raises:
        ValueError: If the field is missing, non-string, or empty.
    """
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Orthographic entry {index} in {path} must define non-empty {key!r}"
        )
    return _nfc(value.strip())


def _require_str_list(
    raw: dict[str, Any],
    key: str,
    *,
    path: Path,
    index: int,
) -> tuple[str, ...]:
    """Read an optional list of strings as NFC-normalized values.

    Args:
        raw: Raw YAML entry mapping.
        key: Field name to read.
        path: YAML file path for error messages.
        index: Entry index for error messages.

    Returns:
        Tuple of stripped, NFC-normalized non-empty strings.

    Raises:
        ValueError: If the field is not a list of strings.
    """
    value = raw.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(
            f"Orthographic entry {index} in {path} must define {key!r} as a list of strings"
        )
    normalized_values = []
    for item in value:
        normalized = _nfc(item.strip())
        if normalized:
            normalized_values.append(normalized)
    return tuple(normalized_values)


def _require_direct_key(
    raw: dict[str, Any],
    key: str,
    *,
    path: Path,
    index: int,
) -> None:
    """Require a key to be directly present in the raw entry.

    Args:
        raw: Raw YAML entry mapping.
        key: Field name to check.
        path: YAML file path for error messages.
        index: Entry index for error messages.

    Returns:
        None.

    Raises:
        ValueError: If the key is absent.
    """
    if key not in raw:
        raise ValueError(
            f"Orthographic entry {index} in {path} must directly define {key!r}"
        )


def _require_bool(raw: dict[str, Any], key: str, *, path: Path, index: int) -> bool:
    """Require a directly defined boolean field.

    Args:
        raw: Raw YAML entry mapping.
        key: Field name to read.
        path: YAML file path for error messages.
        index: Entry index for error messages.

    Returns:
        Boolean field value.

    Raises:
        ValueError: If the key is absent or the value is not a boolean.
    """
    _require_direct_key(raw, key, path=path, index=index)
    value = raw[key]
    if not isinstance(value, bool):
        raise ValueError(
            f"Orthographic entry {index} in {path} must define {key!r} as a boolean"
        )
    return value


def _optional_str(raw: dict[str, Any], key: str, *, path: Path, index: int) -> str:
    """Read an optional string field as stripped NFC text.

    Args:
        raw: Raw YAML entry mapping.
        key: Field name to read.
        path: YAML file path for error messages.
        index: Entry index for error messages.

    Returns:
        Empty string for absent/None values, otherwise stripped NFC text.

    Raises:
        ValueError: If the value is present and not a string.
    """
    value = raw.get(key, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(
            f"Orthographic entry {index} in {path} must define {key!r} as a string"
        )
    return _nfc(value.strip())


def _validate_iso_date(value: str, *, key: str, path: Path, index: int) -> date:
    """Validate and parse an ISO date string.

    Args:
        value: Candidate ISO date string.
        key: Field name being validated.
        path: YAML file path for error messages.
        index: Entry index for error messages.

    Returns:
        Parsed date value.

    Raises:
        ValueError: If value is not an ISO date.
    """
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"Orthographic entry {index} in {path} must define {key!r} as an ISO date"
        ) from exc


def _looks_like_url(value: str) -> bool:
    """Check if a value appears to be a URL.

    Returns True if the value has an explicit scheme and netloc (e.g., http://example.com)
    or starts with 'www.' followed by a domain-like pattern containing another dot.
    """
    parsed = urlparse(value)
    if parsed.scheme and parsed.netloc:
        return True
    lower_value = value.lower()
    if lower_value.startswith("www.") and "." in lower_value[4:]:
        return True
    return False


def _validate_no_urls(
    values: tuple[str, ...],
    *,
    key: str,
    path: Path,
    index: int,
) -> None:
    """Raise ValueError if any value in the tuple appears to be a URL."""
    if any(_looks_like_url(value) for value in values):
        raise ValueError(
            f"Orthographic entry {index} in {path} must keep URLs out of {key!r}; "
            "use 'reference_urls' instead"
        )


def _validate_reference_urls(
    values: tuple[str, ...],
    *,
    path: Path,
    index: int,
) -> None:
    """Validate that all reference URLs use http or https scheme."""
    for value in values:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(
                f"Orthographic entry {index} in {path} must define "
                "'reference_urls' as http(s) URLs"
            )


def _validate_review_metadata(
    raw_entry: dict[str, Any],
    *,
    path: Path,
    index: int,
    kind: str,
    tags: tuple[str, ...],
    references: tuple[str, ...],
) -> _ReviewMetadata:
    """Validate review metadata fields against citation-ready invariants.

    ``kind``, ``tags``, and ``references`` MUST already be parsed and validated
    by the caller; this function uses them only for cross-field invariant
    checks (e.g. pre_403_2_attic requires source_type/ids/references).
    """
    if "evidence_excerpt" in raw_entry:
        raise ValueError(
            f"Orthographic entry {index} in {path} must not define "
            "'evidence_excerpt' in runtime data"
        )
    for key in _REQUIRED_REVIEW_METADATA_KEYS:
        _require_direct_key(raw_entry, key, path=path, index=index)

    review_status = _require_str(raw_entry, "review_status", path=path, index=index)
    if review_status not in _ALLOWED_REVIEW_STATUS:
        raise ValueError(
            f"Orthographic entry {index} in {path} has unsupported review_status "
            f"{review_status!r}"
        )
    if review_status == "rejected":
        raise ValueError(
            f"Orthographic entry {index} in {path} must not use rejected review_status "
            "in runtime data"
        )

    citation_ready = _require_bool(
        raw_entry,
        "citation_ready",
        path=path,
        index=index,
    )
    source_type = _require_str_list(raw_entry, "source_type", path=path, index=index)
    unsupported_source_types = sorted(set(source_type) - _ALLOWED_SOURCE_TYPES)
    if unsupported_source_types:
        raise ValueError(
            f"Orthographic entry {index} in {path} has unsupported source_type "
            f"{unsupported_source_types!r}"
        )
    source_ids = _require_str_list(raw_entry, "source_ids", path=path, index=index)
    _validate_no_urls(source_ids, key="source_ids", path=path, index=index)
    _validate_no_urls(references, key="references", path=path, index=index)
    reference_urls = _require_str_list(
        raw_entry,
        "reference_urls",
        path=path,
        index=index,
    )
    _validate_reference_urls(reference_urls, path=path, index=index)
    review_notes = _optional_str(raw_entry, "review_notes", path=path, index=index)
    reviewed_by = _optional_str(raw_entry, "reviewed_by", path=path, index=index)
    reviewed_at = _optional_str(raw_entry, "reviewed_at", path=path, index=index)
    reviewed_at_datetime: datetime | None = None
    if reviewed_at:
        reviewed_at_date = _validate_iso_date(
            reviewed_at,
            key="reviewed_at",
            path=path,
            index=index,
        )
        reviewed_at_datetime = datetime.combine(
            reviewed_at_date,
            datetime.min.time(),
            tzinfo=UTC,
        )

    if review_status == "source_located" and (
        not source_type or not source_ids or not references
    ):
        raise ValueError(
            f"Orthographic entry {index} in {path} with source_located review_status "
            "must define non-empty source_type, source_ids, and references"
        )
    if review_status == "expert_reviewed" and (
        not source_type
        or not source_ids
        or not references
        or not reviewed_by
        or not reviewed_at
    ):
        raise ValueError(
            f"Orthographic entry {index} in {path} with expert_reviewed review_status "
            "must define non-empty source_type, source_ids, references, "
            "reviewed_by, and reviewed_at"
        )
    if citation_ready and review_status != "expert_reviewed":
        raise ValueError(
            f"Orthographic entry {index} in {path} with citation_ready true must have "
            "expert_reviewed review_status"
        )
    if (kind == "pre_403_2_attic" or "pre_403_2_attic" in tags) and (
        not source_type or not source_ids or not references
    ):
        raise ValueError(
            f"Orthographic entry {index} in {path} with pre_403_2_attic must define "
            "non-empty source_type, source_ids, and references"
        )

    return {
        "review_status": cast(ReviewStatus, review_status),
        "citation_ready": citation_ready,
        "source_type": cast(tuple[SourceType, ...], source_type),
        "source_ids": source_ids,
        "reference_urls": reference_urls,
        "review_notes": review_notes,
        "reviewed_by": reviewed_by,
        "reviewed_at": reviewed_at_datetime,
    }


def _optional_candidate_headwords(
    raw: dict[str, Any],
    *,
    normalized: str,
    path: Path,
    index: int,
) -> tuple[str, ...]:
    """Return optional normalized candidate headwords for one YAML entry.

    Args:
        raw: Raw YAML entry mapping.
        normalized: NFC-normalized primary headword used as the default.
        path: Source YAML path used in validation errors.
        index: Zero-based entry index used in validation errors.

    Returns:
        Tuple of candidate headwords. Returns ``(normalized,)`` when
        ``candidate_headwords`` is absent.

    Raises:
        ValueError: If ``candidate_headwords`` is not a list of strings, or if
        trimming and normalization leave no candidates.

    Items are stripped, normalized through ``_nfc``, and empty strings are
    ignored.
    """
    value = raw.get("candidate_headwords")
    if value is None:
        return (normalized,)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(
            f"Orthographic entry {index} in {path} must define "
            "'candidate_headwords' as a list of strings"
        )
    candidates = tuple(_nfc(item.strip()) for item in value if item.strip())
    if not candidates:
        raise ValueError(
            f"Orthographic entry {index} in {path} must define at least one "
            "candidate headword"
        )
    return candidates
