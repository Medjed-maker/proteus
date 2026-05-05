"""Orthographic-note builder for Ancient Greek."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, TypedDict, cast, get_args
import unicodedata
import warnings

import yaml  # type: ignore[import-untyped]

from phonology._paths import DEFAULT_LANGUAGE_ID, resolve_language_data_dir
from phonology.orthography_notes import (
    OrthographicNoteConfidence,
    OrthographicNoteDataError,
    OrthographicNoteKind,
    OrthographicNotePayload,
    ResponseLanguage,
)
from phonology.transliterate import transliterate


_ORTHOGRAPHY_FILENAME = "orthographic_correspondences.yaml"
_ALLOWED_KINDS = {
    "orthographic_correspondence",
    "beginner_aid",
    "pre_403_2_attic",
}
_ALLOWED_CONFIDENCE = {"low", "medium", "high"}
ReviewStatus = Literal[
    "not_expert_reviewed",
    "source_located",
    "needs_expert_review",
    "expert_reviewed",
    "rejected",  # validated at load time; never stored in entries
]
SourceType = Literal[
    "aio",
    "phi",
    "ig",
    "secondary_literature",
    "expert_note",
]
# Derived from Literals so future updates require changing only the Literal definitions
_ALLOWED_REVIEW_STATUS: set[str] = set(get_args(ReviewStatus))
_ALLOWED_SOURCE_TYPES: set[str] = set(get_args(SourceType))
_REQUIRED_REVIEW_METADATA_KEYS = (
    "review_status",
    "citation_ready",
    "source_type",
    "source_ids",
    "references",
)


@dataclass(frozen=True, slots=True)
class _CorrespondenceEntry:
    """Validated orthographic correspondence entry loaded from YAML."""

    original: str
    normalized: str
    candidate_headwords: tuple[str, ...]
    romanization: str
    kind: OrthographicNoteKind
    tags: tuple[str, ...]
    confidence: OrthographicNoteConfidence
    references: tuple[str, ...]
    review_status: ReviewStatus = "not_expert_reviewed"
    citation_ready: bool = False
    source_type: tuple[SourceType, ...] = ()
    source_ids: tuple[str, ...] = ()
    reference_urls: tuple[str, ...] = ()
    review_notes: str = ""
    reviewed_by: str = ""
    reviewed_at: str = ""


class _ReviewMetadata(TypedDict):
    """Validated review metadata loaded from YAML."""

    review_status: ReviewStatus
    citation_ready: bool
    source_type: tuple[SourceType, ...]
    source_ids: tuple[str, ...]
    reference_urls: tuple[str, ...]
    review_notes: str
    reviewed_by: str
    reviewed_at: str


def _nfc(value: str) -> str:
    """Return NFC-normalized text for exact orthographic matching."""
    return unicodedata.normalize("NFC", value)


def _orthography_data_path() -> Path:
    """Return the packaged orthographic correspondence YAML path."""
    return (
        resolve_language_data_dir(DEFAULT_LANGUAGE_ID, "orthography")
        / _ORTHOGRAPHY_FILENAME
    )


def _require_str(raw: dict[str, Any], key: str, *, path: Path, index: int) -> str:
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
    value = raw.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(
            f"Orthographic entry {index} in {path} must define {key!r} as a list of strings"
        )
    return tuple(item.strip() for item in value if item.strip())


def _require_direct_key(
    raw: dict[str, Any],
    key: str,
    *,
    path: Path,
    index: int,
) -> None:
    if key not in raw:
        raise ValueError(
            f"Orthographic entry {index} in {path} must directly define {key!r}"
        )


def _require_bool(raw: dict[str, Any], key: str, *, path: Path, index: int) -> bool:
    _require_direct_key(raw, key, path=path, index=index)
    value = raw[key]
    if not isinstance(value, bool):
        raise ValueError(
            f"Orthographic entry {index} in {path} must define {key!r} as a boolean"
        )
    return value


def _optional_str(raw: dict[str, Any], key: str, *, path: Path, index: int) -> str:
    value = raw.get(key, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(
            f"Orthographic entry {index} in {path} must define {key!r} as a string"
        )
    return value.strip()


def _validate_iso_date(value: str, *, key: str, path: Path, index: int) -> None:
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"Orthographic entry {index} in {path} must define {key!r} as an ISO date"
        ) from exc


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
    reference_urls = _require_str_list(
        raw_entry,
        "reference_urls",
        path=path,
        index=index,
    )
    review_notes = _optional_str(raw_entry, "review_notes", path=path, index=index)
    reviewed_by = _optional_str(raw_entry, "reviewed_by", path=path, index=index)
    reviewed_at = _optional_str(raw_entry, "reviewed_at", path=path, index=index)
    if reviewed_at:
        _validate_iso_date(reviewed_at, key="reviewed_at", path=path, index=index)

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
        "reviewed_at": reviewed_at,
    }


def _optional_candidate_headwords(
    raw: dict[str, Any],
    *,
    normalized: str,
    path: Path,
    index: int,
) -> tuple[str, ...]:
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


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
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
    if not isinstance(raw_entry, dict):
        raise ValueError(f"Orthographic entry {index} in {path} must be a mapping")

    original = _require_str(raw_entry, "original", path=path, index=index)
    normalized = _require_str(raw_entry, "normalized", path=path, index=index)
    raw_romanization = raw_entry.get("romanization")
    if raw_romanization is None or raw_romanization == "":
        romanization = transliterate(normalized)
    elif isinstance(raw_romanization, str):
        stripped = raw_romanization.strip()
        if stripped == "":
            romanization = transliterate(normalized)
        else:
            romanization = stripped
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
        kind=cast(OrthographicNoteKind, kind),
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
    )


@lru_cache(maxsize=1)
def _load_correspondence_entries() -> tuple[_CorrespondenceEntry, ...]:
    """Load and validate curated Ancient Greek orthographic correspondences."""
    try:
        path = _orthography_data_path()
        document = _load_yaml_mapping(path)
        raw_entries = document.get("entries")
        if not isinstance(raw_entries, list):
            raise ValueError(
                f"Orthography data file {path} must define a list under 'entries'"
            )
        return tuple(
            _parse_entry(raw_entry, path=path, index=index)
            for index, raw_entry in enumerate(raw_entries)
        )
    except (ValueError, yaml.YAMLError, FileNotFoundError, OSError) as exc:
        raise OrthographicNoteDataError(str(exc)) from exc


def _correspondence_message(
    entry: _CorrespondenceEntry,
    *,
    response_language: ResponseLanguage,
) -> str:
    if response_language == "ja":
        return (
            f"{entry.original} は正規化形 {entry.normalized} "
            f"({entry.romanization}) に対応する可能性があります。"
        )
    return (
        f"{entry.original} may correspond to normalized form {entry.normalized} "
        f"({entry.romanization})."
    )


def _beginner_message(
    entry: _CorrespondenceEntry,
    *,
    response_language: ResponseLanguage,
) -> str:
    if response_language == "ja":
        return (
            f"読み替え補助: この形は {entry.normalized} "
            f"({entry.romanization}) に対応する可能性があります。"
        )
    return (
        "Reading aid: this form may correspond to "
        f"{entry.normalized} ({entry.romanization})."
    )


def _historical_message(*, response_language: ResponseLanguage) -> str:
    if response_language == "ja":
        return "この形は、紀元前403/2年以前のアッティカ碑文表記を反映している可能性があります。"
    return "This form may reflect a pre-403/2 BCE Attic inscriptional spelling."


# Reachable only via YAML entries with kind/tag == "pre_403_2_attic" once
# source evidence is confirmed (see citation_ready_plan.md Phase 7+).
# The deprecated orthography_hint code path was removed; do not reintroduce.
def _historical_note(
    *,
    response_language: ResponseLanguage,
    confidence: OrthographicNoteConfidence = "low",
    normalized_form: str | None = None,
    romanization: str | None = None,
    references: tuple[str, ...] = (),
) -> OrthographicNotePayload:
    return OrthographicNotePayload(
        kind="pre_403_2_attic",
        label=(
            "前403/2年以前のアッティカ碑文表記"
            if response_language == "ja"
            else "Pre-403/2 BCE Attic spelling"
        ),
        messages=(
            _historical_message(response_language=response_language),
        ),
        confidence=confidence,
        normalized_form=normalized_form,
        romanization=romanization,
        period_label="pre-403/2 BCE Attic",
        references=references,
    )


def _notes_for_entry(
    entry: _CorrespondenceEntry,
    *,
    response_language: ResponseLanguage,
) -> list[OrthographicNotePayload]:
    notes: list[OrthographicNotePayload] = []
    if entry.kind == "orthographic_correspondence":
        notes.append(
            OrthographicNotePayload(
                kind="orthographic_correspondence",
                label=(
                    "表記対応"
                    if response_language == "ja"
                    else "Orthographic correspondence"
                ),
                messages=(
                    _correspondence_message(
                        entry,
                        response_language=response_language,
                    ),
                ),
                confidence=entry.confidence,
                normalized_form=entry.normalized,
                romanization=entry.romanization,
                references=entry.references,
            )
        )
    if entry.kind == "pre_403_2_attic" or "pre_403_2_attic" in entry.tags:
        notes.append(
            _historical_note(
                response_language=response_language,
                confidence=entry.confidence,
                normalized_form=entry.normalized,
                romanization=entry.romanization,
                references=entry.references,
            )
        )
    if entry.kind == "beginner_aid" or "beginner_aid" in entry.tags:
        notes.append(
            OrthographicNotePayload(
                kind="beginner_aid",
                label="読み替え補助" if response_language == "ja" else "Reading aid",
                messages=(
                    _beginner_message(
                        entry,
                        response_language=response_language,
                    ),
                ),
                confidence=entry.confidence,
                normalized_form=entry.normalized,
                romanization=entry.romanization,
                references=entry.references,
            )
        )
    return notes


def build_orthographic_notes(
    *,
    query_form: str,
    candidate_headword: str,
    candidate_ipa: str,  # Reserved for future IPA-based note generation
    query_ipa: str,  # Reserved for future IPA-based note generation
    response_language: ResponseLanguage,
    orthography_hint: str | None = None,
) -> list[OrthographicNotePayload]:
    """Return Ancient Greek orthographic notes for a search candidate.

    The ``orthography_hint`` parameter is deprecated and is no longer used for
    note generation.
    """
    if orthography_hint is not None:
        warnings.warn(
            "build_orthographic_notes(orthography_hint=...) is deprecated and ignored. "
            "Notes are generated only from curated runtime YAML entries; remove this "
            "argument from call sites.",
            DeprecationWarning,
            stacklevel=2,
        )
    normalized_query = _nfc(query_form)
    normalized_candidate = _nfc(candidate_headword)
    notes: list[OrthographicNotePayload] = []
    for entry in _load_correspondence_entries():
        if (
            normalized_query == entry.original
            and normalized_candidate in entry.candidate_headwords
        ):
            notes.extend(_notes_for_entry(entry, response_language=response_language))

    return notes


def prepare_orthographic_data() -> None:
    """Eagerly load and cache orthographic correspondence data.

    Called at startup to ensure data integrity is validated before the first
    search request. Raises OrthographicNoteDataError if the packaged YAML is
    missing or malformed.
    """
    _load_correspondence_entries()


__all__ = ["build_orthographic_notes", "prepare_orthographic_data"]
