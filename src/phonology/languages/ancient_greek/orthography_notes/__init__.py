"""Orthographic-note builder for Ancient Greek.

Public API:
    - :func:`build_orthographic_notes` — generate per-candidate orthographic notes.
    - :func:`prepare_orthographic_data` — eagerly validate packaged YAML at startup.

Internal symbols (``_load_correspondence_entries``, ``_orthography_data_path``,
``_parse_entry``, ``_CorrespondenceEntry``, ``_nfc``) are re-exported at module
scope so existing tests can ``monkeypatch.setattr`` them. They remain implementation
details and may change without notice.
"""

from __future__ import annotations

from functools import lru_cache
import warnings

import yaml

from phonology.orthography_notes import (
    OrthographicNoteDataError,
    OrthographicNotePayload,
    ResponseLanguage,
)

from .schema import (
    _ALLOWED_CONFIDENCE,
    _ALLOWED_KINDS,
    _ALLOWED_REVIEW_STATUS,
    _ALLOWED_SOURCE_TYPES,
    _ORTHOGRAPHY_FILENAME,
    _REQUIRED_REVIEW_METADATA_KEYS,
    ReviewStatus,
    SourceType,
    _CorrespondenceEntry,
    _ReviewMetadata,
    _nfc,
)
from ._paths import _orthography_data_path
from ._validators import (
    _looks_like_url,
    _optional_candidate_headwords,
    _optional_str,
    _require_bool,
    _require_direct_key,
    _require_str,
    _require_str_list,
    _validate_iso_date,
    _validate_no_urls,
    _validate_reference_urls,
    _validate_review_metadata,
)
from ._parser import _load_yaml_mapping, _parse_entry
from ._messages import (
    _beginner_message,
    _correspondence_message,
    _historical_message,
    _historical_note,
    _notes_for_entry,
)


@lru_cache(maxsize=1)
def _load_correspondence_entries() -> tuple[_CorrespondenceEntry, ...]:
    """Load and validate curated Ancient Greek orthographic correspondences.

    Resolves ``_orthography_data_path`` and ``_load_yaml_mapping``/``_parse_entry``
    via module globals so tests that ``monkeypatch.setattr`` these symbols on the
    package take effect on subsequent invocations.
    """
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
