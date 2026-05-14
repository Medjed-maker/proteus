"""Data classes, constants, and shared helpers for orthographic notes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict, get_args
import unicodedata

from phonology.orthography_notes import (
    OrthographicNoteConfidence,
    OrthographicNoteKind,
)


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
