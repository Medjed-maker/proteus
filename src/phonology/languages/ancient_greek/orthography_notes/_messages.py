"""Language-aware message builders for orthographic-note payloads."""

from __future__ import annotations

from phonology.orthography_notes import (
    OrthographicNoteConfidence,
    OrthographicNotePayload,
    ResponseLanguage,
)

from .schema import _CorrespondenceEntry


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
        messages=(_historical_message(response_language=response_language),),
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
