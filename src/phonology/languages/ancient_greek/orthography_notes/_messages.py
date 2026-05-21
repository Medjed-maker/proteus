"""Language-aware message builders for orthographic-note payloads."""

from __future__ import annotations

from phonology.orthography_notes import (
    OrthographicNoteConfidence,
    OrthographicNotePayload,
    ResponseLanguage,
)

from .schema import _CorrespondenceEntry


def _uses_attic_historical_context(entry: _CorrespondenceEntry) -> bool:
    """Return True when the entry opts into pre-403/2 BCE Attic context messaging.

    The opt-in is conveyed by the YAML tag ``attic_historical_context``. Entries
    already classified under the ``pre_403_2_attic`` kind/tag receive a
    dedicated historical note, so they are excluded here to avoid emitting the
    same pre-reform claim twice within a single candidate's notes.

    Apply the tag to orthographic-correspondence entries whose pre-reform form
    follows the pre-403/2 BCE Attic inscriptional convention (e.g. ``παιδίο``
    standing for ``παιδίου`` with a single Ο representing long /oː/). The full
    tag vocabulary is documented under ``_meta.tag_vocabulary`` in
    ``orthographic_correspondences.yaml``.
    """
    return (
        "attic_historical_context" in entry.tags
        and entry.kind != "pre_403_2_attic"
        and "pre_403_2_attic" not in entry.tags
    )


def _correspondence_messages(
    entry: _CorrespondenceEntry,
    *,
    candidate_headword: str,
    response_language: ResponseLanguage,
    include_historical_context: bool = False,
) -> tuple[str, ...]:
    """Generate localized normalized-form correspondence messages.

    Args:
        entry: _CorrespondenceEntry with original, normalized, and romanization.
        candidate_headword: Current search candidate this note is attached to.
        response_language: ResponseLanguage such as "ja" for Japanese output.
        include_historical_context: Whether to include cautious pre-403/2 BCE
            Attic context in the correspondence message.

    Returns:
        Localized message. When the normalized form differs from the current
        candidate, the message presents it as an alternative orthographic
        reading. The return type is a tuple to leave room for emitting
        follow-up messages alongside the lead correspondence message without
        changing call sites. Currently every branch returns a single-element
        tuple; callers should not rely on the length staying at 1.
    """
    if candidate_headword == entry.normalized:
        if include_historical_context:
            if response_language == "ja":
                return (
                    "前403/2年以前のアッティカ碑文表記では、"
                    f"{entry.original} は正規化形 {entry.normalized} "
                    f"({entry.romanization}) に対応する可能性があります。",
                )
            return (
                "Considering pre-403/2 BCE Attic inscriptional spelling and related "
                f"orthographic systems, {entry.original} may correspond to normalized "
                f"form {entry.normalized} ({entry.romanization}).",
            )
        if response_language == "ja":
            return (
                f"{entry.original} は正規化形 {entry.normalized} "
                f"({entry.romanization}) に対応する可能性があります。",
            )
        return (
            f"{entry.original} may correspond to normalized form {entry.normalized} "
            f"({entry.romanization}).",
        )

    if not include_historical_context:
        if response_language == "ja":
            return (
                "別の表記上の読解として、この形は "
                f"{entry.normalized} ({entry.romanization}) "
                "に対応する可能性があります。",
            )
        return (
            "As an alternative orthographic reading, this form may correspond to "
            f"{entry.normalized} ({entry.romanization}).",
        )

    if response_language == "ja":
        first = (
            "前403/2年以前のアッティカ碑文表記では、"
            f"{entry.original} が後代・標準表記の {entry.normalized} "
            f"({entry.romanization}) に対応する可能性があります。"
        )
    else:
        first = (
            "Considering pre-403/2 BCE Attic inscriptional spelling and related "
            f"orthographic systems, {entry.original} may also correspond to "
            f"{entry.normalized} ({entry.romanization})."
        )

    return (first,)


def _beginner_message(
    entry: _CorrespondenceEntry,
    *,
    candidate_headword: str,
    response_language: ResponseLanguage,
) -> str:
    """Generate a short localized reading-aid message.

    Args:
        entry: _CorrespondenceEntry with normalized and romanization fields.
        candidate_headword: Current search candidate this note is attached to.
        response_language: ResponseLanguage controlling Japanese versus English
            output.

    Returns:
        Reading-aid message string.

    Notes:
        The function branches on response_language because payload messages are
        localized at construction time.
    """
    if candidate_headword == entry.normalized:
        if response_language == "ja":
            return (
                f"読み替え補助: この形は {entry.normalized} "
                f"({entry.romanization}) と読む可能性があります。"
            )
        return (
            "Reading aid: this form may be read as "
            f"{entry.normalized} ({entry.romanization})."
        )

    if response_language == "ja":
        return (
            f"読み替え補助: この別読解は、現在候補 {candidate_headword} "
            "とは別の表記体系上の補助候補で、"
            f"{entry.normalized} ({entry.romanization}) と読む可能性があります。"
        )
    return (
        "Reading aid: this alternative reading is an orthographic aid separate "
        f"from the current candidate {candidate_headword}, and may be read as "
        f"{entry.normalized} ({entry.romanization})."
    )


def _historical_message(*, response_language: ResponseLanguage) -> str:
    """Return a localized pre-403/2 BCE Attic spelling advisory.

    Args:
        response_language: ResponseLanguage such as "ja" for Japanese output.

    Returns:
        Localized advisory message string.
    """
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
    """Construct a pre-403/2 BCE Attic spelling note payload.

    Args:
        response_language: ResponseLanguage controlling localized label/message.
        confidence: OrthographicNoteConfidence for the payload, defaulting to low.
        normalized_form: Optional normalized Greek form to include.
        romanization: Optional romanized normalized form to include.
        references: Reference identifiers supporting the note.

    Returns:
        OrthographicNotePayload containing kind, label, messages, confidence,
        normalized_form, romanization, period_label, and references.
    """
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
    candidate_headword: str,
    response_language: ResponseLanguage,
) -> list[OrthographicNotePayload]:
    """Build note payloads for an orthographic correspondence entry.

    Args:
        entry: _CorrespondenceEntry whose kind and tags determine emitted notes.
        candidate_headword: Current search candidate this note is attached to.
        response_language: ResponseLanguage used for localized labels/messages.

    Returns:
        Payloads for orthographic_correspondence via _correspondence_messages,
        pre_403_2_attic via _historical_note, and/or beginner_aid via
        _beginner_message. Confidence, normalized form, romanization, and
        references are propagated from the entry.
    """
    pre_reform_spelling = entry.pre_reform_spelling or None
    pre_reform_romanization = entry.pre_reform_romanization or None
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
                    *_correspondence_messages(
                        entry,
                        candidate_headword=candidate_headword,
                        response_language=response_language,
                        include_historical_context=_uses_attic_historical_context(
                            entry
                        ),
                    ),
                ),
                confidence=entry.confidence,
                normalized_form=entry.normalized,
                romanization=entry.romanization,
                references=entry.references,
                pre_reform_spelling=pre_reform_spelling,
                pre_reform_romanization=pre_reform_romanization,
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
                        candidate_headword=candidate_headword,
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
