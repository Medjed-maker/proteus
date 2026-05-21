"""Tests for orthographic-note payloads and language builders."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from phonology.languages.ancient_greek import (
    orthography_notes as orthography_notes_module,
)
from phonology.languages.ancient_greek.orthography_notes import (
    _parse_entry,
    build_orthographic_notes,
    prepare_orthographic_data,
)
from phonology.orthography_notes import OrthographicNoteDataError, OrthographicNotePayload


ROOT_DIR = Path(__file__).resolve().parents[1]
ORTHOGRAPHIC_CORRESPONDENCES_PATH = (
    ROOT_DIR
    / "data"
    / "languages"
    / "ancient_greek"
    / "orthography"
    / "orthographic_correspondences.yaml"
)
ALLOWED_KINDS = {
    "orthographic_correspondence",
    "beginner_aid",
    "pre_403_2_attic",
}
ALLOWED_CONFIDENCE = {"low", "medium", "high"}
ALLOWED_REVIEW_STATUS = {
    "not_expert_reviewed",
    "source_located",
    "needs_expert_review",
    "expert_reviewed",
    "rejected",
}
ALLOWED_SOURCE_TYPES = {
    "aio",
    "phi",
    "ig",
    "secondary_literature",
    "expert_note",
}


def _load_orthography_yaml() -> dict[str, Any]:
    document = yaml.safe_load(
        ORTHOGRAPHIC_CORRESPONDENCES_PATH.read_text(encoding="utf-8")
    )
    assert isinstance(document, dict)
    return document


def _valid_raw_entry(**overrides: object) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "original": "παιδίο",
        "normalized": "παιδίου",
        "candidate_headwords": ["παιδίον"],
        "romanization": "paidiou",
        "kind": "orthographic_correspondence",
        "tags": ["beginner_aid", "inscriptional", "attic_historical_context"],
        "confidence": "medium",
        "review_status": "needs_expert_review",
        "citation_ready": False,
        "source_type": [],
        "source_ids": [],
        "references": [],
        "reference_urls": [],
        "review_notes": "",
        "reviewed_by": "",
        "reviewed_at": "",
    }
    entry.update(overrides)
    return entry


def test_orthographic_correspondence_yaml_has_expected_schema() -> None:
    document = _load_orthography_yaml()
    meta = document.get("_meta")
    entries = document.get("entries")

    assert isinstance(meta, dict)
    assert meta["status"] == "provisional"
    assert meta["review_status"] == "not_expert_reviewed"
    assert meta["citation_ready"] is False
    assert isinstance(entries, list)

    for entry in entries:
        assert isinstance(entry, dict)
        assert isinstance(entry.get("original"), str)
        assert isinstance(entry.get("normalized"), str)
        assert isinstance(entry.get("candidate_headwords"), list)
        assert entry["candidate_headwords"]
        assert all(
            isinstance(candidate, str) and candidate.strip()
            for candidate in entry["candidate_headwords"]
        )
        assert isinstance(entry.get("kind"), str)
        assert entry["kind"] in ALLOWED_KINDS
        assert isinstance(entry.get("confidence"), str)
        assert entry["confidence"] in ALLOWED_CONFIDENCE
        assert isinstance(entry.get("tags"), list)
        assert all(isinstance(tag, str) for tag in entry["tags"])
        assert isinstance(entry.get("references"), list)
        assert all(isinstance(reference, str) for reference in entry["references"])
        assert isinstance(entry.get("review_status"), str)
        assert entry["review_status"] in ALLOWED_REVIEW_STATUS - {"rejected"}
        assert isinstance(entry.get("citation_ready"), bool)
        assert isinstance(entry.get("source_type"), list)
        assert all(
            source_type in ALLOWED_SOURCE_TYPES for source_type in entry["source_type"]
        )
        assert isinstance(entry.get("source_ids"), list)
        assert all(isinstance(source_id, str) for source_id in entry["source_ids"])
        assert isinstance(entry.get("reference_urls"), list)
        assert all(isinstance(url, str) for url in entry["reference_urls"])
        assert isinstance(entry.get("review_notes"), str)
        assert isinstance(entry.get("reviewed_by"), str)
        assert isinstance(entry.get("reviewed_at"), str)


def test_orthographic_correspondence_yaml_contains_only_provisional_seed() -> None:
    document = _load_orthography_yaml()
    entries = document["entries"]

    assert entries == [
        {
            "original": "παιδίο",
            "normalized": "παιδίου",
            "candidate_headwords": ["παιδίον"],
            "romanization": "paidiou",
            "pre_reform_spelling": "παιδίο",
            "pre_reform_romanization": "paidiō",
            "kind": "orthographic_correspondence",
            "tags": ["beginner_aid", "inscriptional", "attic_historical_context"],
            "confidence": "medium",
            "review_status": "needs_expert_review",
            "citation_ready": False,
            "source_type": [],
            "source_ids": [],
            "references": [],
            "reference_urls": [],
            "review_notes": (
                "Direct source evidence for a pre-403/2 BCE Attic inscriptional "
                "reading is not yet confirmed."
            ),
            "reviewed_by": "",
            "reviewed_at": "",
        }
    ]


def test_orthography_loader_uses_language_data_dir() -> None:
    assert (
        orthography_notes_module._orthography_data_path()
        == ORTHOGRAPHIC_CORRESPONDENCES_PATH
    )


def test_ancient_greek_builder_returns_curated_correspondence_notes() -> None:
    notes = build_orthographic_notes(
        query_form="παιδίο",
        candidate_headword="παιδίον",
        candidate_ipa="paidiu",
        query_ipa="paidio",
        response_language="en",
        orthography_hint=None,
    )

    kinds = [note.kind for note in notes]

    assert kinds == [
        "orthographic_correspondence",
        "beginner_aid",
    ]
    correspondence = notes[0]
    assert correspondence.normalized_form == "παιδίου"
    assert correspondence.romanization == "paidiou"
    assert correspondence.pre_reform_spelling == "παιδίο"
    assert correspondence.pre_reform_romanization == "paidiō"
    assert correspondence.confidence == "medium"
    assert correspondence.messages == (
        "Considering pre-403/2 BCE Attic inscriptional spelling and related "
        "orthographic systems, παιδίο may also correspond to παιδίου (paidiou).",
    )
    reading_aid = notes[1]
    assert reading_aid.label == "Reading aid"
    assert reading_aid.messages == (
        "Reading aid: this alternative reading is an orthographic aid separate "
        "from the current candidate παιδίον, and may be read as "
        "παιδίου (paidiou).",
    )


def test_ancient_greek_builder_returns_japanese_messages() -> None:
    notes = build_orthographic_notes(
        query_form="παιδίο",
        candidate_headword="παιδίον",
        candidate_ipa="paidiu",
        query_ipa="paidio",
        response_language="ja",
    )

    messages = [message for note in notes for message in note.messages]

    assert (
        "前403/2年以前のアッティカ碑文表記では、"
        "παιδίο が後代・標準表記の παιδίου (paidiou) に対応する可能性があります。"
        in messages
    )
    assert not any("紀元前403/2年" in message for message in messages)
    assert (
        "読み替え補助: この別読解は、現在候補 παιδίον "
        "とは別の表記体系上の補助候補で、παιδίου (paidiou) と読む可能性があります。"
        in messages
    )
    assert any(note.label == "読み替え補助" for note in notes)


def test_ancient_greek_builder_returns_empty_notes_for_unknown_correspondence() -> None:
    notes = build_orthographic_notes(
        query_form="λόγος",
        candidate_headword="λόγος",
        candidate_ipa="loɡos",
        query_ipa="loɡos",
        response_language="en",
        orthography_hint=None,
    )

    assert notes == []


def test_ancient_greek_builder_requires_curated_candidate_headword() -> None:
    notes = build_orthographic_notes(
        query_form="παιδίο",
        candidate_headword="παιδίου",
        candidate_ipa="paidiu",
        query_ipa="paidio",
        response_language="en",
        orthography_hint=None,
    )

    assert notes == []


def test_ancient_greek_builder_warns_for_deprecated_orthography_hint() -> None:
    with pytest.warns(DeprecationWarning, match="orthography_hint"):
        notes = build_orthographic_notes(
            query_form="λόγος",
            candidate_headword="λόγος",
            candidate_ipa="loɡos",
            query_ipa="loɡos",
            response_language="en",
            orthography_hint="standard",
        )

    assert notes == []


def test_pre_403_2_attic_hint_without_yaml_entry_returns_no_note() -> None:
    with pytest.warns(DeprecationWarning, match="orthography_hint"):
        notes = build_orthographic_notes(
            query_form="λόγος",
            candidate_headword="λόγος",
            candidate_ipa="loɡos",
            query_ipa="loɡos",
            response_language="en",
            orthography_hint="pre_403_2_attic",
        )

    assert notes == []


def test_pre_403_2_attic_note_is_not_duplicated_for_tagged_correspondence() -> None:
    with pytest.warns(DeprecationWarning, match="orthography_hint"):
        notes = build_orthographic_notes(
            query_form="παιδίο",
            candidate_headword="παιδίον",
            candidate_ipa="paidiu",
            query_ipa="paidio",
            response_language="en",
            orthography_hint="pre_403_2_attic",
        )

    assert [note.kind for note in notes].count("pre_403_2_attic") == 0


def test_beginner_aid_requires_beginner_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = orthography_notes_module._CorrespondenceEntry(
        original="abc",
        normalized="def",
        candidate_headwords=("def",),
        romanization="def",
        kind="orthographic_correspondence",
        tags=(),
        confidence="medium",
        references=(),
    )
    monkeypatch.setattr(
        orthography_notes_module,
        "_load_correspondence_entries",
        lambda: (entry,),
    )

    notes = build_orthographic_notes(
        query_form="abc",
        candidate_headword="def",
        candidate_ipa="def",
        query_ipa="abc",
        response_language="en",
    )

    assert [note.kind for note in notes] == ["orthographic_correspondence"]


def test_orthographic_note_payload_references_is_immutable() -> None:
    payload = OrthographicNotePayload(
        kind="beginner_aid",
        label="Label",
        messages=("Note.",),
        confidence="low",
    )

    assert payload.references == ()
    assert isinstance(payload.references, tuple)
    assert not hasattr(payload.references, "append")


def test_beginner_aid_note_with_explicit_romanization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that beginner_aid note uses the explicit romanization from the entry."""
    entry = orthography_notes_module._CorrespondenceEntry(
        original="παιδίο",
        normalized="παιδίου",
        candidate_headwords=("παιδίον",),
        romanization="paidiou",
        kind="orthographic_correspondence",
        tags=("beginner_aid",),
        confidence="medium",
        references=(),
    )
    monkeypatch.setattr(
        orthography_notes_module,
        "_load_correspondence_entries",
        lambda: (entry,),
    )

    notes = build_orthographic_notes(
        query_form="παιδίο",
        candidate_headword="παιδίον",
        candidate_ipa="paidiu",
        query_ipa="paidio",
        response_language="en",
    )

    # Should return notes with the explicit romanization from the entry
    note = next(n for n in notes if n.kind == "beginner_aid")
    assert note.romanization == "paidiou"


def test_beginner_aid_kind_returns_note_without_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = orthography_notes_module._CorrespondenceEntry(
        original="abc",
        normalized="def",
        candidate_headwords=("def",),
        romanization="def",
        kind="beginner_aid",
        tags=(),
        confidence="medium",
        references=(),
    )
    monkeypatch.setattr(
        orthography_notes_module,
        "_load_correspondence_entries",
        lambda: (entry,),
    )

    notes = build_orthographic_notes(
        query_form="abc",
        candidate_headword="def",
        candidate_ipa="def",
        query_ipa="abc",
        response_language="en",
    )

    assert [note.kind for note in notes] == ["beginner_aid"]
    assert notes[0].messages == (
        "Reading aid: this form may be read as def (def).",
    )


def test_pre_403_2_attic_kind_returns_note_without_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = orthography_notes_module._CorrespondenceEntry(
        original="abc",
        normalized="def",
        candidate_headwords=("def",),
        romanization="def",
        kind="pre_403_2_attic",
        tags=(),
        confidence="medium",
        references=(),
    )
    monkeypatch.setattr(
        orthography_notes_module,
        "_load_correspondence_entries",
        lambda: (entry,),
    )

    notes = build_orthographic_notes(
        query_form="abc",
        candidate_headword="def",
        candidate_ipa="def",
        query_ipa="abc",
        response_language="en",
    )

    assert [note.kind for note in notes] == ["pre_403_2_attic"]
    assert notes[0].confidence == "medium"


def test_pre_403_2_attic_kind_returns_japanese_historical_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = orthography_notes_module._CorrespondenceEntry(
        original=orthography_notes_module._nfc("παιδίο"),
        normalized=orthography_notes_module._nfc("παιδίου"),
        candidate_headwords=(orthography_notes_module._nfc("παιδίον"),),
        romanization="paidiou",
        kind="pre_403_2_attic",
        tags=(),
        confidence="medium",
        references=("IG I^3 000",),
        review_status="expert_reviewed",
        citation_ready=True,
        source_type=("ig",),
        source_ids=("IG I^3 000",),
        reviewed_by="tm",
        reviewed_at=datetime.datetime(2026, 5, 5, tzinfo=datetime.UTC),
    )
    monkeypatch.setattr(
        orthography_notes_module,
        "_load_correspondence_entries",
        lambda: (entry,),
    )

    notes = build_orthographic_notes(
        query_form="παιδίο",
        candidate_headword="παιδίον",
        candidate_ipa="paidiu",
        query_ipa="paidio",
        response_language="ja",
    )

    assert [note.kind for note in notes] == ["pre_403_2_attic"]
    assert notes[0].messages == (
        "この形は、紀元前403/2年以前のアッティカ碑文表記を反映している可能性があります。",
    )


def test_orthographic_correspondence_with_pre_403_2_attic_tag_omits_historical_second_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tagged entry should not repeat historical text in correspondence note.

    When an orthographic_correspondence entry also carries the pre_403_2_attic
    tag, the dedicated pre_403_2_attic note already conveys the historical
    advisory. The correspondence note therefore should emit only its first
    (candidate-distinction) message to avoid duplicating the historical claim.
    """
    entry = orthography_notes_module._CorrespondenceEntry(
        original="abc",
        normalized="def",
        candidate_headwords=("xyz",),
        romanization="def",
        kind="orthographic_correspondence",
        tags=("pre_403_2_attic",),
        confidence="medium",
        references=(),
    )
    monkeypatch.setattr(
        orthography_notes_module,
        "_load_correspondence_entries",
        lambda: (entry,),
    )

    notes_en = build_orthographic_notes(
        query_form="abc",
        candidate_headword="xyz",
        candidate_ipa="def",
        query_ipa="abc",
        response_language="en",
    )

    kinds = [note.kind for note in notes_en]
    assert kinds == ["orthographic_correspondence", "pre_403_2_attic"]

    correspondence = notes_en[0]
    assert len(correspondence.messages) == 1
    assert correspondence.messages[0] == (
        "As an alternative orthographic reading, this form may correspond to "
        "def (def)."
    )

    historical = notes_en[1]
    # Historical claim appears exactly once across all emitted notes.
    all_messages = [msg for note in notes_en for msg in note.messages]
    historical_mentions = sum(
        1 for msg in all_messages if "pre-403/2 BCE" in msg
    )
    assert historical_mentions == 1
    assert "pre-403/2 BCE" in historical.messages[0]


def test_orthographic_correspondence_with_pre_403_2_attic_tag_japanese_omits_duplication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Japanese variant of the dedup behavior."""
    entry = orthography_notes_module._CorrespondenceEntry(
        original="abc",
        normalized="def",
        candidate_headwords=("xyz",),
        romanization="def",
        kind="orthographic_correspondence",
        tags=("pre_403_2_attic",),
        confidence="medium",
        references=(),
    )
    monkeypatch.setattr(
        orthography_notes_module,
        "_load_correspondence_entries",
        lambda: (entry,),
    )

    notes_ja = build_orthographic_notes(
        query_form="abc",
        candidate_headword="xyz",
        candidate_ipa="def",
        query_ipa="abc",
        response_language="ja",
    )

    correspondence = notes_ja[0]
    assert len(correspondence.messages) == 1
    assert correspondence.messages[0] == (
        "別の表記上の読解として、この形は def (def) に対応する可能性があります。"
    )

    all_messages = [msg for note in notes_ja for msg in note.messages]
    historical_mentions = sum(
        1 for msg in all_messages if "紀元前403/2年" in msg
    )
    assert historical_mentions == 1
    historical = notes_ja[1]
    assert "紀元前403/2年" in historical.messages[0]


def test_orthographic_correspondence_uses_normalized_form_message_for_current_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = orthography_notes_module._CorrespondenceEntry(
        original="abc",
        normalized="def",
        candidate_headwords=("def",),
        romanization="def",
        kind="orthographic_correspondence",
        tags=(),
        confidence="medium",
        references=(),
    )
    monkeypatch.setattr(
        orthography_notes_module,
        "_load_correspondence_entries",
        lambda: (entry,),
    )

    notes = build_orthographic_notes(
        query_form="abc",
        candidate_headword="def",
        candidate_ipa="def",
        query_ipa="abc",
        response_language="en",
    )

    assert [note.kind for note in notes] == ["orthographic_correspondence"]
    assert notes[0].messages == (
        "abc may correspond to normalized form def (def).",
    )
    assert "alternative" not in notes[0].messages[0].lower()


def test_orthographic_correspondence_uses_historical_context_for_current_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = orthography_notes_module._CorrespondenceEntry(
        original="abc",
        normalized="def",
        candidate_headwords=("def",),
        romanization="def",
        kind="orthographic_correspondence",
        tags=("attic_historical_context",),
        confidence="medium",
        references=(),
    )
    monkeypatch.setattr(
        orthography_notes_module,
        "_load_correspondence_entries",
        lambda: (entry,),
    )

    notes = build_orthographic_notes(
        query_form="abc",
        candidate_headword="def",
        candidate_ipa="def",
        query_ipa="abc",
        response_language="en",
    )

    assert [note.kind for note in notes] == ["orthographic_correspondence"]
    assert notes[0].messages == (
        "Considering pre-403/2 BCE Attic inscriptional spelling and related "
        "orthographic systems, abc may correspond to normalized form def (def).",
    )
    assert "alternative" not in notes[0].messages[0].lower()


def test_orthographic_correspondence_uses_japanese_normalized_form_message_for_current_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = orthography_notes_module._CorrespondenceEntry(
        original="abc",
        normalized="def",
        candidate_headwords=("def",),
        romanization="def",
        kind="orthographic_correspondence",
        tags=(),
        confidence="medium",
        references=(),
    )
    monkeypatch.setattr(
        orthography_notes_module,
        "_load_correspondence_entries",
        lambda: (entry,),
    )

    notes = build_orthographic_notes(
        query_form="abc",
        candidate_headword="def",
        candidate_ipa="def",
        query_ipa="abc",
        response_language="ja",
    )

    assert [note.kind for note in notes] == ["orthographic_correspondence"]
    assert notes[0].messages == (
        "abc は正規化形 def (def) に対応する可能性があります。",
    )
    assert "別の表記上の読解" not in notes[0].messages[0]


def test_orthographic_correspondence_uses_japanese_historical_context_for_current_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = orthography_notes_module._CorrespondenceEntry(
        original="abc",
        normalized="def",
        candidate_headwords=("def",),
        romanization="def",
        kind="orthographic_correspondence",
        tags=("attic_historical_context",),
        confidence="medium",
        references=(),
    )
    monkeypatch.setattr(
        orthography_notes_module,
        "_load_correspondence_entries",
        lambda: (entry,),
    )

    notes = build_orthographic_notes(
        query_form="abc",
        candidate_headword="def",
        candidate_ipa="def",
        query_ipa="abc",
        response_language="ja",
    )

    assert [note.kind for note in notes] == ["orthographic_correspondence"]
    assert notes[0].messages == (
        "前403/2年以前のアッティカ碑文表記では、"
        "abc は正規化形 def (def) に対応する可能性があります。",
    )


def test_beginner_aid_kind_and_tag_do_not_duplicate_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = orthography_notes_module._CorrespondenceEntry(
        original="abc",
        normalized="def",
        candidate_headwords=("def",),
        romanization="def",
        kind="beginner_aid",
        tags=("beginner_aid",),
        confidence="medium",
        references=(),
    )
    monkeypatch.setattr(
        orthography_notes_module,
        "_load_correspondence_entries",
        lambda: (entry,),
    )

    notes = build_orthographic_notes(
        query_form="abc",
        candidate_headword="def",
        candidate_ipa="def",
        query_ipa="abc",
        response_language="en",
    )

    assert [note.kind for note in notes] == ["beginner_aid"]


def test_parse_entry_whitespace_romanization_uses_transliteration() -> None:
    """Verify _parse_entry treats whitespace-only romanization as missing."""
    # Entry with whitespace-only romanization should fall back to transliteration
    raw_entry = _valid_raw_entry(romanization="   ")
    entry = _parse_entry(raw_entry, path=Path("test.yaml"), index=0)

    # Should use auto-generated transliteration, not empty string
    assert entry.romanization
    assert entry.romanization.strip() != ""
    assert "paidi" in entry.romanization.lower()


def test_parse_entry_defaults_candidate_headwords_to_normalized_form() -> None:
    raw_entry = _valid_raw_entry(candidate_headwords=None)
    raw_entry.pop("candidate_headwords")

    entry = _parse_entry(raw_entry, path=Path("test.yaml"), index=0)

    assert entry.candidate_headwords == ("παιδίου",)


@pytest.mark.parametrize(
    "candidate_headwords",
    [
        [],
        [""],
        "παιδίον",
        [object()],
    ],
)
def test_parse_entry_rejects_invalid_candidate_headwords(
    candidate_headwords: object,
) -> None:
    raw_entry = _valid_raw_entry(candidate_headwords=candidate_headwords)

    with pytest.raises(ValueError, match="candidate"):
        _parse_entry(raw_entry, path=Path("test.yaml"), index=0)


def test_parse_entry_accepts_valid_review_metadata() -> None:
    raw_entry = _valid_raw_entry(
        review_status="source_located",
        source_type=["aio", "expert_note"],
        source_ids=["AIO IG I^3 000"],
        references=["AIO, IG I^3 000"],
        reference_urls=["https://example.test/aio/ig-i3-000"],
        review_notes="Source located; expert review pending.",
    )

    entry = _parse_entry(raw_entry, path=Path("test.yaml"), index=0)

    assert entry.review_status == "source_located"
    assert entry.citation_ready is False
    assert entry.source_type == ("aio", "expert_note")
    assert entry.source_ids == ("AIO IG I^3 000",)
    assert entry.references == ("AIO, IG I^3 000",)
    assert entry.reference_urls == ("https://example.test/aio/ig-i3-000",)
    assert entry.review_notes == "Source located; expert review pending."
    assert entry.reviewed_at is None


def test_string_list_and_optional_string_validators_normalize_to_nfc() -> None:
    raw_entry = {
        "tags": [" e\u0301 ", " "],
        "review_notes": " cafe\u0301 ",
        "reviewed_by": None,
    }

    assert orthography_notes_module._require_str_list(
        raw_entry,
        "tags",
        path=Path("test.yaml"),
        index=0,
    ) == ("é",)
    assert (
        orthography_notes_module._optional_str(
            raw_entry,
            "review_notes",
            path=Path("test.yaml"),
            index=0,
        )
        == "café"
    )
    assert (
        orthography_notes_module._optional_str(
            raw_entry,
            "reviewed_by",
            path=Path("test.yaml"),
            index=0,
        )
        == ""
    )


@pytest.mark.parametrize(
    ("key", "match"),
    [
        ("review_status", "directly define"),
        ("citation_ready", "directly define"),
        ("source_type", "directly define"),
        ("source_ids", "directly define"),
        ("references", "directly define"),
    ],
)
def test_parse_entry_rejects_missing_required_review_metadata(
    key: str,
    match: str,
) -> None:
    raw_entry = _valid_raw_entry()
    raw_entry.pop(key)

    with pytest.raises(ValueError, match=match):
        _parse_entry(raw_entry, path=Path("test.yaml"), index=0)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"review_status": "reviewed"}, "unsupported review_status"),
        ({"review_status": "rejected"}, "rejected review_status"),
        ({"source_type": ["unknown"]}, "unsupported source_type"),
        ({"source_type": "aio"}, "list of strings"),
        ({"citation_ready": "false"}, "boolean"),
        ({"reviewed_at": "05/05/2026"}, "ISO date"),
        (
            {"review_status": "source_located", "source_type": [], "source_ids": []},
            "source_located",
        ),
        (
            {"review_status": "expert_reviewed", "reviewed_by": "", "reviewed_at": ""},
            "expert_reviewed",
        ),
        (
            {
                "review_status": "expert_reviewed",
                "reviewed_by": "tm",
                "reviewed_at": "2026-05-05",
            },
            "expert_reviewed",
        ),
        (
            {
                "citation_ready": True,
                "review_status": "source_located",
                "source_type": ["ig"],
                "source_ids": ["IG I^3 000"],
                "references": ["IG I^3 000"],
            },
            "citation_ready true",
        ),
        (
            {
                "citation_ready": True,
                "review_status": "expert_reviewed",
                "source_type": ["ig"],
                "source_ids": [],
                "references": ["IG I^3 000"],
                "reviewed_by": "tm",
                "reviewed_at": "2026-05-05",
            },
            "expert_reviewed",
        ),
        (
            {
                "review_status": "source_located",
                "source_type": ["ig"],
                "source_ids": ["IG I^3 000"],
                "references": ["https://example.test/ig-i3-000"],
            },
            "URLs out of 'references'",
        ),
        (
            {
                "review_status": "source_located",
                "source_type": ["ig"],
                "source_ids": ["https://example.test/ig-i3-000"],
                "references": ["IG I^3 000"],
            },
            "URLs out of 'source_ids'",
        ),
        (
            {
                "review_status": "source_located",
                "source_type": ["ig"],
                "source_ids": ["IG I^3 000"],
                "references": ["IG I^3 000"],
                "reference_urls": ["IG I^3 000"],
            },
            "reference_urls",
        ),
        (
            {
                "review_status": "source_located",
                "source_type": ["ig"],
                "source_ids": ["IG I^3 000"],
                "references": ["IG I^3 000"],
                "reference_urls": ["ftp://example.test/ig-i3-000"],
            },
            "reference_urls",
        ),
        (
            {
                "evidence_excerpt": "short excerpt",
            },
            "evidence_excerpt",
        ),
        ({"tags": ["pre_403_2_attic"]}, "pre_403_2_attic"),
        ({"kind": "pre_403_2_attic"}, "pre_403_2_attic"),
    ],
)
def test_parse_entry_rejects_invalid_review_metadata(
    overrides: dict[str, object],
    match: str,
) -> None:
    raw_entry = _valid_raw_entry(**overrides)

    with pytest.raises(ValueError, match=match):
        _parse_entry(raw_entry, path=Path("test.yaml"), index=0)


def test_parse_entry_accepts_citation_ready_entry_with_complete_review_metadata() -> None:
    raw_entry = _valid_raw_entry(
        review_status="expert_reviewed",
        citation_ready=True,
        source_type=["ig", "expert_note"],
        source_ids=["IG I^3 000"],
        references=["IG I^3 000"],
        reviewed_by="tm",
        reviewed_at="2026-05-05",
    )

    entry = _parse_entry(raw_entry, path=Path("test.yaml"), index=0)

    assert entry.review_status == "expert_reviewed"
    assert entry.citation_ready is True
    assert entry.reviewed_by == "tm"
    assert entry.reviewed_at == datetime.datetime(2026, 5, 5, tzinfo=datetime.UTC)


def test_orthographic_note_payload_normalizes_list_messages_to_tuple() -> None:
    payload = OrthographicNotePayload(
        kind="beginner_aid",
        label="Label",
        messages=["a", "b"],  # type: ignore[arg-type]
        confidence="low",
    )

    assert isinstance(payload.messages, tuple)
    assert payload.messages == ("a", "b")


def test_orthographic_note_data_error_is_raised_on_invalid_yaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad_yaml = tmp_path / "orthographic_correspondences.yaml"
    bad_yaml.write_text(
        "_meta:\n  status: provisional\n  review_status: not_expert_reviewed\n"
        "  citation_ready: false\n  references: []\n  note: test\n"
        "entries:\n  - original: x\n    normalized: y\n    kind: invalid_kind\n"
        "    confidence: medium\n    tags: []\n    review_status: needs_expert_review\n"
        "    citation_ready: false\n    source_type: []\n    source_ids: []\n"
        "    references: []\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        orthography_notes_module,
        "_orthography_data_path",
        lambda: bad_yaml,
    )
    orthography_notes_module._load_correspondence_entries.cache_clear()

    with pytest.raises(OrthographicNoteDataError):
        orthography_notes_module._load_correspondence_entries()

    orthography_notes_module._load_correspondence_entries.cache_clear()


def test_orthographic_note_data_error_is_raised_on_missing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        orthography_notes_module,
        "_orthography_data_path",
        lambda: tmp_path / "does_not_exist.yaml",
    )
    orthography_notes_module._load_correspondence_entries.cache_clear()

    with pytest.raises(OrthographicNoteDataError):
        orthography_notes_module._load_correspondence_entries()

    orthography_notes_module._load_correspondence_entries.cache_clear()


def test_prepare_orthographic_data_succeeds_with_valid_data() -> None:
    prepare_orthographic_data()  # Should not raise


def test_parse_entry_reads_optional_pre_reform_fields() -> None:
    raw_entry = _valid_raw_entry(
        pre_reform_spelling="παιδίο",
        pre_reform_romanization="paidiō",
    )

    entry = _parse_entry(raw_entry, path=Path("test.yaml"), index=0)

    assert entry.pre_reform_spelling == "παιδίο"
    assert entry.pre_reform_romanization == "paidiō"


@pytest.mark.parametrize(
    "pre_reform_spelling",
    ["παιδίō", "παιδē", "paidiou", "παιδίou", "абв"],
)
def test_parse_entry_rejects_latin_letters_in_pre_reform_spelling(
    pre_reform_spelling: str,
) -> None:
    raw_entry = _valid_raw_entry(
        pre_reform_spelling=pre_reform_spelling,
        pre_reform_romanization="paidiō",
    )

    with pytest.raises(
        ValueError,
        match="pre_reform_spelling.*Greek script",
    ):
        _parse_entry(raw_entry, path=Path("test.yaml"), index=0)


@pytest.mark.parametrize("pre_reform_spelling", ["", "παιδίο", "ἀγαθός"])
def test_parse_entry_accepts_valid_greek_pre_reform_spelling(
    pre_reform_spelling: str,
) -> None:
    raw_entry = _valid_raw_entry(
        pre_reform_spelling=pre_reform_spelling,
        pre_reform_romanization="paidiō",
    )

    entry = _parse_entry(raw_entry, path=Path("test.yaml"), index=0)

    assert entry.pre_reform_spelling == pre_reform_spelling


def test_parse_entry_defaults_pre_reform_fields_to_empty_string() -> None:
    raw_entry = _valid_raw_entry()
    raw_entry.pop("pre_reform_spelling", None)
    raw_entry.pop("pre_reform_romanization", None)

    entry = _parse_entry(raw_entry, path=Path("test.yaml"), index=0)

    assert entry.pre_reform_spelling == ""
    assert entry.pre_reform_romanization == ""


def test_parse_entry_nfc_normalizes_pre_reform_fields() -> None:
    """NFD-encoded pre-reform inputs are stored as their NFC-equivalent strings."""
    import unicodedata

    nfd_spelling = unicodedata.normalize("NFD", "παιδίο")
    nfd_romanization = unicodedata.normalize("NFD", "paidiō")
    assert nfd_spelling != "παιδίο"  # sanity: precomposed differs from decomposed
    raw_entry = _valid_raw_entry(
        pre_reform_spelling=nfd_spelling,
        pre_reform_romanization=nfd_romanization,
    )

    entry = _parse_entry(raw_entry, path=Path("test.yaml"), index=0)

    assert entry.pre_reform_spelling == "παιδίο"
    assert entry.pre_reform_romanization == "paidiō"
    assert entry.pre_reform_spelling == unicodedata.normalize(
        "NFC", entry.pre_reform_spelling
    )
    assert entry.pre_reform_romanization == unicodedata.normalize(
        "NFC", entry.pre_reform_romanization
    )


def test_attic_historical_context_tag_triggers_historical_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Entries tagged ``attic_historical_context`` get the pre-403/2 BCE phrasing."""
    entry = orthography_notes_module._CorrespondenceEntry(
        original="abc",
        normalized="def",
        candidate_headwords=("xyz",),
        romanization="def",
        kind="orthographic_correspondence",
        tags=("attic_historical_context",),
        confidence="medium",
        references=(),
    )
    monkeypatch.setattr(
        orthography_notes_module,
        "_load_correspondence_entries",
        lambda: (entry,),
    )

    notes = build_orthographic_notes(
        query_form="abc",
        candidate_headword="xyz",
        candidate_ipa="def",
        query_ipa="abc",
        response_language="en",
    )

    correspondence = notes[0]
    assert correspondence.messages == (
        "Considering pre-403/2 BCE Attic inscriptional spelling and related "
        "orthographic systems, abc may also correspond to def (def).",
    )


def test_attic_historical_context_suppressed_when_pre_403_2_attic_tag_also_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When both tags coexist, the dedicated historical note takes precedence."""
    entry = orthography_notes_module._CorrespondenceEntry(
        original="abc",
        normalized="def",
        candidate_headwords=("xyz",),
        romanization="def",
        kind="orthographic_correspondence",
        tags=("attic_historical_context", "pre_403_2_attic"),
        confidence="medium",
        references=(),
    )
    monkeypatch.setattr(
        orthography_notes_module,
        "_load_correspondence_entries",
        lambda: (entry,),
    )

    notes = build_orthographic_notes(
        query_form="abc",
        candidate_headword="xyz",
        candidate_ipa="def",
        query_ipa="abc",
        response_language="en",
    )

    correspondence = notes[0]
    # Falls back to the non-historical alternative-reading phrasing because the
    # pre_403_2_attic note already carries the historical advisory.
    assert correspondence.messages == (
        "As an alternative orthographic reading, this form may correspond to "
        "def (def).",
    )


def test_correspondence_note_propagates_pre_reform_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = orthography_notes_module._CorrespondenceEntry(
        original="abc",
        normalized="def",
        candidate_headwords=("xyz",),
        romanization="def",
        kind="orthographic_correspondence",
        tags=(),
        confidence="medium",
        references=(),
        pre_reform_spelling="dEf",
        pre_reform_romanization="dEf",
    )
    monkeypatch.setattr(
        orthography_notes_module,
        "_load_correspondence_entries",
        lambda: (entry,),
    )

    notes = build_orthographic_notes(
        query_form="abc",
        candidate_headword="xyz",
        candidate_ipa="def",
        query_ipa="abc",
        response_language="en",
    )

    assert notes[0].pre_reform_spelling == "dEf"
    assert notes[0].pre_reform_romanization == "dEf"


def test_correspondence_note_pre_reform_fields_none_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = orthography_notes_module._CorrespondenceEntry(
        original="abc",
        normalized="def",
        candidate_headwords=("xyz",),
        romanization="def",
        kind="orthographic_correspondence",
        tags=(),
        confidence="medium",
        references=(),
    )
    monkeypatch.setattr(
        orthography_notes_module,
        "_load_correspondence_entries",
        lambda: (entry,),
    )

    notes = build_orthographic_notes(
        query_form="abc",
        candidate_headword="xyz",
        candidate_ipa="def",
        query_ipa="abc",
        response_language="en",
    )

    assert notes[0].pre_reform_spelling is None
    assert notes[0].pre_reform_romanization is None
