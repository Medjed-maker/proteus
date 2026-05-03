"""Tests for orthographic-note payloads and language builders."""

from __future__ import annotations

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


def _load_orthography_yaml() -> dict[str, Any]:
    document = yaml.safe_load(
        ORTHOGRAPHIC_CORRESPONDENCES_PATH.read_text(encoding="utf-8")
    )
    assert isinstance(document, dict)
    return document


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


def test_orthographic_correspondence_yaml_contains_only_provisional_seed() -> None:
    document = _load_orthography_yaml()
    entries = document["entries"]

    assert entries == [
        {
            "original": "παιδίο",
            "normalized": "παιδίου",
            "candidate_headwords": ["παιδίον"],
            "romanization": "paidiou",
            "kind": "orthographic_correspondence",
            "tags": ["beginner_aid", "inscriptional", "pre_403_2_attic"],
            "confidence": "medium",
            "references": [],
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
        "pre_403_2_attic",
        "beginner_aid",
    ]
    correspondence = notes[0]
    assert correspondence.normalized_form == "παιδίου"
    assert correspondence.romanization == "paidiou"
    assert correspondence.confidence == "medium"
    assert correspondence.messages == (
        "παιδίο may correspond to normalized form παιδίου (paidiou).",
    )
    reading_aid = notes[2]
    assert reading_aid.label == "Reading aid"
    assert reading_aid.messages == (
        "Reading aid: this form may correspond to παιδίου (paidiou).",
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

    assert "παιδίο は正規化形 παιδίου (paidiou) に対応する可能性があります。" in messages
    assert (
        "この形は、紀元前403/2年以前のアッティカ碑文表記を反映している可能性があります。"
        in messages
    )
    assert (
        "読み替え補助: この形は παιδίου (paidiou) に対応する可能性があります。"
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


def test_pre_403_2_attic_hint_returns_historical_note_without_correspondence() -> None:
    notes = build_orthographic_notes(
        query_form="λόγος",
        candidate_headword="λόγος",
        candidate_ipa="loɡos",
        query_ipa="loɡos",
        response_language="en",
        orthography_hint="pre_403_2_attic",
    )

    assert [note.kind for note in notes] == ["pre_403_2_attic"]
    assert notes[0].confidence == "low"
    assert notes[0].messages == (
        "This form may reflect a pre-403/2 BCE Attic inscriptional spelling.",
    )


def test_pre_403_2_attic_note_is_not_duplicated_for_tagged_correspondence() -> None:
    notes = build_orthographic_notes(
        query_form="παιδίο",
        candidate_headword="παιδίον",
        candidate_ipa="paidiu",
        query_ipa="paidio",
        response_language="en",
        orthography_hint="pre_403_2_attic",
    )

    assert [note.kind for note in notes].count("pre_403_2_attic") == 1


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
    assert notes[0].messages == ("Reading aid: this form may correspond to def (def).",)


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
    raw_entry = {
        "original": "παιδίο",
        "normalized": "παιδίου",
        "romanization": "   ",  # whitespace-only
        "kind": "orthographic_correspondence",
        "tags": [],
        "confidence": "medium",
        "references": [],
    }
    entry = _parse_entry(raw_entry, path=Path("test.yaml"), index=0)

    # Should use auto-generated transliteration, not empty string
    assert entry.romanization
    assert entry.romanization.strip() != ""
    assert "paidi" in entry.romanization.lower()


def test_parse_entry_defaults_candidate_headwords_to_normalized_form() -> None:
    raw_entry = {
        "original": "παιδίο",
        "normalized": "παιδίου",
        "kind": "orthographic_correspondence",
        "tags": [],
        "confidence": "medium",
        "references": [],
    }

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
    raw_entry = {
        "original": "παιδίο",
        "normalized": "παιδίου",
        "candidate_headwords": candidate_headwords,
        "kind": "orthographic_correspondence",
        "tags": [],
        "confidence": "medium",
        "references": [],
    }

    with pytest.raises(ValueError, match="candidate"):
        _parse_entry(raw_entry, path=Path("test.yaml"), index=0)


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
        "    confidence: medium\n    tags: []\n    references: []\n",
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
