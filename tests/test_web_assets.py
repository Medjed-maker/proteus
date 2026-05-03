from pathlib import Path
from typing import Any
from .test_helpers import WEB_ASSET_KEYS


ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_HTML_PATH = ROOT_DIR / "src" / "web" / "index.html"


def test_translations_json_valid(translations_data: dict[str, Any]) -> None:
    """Verify that translations.json is valid JSON and contains required locales."""
    assert isinstance(translations_data, dict)
    assert "en" in translations_data
    assert "ja" in translations_data


def test_translations_keys_consistent(translations_data: dict[str, Any]) -> None:
    """Verify that 'en' and 'ja' locales have identical key sets."""
    assert "en" in translations_data
    assert "ja" in translations_data

    en_keys = set(translations_data["en"].keys())
    ja_keys = set(translations_data["ja"].keys())

    missing_in_ja = en_keys - ja_keys
    missing_in_en = ja_keys - en_keys

    assert not missing_in_ja, (
        f"Keys present in 'en' but missing in 'ja': {missing_in_ja}"
    )
    assert not missing_in_en, (
        f"Keys present in 'ja' but missing in 'en': {missing_in_en}"
    )


def test_translations_required_keys_present(translations_data: dict[str, Any]) -> None:
    """Verify that specific keys required by the frontend are present in all locales."""
    for locale in ["en", "ja"]:
        assert locale in translations_data, (
            f"Locale '{locale}' missing in translations_data"
        )
        for key in WEB_ASSET_KEYS:
            assert key in translations_data[locale], (
                f"Required key '{key}' missing in locale '{locale}'"
            )


def test_translations_values_not_empty(translations_data: dict[str, Any]) -> None:
    """Verify that all translation values are non-empty strings."""
    for locale in ["en", "ja"]:
        assert locale in translations_data, (
            f"Locale '{locale}' missing in translations_data"
        )
        assert isinstance(translations_data[locale], dict), (
            f"Locale '{locale}' is not a dict"
        )
        for key, value in translations_data[locale].items():
            assert isinstance(value, str), (
                f"Translation for '{key}' in '{locale}' is not a string"
            )
            assert value.strip() != "", (
                f"Translation for '{key}' in '{locale}' is empty"
            )


def test_orthographic_note_translation_key_present_in_fallback_html(
    translations_data: dict[str, Any],
) -> None:
    html = FRONTEND_HTML_PATH.read_text(encoding="utf-8")

    assert translations_data["en"]["sectionOrthographicNote"] == "Orthographic note"
    assert translations_data["ja"]["sectionOrthographicNote"] == "表記体系コメント"
    assert 'sectionOrthographicNote: "Orthographic note"' in html
    assert "orthographicNoteEmpty" not in html


def test_frontend_renders_orthographic_notes_before_alignment() -> None:
    html = FRONTEND_HTML_PATH.read_text(encoding="utf-8")

    assert "function renderOrthographicNotes(notes)" in html
    assert "if (!Array.isArray(notes) || notes.length === 0) return null;" in html
    assert "renderOrthographicNotes(hit.orthographic_notes)" in html
    assert (
        "hit.explanation"
        not in html[
            html.index("function renderOrthographicNotes(notes)") : html.index(
                "function splitRules(hit)"
            )
        ]
    )
    orthographic_renderer = html[
        html.index("function renderOrthographicNotes(notes)") : html.index(
            "function splitRules(hit)"
        )
    ]
    assert "references" not in orthographic_renderer
    assert "confidence" not in orthographic_renderer

    append_body = html[
        html.index("function appendCardBody") : html.index(
            "function renderDetailedCard"
        )
    ]
    assert append_body.index('t("sectionMatchedRules")') < append_body.index(
        "renderOrthographicNotes(hit.orthographic_notes)"
    )
    assert append_body.index('t("sectionOrthographicNote")') < append_body.index(
        't("sectionAlignment")'
    )
    assert append_body.index('t("sectionWhyCandidate")') < append_body.index(
        "renderNotesDetails(hit)"
    )
