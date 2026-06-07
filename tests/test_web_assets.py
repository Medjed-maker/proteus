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


def test_orthographic_note_translation_keys_present_in_static_json(
    translations_data: dict[str, Any],
) -> None:
    html = FRONTEND_HTML_PATH.read_text(encoding="utf-8")

    assert translations_data["en"]["sectionOrthographicNote"] == "Orthographic note"
    assert translations_data["ja"]["sectionOrthographicNote"] == "表記体系コメント"
    assert (
        translations_data["en"]["orthographicGroupTitle"]
        == "Orthographically annotated candidates"
    )
    assert translations_data["ja"]["orthographicGroupTitle"] == "表記体系コメントあり"
    assert (
        translations_data["en"]["orthographicCurrentCandidate"]
        == "Current candidate:"
    )
    assert translations_data["ja"]["orthographicCurrentCandidate"] == "現在候補:"
    assert (
        translations_data["en"]["orthographicAlternativeReading"]
        == "Writing-system alternative reading:"
    )
    assert (
        translations_data["ja"]["orthographicAlternativeReading"]
        == "表記体系上の別読解:"
    )
    assert (
        translations_data["en"]["orthographicPreReformSpelling"]
        == "Pre-403/2 BCE Attic inscriptional spelling:"
    )
    assert (
        translations_data["ja"]["orthographicPreReformSpelling"]
        == "前403/2年以前のアッティカ碑文表記として:"
    )
    # The values now live in translations.json; assert each key is actually
    # referenced in the frontend (via t() or group config), not merely present.
    assert 't("sectionOrthographicNote")' in html
    assert 'titleKey: "orthographicGroupTitle"' in html
    assert 't("orthographicCurrentCandidate")' in html
    assert 't("orthographicAlternativeReading")' in html
    assert 't("orthographicPreReformSpelling")' in html
    assert "Pre-403/2 BCE Attic inscriptional spelling" not in html
    assert "orthographicNoteEmpty" not in html


def test_frontend_renders_orthographic_notes_before_alignment() -> None:
    html = FRONTEND_HTML_PATH.read_text(encoding="utf-8")

    assert "function renderOrthographicNotes(notes, currentCandidate, queryForm)" in html
    assert "if (!Array.isArray(notes) || notes.length === 0) return null;" in html
    assert (
        "renderOrthographicNotes(\n        hit.orthographic_notes,"
        in html
    )
    renderer_start = html.index(
        "function renderOrthographicNotes(notes, currentCandidate, queryForm)"
    )
    renderer_end = html.index("function splitRules(hit)")
    assert (
        "hit.explanation"
        not in html[renderer_start:renderer_end]
    )
    orthographic_renderer = html[renderer_start:renderer_end]
    assert "references" not in orthographic_renderer
    assert "confidence" not in orthographic_renderer
    assert "function appendOrthographicRelation" in html
    assert 'text: " \\u2192 "' in html
    assert 'const query = typeof queryForm === "string" ? queryForm.trim() : "";' in orthographic_renderer
    assert "appendOrthographicRelation(currentText, query, currentCandidate.trim())" in orthographic_renderer
    assert "note.kind === ORTHO_KIND_CORRESPONDENCE" in orthographic_renderer
    assert (
        'const normalizedForm = note.normalized_form.trim();'
        in orthographic_renderer
    )
    assert (
        'const candidateForm ='
        in orthographic_renderer
    )
    assert (
        '(!candidateForm || normalizedForm !== candidateForm)'
        in orthographic_renderer
    )
    assert orthographic_renderer.index(
        'const isAlternativeReading ='
    ) < orthographic_renderer.index('text: t("orthographicAlternativeReading")')
    assert "note.pre_reform_spelling" in orthographic_renderer
    assert "note.pre_reform_romanization" in orthographic_renderer
    assert (
        'text: t("orthographicPreReformSpelling")'
        in orthographic_renderer
    )
    assert "isAlternativeReading ? query : \"\"" in orthographic_renderer

    append_body = html[
        html.index("function appendCardBody") : html.index("function renderDetailedCard")
    ]
    assert append_body.index('t("sectionMatchedRules")') < append_body.index(
        "renderOrthographicNotes("
    )
    assert append_body.index('t("sectionOrthographicNote")') < append_body.index(
        't("sectionAlignment")'
    )
    assert append_body.index('t("sectionWhyCandidate")') < append_body.index(
        "renderNotesDetails(hit)"
    )


def test_frontend_highlights_orthographic_candidates_above_supported_group() -> None:
    """Verify that candidates with orthographic notes are displayed before the supported group."""
    html = FRONTEND_HTML_PATH.read_text(encoding="utf-8")

    assert "function hasOrthographicNotes(hit)" in html
    assert (
        "return Array.isArray(hit?.orthographic_notes) && "
        "hit.orthographic_notes.length > 0;"
        in html
    )
    assert "function partitionHits(indexedHits)" in html
    assert "orthographic: indexedHits.filter(({ hit }) => hasOrthographicNotes(hit))" in html
    assert (
        "({ hit }) => !hasOrthographicNotes(hit) && isSupportedCandidate(hit)"
        in html
    )
    assert (
        "({ hit }) => !hasOrthographicNotes(hit) && !isSupportedCandidate(hit)"
        in html
    )

    assert 'const ORTHO_KIND_CORRESPONDENCE = "orthographic_correspondence";' in html

    results_renderer = html[
        html.index("function renderResults") : html.index("/* ------------------------------------------------------------------ */", html.index("function renderResults"))
    ]
    assert 'const queryForm = data.query || "";' in results_renderer
    assert "const groups = partitionHits(indexedHits);" in results_renderer
    assert "items: groups.orthographic" in results_renderer
    assert "items: groups.supported" in results_renderer
    assert "items: groups.exploratory" in results_renderer
    assert "renderCandidateGroup(container, group, queryForm)" in results_renderer
    assert results_renderer.index('titleKey: "orthographicGroupTitle"') < results_renderer.index(
        'titleKey: "supportedGroupTitle"'
    )
    assert 'descriptionKey: "orthographicGroupDesc"' in results_renderer


def test_frontend_greek_keyboard_markup_and_i18n(
    translations_data: dict[str, Any],
) -> None:
    """Verify that the search form exposes the Greek keyboard controls."""
    html = FRONTEND_HTML_PATH.read_text(encoding="utf-8")

    assert 'id="greekKeyboardToggle"' in html
    assert 'aria-controls="greekKeyboard"' in html
    assert 'aria-expanded="false"' in html
    assert 'aria-pressed="false"' in html
    assert 'aria-label="Show Greek keyboard"' in html
    assert '<span aria-hidden="true">⌨</span>' in html

    assert 'id="greekKeyboard"' in html
    assert 'role="group"' in html
    assert 'aria-label="Greek keyboard"' in html
    assert "hidden>" in html
    assert 'id="greekKeyboardCaseGroup"' in html
    assert 'aria-label="Greek keyboard letter case"' in html
    assert 'id="greekKeyboardLower"' in html
    assert 'id="greekKeyboardUpper"' in html
    assert 'id="greekKeyboardKeys"' in html
    assert 'aria-label="Greek keyboard keys"' in html

    for key in [
        "keyboardToggleShow",
        "keyboardToggleHide",
        "keyboardLower",
        "keyboardUpper",
        "keyboardCaseLabel",
        "keyboardPanelLabel",
        "keyboardKeysLabel",
        "keyboardAcute",
        "keyboardGrave",
        "keyboardCircumflex",
        "keyboardSmoothBreathing",
        "keyboardRoughBreathing",
        "keyboardDiaeresis",
        "keyboardIotaSubscript",
        "keyboardBackspace",
        "keyboardWildcardHyphen",
        "keyboardWildcardAsterisk",
        "keyboardWildcardTilde",
    ]:
        assert key in translations_data["en"]
        assert key in translations_data["ja"]
    for key in [
        "keyboardAcute",
        "keyboardGrave",
        "keyboardCircumflex",
        "keyboardSmoothBreathing",
        "keyboardRoughBreathing",
        "keyboardDiaeresis",
        "keyboardIotaSubscript",
        "keyboardBackspace",
        "keyboardWildcardHyphen",
        "keyboardWildcardAsterisk",
        "keyboardWildcardTilde",
    ]:
        assert f'ariaKey: "{key}"' in html

    # The on-screen keyboard is operable offline (no /search round-trip), so its
    # per-key aria-labels must survive a translations.json load failure by living
    # in the inline English fallback object.
    fallback = html[
        html.index("let TRANSLATIONS = {") : html.index(
            "async function loadTranslations"
        )
    ]
    for key in [
        "keyboardAcute",
        "keyboardGrave",
        "keyboardCircumflex",
        "keyboardSmoothBreathing",
        "keyboardRoughBreathing",
        "keyboardDiaeresis",
        "keyboardIotaSubscript",
        "keyboardBackspace",
        "keyboardWildcardHyphen",
        "keyboardWildcardAsterisk",
        "keyboardWildcardTilde",
    ]:
        assert f"{key}:" in fallback


def test_frontend_greek_keyboard_key_sets_and_insertion_contract() -> None:
    """Verify the Greek keyboard key sets and input-editing contract."""
    html = FRONTEND_HTML_PATH.read_text(encoding="utf-8")

    lower_start = html.index("lower: [")
    upper_start = html.index("upper: [")
    rows_end = html.index("};", upper_start)
    lower_rows = html[lower_start:upper_start]
    upper_rows = html[upper_start:rows_end]

    assert '["α", "β", "γ", "δ", "ε", "ζ", "η", "θ"]' in lower_rows
    assert '["ι", "κ", "λ", "μ", "ν", "ξ", "ο", "π"]' in lower_rows
    assert '["ρ", "σ", "ς", "τ", "υ", "φ", "χ", "ψ", "ω"]' in lower_rows
    assert '"ς"' in lower_rows

    assert '["Α", "Β", "Γ", "Δ", "Ε", "Ζ", "Η", "Θ"]' in upper_rows
    assert '["Ι", "Κ", "Λ", "Μ", "Ν", "Ξ", "Ο", "Π"]' in upper_rows
    assert '["Ρ", "Σ", "Τ", "Υ", "Φ", "Χ", "Ψ", "Ω"]' in upper_rows
    assert '"ς"' not in upper_rows

    assert 'const GREEK_DIACRITIC_KEYS = [' in html
    assert 'value: "\\u0301", ariaKey: "keyboardAcute"' in html
    assert 'value: "\\u0300", ariaKey: "keyboardGrave"' in html
    assert 'value: "\\u0342", ariaKey: "keyboardCircumflex"' in html
    assert 'value: "\\u0313", ariaKey: "keyboardSmoothBreathing"' in html
    assert 'value: "\\u0314", ariaKey: "keyboardRoughBreathing"' in html
    assert 'value: "\\u0308", ariaKey: "keyboardDiaeresis"' in html
    assert 'value: "\\u0345", ariaKey: "keyboardIotaSubscript"' in html

    # Regression guard: diacritic labels must NOT use the dotted circle
    # (U+25CC) + combining mark, which tofus in fonts lacking Greek combining
    # glyphs (e.g. Palatino in Chrome). Labels use spacing diacritic glyphs
    # while the inserted values stay combining marks.
    assert "◌" not in html
    assert '{ label: "´", value: "\\u0301", ariaKey: "keyboardAcute" }' in html
    assert '{ label: "ˋ", value: "\\u0300", ariaKey: "keyboardGrave" }' in html
    assert '{ label: "῀", value: "\\u0342", ariaKey: "keyboardCircumflex" }' in html
    assert '{ label: "᾿", value: "\\u0313", ariaKey: "keyboardSmoothBreathing" }' in html
    assert '{ label: "῾", value: "\\u0314", ariaKey: "keyboardRoughBreathing" }' in html
    assert '{ label: "¨", value: "\\u0308", ariaKey: "keyboardDiaeresis" }' in html
    assert '{ label: "ͺ", value: "\\u0345", ariaKey: "keyboardIotaSubscript" }' in html

    assert 'const GREEK_UTILITY_KEYS = [' in html
    assert '{ label: "-", value: "-", ariaKey: "keyboardWildcardHyphen" }' in html
    assert '{ label: "*", value: "*", ariaKey: "keyboardWildcardAsterisk" }' in html
    assert '{ label: "~", value: "~", ariaKey: "keyboardWildcardTilde" }' in html
    assert '{ label: "⌫", action: "backspace", ariaKey: "keyboardBackspace" }' in html

    assert 'input.setRangeText(value, start, end, "end");' in html
    assert 'input.setRangeText("", start, end, "end");' in html
    assert 'input.setRangeText("", start - 1, end, "end");' in html
    assert 'input.dispatchEvent(new Event("input", { bubbles: true }));' in html
    assert 'function handleGreekKeyboardKeyClick(event)' in html
    assert 'button.dataset.keyboardAction === "backspace"' in html


def test_frontend_greek_keyboard_uses_event_listeners_and_closes_on_search() -> None:
    """Verify keyboard events are registered progressively and closed on submit."""
    html = FRONTEND_HTML_PATH.read_text(encoding="utf-8")
    register_listeners = html[
        html.index("function registerGreekKeyboardListeners")
        : html.index(
            "/* ------------------------------------------------------------------ */",
            html.index("function registerGreekKeyboardListeners"),
        )
    ]

    assert "onclick=" not in html
    assert 'const keyboardToggle = document.getElementById("greekKeyboardToggle");' in register_listeners
    assert "if (keyboardToggle)" in register_listeners
    assert "keyboardToggle.addEventListener(\"click\", toggleGreekKeyboard);" in register_listeners
    assert '["greekKeyboardLower", "lower"]' in register_listeners
    assert '["greekKeyboardUpper", "upper"]' in register_listeners
    assert "document.getElementById(id)" in register_listeners
    assert "setGreekKeyboardCase(nextCase);" in register_listeners
    assert 'const keyboardKeys = document.getElementById("greekKeyboardKeys");' in register_listeners
    assert "if (keyboardKeys)" in register_listeners
    assert 'keyboardKeys.addEventListener("click", handleGreekKeyboardKeyClick);' in register_listeners
    assert "greekKeyboardListenersRegistered = true;" in register_listeners
    assert "registerGreekKeyboardListeners();" in html

    run_search = html[html.index("async function runSearch") : html.index(
        "/* ------------------------------------------------------------------ */",
        html.index("async function runSearch"),
    )]
    assert "if (event) event.preventDefault();" in run_search
    assert "hideGreekKeyboard();" in run_search
    assert run_search.index("hideGreekKeyboard();") < run_search.index(
        'const query = document.getElementById("query").value.trim();'
    )


def test_frontend_search_payload_normalizes_query_to_nfc() -> None:
    """Verify the search payload normalizes the query to NFC.

    The Greek keyboard inserts combining diacritics (base letter + mark),
    yielding NFD text. Normalizing at the payload boundary guards against
    NFC-keyed lexicon lookups missing such input.
    """
    html = FRONTEND_HTML_PATH.read_text(encoding="utf-8")

    payload_builder = html[
        html.index("function buildSearchPayload(query)") : html.index(
            "}", html.index("function buildSearchPayload(query)")
        )
    ]
    assert 'query: query.normalize("NFC")' in payload_builder
