"""Tests for EN/JA language switching (i18n) support."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api import _hit_formatting as hit_fmt
from api._models import SearchRequest
from phonology.explainer import RuleApplication
from phonology.search import SearchResult


@pytest.fixture
def capture_lang(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Patch the endpoint formatter and capture the forwarded lang argument."""
    import api.main as api_main_module

    captured: dict[str, str] = {}
    original_build = api_main_module._build_search_hit

    def patched_build(result, query_ipa, rules_registry, query_mode, lang="en"):
        captured["lang"] = lang
        return original_build(result, query_ipa, rules_registry, query_mode, lang=lang)

    monkeypatch.setattr(api_main_module, "_build_search_hit", patched_build)
    return captured


# ---------------------------------------------------------------------------
# SearchRequest model validation
# ---------------------------------------------------------------------------


class TestSearchRequestLang:
    def test_lang_defaults_to_english(self) -> None:
        req = SearchRequest(query="ἄνθρωπος")

        assert req.lang == "en"

    def test_lang_accepts_japanese(self) -> None:
        req = SearchRequest(**{"query": "ἄνθρωπος", "lang": "ja"})

        assert req.lang == "ja"
        assert req.response_language == "ja"

    def test_response_language_accepts_japanese(self) -> None:
        req = SearchRequest(**{"query": "ἄνθρωπος", "response_language": "ja"})

        assert req.response_language == "ja"
        assert req.lang == "ja"

    def test_lang_accepts_english_explicit(self) -> None:
        req = SearchRequest(**{"query": "ἄνθρωπος", "lang": "en"})

        assert req.lang == "en"

    def test_lang_rejects_unsupported_locale(self) -> None:
        with pytest.raises(ValidationError):
            SearchRequest(**{"query": "ἄνθρωπος", "lang": "zh"})

    def test_lang_alias_language_is_accepted(self) -> None:
        req = SearchRequest(**{"query": "ἄνθρωπος", "language": "ja"})

        assert req.lang == "ja"
        assert req.response_language == "ja"
        assert req.language == "ancient_greek"
        assert req.legacy_language_alias_used is True

    def test_language_accepts_profile_id(self) -> None:
        req = SearchRequest(**{"query": "ἄνθρωπος", "language": "ancient_greek"})

        assert req.language == "ancient_greek"
        assert req.lang == "en"
        assert req.legacy_language_alias_used is False

    def test_language_alias_conflict_precedence(self) -> None:
        # 1. response_language vs legacy language alias
        req1 = SearchRequest(**{"query": "ἄνθρωπος", "language": "ja", "response_language": "en"})
        assert req1.response_language == "en"
        assert req1.lang == "en"
        assert req1.language == "ancient_greek"
        assert req1.legacy_language_alias_used is True

        # 2. lang vs legacy language alias
        req2 = SearchRequest(**{"query": "ἄνθρωπος", "language": "ja", "lang": "en"})
        assert req2.response_language == "en"
        assert req2.lang == "en"
        assert req2.language == "ancient_greek"
        assert req2.legacy_language_alias_used is True

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("EN", "en"),
            (" en ", "en"),
            (" JA ", "ja"),
        ],
    )
    def test_lang_normalizes_case_and_whitespace(self, value: str, expected: str) -> None:
        req = SearchRequest(**{"query": "ἄνθρωπος", "lang": value})

        assert req.lang == expected


# ---------------------------------------------------------------------------
# _similarity_line
# ---------------------------------------------------------------------------


class TestSimilarityLineI18n:
    @pytest.mark.parametrize(
        ("confidence", "expected_ja"),
        [
            (0.9, "音韻的類似度：高。"),
            (0.8, "音韻的類似度：高。"),
            (0.7, "音韻的類似度：中程度。"),
            (0.55, "音韻的類似度：中程度。"),
            (0.5, "音韻的類似度：低。"),
            (0.3, "音韻的類似度：低。"),
        ],
    )
    def test_returns_japanese_strings(self, confidence: float, expected_ja: str) -> None:
        assert hit_fmt._similarity_line(confidence, lang="ja") == expected_ja

    @pytest.mark.parametrize(
        ("confidence", "expected_en"),
        [
            (0.9, "High phonological similarity."),
            (0.8, "High phonological similarity."),
            (0.55, "Moderate phonological similarity."),
            (0.5, "Weak phonological similarity."),
            (0.3, "Weak phonological similarity."),
        ],
    )
    def test_returns_english_strings_by_default(self, confidence: float, expected_en: str) -> None:
        assert hit_fmt._similarity_line(confidence) == expected_en
        assert hit_fmt._similarity_line(confidence, lang="en") == expected_en


# ---------------------------------------------------------------------------
# _build_alignment_summary
# ---------------------------------------------------------------------------


class TestBuildAlignmentSummaryI18n:
    def test_exact_match_japanese(self) -> None:
        result = hit_fmt._build_alignment_summary(
            source_ipa="antʰrɔːpos",
            query_ipa="antʰrɔːpos",
            steps=[],
            lang="ja",
        )

        assert result == "音韻的差異なし。"

    def test_exact_match_english_default(self) -> None:
        result = hit_fmt._build_alignment_summary(
            source_ipa="antʰrɔːpos",
            query_ipa="antʰrɔːpos",
            steps=[],
        )

        assert result == "No phonological difference."

    def test_no_steps_japanese(self) -> None:
        result = hit_fmt._build_alignment_summary(
            source_ipa="loɡos",
            query_ipa="loɡas",
            steps=[],
            lang="ja",
        )

        assert result == "完全なアラインメントで差異を確認できます。"

    def test_explicit_rules_only_japanese(self) -> None:
        steps = [
            RuleApplication(
                rule_id="VSH-001",
                rule_name="Vowel shift",
                from_phone="e",
                to_phone="a",
                position=2,
            )
        ]
        result = hit_fmt._build_alignment_summary(
            source_ipa="loɡos",
            query_ipa="laɡos",
            steps=steps,
            lang="ja",
        )

        assert "マッチしたルール" in result
        assert "1" in result

    def test_explicit_rules_only_english(self) -> None:
        steps = [
            RuleApplication(
                rule_id="VSH-001",
                rule_name="Vowel shift",
                from_phone="e",
                to_phone="a",
                position=2,
            )
        ]
        result = hit_fmt._build_alignment_summary(
            source_ipa="loɡos",
            query_ipa="laɡos",
            steps=steps,
        )

        assert "1 matched rule" in result

    def test_mixed_positions_english_uses_full_position_phrases(self) -> None:
        steps = [
            RuleApplication(
                rule_id="VSH-001",
                rule_name="Vowel shift",
                from_phone="e",
                to_phone="a",
                position=2,
            ),
            RuleApplication(
                rule_id="OBS-001",
                rule_name="Observed change",
                from_phone="o",
                to_phone="a",
                position=-1,
            ),
        ]
        result = hit_fmt._build_alignment_summary(
            source_ipa="loɡos",
            query_ipa="laɡas",
            steps=steps,
        )

        assert "across 1 known position and 1 unknown position" in result
        assert "known known" not in result
        assert "unknown unknown" not in result


# ---------------------------------------------------------------------------
# _build_why_candidate
# ---------------------------------------------------------------------------


class TestBuildWhyCandidateI18n:
    def test_exact_match_japanese(self) -> None:
        bullets = hit_fmt._build_why_candidate(
            "Exact",
            applied_rule_count=0,
            observed_change_count=0,
            confidence=1.0,
            lang="ja",
        )

        assert bullets[0] == "完全な音韻的一致。"
        assert bullets[1] == "音韻的類似度：高。"
        assert bullets[2] == "未説明の差異はありません。"

    def test_exact_match_english_default(self) -> None:
        bullets = hit_fmt._build_why_candidate(
            "Exact",
            applied_rule_count=0,
            observed_change_count=0,
            confidence=1.0,
        )

        assert bullets[0] == "Exact phonological match."
        assert bullets[1] == "High phonological similarity."
        assert bullets[2] == "No remaining unexplained differences."

    def test_rule_based_japanese(self) -> None:
        bullets = hit_fmt._build_why_candidate(
            "Rule-based",
            applied_rule_count=2,
            observed_change_count=0,
            confidence=0.85,
            lang="ja",
        )

        assert bullets[0] == "2明示的ルールによりマッチが支持されます。"
        assert "がにより" not in bullets[0]
        assert bullets[2] == "フォールバック編集は不要です。"

    def test_distance_only_japanese(self) -> None:
        bullets = hit_fmt._build_why_candidate(
            "Distance-only",
            applied_rule_count=0,
            observed_change_count=1,
            confidence=0.6,
            lang="ja",
        )

        assert bullets[0] == "明示的なルールなしで音韻距離によりランキングされました。"
        assert "フォールバック編集" in bullets[2]
        assert "未登録" in bullets[2]

    def test_no_fallback_japanese(self) -> None:
        bullets = hit_fmt._build_why_candidate(
            "Rule-based",
            applied_rule_count=1,
            observed_change_count=0,
            confidence=0.7,
            lang="ja",
        )

        assert bullets[2] == "フォールバック編集は不要です。"

    def test_see_alignment_japanese(self) -> None:
        bullets = hit_fmt._build_why_candidate(
            "Distance-only",
            applied_rule_count=0,
            observed_change_count=0,
            confidence=0.6,
            lang="ja",
        )

        assert bullets[2] == "局所的な差異はアラインメントを参照してください。"


# ---------------------------------------------------------------------------
# _build_search_hit with lang="ja" (integration)
# ---------------------------------------------------------------------------


class TestBuildSearchHitI18n:
    def test_japanese_prose_in_hit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(hit_fmt, "to_prose", lambda explanation: "example prose")

        hit = hit_fmt._build_search_hit(
            SearchResult(
                lemma="λόγος",
                confidence=1.0,
                dialect_attribution="lemma dialect: attic",
                applied_rules=[],
                rule_applications=[],
                ipa="loɡos",
            ),
            query_ipa="loɡos",
            rules_registry={},
            query_mode="Full-form",
            lang="ja",
        )

        assert hit.alignment_summary == "音韻的差異なし。"
        assert hit.why_candidate[0] == "完全な音韻的一致。"
        assert hit.why_candidate[1] == "音韻的類似度：高。"
        assert hit.why_candidate[2] == "未説明の差異はありません。"

    def test_english_prose_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(hit_fmt, "to_prose", lambda explanation: "example prose")

        hit = hit_fmt._build_search_hit(
            SearchResult(
                lemma="λόγος",
                confidence=1.0,
                dialect_attribution="lemma dialect: attic",
                applied_rules=[],
                rule_applications=[],
                ipa="loɡos",
            ),
            query_ipa="loɡos",
            rules_registry={},
            query_mode="Full-form",
        )

        assert hit.alignment_summary == "No phonological difference."
        assert hit.why_candidate[0] == "Exact phonological match."


# ---------------------------------------------------------------------------
# API endpoint: POST /search with lang parameter
# ---------------------------------------------------------------------------


class TestSearchEndpointI18n:
    def test_invalid_lang_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/search",
            json={"query": "ἄνθρωπος", "lang": "zh"},
        )

        assert response.status_code == 422

    def test_missing_lang_defaults_to_english(
        self, client: TestClient, capture_lang: dict[str, str]
    ) -> None:
        """POST without lang returns English prose."""
        client.post("/search", json={"query": "λόγος"})

        assert capture_lang.get("lang") == "en"

    def test_lang_ja_is_forwarded_to_build_search_hit(
        self, client: TestClient, capture_lang: dict[str, str]
    ) -> None:
        """POST with lang=ja forwards the parameter to hit formatting."""
        client.post("/search", json={"query": "λόγος", "lang": "ja"})

        assert capture_lang.get("lang") == "ja"

    def test_response_language_ja_is_forwarded_to_build_search_hit(
        self, client: TestClient, capture_lang: dict[str, str]
    ) -> None:
        client.post("/search", json={"query": "λόγος", "response_language": "ja"})

        assert capture_lang.get("lang") == "ja"

    def test_legacy_language_locale_adds_deprecation_headers(
        self, client: TestClient, capture_lang: dict[str, str]
    ) -> None:
        response = client.post("/search", json={"query": "λόγος", "language": "ja"})

        assert response.status_code == 200
        assert capture_lang.get("lang") == "ja"
        assert response.headers["deprecation"] == "true"
        assert "link" not in response.headers
        assert "response_language" in response.headers["x-proteus-migration"]

    def test_legacy_language_locale_adds_deprecation_link_when_docs_are_enabled(
        self,
        client: TestClient,
        capture_lang: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import api.main as api_main_module

        monkeypatch.setattr(api_main_module.app, "docs_url", "/docs")

        response = client.post("/search", json={"query": "λόγος", "language": "ja"})

        assert response.status_code == 200
        assert capture_lang.get("lang") == "ja"
        assert response.headers["deprecation"] == "true"
        assert response.headers["link"] == '<http://testserver/docs>; rel="deprecation"'
        assert "response_language" in response.headers["x-proteus-migration"]


# ---------------------------------------------------------------------------
# Frontend HTML: language toggle button exists
# ---------------------------------------------------------------------------


class TestFrontendI18nHtml:
    def test_lang_toggle_button_is_present(self, client: TestClient) -> None:
        response = client.get("/")

        assert response.status_code == 200
        assert 'id="langToggle"' in response.text

    def test_lang_toggle_uses_event_listener_not_onclick(self, client: TestClient) -> None:
        response = client.get("/")

        assert response.status_code == 200
        assert "langToggle" in response.text
        assert 'addEventListener("click"' in response.text

    def test_translations_are_loaded_from_static_json(self, client: TestClient) -> None:
        response = client.get("/")

        assert response.status_code == 200
        assert 'fetch("/static/translations.json")' in response.text
        assert "古代ギリシャ語パイロットを備えた歴史音韻フレームワーク" not in response.text

        translations_response = client.get("/static/translations.json")

        assert translations_response.status_code == 200
        translations = translations_response.json()
        assert translations["ja"]["subtitle"] == "古代ギリシャ語パイロットを備えた歴史音韻フレームワーク"
        assert translations["en"]["footerLexiconLabel"] == "Lexicon data from:"
        assert translations["en"]["searchLabel"] == "Search query"

    def test_header_subtitle_uses_data_i18n_selector(self, client: TestClient) -> None:
        response = client.get("/")

        assert response.status_code == 200
        assert 'data-i18n="subtitle"' in response.text
        assert 'querySelectorAll("[data-i18n]")' in response.text
        assert 'querySelector("header p")' not in response.text

    def test_applyStaticTranslations_called_on_init(self, client: TestClient) -> None:
        response = client.get("/")

        assert response.status_code == 200
        assert "applyStaticTranslations()" in response.text

    def test_lang_sent_in_fetch_body(self, client: TestClient) -> None:
        response = client.get("/")

        assert response.status_code == 200
        assert "const requestedLang = _currentLang;" in response.text
        assert "body: JSON.stringify({ ...payload, lang: requestedLang })" in response.text

    def test_last_successful_search_state_is_kept_for_language_switch(
        self, client: TestClient
    ) -> None:
        response = client.get("/")

        assert response.status_code == 200
        assert "let lastSuccessfulSearch = null;" in response.text
        assert "let searchRequestSeq = 0;" in response.text
        assert "lastSuccessfulSearch = { ...payload };" in response.text

    def test_language_switch_refreshes_last_search(self, client: TestClient) -> None:
        response = client.get("/")

        assert response.status_code == 200
        assert "applyStaticTranslations();" in response.text
        assert "refreshLastSearchForLanguage();" in response.text
        assert "if (!lastSuccessfulSearch) return;" in response.text
        assert "executeSearch(lastSuccessfulSearch);" in response.text

    def test_language_refresh_fetches_with_current_lang_not_cached_response(
        self, client: TestClient
    ) -> None:
        response = client.get("/")

        assert response.status_code == 200
        assert "const requestedLang = _currentLang;" in response.text
        assert "body: JSON.stringify({ ...payload, lang: requestedLang })" in response.text
        assert "renderResults(lastSuccessfulSearch" not in response.text

    def test_stale_language_responses_are_not_rendered(self, client: TestClient) -> None:
        response = client.get("/")

        assert response.status_code == 200
        assert "const requestSeq = ++searchRequestSeq;" in response.text
        assert response.text.count(
            "if (requestedLang !== _currentLang || requestSeq !== searchRequestSeq)"
        ) >= 3
        assert "renderResults(data);" in response.text

    def test_only_latest_search_clears_loading_state(self, client: TestClient) -> None:
        response = client.get("/")

        assert response.status_code == 200
        assert "if (requestSeq === searchRequestSeq)" in response.text
        assert "setLoadingState(false);" in response.text
