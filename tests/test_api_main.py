"""Tests for api.main."""

import logging
import unicodedata
from collections.abc import Callable, Generator
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api import main as api_main
from api import _hit_formatting as api_hit_formatting
from api._models import RuleStep, SearchHit, SearchRequest
from phonology import search as search_module
from phonology.explainer import RuleApplication
from phonology.profiles import (
    get_default_language_profile,
    LanguageProfile,
    register_language_profile,
)
from phonology.search import LexiconRecord, SearchResult


class FakeExplanation:
    """Minimal explanation stub for explain_alignment monkeypatches."""

    def __init__(self, steps: list[RuleApplication] | None = None) -> None:
        self.steps = list(steps or [])


def _clear_loader_caches() -> None:
    """Reset cached API loader state."""
    for loader_name in (
        "_load_lexicon_document",
        "_load_lexicon_entries",
        "_load_distance_matrix_with_meta",
        "_load_distance_matrix",
        "_load_rules_registry",
        "_load_search_index",
        "_load_unigram_index",
        "_load_lexicon_map",
        "_load_ipa_index",
    ):
        cache_clear = getattr(getattr(api_main, loader_name), "cache_clear", None)
        if cache_clear is not None:
            cache_clear()
    search_cache_clear = getattr(api_main.phonology_search.get_rules_registry, "cache_clear", None)
    if search_cache_clear is not None:
        search_cache_clear()


def _install_invalid_query_to_ipa(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch to_ipa to raise the same query validation error used in search tests."""

    def fake_to_ipa(query: str, dialect: str = "attic") -> str:
        raise ValueError("query must be a non-empty string")

    monkeypatch.setattr(api_main, "to_ipa", fake_to_ipa)


def _make_fake_search_execution(
    captured: dict[str, object],
    *,
    results: list[SearchResult] | None = None,
    truncated: bool = False,
) -> Callable[..., search_module.SearchExecutionResult]:
    """Build a fake public phonology search callable that records API arguments."""

    def fake_search_execution(
        query: str,
        lexicon: tuple[dict[str, object], ...],
        matrix: dict[str, dict[str, float]],
        max_results: int,
        dialect: str,
        index: dict[str, list[str]],
        unigram_index: dict[str, list[str]] | None = None,
        prebuilt_lexicon_map: dict[str, object] | None = None,
        language: str = "ancient_greek",
        converter: Callable[..., str] | None = None,
        phone_inventory: tuple[str, ...] | None = None,
        dialect_skeleton_builders: object | None = None,
        query_ipa: str | None = None,
        prepared_query: object | None = None,
        prebuilt_ipa_index: dict[str, list[str]] | None = None,
        similarity_fallback_limit: int | None = None,
        unigram_fallback_limit: int | None = None,
    ) -> search_module.SearchExecutionResult:
        captured["query"] = query
        captured["lexicon"] = lexicon
        captured["matrix"] = matrix
        captured["max_results"] = max_results
        captured["dialect"] = dialect
        captured["index"] = index
        captured["unigram_index"] = unigram_index
        captured["prebuilt_lexicon_map"] = prebuilt_lexicon_map
        captured["language"] = language
        captured["converter"] = converter
        captured["phone_inventory"] = phone_inventory
        # Extract query_ipa from prepared_query if available, otherwise use query_ipa param
        effective_query_ipa = query_ipa
        if prepared_query is not None:
            effective_query_ipa = getattr(prepared_query, "query_ipa", query_ipa)
        captured["query_ipa"] = effective_query_ipa
        captured["prebuilt_ipa_index"] = prebuilt_ipa_index
        captured["similarity_fallback_limit"] = similarity_fallback_limit
        captured["unigram_fallback_limit"] = unigram_fallback_limit
        search_results = results
        if search_results is None:
            search_results = [
                SearchResult(
                    lemma="λόγος",
                    confidence=0.75,
                    dialect_attribution="lemma dialect: attic",
                    applied_rules=["CCH-001"],
                    rule_applications=[
                        RuleApplication(
                            rule_id="CCH-001",
                            rule_name="CCH-001",
                            from_phone="s",
                            to_phone="h",
                            position=2,
                        )
                    ],
                    ipa="lóɡos",
                )
            ]
        return search_module.SearchExecutionResult(
            results=search_results,
            truncated=truncated,
        )

    return fake_search_execution


class TestDataVersionsModel:
    def test_accepts_empty_and_iso_timestamps(self) -> None:
        versions = api_main.DataVersions(
            lexicon_updated_at="",
            matrix_generated_at="2026-04-20T09:10:11.123+00:00",
        )

        assert versions.lexicon_updated_at == ""
        assert versions.matrix_generated_at == "2026-04-20T09:10:11.123+00:00"

        normalized = api_main.DataVersions(lexicon_updated_at="2026-04-20T09:10:11Z")

        assert normalized.lexicon_updated_at == "2026-04-20T09:10:11+00:00"

    def test_rejects_invalid_timestamps(self) -> None:
        with pytest.raises(ValidationError):
            api_main.DataVersions(lexicon_updated_at="not-a-timestamp")

        with pytest.raises(ValidationError):
            api_main.DataVersions(matrix_generated_at="2026/04/20")


@pytest.fixture(autouse=True)
def clear_api_caches() -> Generator[None, None, None]:
    """Reset cached API assets between tests."""
    _clear_loader_caches()

    yield

    _clear_loader_caches()


class TestSearchHit:
    @pytest.mark.parametrize("distance", [-0.1, 1.1])
    def test_distance_must_stay_within_unit_interval(self, distance: float) -> None:
        with pytest.raises(ValidationError):
            SearchHit(
                headword="λόγος",
                ipa="loɡos",
                distance=distance,
                confidence=0.5,
                rules_applied=[],
                explanation="example",
            )

    @pytest.mark.parametrize("distance", [0.0, 1.0])
    def test_distance_accepts_unit_interval_boundaries(self, distance: float) -> None:
        hit = SearchHit(
            headword="λόγος",
            ipa="loɡos",
            distance=distance,
            confidence=0.5,
            rules_applied=[],
            explanation="example",
        )

        assert hit.distance == distance

    @pytest.mark.parametrize("confidence", [-0.1, 1.1])
    def test_confidence_must_stay_within_unit_interval(self, confidence: float) -> None:
        with pytest.raises(ValidationError):
            SearchHit(
                headword="λόγος",
                ipa="loɡos",
                distance=0.5,
                confidence=confidence,
                rules_applied=[],
                explanation="example",
            )

    @pytest.mark.parametrize("confidence", [0.0, 1.0])
    def test_confidence_accepts_unit_interval_boundaries(self, confidence: float) -> None:
        hit = SearchHit(
            headword="λόγος",
            ipa="loɡos",
            distance=0.5,
            confidence=confidence,
            rules_applied=[],
            explanation="example",
        )

        assert hit.confidence == confidence

    def test_rules_applied_defaults_to_empty_list(self) -> None:
        hit = SearchHit(
            headword="λόγος",
            ipa="loɡos",
            distance=0.5,
            confidence=0.5,
            explanation="example",
        )

        assert hit.rules_applied == []

    @pytest.mark.parametrize("position", [-1, 0])
    def test_rule_step_position_accepts_unknown_and_known_positions(
        self, position: int
    ) -> None:
        step = RuleStep(
            rule_id="OBS-SUB",
            rule_name="Observed substitution",
            from_phone="a",
            to_phone="b",
            position=position,
        )

        assert step.position == position

    def test_rule_step_position_rejects_values_below_unknown_sentinel(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            RuleStep(
                rule_id="OBS-SUB",
                rule_name="Observed substitution",
                from_phone="a",
                to_phone="b",
                position=-2,
            )

        assert "position" in str(exc_info.value)

    @pytest.mark.parametrize("field_name", ["applied_rule_count", "observed_change_count"])
    def test_rule_counts_must_be_non_negative(self, field_name: str) -> None:
        payload = {
            "headword": "λόγος",
            "ipa": "loɡos",
            "distance": 0.1,
            "confidence": 0.5,
            "rules_applied": [],
            "explanation": "example",
            field_name: -1,
        }

        with pytest.raises(ValidationError) as exc_info:
            SearchHit(**payload)

        assert field_name in str(exc_info.value)

    @pytest.mark.parametrize(
        ("field_name", "value"),
        [
            ("match_type", "Normalized"),
            ("uncertainty", "Unknown"),
            ("candidate_bucket", "Maybe"),
            ("why_candidate", "not-a-list"),
        ],
    )
    def test_new_structured_fields_validate_supported_values(
        self, field_name: str, value: object
    ) -> None:
        payload = {
            "headword": "λόγος",
            "ipa": "loɡos",
            "distance": 0.1,
            "confidence": 0.5,
            "rules_applied": [],
            "explanation": "example",
            field_name: value,
        }

        with pytest.raises(ValidationError) as exc_info:
            SearchHit(**payload)

        assert field_name in str(exc_info.value)

    def test_ipa_rejects_none(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            SearchHit(
                headword="λόγος",
                ipa=None,
                distance=0.1,
                confidence=0.5,
                rules_applied=[],
                explanation="example",
            )

        assert "ipa" in str(exc_info.value)

    @pytest.mark.parametrize(
        ("payload", "field_name"),
        [
            (
                {
                    "headword": None,
                    "ipa": "loɡos",
                    "distance": 0.1,
                    "confidence": 0.5,
                    "rules_applied": [],
                    "explanation": "example",
                },
                "headword",
            ),
            (
                {
                    "headword": "λόγος",
                    "ipa": "loɡos",
                    "distance": 0.1,
                    "confidence": 0.5,
                    "rules_applied": "not-a-list",
                    "explanation": "example",
                },
                "rules_applied",
            ),
            (
                {
                    "headword": "λόγος",
                    "ipa": "loɡos",
                    "distance": 0.1,
                    "confidence": 0.5,
                    "rules_applied": [],
                    "explanation": None,
                },
                "explanation",
            ),
            (
                {
                    "headword": "λόγος",
                    "ipa": "loɡos",
                    "distance": 0.1,
                    "confidence": None,
                    "rules_applied": [],
                    "explanation": "example",
                },
                "confidence",
            ),
        ],
    )
    def test_other_fields_reject_invalid_values(
        self, payload: dict[str, object], field_name: str
    ) -> None:
        with pytest.raises(ValidationError) as exc_info:
            SearchHit(**payload)

        assert field_name in str(exc_info.value)

    def test_build_search_hit_uses_empty_string_when_result_ipa_is_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}

        def fake_explain_alignment(**kwargs: object) -> FakeExplanation:
            captured.update(kwargs)
            return FakeExplanation()

        monkeypatch.setattr(api_hit_formatting, "explain_alignment", fake_explain_alignment)
        monkeypatch.setattr(api_hit_formatting, "to_prose", lambda explanation: "example")

        hit = api_main._build_search_hit(
            SearchResult(
                lemma="λόγος",
                confidence=0.75,
                dialect_attribution="lemma dialect: attic",
                applied_rules=[],
                ipa=None,
            ),
            query_ipa="loɡos",
            rules_registry={},
            query_mode="Full-form",
        )

        assert captured["source_ipa"] == ""
        assert captured["target_ipa"] == "loɡos"
        assert hit.ipa == ""

    def test_build_search_hit_uses_rule_applications_for_positions_and_prose(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            api_hit_formatting,
            "explain_alignment",
            lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not use fallback")),
        )

        hit = api_main._build_search_hit(
            SearchResult(
                lemma="χώρα",
                confidence=0.725,
                dialect_attribution="lemma dialect: attic; query-compatible dialects: attic",
                applied_rules=["VSH-010"],
                rule_applications=[
                    RuleApplication(
                        rule_id="VSH-010",
                        rule_name="アッティカ方言の e・i・r 後における長母音 ā 保持",
                        from_phone="ɛː",
                        to_phone="aː",
                        position=1,
                        dialects=["attic"],
                    )
                ],
                ipa="rɛː",
            ),
            query_ipa="raː",
            rules_registry={},
            query_mode="Full-form",
        )

        assert [step.model_dump() for step in hit.rules_applied] == [
            {
                "rule_id": "VSH-010",
                "rule_name": "アッティカ方言の e・i・r 後における長母音 ā 保持",
                "rule_name_en": "",
                "from_phone": "ɛː",
                "to_phone": "aː",
                "position": 1,
            }
        ]
        assert "distance 0.275" in hit.explanation
        assert "position 1" in hit.explanation
        assert hit.match_type == "Rule-based"
        assert hit.rule_support is True
        assert hit.applied_rule_count == 1
        assert hit.observed_change_count == 0
        assert hit.alignment_summary == "1 matched rule across 1 position."
        assert hit.why_candidate == [
            "1 explicit rule supports the match.",
            "Moderate phonological similarity.",
            "No fallback edits required.",
        ]
        assert hit.uncertainty == "Medium"
        assert hit.candidate_bucket == "Supported"

    def test_build_search_hit_accepts_morphophonemic_rule_ids_without_schema_changes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            api_hit_formatting,
            "explain_alignment",
            lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not use fallback")),
        )

        hit = api_main._build_search_hit(
            SearchResult(
                lemma="Δημοσθένης",
                confidence=0.7,
                dialect_attribution="lemma dialect: attic; query-compatible dialects: attic",
                applied_rules=["MPH-001"],
                rule_applications=[
                    RuleApplication(
                        rule_id="MPH-001",
                        rule_name="アッティカ語の男性語尾 -ās > -ēs 交替",
                        rule_name_en="Attic masculine ending -ās -> -ēs",
                        from_phone="aːs",
                        to_phone="ɛːs",
                        position=4,
                        dialects=["attic"],
                    )
                ],
                ipa="dɛːmostʰenɛːs",
            ),
            query_ipa="damostʰenaːs",
            rules_registry={},
            query_mode="Full-form",
        )

        assert [step.model_dump() for step in hit.rules_applied] == [
            {
                "rule_id": "MPH-001",
                "rule_name": "アッティカ語の男性語尾 -ās > -ēs 交替",
                "rule_name_en": "Attic masculine ending -ās -> -ēs",
                "from_phone": "aːs",
                "to_phone": "ɛːs",
                "position": 4,
            }
        ]

    def test_build_search_hit_derives_exact_match_metadata(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(api_hit_formatting, "explain_alignment", lambda **kwargs: FakeExplanation())
        monkeypatch.setattr(api_hit_formatting, "to_prose", lambda explanation: "exact explanation")

        hit = api_main._build_search_hit(
            SearchResult(
                lemma="λόγος",
                confidence=1.0,
                dialect_attribution="lemma dialect: attic",
                applied_rules=[],
                ipa="loɡos",
            ),
            query_ipa="loɡos",
            rules_registry={},
            query_mode="Full-form",
        )

        assert hit.match_type == "Exact"
        assert hit.rule_support is False
        assert hit.applied_rule_count == 0
        assert hit.observed_change_count == 0
        assert hit.alignment_summary == "No phonological difference."
        assert hit.why_candidate == [
            "Exact phonological match.",
            "High phonological similarity.",
            "No remaining unexplained differences.",
        ]
        assert hit.uncertainty == "Low"
        assert hit.candidate_bucket == "Supported"

    def test_build_search_hit_derives_exact_match_metadata_ignoring_ipa_accents(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(api_hit_formatting, "explain_alignment", lambda **kwargs: FakeExplanation())
        monkeypatch.setattr(api_hit_formatting, "to_prose", lambda explanation: "exact explanation")

        hit = api_main._build_search_hit(
            SearchResult(
                lemma="νῦν",
                confidence=1.0,
                dialect_attribution="lemma dialect: attic",
                applied_rules=[],
                ipa="nýn",
            ),
            query_ipa="nyn",
            rules_registry={},
            query_mode="Short-query",
        )

        assert hit.match_type == "Exact"
        assert hit.alignment_summary == "No phonological difference."
        assert hit.why_candidate == [
            "Exact phonological match.",
            "High phonological similarity.",
            "No remaining unexplained differences.",
        ]
        assert hit.uncertainty == "Low"
        assert hit.candidate_bucket == "Supported"

    def test_build_search_hit_derives_distance_only_metadata_from_observed_changes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            api_hit_formatting,
            "explain_alignment",
            lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not use fallback")),
        )

        hit = api_main._build_search_hit(
            SearchResult(
                lemma="λόγος",
                confidence=0.72,
                dialect_attribution="lemma dialect: attic",
                rule_applications=[
                    RuleApplication(
                        rule_id="OBS-SUB",
                        rule_name="Observed substitution",
                        from_phone="a",
                        to_phone="x",
                        position=0,
                    )
                ],
                ipa="a",
            ),
            query_ipa="x",
            rules_registry={},
            query_mode="Full-form",
        )

        assert hit.match_type == "Distance-only"
        assert hit.rule_support is False
        assert hit.applied_rule_count == 0
        assert hit.observed_change_count == 1
        assert hit.alignment_summary == "1 fallback edit across 1 position."
        assert hit.why_candidate == [
            "Ranked by phonological distance without explicit rule support.",
            "Moderate phonological similarity.",
            "1 fallback edit remains uncatalogued.",
        ]
        assert hit.uncertainty == "Medium"
        assert hit.candidate_bucket == "Exploratory"

    def test_build_search_hit_derives_low_confidence_metadata_without_rules(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(api_hit_formatting, "explain_alignment", lambda **kwargs: FakeExplanation())
        monkeypatch.setattr(api_hit_formatting, "to_prose", lambda explanation: "distance-only explanation")

        hit = api_main._build_search_hit(
            SearchResult(
                lemma="λόγος",
                confidence=0.4,
                dialect_attribution="lemma dialect: attic",
                applied_rules=[],
                ipa="a",
            ),
            query_ipa="x",
            rules_registry={},
            query_mode="Full-form",
        )

        assert hit.match_type == "Low-confidence"
        assert hit.rule_support is False
        assert hit.applied_rule_count == 0
        assert hit.observed_change_count == 0
        assert hit.alignment_summary == "Differences visible in full alignment."
        assert hit.why_candidate == [
            "Ranked by phonological distance without explicit rule support.",
            "Weak phonological similarity.",
            "See alignment for localized differences.",
        ]
        assert hit.uncertainty == "High"
        assert hit.candidate_bucket == "Exploratory"

    def test_build_search_hit_coalesces_missing_dialect_attribution_to_empty_string(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(api_hit_formatting, "explain_alignment", lambda **kwargs: FakeExplanation())
        monkeypatch.setattr(api_hit_formatting, "to_prose", lambda explanation: "example")

        hit = api_main._build_search_hit(
            SearchResult(
                lemma="λόγος",
                confidence=0.75,
                dialect_attribution=None,
                applied_rules=[],
                ipa="loɡos",
            ),
            query_ipa="loɡos",
            rules_registry={},
            query_mode="Full-form",
        )

        assert hit.dialect_attribution == ""


class TestLoadFrontendHtml:
    def test_returns_none_when_frontend_read_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class BrokenFrontendPath:
            def exists(self) -> bool:
                return True

            def read_text(self, *, encoding: str) -> str:
                raise OSError("permission denied")

            def __str__(self) -> str:
                return "broken/index.html"

        monkeypatch.setattr(api_main, "_FRONTEND_PATH", BrokenFrontendPath())

        assert api_main._load_frontend_html() is None


class TestLoadChangelogHtml:
    def test_returns_none_when_changelog_read_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class BrokenChangelogPath:
            def exists(self) -> bool:
                return True

            def read_text(self, *, encoding: str) -> str:
                raise OSError("permission denied")

            def __str__(self) -> str:
                return "broken/changelog.html"

        monkeypatch.setattr(api_main, "_CHANGELOG_PATH", BrokenChangelogPath())

        assert api_main._load_changelog_html() is None


class TestFrontendHtml:
    def test_root_head_returns_success_for_uptime_probes(
        self, client: TestClient
    ) -> None:
        response = client.head("/")

        assert response.status_code == 200
        assert response.content == b""

    def test_root_head_returns_404_when_asset_is_missing(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(api_main, "_FRONTEND_HTML", None)

        response = client.head("/")

        assert response.status_code == 404
        assert response.content == b""

    def test_root_html_accessibility(self, client: TestClient) -> None:
        response = client.get("/")

        assert response.status_code == 200
        assert 'aria-live="polite"' in response.text
        assert 'role="region"' in response.text
        assert 'aria-label="Search results"' in response.text

    def test_root_html_form_structure(self, client: TestClient) -> None:
        response = client.get("/")

        assert response.status_code == 200
        assert 'id="searchForm"' in response.text
        assert 'id="searchBtn"' in response.text
        assert 'type="submit"' in response.text
        assert "Please enter a search term." in response.text

    def test_root_html_uses_progressive_js_without_inline_handlers(
        self, client: TestClient
    ) -> None:
        response = client.get("/")

        assert response.status_code == 200
        assert 'addEventListener("submit", runSearch)' in response.text
        assert "onclick=" not in response.text

    def test_root_html_injects_application_version(self, client: TestClient) -> None:
        response = client.get("/")

        assert response.status_code == 200
        assert "{{APP_VERSION}}" not in response.text
        assert f"Version</span> {api_main._APP_VERSION}" in response.text

    def test_changelog_html_injects_application_version(
        self, client: TestClient
    ) -> None:
        response = client.get("/changelog")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "{{APP_VERSION}}" not in response.text
        assert f"Current version: {api_main._APP_VERSION}" in response.text
        assert "<h1" in response.text
        assert "Changelog" in response.text
        assert 'href="/"' in response.text
        assert "/static/styles.css" in response.text
        assert "cdn.tailwindcss.com" not in response.text

    def test_changelog_returns_404_when_asset_is_missing(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(api_main, "_CHANGELOG_HTML", None)

        response = client.get("/changelog")

        assert response.status_code == 404
        assert response.json() == {"detail": "Changelog asset not found"}

    def test_changelog_head_returns_success_for_preflight(
        self, client: TestClient
    ) -> None:
        response = client.head("/changelog")

        assert response.status_code == 200
        assert response.content == b""

    def test_changelog_head_returns_404_when_asset_is_missing(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(api_main, "_CHANGELOG_HTML", None)

        response = client.head("/changelog")

        assert response.status_code == 404
        assert response.content == b""

    def test_root_html_contains_search_result_ui_labels(self, client: TestClient) -> None:
        response = client.get("/")
        translations_response = client.get("/static/translations.json")

        assert response.status_code == 200
        assert translations_response.status_code == 200
        data = translations_response.json()
        assert "en" in data, "translations.json should contain 'en' key"
        translations = data["en"]
        assert translations["badgeMatchType"] == "Match type"
        assert translations["badgeMatchedRules"] == "Matched rules"
        assert translations["badgeFallbackEdits"] == "Fallback edits"
        assert translations["badgeRuleSupport"] == "Rule support"
        assert translations["sectionUncatalogued"] == "Uncatalogued differences"
        assert translations["badgeUncertainty"] == "Uncertainty"
        assert translations["supportedGroupTitle"] == "Supported candidates"
        assert translations["exploratoryGroupTitle"] == "Exploratory candidates"
        # "Exploratory candidates (" is now built dynamically via t() + count suffix;
        # verify the translation key and the details.open render pattern instead.
        assert 'exploratoryGroupTitle' in response.text
        assert "details.open = true;" in response.text
        assert "collapsed by default" not in response.text
        assert translations["queryModeLabel"] == "Query mode"
        assert "full wildcard and fragment matches" in translations["queryModePartialDetail"]
        assert translations["errTooManyWildcards"].startswith("Only one wildcard marker")
        assert translations["showAlignment"] == "Show full alignment"
        assert translations["versionLabel"] == "Version"
        assert translations["changelogLink"] == "Changelog"
        assert 'versionLabel: "Version"' in response.text
        assert 'changelogLink: "Changelog"' in response.text
        assert 'href="/changelog"' in response.text

    def test_html_references_local_css_not_cdn(self, client: TestClient) -> None:
        response = client.get("/")

        assert response.status_code == 200
        assert "/static/styles.css" in response.text
        assert "cdn.tailwindcss.com" not in response.text
        assert "fonts.googleapis.com" not in response.text

    def test_root_html_footer_uses_sticky_layout_and_announces_external_links(
        self, client: TestClient
    ) -> None:
        response = client.get("/")

        assert response.status_code == 200
        assert 'footer class="mt-auto text-center text-xs text-ink-light"' in response.text
        assert 'footer class="mt-16 text-center text-xs text-ink-light"' not in response.text
        assert 'Perseus Digital Library<span class="sr-only"> (opens in a new tab)</span>' in response.text
        assert 'PerseusDL/lexica<span class="sr-only"> (opens in a new tab)</span>' in response.text
        assert 'CC BY-SA 4.0<span class="sr-only"> (opens in a new tab)</span>' in response.text
        assert 'https://github.com/PerseusDL/morpheus' not in response.text

    def test_missing_route_returns_not_found_json(self, client: TestClient) -> None:
        response = client.get("/missing-route")

        assert response.status_code == 404
        assert response.json() == {"detail": "Not Found"}


class TestStaticAssets:
    def test_css_is_served(self, client: TestClient) -> None:
        response = client.get("/static/styles.css")

        assert response.status_code == 200
        assert "text/css" in response.headers["content-type"]
        assert ".mt-auto" in response.text


class TestHealthEndpoint:
    def test_api_health(self, client: TestClient) -> None:
        """Health endpoint returns 200 ok."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_api_health_head_returns_success(self, client: TestClient) -> None:
        """Health endpoint supports HEAD probes."""
        response = client.head("/health")

        assert response.status_code == 204
        assert response.content == b""


class TestReadyEndpoint:
    def test_ready_returns_ok(self, client: TestClient) -> None:
        """Readiness probe returns 200 when all dependencies load."""
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_ready_returns_503_with_lexicon_setup_guidance_when_lexicon_loader_fails(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Readiness probe should explain how to generate the lexicon."""

        def failing_loader() -> None:
            raise ValueError("lexicon unavailable")

        monkeypatch.setattr(api_main, "_load_lexicon_entries", failing_loader)

        response = client.get("/ready")

        assert response.status_code == 503
        assert response.json()["detail"] == api_main._SEARCH_DEPENDENCIES_LEXICON_NOT_READY_DETAIL

    def test_ready_logs_warning_without_traceback_for_expected_not_ready_state(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Readiness probe should log expected 503 states without stack traces."""

        def failing_loader() -> None:
            raise ValueError("lexicon unavailable")

        monkeypatch.setattr(api_main, "_load_lexicon_entries", failing_loader)
        caplog.set_level(logging.WARNING, logger="api.main")

        response = client.get("/ready")

        assert response.status_code == 503
        assert "Readiness probe skipped; dependencies not ready" in caplog.text
        assert api_main._SEARCH_DEPENDENCIES_LEXICON_NOT_READY_DETAIL in caplog.text
        assert "Traceback" not in caplog.text

    @pytest.mark.parametrize(
        ("loader_name", "error"),
        [
            ("_load_distance_matrix", FileNotFoundError("matrix not found")),
            ("_load_rules_registry", yaml.YAMLError("bad rules")),
            ("_load_search_index", OSError("index error")),
            ("_load_unigram_index", OSError("unigram index error")),
            ("_load_lexicon_map", OSError("lexicon map error")),
        ],
    )
    def test_ready_returns_generic_503_when_non_lexicon_loader_fails(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        loader_name: str,
        error: Exception,
    ) -> None:
        """Readiness probe returns 503 when a dependency loader raises."""

        def failing_loader() -> None:
            raise error

        monkeypatch.setattr(api_main, loader_name, failing_loader)

        response = client.get("/ready")

        assert response.status_code == 503
        assert response.json() == {
            "detail": api_main._SEARCH_DEPENDENCIES_GENERIC_NOT_READY_DETAIL
        }


def _make_test_dependencies() -> dict[str, object]:
    """Return standard test dependency fixtures shared across tests."""
    return {
        "lexicon": (
            {
                "id": "L1",
                "headword": "λόγος",
                "ipa": "lóɡos",
                "dialect": "attic",
            },
        ),
        "matrix": {"l": {"l": 0.0}},
        "rules_registry": {
            "CCH-001": {
                "id": "CCH-001",
                "input": "s",
                "output": "h",
                "dialects": ["attic"],
            }
        },
        "search_index": {"l ɡ": ["L1"]},
        "unigram_index": {"l": ["L1"]},
        "lexicon_map": {
            "L1": LexiconRecord(
                entry={"id": "L1", "headword": "λόγος", "ipa": "lóɡos", "dialect": "attic"},
                token_count=4,
            )
        },
        "ipa_index": {"lóɡos": ["L1"]},
    }


class TestSearchDependenciesLoader:
    def test_build_data_versions_uses_cached_loaders_and_semantic_rule_max(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        lexicon_calls = 0

        def fake_lexicon_document() -> dict[str, object]:
            nonlocal lexicon_calls
            lexicon_calls += 1
            return {
                "schema_version": "2.0.0",
                "_meta": {"last_updated": "2026-04-20T09:10:11Z"},
                "lemmas": [],
            }

        monkeypatch.setattr(api_main, "_load_lexicon_document", fake_lexicon_document)
        monkeypatch.setattr(
            api_main,
            "_load_distance_matrix_with_meta",
            lambda: (
                {},
                {
                    "version": "1.0.0",
                    "generated_at": "2026-04-20T09:10:11.123+00:00",
                },
            ),
        )
        monkeypatch.setattr(
            api_main,
            "get_rules_version",
            lambda _: {"older": "9.9.9", "newer": "10.0.0"},
        )

        versions = api_main._build_data_versions()

        assert lexicon_calls == 1
        assert versions.lexicon == "2.0.0"
        assert versions.lexicon_updated_at == "2026-04-20T09:10:11+00:00"
        assert versions.matrix == "1.0.0"
        assert versions.matrix_generated_at == "2026-04-20T09:10:11.123+00:00"
        assert versions.rules == "10.0.0"

    def test_build_data_versions_logs_expected_metadata_errors(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(
            api_main,
            "_load_lexicon_document",
            lambda: (_ for _ in ()).throw(ValueError("bad lexicon")),
        )
        monkeypatch.setattr(
            api_main,
            "_load_distance_matrix_with_meta",
            lambda: (_ for _ in ()).throw(ValueError("bad matrix")),
        )
        monkeypatch.setattr(
            api_main,
            "get_rules_version",
            lambda _: {"bad": "not-semver"},
        )
        caplog.set_level(logging.ERROR, logger="api.main")

        versions = api_main._build_data_versions()

        assert versions == api_main.DataVersions()
        assert "Failed to load lexicon data version metadata" in caplog.text
        assert "Failed to load matrix data version metadata" in caplog.text
        assert "Failed to load rules data version metadata" in caplog.text

    def test_load_search_dependencies_returns_named_tuple_with_named_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        td = _make_test_dependencies()

        monkeypatch.setattr(api_main, "load_lexicon_entries", lambda: td["lexicon"])
        monkeypatch.setattr(api_main, "_load_distance_matrix", lambda: td["matrix"])
        monkeypatch.setattr(api_main, "_load_rules_registry", lambda: td["rules_registry"])
        monkeypatch.setattr(api_main, "_load_search_index", lambda: td["search_index"])
        monkeypatch.setattr(api_main, "_load_unigram_index", lambda: td["unigram_index"])
        monkeypatch.setattr(api_main, "_load_lexicon_map", lambda: td["lexicon_map"])
        monkeypatch.setattr(api_main, "_load_ipa_index", lambda: td["ipa_index"])
        monkeypatch.setattr(
            api_main,
            "_build_data_versions",
            lambda: api_main.DataVersions(rules="10.0.0"),
        )

        deps = api_main._load_search_dependencies()

        assert isinstance(deps, api_main.SearchDependencies)
        assert deps.lexicon == td["lexicon"]
        assert deps.matrix == td["matrix"]
        assert deps.rules_registry == td["rules_registry"]
        assert deps.search_index == td["search_index"]
        assert deps.unigram_index == td["unigram_index"]
        assert deps.lexicon_map == td["lexicon_map"]
        assert deps.ipa_index == td["ipa_index"]
        assert deps.data_versions.rules == "10.0.0"

    def test_warm_search_dependencies_logs_info_without_traceback_when_not_ready(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        detail = api_main._SEARCH_DEPENDENCIES_LEXICON_NOT_READY_DETAIL

        def fail_dependencies() -> api_main.SearchDependencies:
            raise api_main.SearchDependenciesNotReadyError(detail)

        monkeypatch.setattr(api_main, "_load_search_dependencies", fail_dependencies)
        caplog.set_level(logging.INFO, logger="api.main")

        api_main._warm_search_dependencies()

        assert "Background search warmup skipped; dependencies not ready" in caplog.text
        assert detail in caplog.text
        assert "Traceback" not in caplog.text


class TestDocumentationAndCors:
    def test_docs_route_is_gated_off_by_default(self, client: TestClient) -> None:
        """/docs must stay hidden unless the operator opts in."""
        response = client.get("/docs")

        assert response.status_code == 404

    def test_openapi_schema_route_is_gated_off_by_default(
        self, client: TestClient
    ) -> None:
        """/openapi.json must stay hidden unless the operator opts in."""
        response = client.get("/openapi.json")

        assert response.status_code == 404

    def test_env_flag_enabled_reads_docs_env_variable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The docs env var is parsed by the shared flag helper."""
        # The FastAPI app is instantiated at import time, so route-level docs
        # exposure cannot be changed here; this only verifies _env_flag_enabled
        # reads _ENABLE_API_DOCS_ENV_VAR correctly.
        monkeypatch.setenv(api_main._ENABLE_API_DOCS_ENV_VAR, "1")
        assert api_main._env_flag_enabled(api_main._ENABLE_API_DOCS_ENV_VAR) is True

        monkeypatch.delenv(api_main._ENABLE_API_DOCS_ENV_VAR, raising=False)
        assert api_main._env_flag_enabled(api_main._ENABLE_API_DOCS_ENV_VAR) is False

    def test_allowed_origins_warns_when_env_is_unset(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.delenv("PROTEUS_ALLOWED_ORIGINS", raising=False)

        with caplog.at_level(logging.WARNING):
            allowed_origins = api_main._get_allowed_origins()

        assert allowed_origins == []
        assert "_get_allowed_origins found PROTEUS_ALLOWED_ORIGINS unset" in caplog.text
        assert "may block cross-origin requests" in caplog.text

    def test_allowed_origins_are_loaded_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "PROTEUS_ALLOWED_ORIGINS",
            " https://example.com, https://app.example.com , ,",
        )

        assert api_main._get_allowed_origins() == [
            "https://example.com",
            "https://app.example.com",
        ]

    def test_search_preflight_rejects_unconfigured_origin(
        self, client: TestClient
    ) -> None:
        response = client.options(
            "/search",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "POST",
            },
        )

        assert response.status_code == 400
        assert "access-control-allow-origin" not in response.headers
        assert "POST" in response.headers["access-control-allow-methods"]



def mock_search_dependencies(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Stub all search dependencies and return a capture dict."""
    td = _make_test_dependencies()
    captured: dict[str, object] = {}

    monkeypatch.setattr(api_main, "_load_lexicon_entries", lambda: td["lexicon"])
    monkeypatch.setattr(api_main, "_load_distance_matrix", lambda: td["matrix"])
    monkeypatch.setattr(api_main, "_load_rules_registry", lambda: td["rules_registry"])
    monkeypatch.setattr(api_main, "to_ipa", lambda query, dialect="attic": "loɡos")
    monkeypatch.setattr(api_main, "_load_search_index", lambda: td["search_index"])
    monkeypatch.setattr(api_main, "_load_unigram_index", lambda: td["unigram_index"])
    monkeypatch.setattr(api_main, "_load_lexicon_map", lambda: td["lexicon_map"])
    monkeypatch.setattr(api_main, "_load_ipa_index", lambda: td["ipa_index"])
    monkeypatch.setattr(
        api_main.phonology_search,
        "search_execution",
        _make_fake_search_execution(captured),
    )
    return captured


class TestSearchEndpoint:
    def test_search_uses_named_search_dependencies_fields(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        td = _make_test_dependencies()
        captured: dict[str, object] = {}

        monkeypatch.setattr(
            api_main,
            "_load_search_dependencies",
            lambda: api_main.SearchDependencies(
                lexicon=td["lexicon"],
                matrix=td["matrix"],
                rules_registry=td["rules_registry"],
                search_index=td["search_index"],
                unigram_index=td["unigram_index"],
                lexicon_map=td["lexicon_map"],
                ipa_index=td["ipa_index"],
                data_versions=api_main.DataVersions(),
            ),
        )
        monkeypatch.setattr(api_main, "to_ipa", lambda query, dialect="attic": "loɡos")

        monkeypatch.setattr(
            api_main.phonology_search,
            "search_execution",
            _make_fake_search_execution(captured),
        )

        response = client.post(
            "/search",
            json={"query_form": "λόγος", "dialect_hint": "attic", "max_candidates": 3},
        )

        assert response.status_code == 200
        assert captured["query"] == "λόγος"
        assert captured["lexicon"] is td["lexicon"]
        assert captured["matrix"] is td["matrix"]
        assert captured["max_results"] == 3
        assert captured["dialect"] == "attic"
        assert captured["index"] is td["search_index"]
        assert captured["unigram_index"] is td["unigram_index"]
        assert captured["prebuilt_lexicon_map"] is td["lexicon_map"]
        assert captured["query_ipa"] == "loɡos"
        assert captured["similarity_fallback_limit"] == 2000
        assert captured["unigram_fallback_limit"] == 2000
        assert captured["prebuilt_ipa_index"] is td["ipa_index"]
        assert response.json()["hits"][0]["rules_applied"][0]["rule_id"] == "CCH-001"

    def test_search_accepts_public_request_shape_and_returns_hits(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = mock_search_dependencies(monkeypatch)

        response = client.post(
            "/search",
            json={"query_form": "  λόγος  ", "dialect_hint": "attic", "max_candidates": 3},
        )

        assert response.status_code == 200
        assert captured["query"] == "λόγος"
        assert captured["max_results"] == 3
        assert captured["dialect"] == "attic"
        assert captured["index"] == {"l ɡ": ["L1"]}
        assert captured["unigram_index"] == {"l": ["L1"]}
        assert captured["prebuilt_ipa_index"] == {"lóɡos": ["L1"]}
        assert captured["similarity_fallback_limit"] == 2000
        assert captured["unigram_fallback_limit"] == 2000
        assert captured["prebuilt_lexicon_map"] == {
            "L1": LexiconRecord(
                entry={"id": "L1", "headword": "λόγος", "ipa": "lóɡos", "dialect": "attic"},
                token_count=4,
            )
        }
        assert captured["lexicon"] == (
            {"id": "L1", "headword": "λόγος", "ipa": "lóɡos", "dialect": "attic"},
        )
        payload = response.json()
        assert payload["query"] == "λόγος"
        assert payload["query_ipa"] == "loɡos"
        assert payload["query_mode"] == "Full-form"
        assert len(payload["hits"]) == 1
        assert payload["hits"][0]["headword"] == "λόγος"
        assert payload["hits"][0]["ipa"] == "lóɡos"
        assert payload["hits"][0]["distance"] == pytest.approx(0.25)
        assert payload["hits"][0]["confidence"] == pytest.approx(0.75)
        assert payload["hits"][0]["dialect_attribution"] == "lemma dialect: attic"
        assert payload["hits"][0]["alignment_visualization"] == ""
        assert payload["hits"][0]["match_type"] == "Rule-based"
        assert payload["hits"][0]["rule_support"] is True
        assert payload["hits"][0]["applied_rule_count"] == 1
        assert payload["hits"][0]["observed_change_count"] == 0
        assert payload["hits"][0]["alignment_summary"] == "1 matched rule across 1 position."
        assert payload["hits"][0]["why_candidate"] == [
            "1 explicit rule supports the match.",
            "Moderate phonological similarity.",
            "No fallback edits required.",
        ]
        assert payload["hits"][0]["uncertainty"] == "Medium"
        assert payload["hits"][0]["candidate_bucket"] == "Supported"
        assert payload["hits"][0]["rules_applied"] == [
            {
                "rule_id": "CCH-001",
                "rule_name": "CCH-001",
                "rule_name_en": "",
                "from_phone": "s",
                "to_phone": "h",
                "position": 2,
            }
        ]
        assert "Applied rules:" in payload["hits"][0]["explanation"]
        assert "CCH-001" in payload["hits"][0]["explanation"]
        assert "distance 0.250" in payload["hits"][0]["explanation"]

    def test_search_marks_short_query_distance_only_hits_as_exploratory(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_search_dependencies(monkeypatch)
        monkeypatch.setattr(api_main, "to_ipa", lambda query, dialect="attic": "nyn")
        monkeypatch.setattr(
            api_main.phonology_search,
            "search_execution",
            lambda *args, **kwargs: search_module.SearchExecutionResult(
                results=[
                    SearchResult(
                        lemma="ἄγνυον",
                        confidence=0.72,
                        dialect_attribution="lemma dialect: attic",
                        applied_rules=[],
                        ipa="aɡnyon",
                    )
                ],
                truncated=False,
            ),
        )
        monkeypatch.setattr(api_hit_formatting, "explain_alignment", lambda **kwargs: FakeExplanation())
        monkeypatch.setattr(api_hit_formatting, "to_prose", lambda explanation: "short-query exploratory")

        response = client.post("/search", json={"query_form": "νυν"})

        assert response.status_code == 200
        payload = response.json()
        assert payload["query_mode"] == "Short-query"
        assert payload["hits"][0]["match_type"] == "Distance-only"
        assert payload["hits"][0]["candidate_bucket"] == "Exploratory"

    def test_search_sets_api_fallback_caps_for_short_queries(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = mock_search_dependencies(monkeypatch)

        response = client.post("/search", json={"query_form": "νυν"})

        assert response.status_code == 200
        assert response.json()["query_mode"] == "Short-query"
        assert captured["similarity_fallback_limit"] == 2000
        assert captured["unigram_fallback_limit"] == 2000

    def test_search_sets_short_query_caps_for_decomposed_unicode(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = mock_search_dependencies(monkeypatch)
        query_form = unicodedata.normalize("NFD", "νῦν")

        response = client.post("/search", json={"query_form": query_form})

        assert response.status_code == 200
        assert response.json()["query_mode"] == "Short-query"
        assert captured["similarity_fallback_limit"] == 2000
        assert captured["unigram_fallback_limit"] == 2000

    def test_search_sets_api_fallback_caps_for_partial_queries(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = mock_search_dependencies(monkeypatch)

        response = client.post("/search", json={"query_form": "ζηταω-"})

        assert response.status_code == 200
        assert response.json()["query_mode"] == "Partial-form"
        assert captured["similarity_fallback_limit"] == 2000
        assert captured["unigram_fallback_limit"] == 2000

    def test_search_sets_api_fallback_caps_for_fullform_queries(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = mock_search_dependencies(monkeypatch)

        response = client.post("/search", json={"query_form": "λόγος"})

        assert response.status_code == 200
        assert response.json()["query_mode"] == "Full-form"
        assert captured["similarity_fallback_limit"] == 2000
        assert captured["unigram_fallback_limit"] == 2000

    def test_search_marks_partial_query_distance_only_hits_as_exploratory(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_search_dependencies(monkeypatch)
        captured: dict[str, object] = {}

        def fake_to_ipa(query: str, dialect: str = "attic") -> str:
            captured["query"] = query
            return "zɛːtaɔ"

        monkeypatch.setattr(api_main, "to_ipa", fake_to_ipa)
        monkeypatch.setattr(
            api_main.phonology_search,
            "search_execution",
            lambda *args, **kwargs: search_module.SearchExecutionResult(
                results=[
                    SearchResult(
                        lemma="ζητέω",
                        confidence=0.72,
                        dialect_attribution="lemma dialect: attic",
                        applied_rules=[],
                        ipa="zɛːtɛɔ",
                    )
                ],
                truncated=False,
            ),
        )
        monkeypatch.setattr(api_hit_formatting, "explain_alignment", lambda **kwargs: FakeExplanation())
        monkeypatch.setattr(api_hit_formatting, "to_prose", lambda explanation: "partial-query exploratory")

        response = client.post("/search", json={"query_form": "ζηταω-"})

        assert response.status_code == 200
        payload = response.json()
        assert captured["query"] == "ζηταω"
        assert payload["query_ipa"] == "zɛːtaɔ"
        assert payload["query_mode"] == "Partial-form"
        assert payload["hits"][0]["match_type"] == "Distance-only"
        assert payload["hits"][0]["candidate_bucket"] == "Exploratory"

    @pytest.mark.parametrize(
        ("query_form", "expected_queries", "expected_ipa"),
        [
            ("*λόγ", ["λόγ"], "loɡ"),
            ("λ*γ", ["λ", "γ"], "l ɡ"),
            ("α*ι", ["α", "ι"], "a i"),
            # Unicode dash suffix markers (en dash, em dash, fullwidth hyphen)
            # must be treated identically to ASCII hyphen — users often paste
            # these from word processors or type them via locale keyboards.
            ("λόγ\u2013", ["λόγ"], "loɡ"),  # U+2013 EN DASH
            ("λόγ\u2014", ["λόγ"], "loɡ"),  # U+2014 EM DASH
            ("λόγ\uFF0D", ["λόγ"], "loɡ"),  # U+FF0D FULLWIDTH HYPHEN-MINUS
        ],
    )
    def test_search_normalizes_suffix_and_infix_partial_queries(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        query_form: str,
        expected_queries: list[str],
        expected_ipa: str,
    ) -> None:
        captured_search = mock_search_dependencies(monkeypatch)
        captured: list[str] = []

        def fake_to_ipa(query: str, dialect: str = "attic") -> str:
            captured.append(query)
            return {
                "λόγ": "loɡ",
                "λ": "l",
                "γ": "ɡ",
                "α": "a",
                "ι": "i",
            }.get(query, query)

        monkeypatch.setattr(api_main, "to_ipa", fake_to_ipa)
        monkeypatch.setattr(api_hit_formatting, "to_prose", lambda explanation: "partial-query exploratory")

        response = client.post("/search", json={"query_form": query_form})

        assert response.status_code == 200
        payload = response.json()
        assert payload["query_mode"] == "Partial-form"
        assert payload["query_ipa"] == expected_ipa
        assert captured == expected_queries
        assert captured_search["query_ipa"] == expected_ipa

    @pytest.mark.parametrize("query_form", ["*", "-", "-*", "a*b*c"])
    def test_search_rejects_invalid_partial_query_syntax(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, query_form: str
    ) -> None:
        mock_search_dependencies(monkeypatch)
        monkeypatch.setattr(
            api_main,
            "to_ipa",
            lambda *_args, **_kwargs: pytest.fail("to_ipa should not run"),
        )
        monkeypatch.setattr(
            api_main.phonology_search,
            "search_execution",
            lambda *_args, **_kwargs: pytest.fail("search_execution should not run"),
        )

        response = client.post("/search", json={"query_form": query_form})

        assert response.status_code == 400
        assert response.json()["detail"] == "Invalid search query"

    def test_search_rejects_dialect_not_supported_by_profile(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        td = _make_test_dependencies()
        profile = replace(
            get_default_language_profile(),
            supported_dialects=("attic",),
        )
        monkeypatch.setattr(
            api_main,
            "_load_search_dependencies",
            lambda: api_main.SearchDependencies(
                lexicon=td["lexicon"],
                matrix=td["matrix"],
                rules_registry=td["rules_registry"],
                search_index=td["search_index"],
                unigram_index=td["unigram_index"],
                lexicon_map=td["lexicon_map"],
                ipa_index=td["ipa_index"],
                data_versions=api_main.DataVersions(),
                profile=profile,
            ),
        )

        response = client.post(
            "/search",
            json={"query_form": "λόγος", "dialect_hint": "koine"},
        )

        assert response.status_code == 400
        assert response.json()["detail"] == (
            "Invalid dialect_hint 'koine'; expected one of ('attic')"
        )

    def test_search_uses_profile_default_dialect_and_phone_inventory(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        isolated_language_registry: None,
    ) -> None:
        td = _make_test_dependencies()
        rules_dir = tmp_path / "rules"
        matrix_dir = tmp_path / "matrices"
        rules_dir.mkdir()
        matrix_dir.mkdir()
        profile = LanguageProfile(
            language_id="toy_api",
            display_name="Toy API",
            default_dialect="toy",
            supported_dialects=("toy",),
            converter=lambda text, *, dialect: text if dialect == "toy" else "",
            phone_inventory=("ts", "p", "a"),
            lexicon_path=tmp_path / "lexicon.json",
            matrix_path=matrix_dir / "matrix.json",
            rules_dir=rules_dir,
        )
        captured: dict[str, object] = {}
        monkeypatch.setattr(
            api_main.phonology_search,
            "search_execution",
            _make_fake_search_execution(captured),
        )
        monkeypatch.setattr(
            api_main,
            "_load_search_dependencies",
            lambda _language: api_main.SearchDependencies(
                lexicon=td["lexicon"],
                matrix=td["matrix"],
                rules_registry=td["rules_registry"],
                search_index=td["search_index"],
                unigram_index=td["unigram_index"],
                lexicon_map=td["lexicon_map"],
                ipa_index=td["ipa_index"],
                data_versions=api_main.DataVersions(),
                profile=profile,
            ),
        )
        register_language_profile(profile)
        response = client.post(
            "/search",
            json={"query_form": "pa", "language": "toy_api"},
        )

        assert response.status_code == 200
        assert captured["dialect"] == "toy"
        assert captured["language"] == "toy_api"
        assert captured["phone_inventory"] == ("ts", "p", "a")

    def test_search_returns_empty_hits_when_short_query_quality_filter_drops_candidates(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_search_dependencies(monkeypatch)
        monkeypatch.setattr(api_main, "to_ipa", lambda query, dialect="attic": "nyn")
        monkeypatch.setattr(
            api_main.phonology_search,
            "search_execution",
            lambda *args, **kwargs: search_module.SearchExecutionResult(results=[], truncated=False),
        )

        response = client.post("/search", json={"query_form": "νυν"})

        assert response.status_code == 200
        payload = response.json()
        assert payload["query_mode"] == "Short-query"
        assert payload["query_ipa"] == "nyn"
        assert payload["hits"] == []
        assert payload["truncated"] is False

    def test_search_returns_truncated_true_for_empty_incomplete_short_query_results(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_search_dependencies(monkeypatch)
        monkeypatch.setattr(api_main, "to_ipa", lambda query, dialect="attic": "nyn")
        monkeypatch.setattr(
            api_main.phonology_search,
            "search_execution",
            lambda *args, **kwargs: search_module.SearchExecutionResult(results=[], truncated=True),
        )

        response = client.post("/search", json={"query_form": "νυν"})

        assert response.status_code == 200
        payload = response.json()
        assert payload["hits"] == []
        assert payload["truncated"] is True

    def test_search_short_query_keeps_exact_or_rule_supported_candidates(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_search_dependencies(monkeypatch)
        monkeypatch.setattr(api_main, "to_ipa", lambda query, dialect="attic": "nyn")
        monkeypatch.setattr(
            api_main.phonology_search,
            "search_execution",
            lambda *args, **kwargs: search_module.SearchExecutionResult(
                results=[
                    SearchResult(
                        lemma="νυν",
                        confidence=0.40,
                        dialect_attribution="lemma dialect: attic",
                        applied_rules=["RULE-001"],
                        rule_applications=[
                            RuleApplication(
                                rule_id="RULE-001",
                                rule_name="Rule 001",
                                from_phone="y",
                                to_phone="u",
                                position=1,
                            )
                        ],
                        ipa="nyn",
                    )
                ],
                truncated=True,
            ),
        )
        monkeypatch.setattr(api_hit_formatting, "to_prose", lambda explanation: "short-query supported")

        response = client.post("/search", json={"query_form": "νυν"})

        assert response.status_code == 200
        payload = response.json()
        assert payload["query_mode"] == "Short-query"
        assert payload["hits"][0]["headword"] == "νυν"
        assert payload["hits"][0]["candidate_bucket"] == "Supported"
        assert payload["truncated"] is True

    def test_search_accepts_koine_dialect_hint_and_returns_koine_rule(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = mock_search_dependencies(monkeypatch)
        monkeypatch.setattr(
            api_main,
            "to_ipa",
            lambda query, dialect="attic": "loɣos" if dialect == "koine" else "loɡos",
        )
        monkeypatch.setattr(
            api_main,
            "_load_rules_registry",
            lambda: {
                "CCH-009": {
                    "id": "CCH-009",
                    "name_ja": "コイネー期の γ 摩擦音化",
                    "name_en": "Koine spirantization: gamma",
                    "input": "ɡ",
                    "output": "ɣ",
                    "dialects": ["koine"],
                }
            },
        )
        monkeypatch.setattr(
            api_main.phonology_search,
            "search_execution",
            lambda *args, **kwargs: captured.update(
                {"dialect": kwargs.get("dialect", args[4] if len(args) > 4 else None)}
            )
            or search_module.SearchExecutionResult(
                results=[
                    SearchResult(
                        lemma="λόγος",
                        confidence=0.8,
                        dialect_attribution="lemma dialect: attic; query-compatible dialects: koine",
                        applied_rules=["CCH-009"],
                        rule_applications=[
                            RuleApplication(
                                rule_id="CCH-009",
                                rule_name="コイネー期の γ 摩擦音化",
                                rule_name_en="Koine spirantization: gamma",
                                from_phone="ɡ",
                                to_phone="ɣ",
                                position=2,
                                dialects=["koine"],
                            )
                        ],
                        ipa="lóɡos",
                    )
                ],
                truncated=False,
            ),
        )

        response = client.post(
            "/search",
            json={"query_form": "λόγος", "dialect_hint": "koine", "max_candidates": 3},
        )

        assert response.status_code == 200
        assert captured["dialect"] == "koine"
        payload = response.json()
        assert payload["query_ipa"] == "loɣos"
        assert payload["query_mode"] == "Full-form"
        assert payload["hits"][0]["match_type"] == "Rule-based"
        assert payload["hits"][0]["rule_support"] is True
        assert payload["hits"][0]["applied_rule_count"] == 1
        assert payload["hits"][0]["uncertainty"] == "Low"
        assert payload["hits"][0]["candidate_bucket"] == "Supported"
        assert payload["hits"][0]["rules_applied"][0]["rule_id"] == "CCH-009"
        assert payload["hits"][0]["dialect_attribution"] == (
            "lemma dialect: attic; query-compatible dialects: koine"
        )

    def test_search_uses_contextual_rule_ids_and_matching_distance_in_api_response(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            api_main,
            "_load_lexicon_entries",
            lambda: (
                {
                    "id": "L1",
                    "headword": "χώρα",
                    "ipa": "rɛː",
                    "dialect": "attic",
                },
            ),
        )
        monkeypatch.setattr(
            api_main,
            "_load_distance_matrix",
            lambda: {"ɛː": {"aː": 0.1}},
        )
        monkeypatch.setattr(api_main, "_load_search_index", lambda: {})
        monkeypatch.setattr(
            api_main,
            "_load_rules_registry",
            lambda: {
                "VSH-010": {
                    "id": "VSH-010",
                    "name_ja": "アッティカ方言の e・i・r 後における長母音 ā 保持",
                    "name_en": "Attic retention of long alpha after e, i, or r",
                    "input": "ɛː",
                    "output": "aː",
                    "dialects": ["attic"],
                }
            },
        )
        monkeypatch.setattr(api_main, "to_ipa", lambda query, dialect="attic": "raː")
        monkeypatch.setattr(
            api_main.phonology_search,
            "to_ipa",
            lambda query, dialect="attic": "raː",
        )

        response = client.post(
            "/search",
            json={"query_form": "χωρα", "dialect_hint": "attic", "max_candidates": 5},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["query_mode"] == "Full-form"
        assert payload["hits"][0]["rules_applied"] == [
            {
                "rule_id": "VSH-010",
                "rule_name": "アッティカ方言の e・i・r 後における長母音 ā 保持",
                "rule_name_en": "Attic retention of long alpha after e, i, or r",
                "from_phone": "ɛː",
                "to_phone": "aː",
                "position": 1,
            }
        ]
        assert payload["hits"][0]["match_type"] == "Rule-based"
        assert payload["hits"][0]["alignment_summary"] == "1 matched rule across 1 position."
        assert payload["hits"][0]["distance"] == pytest.approx(0.275)
        assert "distance 0.275" in payload["hits"][0]["explanation"]

    def test_search_hit_falls_back_to_explain_alignment_when_rule_details_are_missing(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}

        monkeypatch.setattr(
            api_main,
            "_load_lexicon_entries",
            lambda: (
                {
                    "id": "L1",
                    "headword": "λόγος",
                    "ipa": "lóɡos",
                    "dialect": "attic",
                },
            ),
        )
        monkeypatch.setattr(api_main, "_load_distance_matrix", lambda: {"l": {"l": 0.0}})
        monkeypatch.setattr(api_main, "_load_rules_registry", lambda: {})
        monkeypatch.setattr(api_main, "_load_search_index", lambda: {"l ɡ": ["L1"]})
        monkeypatch.setattr(api_main, "to_ipa", lambda query, dialect="attic": "loɡos")
        monkeypatch.setattr(
            api_main.phonology_search,
            "search_execution",
            lambda *args, **kwargs: search_module.SearchExecutionResult(
                results=[
                    SearchResult(
                        lemma="λόγος",
                        confidence=0.75,
                        dialect_attribution="lemma dialect: attic",
                        applied_rules=["CCH-001"],
                        ipa="lóɡos",
                    )
                ],
                truncated=False,
            ),
        )

        def fake_explain_alignment(**kwargs: object) -> FakeExplanation:
            captured.update(kwargs)
            return FakeExplanation(
                steps=[
                    RuleApplication(
                        rule_id="CCH-001",
                        rule_name="Fallback Rule",
                        from_phone="s",
                        to_phone="h",
                        position=-1,
                    )
                ]
            )

        monkeypatch.setattr(api_hit_formatting, "explain_alignment", fake_explain_alignment)
        monkeypatch.setattr(api_hit_formatting, "to_prose", lambda explanation: "fallback explanation")

        response = client.post("/search", json={"query_form": "λόγος"})

        assert response.status_code == 200
        assert captured["rule_ids"] == ["CCH-001"]
        payload = response.json()["hits"][0]
        assert payload["rules_applied"][0]["position"] == -1
        assert payload["match_type"] == "Rule-based"
        assert payload["rule_support"] is True
        assert payload["applied_rule_count"] == 1
        assert payload["candidate_bucket"] == "Supported"
        assert payload["alignment_summary"] == "1 matched rule at an unknown position."

    def test_search_hit_prefers_observed_rule_applications_over_fallback_ids(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            api_main,
            "_load_lexicon_entries",
            lambda: (
                {
                    "id": "L1",
                    "headword": "λόγος",
                    "ipa": "a",
                    "dialect": "attic",
                },
            ),
        )
        monkeypatch.setattr(api_main, "_load_distance_matrix", lambda: {"a": {"x": 0.5}})
        monkeypatch.setattr(api_main, "_load_rules_registry", lambda: {})
        monkeypatch.setattr(api_main, "_load_search_index", lambda: {"l ɡ": ["L1"]})
        monkeypatch.setattr(api_main, "to_ipa", lambda query, dialect="attic": "x")
        monkeypatch.setattr(
            api_main.phonology_search,
            "search",
            lambda *args, **kwargs: [
                SearchResult(
                    lemma="λόγος",
                    confidence=0.5,
                    dialect_attribution="lemma dialect: attic",
                    applied_rules=[],
                    rule_applications=[
                        RuleApplication(
                            rule_id="OBS-SUB",
                            rule_name="観測された置換",
                            rule_name_en="Observed substitution",
                            from_phone="a",
                            to_phone="x",
                            position=0,
                        )
                    ],
                    ipa="a",
                )
            ],
        )

        def fail_explain_alignment(**_kwargs: object) -> object:
            raise AssertionError("explain_alignment should not run when rule_applications are present")

        monkeypatch.setattr(api_hit_formatting, "explain_alignment", fail_explain_alignment)

        response = client.post("/search", json={"query_form": "λόγος"})

        assert response.status_code == 200
        payload = response.json()["hits"][0]
        assert payload["rules_applied"] == [
            {
                "rule_id": "OBS-SUB",
                "rule_name": "観測された置換",
                "rule_name_en": "Observed substitution",
                "from_phone": "a",
                "to_phone": "x",
                "position": 0,
            }
        ]
        assert payload["match_type"] == "Low-confidence"
        assert payload["rule_support"] is False
        assert payload["applied_rule_count"] == 0
        assert payload["observed_change_count"] == 1
        assert payload["candidate_bucket"] == "Exploratory"

    def test_search_accepts_legacy_request_shape(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = mock_search_dependencies(monkeypatch)

        response = client.post(
            "/search",
            json={"query": "λόγος", "dialect": "attic", "max_results": 2},
        )

        assert response.status_code == 200
        assert captured["query"] == "λόγος"
        assert captured["max_results"] == 2
        assert response.json()["query"] == "λόγος"
        assert response.json()["query_mode"] == "Full-form"

    def test_search_returns_bad_request_for_invalid_query_runtime(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_search_dependencies(monkeypatch)

        def fake_to_ipa(query: str, dialect: str = "attic") -> str:
            raise ValueError("query must be a non-empty string")

        monkeypatch.setattr(
            api_main,
            "to_ipa",
            fake_to_ipa,
        )

        response = client.post(
            "/search",
            json={"query_form": "λόγος", "dialect_hint": "attic"},
        )

        assert response.status_code == 400
        assert response.json() == {"detail": "Invalid search query"}

    def test_search_error_logs_redacted_query_by_default(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        mock_search_dependencies(monkeypatch)
        _install_invalid_query_to_ipa(monkeypatch)
        monkeypatch.delenv(api_main._LOG_RAW_QUERY_ENV_VAR, raising=False)
        caplog.set_level(logging.DEBUG, logger="api.main")

        query = "λόγος"
        response = client.post(
            "/search",
            json={"query_form": query, "dialect_hint": "attic"},
        )

        assert response.status_code == 400
        assert query not in caplog.text
        assert api_main._summarize_query_for_logs(query) in caplog.text

    @pytest.mark.parametrize(
        ("env_value", "expect_raw_query"),
        [
            ("", False),
            ("0", False),
            ("false", False),
            ("FALSE", False),
            ("1", True),
            ("TRUE", True),
            ("true", True),
        ],
    )
    def test_search_debug_logs_raw_query_based_on_opt_in_env_value(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        env_value: str,
        expect_raw_query: bool,
    ) -> None:
        mock_search_dependencies(monkeypatch)
        _install_invalid_query_to_ipa(monkeypatch)
        monkeypatch.setenv(api_main._LOG_RAW_QUERY_ENV_VAR, env_value)
        caplog.set_level(logging.DEBUG, logger="api.main")

        query = "λόγος"
        response = client.post(
            "/search",
            json={"query_form": query, "dialect_hint": "attic"},
        )

        assert response.status_code == 400
        summary = api_main._summarize_query_for_logs(query)
        assert summary in caplog.text
        debug_message = f"Full ValueError details for query '{query}'"
        if expect_raw_query:
            assert debug_message in caplog.text
        else:
            assert debug_message not in caplog.text
            assert query not in caplog.text

    def test_search_returns_503_with_lexicon_setup_guidance_when_lexicon_loader_fails(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail_lexicon_entries() -> tuple[dict[str, object], ...]:
            raise ValueError("lexicon unavailable")

        monkeypatch.setattr(api_main, "_load_lexicon_entries", fail_lexicon_entries)

        response = client.post(
            "/search",
            json={"query_form": "λόγος", "dialect_hint": "attic"},
        )

        assert response.status_code == 503
        assert response.json()["detail"] == api_main._SEARCH_DEPENDENCIES_LEXICON_NOT_READY_DETAIL

    def test_search_logs_warning_with_redacted_query_without_traceback_when_not_ready(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        def fail_lexicon_entries() -> tuple[dict[str, object], ...]:
            raise ValueError("lexicon unavailable")

        monkeypatch.setattr(api_main, "_load_lexicon_entries", fail_lexicon_entries)
        monkeypatch.delenv(api_main._LOG_RAW_QUERY_ENV_VAR, raising=False)
        caplog.set_level(logging.WARNING, logger="api.main")

        query = "λόγος"
        response = client.post(
            "/search",
            json={"query_form": query, "dialect_hint": "attic"},
        )

        assert response.status_code == 503
        assert "Search dependencies are not ready" in caplog.text
        assert api_main._SEARCH_DEPENDENCIES_LEXICON_NOT_READY_DETAIL in caplog.text
        assert api_main._summarize_query_for_logs(query) in caplog.text
        assert query not in caplog.text
        assert "Traceback" not in caplog.text

    def test_search_returns_generic_503_when_distance_matrix_loader_fails(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail_distance_matrix() -> dict[str, dict[str, float]]:
            raise ValueError("matrix unavailable")

        monkeypatch.setattr(
            api_main,
            "_load_distance_matrix",
            fail_distance_matrix,
        )

        response = client.post(
            "/search",
            json={"query_form": "λόγος", "dialect_hint": "attic"},
        )

        assert response.status_code == 503
        assert response.json() == {
            "detail": api_main._SEARCH_DEPENDENCIES_GENERIC_NOT_READY_DETAIL
        }

    def test_search_returns_generic_503_when_unigram_loader_fails(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_search_dependencies(monkeypatch)

        def fail_unigram_index() -> dict[str, list[str]]:
            raise ValueError("unigram unavailable")

        monkeypatch.setattr(api_main, "_load_unigram_index", fail_unigram_index)

        response = client.post(
            "/search",
            json={"query_form": "λόγος", "dialect_hint": "attic"},
        )

        assert response.status_code == 503
        assert response.json() == {
            "detail": api_main._SEARCH_DEPENDENCIES_GENERIC_NOT_READY_DETAIL
        }

    def test_search_returns_generic_503_when_lexicon_map_loader_fails(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_search_dependencies(monkeypatch)

        def fail_lexicon_map() -> dict[str, object]:
            raise ValueError("lexicon map unavailable")

        monkeypatch.setattr(api_main, "_load_lexicon_map", fail_lexicon_map)

        response = client.post(
            "/search",
            json={"query_form": "λόγος", "dialect_hint": "attic"},
        )

        assert response.status_code == 503
        assert response.json() == {
            "detail": api_main._SEARCH_DEPENDENCIES_GENERIC_NOT_READY_DETAIL
        }


class TestSearchValidation:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, "attic"),
            (" attic ", "attic"),
            (" Koine ", "koine"),
        ],
    )
    def test_normalize_dialect_hint_handles_none_and_str_values(
        self, value: object, expected: str
    ) -> None:
        assert api_main.SearchRequest._normalize_dialect_hint(value) == expected

    def test_normalize_dialect_hint_preserves_non_string_values(self) -> None:
        marker = object()

        assert api_main.SearchRequest._normalize_dialect_hint(marker) is marker

    def test_normalize_dialect_hint_accepts_profile_specific_string(self) -> None:
        assert api_main.SearchRequest._normalize_dialect_hint(" ionic ") == "ionic"

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, "en"),
            (" EN ", "en"),
            (" ja ", "ja"),
        ],
    )
    def test_normalize_lang_handles_none_and_str_values(
        self, value: object, expected: str
    ) -> None:
        assert api_main.SearchRequest._normalize_lang(value) == expected

    def test_normalize_lang_preserves_non_string_values(self) -> None:
        marker = object()

        assert api_main.SearchRequest._normalize_lang(marker) is marker

    def test_normalize_lang_rejects_unsupported_string(self) -> None:
        with pytest.raises(ValueError, match="lang"):
            api_main.SearchRequest._normalize_lang(" fr ")

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, "ancient_greek"),
            (" Ancient_Greek ", "ancient_greek"),
            ("   ", "ancient_greek"),
        ],
    )
    def test_normalize_language_handles_none_and_str_values(
        self, value: object, expected: str
    ) -> None:
        assert api_main.SearchRequest._normalize_language(value) == expected

    def test_normalize_language_rejects_non_string_values(self) -> None:
        with pytest.raises(ValueError, match="language must be a string"):
            api_main.SearchRequest._normalize_language(object())

    def test_search_rejects_blank_query_form(self, client: TestClient) -> None:
        response = client.post("/search", json={"query_form": "   "})

        assert response.status_code == 422
        assert "query_form" in response.text

    @pytest.mark.parametrize("max_candidates", [-1, 0, 101])
    def test_search_rejects_invalid_max_candidates(
        self, client: TestClient, max_candidates: int
    ) -> None:
        response = client.post(
            "/search",
            json={
                "query_form": "λόγος",
                "dialect_hint": "attic",
                "max_candidates": max_candidates,
            },
        )

        assert response.status_code == 422
        assert "max_candidates" in response.text

    def test_search_rejects_unsupported_dialect_hint(self, client: TestClient) -> None:
        response = client.post(
            "/search",
            json={"query_form": "λόγος", "dialect_hint": "ionic"},
        )

        assert response.status_code == 422
        assert "dialect_hint" in response.text

    @pytest.mark.parametrize("max_candidates", [1, 100])
    def test_search_accepts_boundary_max_candidates(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        max_candidates: int,
    ) -> None:
        mock_search_dependencies(monkeypatch)
        response = client.post(
            "/search",
            json={
                "query_form": "λόγος",
                "dialect_hint": "attic",
                "max_candidates": max_candidates,
            },
        )

        assert response.status_code == 200
        assert "max_candidates" not in response.text


class TestMatchHelpers:
    def test_is_observed_rule_step(self):
        assert api_hit_formatting._is_observed_rule_step(RuleApplication(rule_id="OBS-SUB", rule_name="", from_phone="", to_phone="", position=0)) is True
        assert api_hit_formatting._is_observed_rule_step(RuleApplication(rule_id="VSH-010", rule_name="", from_phone="", to_phone="", position=0)) is False

    def test_count_explicit_and_observed_steps(self):
        assert api_hit_formatting._count_explicit_and_observed_steps([]) == (0, 0)
        steps = [
            RuleApplication(rule_id="VSH-010", rule_name="", from_phone="", to_phone="", position=0),
            RuleApplication(rule_id="OBS-SUB", rule_name="", from_phone="", to_phone="", position=1)
        ]
        assert api_hit_formatting._count_explicit_and_observed_steps(steps) == (1, 1)

    def test_build_match_type(self):
        assert api_hit_formatting._build_match_type(source_ipa="a", query_ipa="a", steps=[], applied_rule_count=0, confidence=1.0) == "Exact"
        assert api_hit_formatting._build_match_type(source_ipa="nýn", query_ipa="nyn", steps=[], applied_rule_count=0, confidence=1.0) == "Exact"
        steps = [RuleApplication(rule_id="OBS-SUB", rule_name="", from_phone="", to_phone="", position=0)]
        assert api_hit_formatting._build_match_type(source_ipa="a", query_ipa="a", steps=steps, applied_rule_count=0, confidence=1.0) == "Distance-only"
        assert api_hit_formatting._build_match_type(source_ipa="a", query_ipa="b", steps=[], applied_rule_count=1, confidence=0.8) == "Rule-based"
        assert api_hit_formatting._build_match_type(source_ipa="a", query_ipa="b", steps=[], applied_rule_count=0, confidence=0.80) == "Distance-only"
        assert api_hit_formatting._build_match_type(source_ipa="a", query_ipa="b", steps=[], applied_rule_count=0, confidence=0.70) == "Distance-only"
        assert api_hit_formatting._build_match_type(source_ipa="a", query_ipa="b", steps=[], applied_rule_count=0, confidence=0.55) == "Distance-only"
        assert api_hit_formatting._build_match_type(source_ipa="a", query_ipa="b", steps=[], applied_rule_count=0, confidence=0.54) == "Low-confidence"

    def test_build_uncertainty(self):
        assert api_hit_formatting._build_uncertainty("Exact", applied_rule_count=0, confidence=1.0) == "Low"
        assert api_hit_formatting._build_uncertainty("Rule-based", applied_rule_count=1, confidence=0.80) == "Low"
        assert api_hit_formatting._build_uncertainty("Rule-based", applied_rule_count=1, confidence=0.79) == "Medium"
        assert api_hit_formatting._build_uncertainty("Rule-based", applied_rule_count=1, confidence=0.69) == "High"
        assert api_hit_formatting._build_uncertainty("Distance-only", applied_rule_count=0, confidence=0.70) == "Medium"
        assert api_hit_formatting._build_uncertainty("Distance-only", applied_rule_count=0, confidence=0.69) == "High"

    @pytest.mark.parametrize(
        ("match_type", "query_mode", "uncertainty", "expected"),
        [
            ("Exact", "Full-form", "Low", "Supported"),
            ("Rule-based", "Full-form", "Medium", "Supported"),
            ("Distance-only", "Full-form", "Low", "Supported"),
            ("Distance-only", "Full-form", "Medium", "Exploratory"),
            ("Distance-only", "Short-query", "Low", "Exploratory"),
            ("Distance-only", "Partial-form", "Low", "Exploratory"),
            ("Rule-based", "Short-query", "Medium", "Supported"),
            ("Low-confidence", "Partial-form", "High", "Exploratory"),
            ("Low-confidence", "Full-form", "Low", "Supported"),
            ("Low-confidence", "Full-form", "High", "Exploratory"),
        ],
    )
    def test_build_candidate_bucket(
        self,
        match_type: str,
        query_mode: str,
        uncertainty: str,
        expected: str,
    ) -> None:
        assert (
            api_hit_formatting._build_candidate_bucket(
                match_type,
                query_mode=query_mode,
                uncertainty=uncertainty,
            )
            == expected
        )

    def test_build_alignment_summary(self):
        assert api_hit_formatting._build_alignment_summary(source_ipa="a", query_ipa="a", steps=[]) == "No phonological difference."
        assert api_hit_formatting._build_alignment_summary(source_ipa="a", query_ipa="b", steps=[]) == "Differences visible in full alignment."
        explicit_step = RuleApplication(rule_id="VSH-010", rule_name="", from_phone="a", to_phone="b", position=0)
        assert api_hit_formatting._build_alignment_summary(source_ipa="a", query_ipa="b", steps=[explicit_step]) == "1 matched rule across 1 position."
        step = RuleApplication(rule_id="OBS-DEL", rule_name="", from_phone="a", to_phone="", position=0)
        assert api_hit_formatting._build_alignment_summary(source_ipa="a", query_ipa="b", steps=[step]) == "1 fallback edit across 1 position."
        steps_repeated = [
            RuleApplication(rule_id="OBS-DEL", rule_name="", from_phone="a", to_phone="", position=0),
            RuleApplication(rule_id="OBS-DEL", rule_name="", from_phone="b", to_phone="", position=0),
        ]
        assert api_hit_formatting._build_alignment_summary(source_ipa="a", query_ipa="b", steps=steps_repeated) == "2 deletions across 1 position."
        steps_distinct = [
            RuleApplication(rule_id="OBS-DEL", rule_name="", from_phone="a", to_phone="", position=0),
            RuleApplication(rule_id="OBS-SUB", rule_name="", from_phone="b", to_phone="c", position=1),
        ]
        assert api_hit_formatting._build_alignment_summary(source_ipa="a", query_ipa="b", steps=steps_distinct) == "1 deletion, 1 substitution across 2 positions."
        steps_mixed = [
            RuleApplication(rule_id="VSH-010", rule_name="", from_phone="a", to_phone="b", position=0),
            RuleApplication(rule_id="OBS-SUB", rule_name="", from_phone="c", to_phone="d", position=1),
        ]
        assert api_hit_formatting._build_alignment_summary(source_ipa="a", query_ipa="b", steps=steps_mixed) == "1 matched rule and 1 fallback edit across 2 positions."

    def test_count_distinct_positions_excludes_unknown_sentinel(self):
        step_unknown = RuleApplication(rule_id="OBS-DEL", rule_name="", from_phone="a", to_phone="", position=-1)
        assert api_hit_formatting._count_distinct_positions([step_unknown]) is None

        step_known = RuleApplication(rule_id="OBS-DEL", rule_name="", from_phone="a", to_phone="", position=0)
        assert api_hit_formatting._count_distinct_positions([step_known, step_unknown]) == 1

        assert api_hit_formatting._count_distinct_positions([]) == 0

    def test_build_alignment_summary_with_unknown_position(self):
        step_unknown = RuleApplication(rule_id="OBS-DEL", rule_name="", from_phone="a", to_phone="", position=-1)
        assert api_hit_formatting._build_alignment_summary(source_ipa="a", query_ipa="b", steps=[step_unknown]) == "1 fallback edit at an unknown position."

        unknown_pair = [
            RuleApplication(rule_id="OBS-DEL", rule_name="", from_phone="a", to_phone="", position=-1),
            RuleApplication(rule_id="OBS-SUB", rule_name="", from_phone="b", to_phone="c", position=-1),
        ]
        assert api_hit_formatting._build_alignment_summary(
            source_ipa="a", query_ipa="b", steps=unknown_pair
        ) == "1 deletion, 1 substitution at 2 unknown positions."

        step_known = RuleApplication(rule_id="VSH-010", rule_name="", from_phone="a", to_phone="b", position=0)
        assert api_hit_formatting._build_alignment_summary(
            source_ipa="a", query_ipa="b", steps=[step_known, step_unknown]
        ) == "1 matched rule and 1 fallback edit across 1 known position and 1 unknown position."

    def test_build_why_candidate(self):
        exact = api_hit_formatting._build_why_candidate("Exact", applied_rule_count=0, observed_change_count=0, confidence=1.0)
        assert exact == ["Exact phonological match.", "High phonological similarity.", "No remaining unexplained differences."]

        rule_based = api_hit_formatting._build_why_candidate("Rule-based", applied_rule_count=1, observed_change_count=0, confidence=0.8)
        assert rule_based == ["1 explicit rule supports the match.", "High phonological similarity.", "No fallback edits required."]

        observed = api_hit_formatting._build_why_candidate("Distance-only", applied_rule_count=0, observed_change_count=1, confidence=0.55)
        assert observed == ["Ranked by phonological distance without explicit rule support.", "Moderate phonological similarity.", "1 fallback edit remains uncatalogued."]

        weak = api_hit_formatting._build_why_candidate("Low-confidence", applied_rule_count=0, observed_change_count=2, confidence=0.54)
        assert weak == ["Ranked by phonological distance without explicit rule support.", "Weak phonological similarity.", "2 fallback edits remain uncatalogued."]

    def test_fallback_edit_label_order_en(self) -> None:
        """_FALLBACK_EDIT_LABELS order governs the summary output in EN; catch tuple regressions."""
        # deletion: from_phone set, to_phone empty
        # insertion: to_phone set, from_phone empty
        # substitution: both set
        deletion_step = RuleApplication(rule_id="OBS-DEL", rule_name="", from_phone="a", to_phone="", position=0)
        insertion_step = RuleApplication(rule_id="OBS-INS", rule_name="", from_phone="", to_phone="b", position=1)
        substitution_step = RuleApplication(rule_id="OBS-SUB", rule_name="", from_phone="c", to_phone="d", position=2)

        summary = api_hit_formatting._build_alignment_summary(
            source_ipa="a", query_ipa="bd", steps=[deletion_step, insertion_step, substitution_step], lang="en"
        )

        # deletion must come before insertion, insertion before substitution
        assert summary.index("deletion") < summary.index("insertion")
        assert summary.index("insertion") < summary.index("substitution")

    def test_fallback_edit_label_order_ja(self) -> None:
        """_FALLBACK_EDIT_LABELS order governs the summary output in JA; catch tuple regressions."""
        deletion_step = RuleApplication(rule_id="OBS-DEL", rule_name="", from_phone="a", to_phone="", position=0)
        insertion_step = RuleApplication(rule_id="OBS-INS", rule_name="", from_phone="", to_phone="b", position=1)
        substitution_step = RuleApplication(rule_id="OBS-SUB", rule_name="", from_phone="c", to_phone="d", position=2)

        summary = api_hit_formatting._build_alignment_summary(
            source_ipa="a", query_ipa="bd", steps=[deletion_step, insertion_step, substitution_step], lang="ja"
        )

        # The JA summary joins with "・" — verify the string contains all three labels in order
        deletion_label = api_hit_formatting._p("ja", "deletion_s")
        insertion_label = api_hit_formatting._p("ja", "insertion_s")
        substitution_label = api_hit_formatting._p("ja", "substitution_s")

        assert summary.index(deletion_label) < summary.index(insertion_label)
        assert summary.index(insertion_label) < summary.index(substitution_label)


class TestQueryLengthBound:
    def test_query_form_returns_nfc_normalized_value(self) -> None:
        query = unicodedata.normalize("NFD", "νῦν")

        request = SearchRequest(query_form=query)

        assert request.query_form == unicodedata.normalize("NFC", query)

    def test_query_form_rejects_payload_above_64_characters(
        self, client: TestClient
    ) -> None:
        """Unbounded queries would blow up the Needleman-Wunsch O(n^2) table."""
        payload = {"query_form": "α" * 65, "dialect_hint": "attic"}

        response = client.post("/search", json=payload)

        assert response.status_code == 422

    def test_query_form_accepts_payload_at_the_limit(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """64-character queries must still pass validation."""
        mock_search_dependencies(monkeypatch)
        monkeypatch.setattr(
            api_main.phonology_search,
            "search_execution",
            _make_fake_search_execution({}),
        )
        monkeypatch.setattr(
            api_main.phonology_search,
            "prepare_query_ipa",
            lambda *args, **kwargs: api_main.phonology_search.PreparedQueryIpa(
                query_mode="Full-form",
                normalized_query="",
                partial_query=None,
                query_ipa="",
                partial_query_tokens=None,
            ),
        )

        payload = {"query_form": "α" * 64, "dialect_hint": "attic"}

        response = client.post("/search", json=payload)

        assert response.status_code == 200

    def test_query_form_accepts_nfd_payload_at_nfc_limit(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """NFD combining marks must not make a 64-character NFC query invalid."""
        mock_search_dependencies(monkeypatch)
        monkeypatch.setattr(
            api_main.phonology_search,
            "search_execution",
            _make_fake_search_execution({}),
        )
        monkeypatch.setattr(
            api_main.phonology_search,
            "prepare_query_ipa",
            lambda *args, **kwargs: api_main.phonology_search.PreparedQueryIpa(
                query_mode="Full-form",
                normalized_query="",
                partial_query=None,
                query_ipa="",
                partial_query_tokens=None,
            ),
        )
        query = unicodedata.normalize("NFD", "ά" * 64)
        assert len(query) > 64
        assert len(unicodedata.normalize("NFC", query)) == 64

        response = client.post(
            "/search",
            json={"query_form": query, "dialect_hint": "attic"},
        )

        assert response.status_code == 200
        assert response.json()["query"] == unicodedata.normalize("NFC", query)


class TestSecurityHeaders:
    def test_security_headers_present_on_every_response(
        self, client: TestClient
    ) -> None:
        """Minimum hardening set flows through the custom middleware."""
        response = client.get("/health")

        assert response.status_code == 200
        assert response.headers.get("x-content-type-options") == "nosniff"
        assert response.headers.get("x-frame-options") == "DENY"
        assert (
            response.headers.get("referrer-policy")
            == "strict-origin-when-cross-origin"
        )
        assert "camera=()" in response.headers.get("permissions-policy", "")


class TestRuleRegistrySingleSource:
    def test_api_loader_delegates_to_search_registry(self) -> None:
        """The API's rule loader must share a single cached dict with search."""
        api_registry = api_main._load_rules_registry()
        search_registry = search_module.get_rules_registry("ancient_greek")

        assert api_registry is search_registry


class TestLoadAppVersion:
    def test_returns_env_var_value_when_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(api_main._APP_VERSION_ENV_VAR, "1.2.3-env")

        result = api_main._load_app_version()

        assert result == "1.2.3-env"

    def test_strips_whitespace_from_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(api_main._APP_VERSION_ENV_VAR, "  1.2.3  ")

        result = api_main._load_app_version()

        assert result == "1.2.3"

    def test_strips_leading_v_prefix_from_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(api_main._APP_VERSION_ENV_VAR, "v1.2.3")

        result = api_main._load_app_version()

        assert result == "1.2.3"

    def test_falls_back_to_metadata_version(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from importlib import metadata

        monkeypatch.delenv(api_main._APP_VERSION_ENV_VAR, raising=False)
        monkeypatch.setattr(metadata, "version", lambda _pkg: "2.0.0-meta")

        result = api_main._load_app_version()

        assert result == "2.0.0-meta"

    def test_falls_back_to_pyproject_toml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from importlib import metadata

        monkeypatch.delenv(api_main._APP_VERSION_ENV_VAR, raising=False)
        monkeypatch.setattr(
            metadata,
            "version",
            lambda _pkg: (_ for _ in ()).throw(metadata.PackageNotFoundError()),
        )

        pyproject_path = tmp_path / "pyproject.toml"
        pyproject_path.write_text(
            '[project]\nversion = "3.0.0-toml"\n',
            encoding="utf-8",
        )
        # Patch __file__ resolution to point to our tmp_path
        # _load_app_version resolves Path(__file__).resolve().parents[2] / "pyproject.toml"
        # We need to make that resolve to our tmp pyproject
        fake_file = tmp_path / "src" / "api" / "main.py"
        fake_file.parent.mkdir(parents=True, exist_ok=True)
        fake_file.touch()
        monkeypatch.setattr(api_main, "__file__", str(fake_file))

        result = api_main._load_app_version()

        assert result == "3.0.0-toml"

    def test_returns_unknown_when_all_sources_fail(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from importlib import metadata

        monkeypatch.delenv(api_main._APP_VERSION_ENV_VAR, raising=False)
        monkeypatch.setattr(
            metadata,
            "version",
            lambda _pkg: (_ for _ in ()).throw(metadata.PackageNotFoundError()),
        )
        # Point __file__ to a tmp dir with no pyproject.toml
        fake_file = tmp_path / "src" / "api" / "main.py"
        fake_file.parent.mkdir(parents=True, exist_ok=True)
        fake_file.touch()
        monkeypatch.setattr(api_main, "__file__", str(fake_file))

        result = api_main._load_app_version()

        assert result == "unknown"


class TestLoadAppVersionIntegration:
    def test_env_takes_priority_over_metadata(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(api_main._APP_VERSION_ENV_VAR, "env-version")

        assert api_main._load_app_version() == "env-version"

    def test_empty_env_skips_to_next_source(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from importlib import metadata

        monkeypatch.setenv(api_main._APP_VERSION_ENV_VAR, "   ")
        monkeypatch.setattr(metadata, "version", lambda _pkg: "9.8.7-mock")

        result = api_main._load_app_version()

        assert result == "9.8.7-mock"


class TestBuildDataVersionsPartialFailure:
    def test_lexicon_failure_still_loads_matrix_and_rules(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(
            api_main,
            "_load_lexicon_document",
            lambda: (_ for _ in ()).throw(OSError("lexicon read error")),
        )
        monkeypatch.setattr(
            api_main,
            "_load_distance_matrix_with_meta",
            lambda: ({}, {"version": "1.0.0", "generated_at": "2026-04-20T09:00:00.000+00:00"}),
        )
        monkeypatch.setattr(
            api_main,
            "get_rules_version",
            lambda _: {"vowels": "1.0.0"},
        )
        caplog.set_level(logging.ERROR, logger="api.main")

        versions = api_main._build_data_versions()

        assert versions.lexicon == "unknown"
        assert versions.matrix == "1.0.0"
        assert versions.rules == "1.0.0"
        assert "Failed to load lexicon data version metadata" in caplog.text

    def test_matrix_failure_still_loads_lexicon_and_rules(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(
            api_main,
            "_load_lexicon_document",
            lambda: {"schema_version": "2.0.0", "_meta": {}, "lemmas": []},
        )
        monkeypatch.setattr(
            api_main,
            "_load_distance_matrix_with_meta",
            lambda: (_ for _ in ()).throw(OSError("matrix read error")),
        )
        monkeypatch.setattr(
            api_main,
            "get_rules_version",
            lambda _: {"consonants": "1.0.0"},
        )
        caplog.set_level(logging.ERROR, logger="api.main")

        versions = api_main._build_data_versions()

        assert versions.lexicon == "2.0.0"
        assert versions.matrix == "unknown"
        assert versions.rules == "1.0.0"
        assert "Failed to load matrix data version metadata" in caplog.text

    def test_total_failure_returns_default_versions(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(
            api_main,
            "_load_lexicon_document",
            lambda: (_ for _ in ()).throw(ValueError("bad")),
        )
        monkeypatch.setattr(
            api_main,
            "_load_distance_matrix_with_meta",
            lambda: (_ for _ in ()).throw(ValueError("bad")),
        )
        monkeypatch.setattr(
            api_main,
            "get_rules_version",
            lambda _: (_ for _ in ()).throw(ValueError("bad")),
        )
        caplog.set_level(logging.ERROR, logger="api.main")

        versions = api_main._build_data_versions()

        assert versions == api_main.DataVersions()
        assert "Failed to load lexicon data version metadata" in caplog.text
        assert "Failed to load matrix data version metadata" in caplog.text
        assert "Failed to load rules data version metadata" in caplog.text


class TestCallWithLanguage:
    """Regression tests for the _call_with_language cache-key coalescing invariant."""

    def test_routes_default_language_to_zero_arg_cache_slot(self) -> None:
        """Default-language calls must reuse the zero-arg lru_cache slot.

        The _call_with_language wrapper is the seam that lets test fixtures
        patch ``api.main._load_lexicon_entries()`` (zero-arg) and have explicit
        ``language='ancient_greek'`` calls route to the same cache. Splitting
        the cache would silently bypass test patches and double memory usage.
        """
        default_language_id = api_main.get_default_language_profile().language_id
        call_count = 0

        @lru_cache(maxsize=8)
        def _mock_loader(language: str = default_language_id) -> int:
            nonlocal call_count
            call_count += 1
            return call_count

        result1 = api_main._call_with_language(_mock_loader, default_language_id)
        info_after_first = _mock_loader.cache_info()

        result2 = api_main._call_with_language(_mock_loader, default_language_id)
        info_after_second = _mock_loader.cache_info()

        assert info_after_first.misses == 1
        assert info_after_second.misses == 1, (
            "Explicit default language must reuse the zero-arg cache slot"
        )
        assert info_after_second.hits == 1
        assert result1 == result2

    def test_load_ipa_index_uses_default_lexicon_map_cache_slot(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Default IPA index loading must reuse the zero-arg lexicon-map slot."""
        default_language_id = api_main.get_default_language_profile().language_id
        call_count = 0

        @lru_cache(maxsize=8)
        def fake_load_lexicon_map(language: str = default_language_id) -> dict[str, object]:
            nonlocal call_count
            call_count += 1
            return {"language": language}

        monkeypatch.setattr(api_main, "_load_lexicon_map", fake_load_lexicon_map)
        monkeypatch.setattr(
            api_main.phonology_search,
            "build_ipa_index",
            lambda lexicon_map: {"lexicon_map": lexicon_map},
        )
        api_main._load_ipa_index.cache_clear()

        cached_map = fake_load_lexicon_map()
        result = api_main._load_ipa_index(default_language_id)

        assert call_count == 1
        assert result == {"lexicon_map": cached_map}
