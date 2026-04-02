"""Tests for api.main."""

import logging
from collections.abc import Generator

import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api import main as api_main
from api.main import SearchHit
from phonology.explainer import RuleApplication
from phonology.search import SearchResult


def _clear_loader_caches() -> None:
    """Reset cached API loader state."""
    for loader_name in (
        "_load_lexicon_entries",
        "_load_distance_matrix",
        "_load_rules_registry",
        "_load_search_index",
    ):
        cache_clear = getattr(getattr(api_main, loader_name), "cache_clear", None)
        if cache_clear is not None:
            cache_clear()
    search_cache_clear = getattr(api_main.phonology_search._get_rules_registry, "cache_clear", None)
    if search_cache_clear is not None:
        search_cache_clear()


def _install_invalid_query_to_ipa(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch to_ipa to raise the same query validation error used in search tests."""

    def fake_to_ipa(query: str, dialect: str = "attic") -> str:
        raise ValueError("query must be a non-empty string")

    monkeypatch.setattr(api_main, "to_ipa", fake_to_ipa)


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

        class FakeExplanation:
            steps: list[object] = []

        def fake_explain_alignment(**kwargs: object) -> FakeExplanation:
            captured.update(kwargs)
            return FakeExplanation()

        monkeypatch.setattr(api_main, "explain_alignment", fake_explain_alignment)
        monkeypatch.setattr(api_main, "to_prose", lambda explanation: "example")

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
        )

        assert captured["source_ipa"] == ""
        assert captured["target_ipa"] == "loɡos"
        assert hit.ipa == ""

    def test_build_search_hit_uses_rule_applications_for_positions_and_prose(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            api_main,
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

    def test_build_search_hit_accepts_morphophonemic_rule_ids_without_schema_changes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            api_main,
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

    def test_build_search_hit_coalesces_missing_dialect_attribution_to_empty_string(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeExplanation:
            steps: list[object] = []

        monkeypatch.setattr(api_main, "explain_alignment", lambda **kwargs: FakeExplanation())
        monkeypatch.setattr(api_main, "to_prose", lambda explanation: "example")

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


class TestFrontendHtml:
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

    def test_html_references_local_css_not_cdn(self, client: TestClient) -> None:
        response = client.get("/")

        assert response.status_code == 200
        assert "/static/styles.css" in response.text
        assert "cdn.tailwindcss.com" not in response.text
        assert "fonts.googleapis.com" not in response.text

    def test_missing_route_returns_not_found_json(self, client: TestClient) -> None:
        response = client.get("/missing-route")

        assert response.status_code == 404
        assert response.json() == {"detail": "Not Found"}


class TestStaticAssets:
    def test_css_is_served(self, client: TestClient) -> None:
        response = client.get("/static/styles.css")

        assert response.status_code == 200
        assert "text/css" in response.headers["content-type"]


class TestHealthEndpoint:
    def test_api_health(self, client: TestClient) -> None:
        """Health endpoint returns 200 ok."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestReadyEndpoint:
    def test_ready_returns_ok(self, client: TestClient) -> None:
        """Readiness probe returns 200 when all dependencies load."""
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    @pytest.mark.parametrize(
        ("loader_name", "error"),
        [
            ("_load_lexicon_entries", ValueError("lexicon unavailable")),
            ("_load_distance_matrix", FileNotFoundError("matrix not found")),
            ("_load_rules_registry", yaml.YAMLError("bad rules")),
            ("_load_search_index", OSError("index error")),
        ],
    )
    def test_ready_returns_503_when_loader_fails(
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
        assert response.json() == {"detail": "Dependencies not ready"}


class TestDocumentationAndCors:
    def test_docs_route_is_available(self, client: TestClient) -> None:
        response = client.get("/docs")

        assert response.status_code == 200
        assert "Swagger UI" in response.text

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
    captured: dict[str, object] = {}

    def fake_search(
        query: str,
        lexicon: tuple[dict[str, object], ...],
        matrix: dict[str, dict[str, float]],
        max_results: int,
        dialect: str,
        index: dict[str, list[str]],
    ) -> list[SearchResult]:
        captured["query"] = query
        captured["lexicon"] = lexicon
        captured["matrix"] = matrix
        captured["max_results"] = max_results
        captured["dialect"] = dialect
        captured["index"] = index
        return [
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
    monkeypatch.setattr(
        api_main,
        "_load_rules_registry",
        lambda: {
            "CCH-001": {
                "id": "CCH-001",
                "input": "s",
                "output": "h",
                "dialects": ["attic"],
            }
        },
    )
    monkeypatch.setattr(api_main, "to_ipa", lambda query, dialect="attic": "loɡos")
    monkeypatch.setattr(api_main, "_load_search_index", lambda: {"l ɡ": ["L1"]})
    monkeypatch.setattr(api_main.phonology_search, "search", fake_search)
    return captured


class TestSearchEndpoint:

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
        assert captured["lexicon"] == (
            {"id": "L1", "headword": "λόγος", "ipa": "lóɡos", "dialect": "attic"},
        )
        payload = response.json()
        assert payload["query"] == "λόγος"
        assert payload["query_ipa"] == "loɡos"
        assert len(payload["hits"]) == 1
        assert payload["hits"][0]["headword"] == "λόγος"
        assert payload["hits"][0]["ipa"] == "lóɡos"
        assert payload["hits"][0]["distance"] == pytest.approx(0.25)
        assert payload["hits"][0]["confidence"] == pytest.approx(0.75)
        assert payload["hits"][0]["dialect_attribution"] == "lemma dialect: attic"
        assert payload["hits"][0]["alignment_visualization"] == ""
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
            "search",
            lambda *args, **kwargs: captured.update(
                {"dialect": kwargs.get("dialect", args[4] if len(args) > 4 else None)}
            ) or [
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
        )

        response = client.post(
            "/search",
            json={"query_form": "λόγος", "dialect_hint": "koine", "max_candidates": 3},
        )

        assert response.status_code == 200
        assert captured["dialect"] == "koine"
        payload = response.json()
        assert payload["query_ipa"] == "loɣos"
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
        assert payload["hits"][0]["distance"] == pytest.approx(0.275)
        assert "distance 0.275" in payload["hits"][0]["explanation"]

    def test_search_hit_falls_back_to_explain_alignment_when_rule_details_are_missing(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}

        class FakeExplanation:
            def __init__(self) -> None:
                self.steps = [
                    RuleApplication(
                        rule_id="CCH-001",
                        rule_name="Fallback Rule",
                        from_phone="s",
                        to_phone="h",
                        position=-1,
                    )
                ]

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
            "search",
            lambda *args, **kwargs: [
                SearchResult(
                    lemma="λόγος",
                    confidence=0.75,
                    dialect_attribution="lemma dialect: attic",
                    applied_rules=["CCH-001"],
                    ipa="lóɡos",
                )
            ],
        )

        def fake_explain_alignment(**kwargs: object) -> FakeExplanation:
            captured.update(kwargs)
            return FakeExplanation()

        monkeypatch.setattr(api_main, "explain_alignment", fake_explain_alignment)
        monkeypatch.setattr(api_main, "to_prose", lambda explanation: "fallback explanation")

        response = client.post("/search", json={"query_form": "λόγος"})

        assert response.status_code == 200
        assert captured["rule_ids"] == ["CCH-001"]
        assert response.json()["hits"][0]["rules_applied"][0]["position"] == -1

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

        monkeypatch.setattr(api_main, "explain_alignment", fail_explain_alignment)

        response = client.post("/search", json={"query_form": "λόγος"})

        assert response.status_code == 200
        assert response.json()["hits"][0]["rules_applied"] == [
            {
                "rule_id": "OBS-SUB",
                "rule_name": "観測された置換",
                "rule_name_en": "Observed substitution",
                "from_phone": "a",
                "to_phone": "x",
                "position": 0,
            }
        ]

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

    def test_search_returns_server_error_when_loader_fails(
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

        assert response.status_code == 500
        assert response.json() == {
            "detail": "Search is temporarily unavailable. Please try again later."
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
