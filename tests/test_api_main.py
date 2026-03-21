"""Tests for proteus.api.main."""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from proteus.api import main as api_main
from proteus.api.main import SearchHit


class TestSearchHit:
    @pytest.mark.parametrize("distance", [-0.1, 1.1])
    def test_distance_must_stay_within_unit_interval(self, distance: float) -> None:
        with pytest.raises(ValidationError):
            SearchHit(
                headword="λόγος",
                ipa="loɡos",
                distance=distance,
                rules_applied=[],
                explanation="example",
            )

    @pytest.mark.parametrize("distance", [0.0, 1.0])
    def test_distance_accepts_unit_interval_boundaries(self, distance: float) -> None:
        hit = SearchHit(
            headword="λόγος",
            ipa="loɡos",
            distance=distance,
            rules_applied=[],
            explanation="example",
        )

        assert hit.distance == distance

    @pytest.mark.parametrize(
        ("payload", "field_name"),
        [
            (
                {
                    "headword": None,
                    "ipa": "loɡos",
                    "distance": 0.1,
                    "rules_applied": [],
                    "explanation": "example",
                },
                "headword",
            ),
            (
                {
                    "headword": "λόγος",
                    "ipa": None,
                    "distance": 0.1,
                    "rules_applied": [],
                    "explanation": "example",
                },
                "ipa",
            ),
            (
                {
                    "headword": "λόγος",
                    "ipa": "loɡos",
                    "distance": 0.1,
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
                    "rules_applied": [],
                    "explanation": None,
                },
                "explanation",
            ),
        ],
    )
    def test_other_fields_reject_invalid_values(
        self, payload: dict[str, object], field_name: str
    ) -> None:
        with pytest.raises(ValidationError) as exc_info:
            SearchHit(**payload)

        assert field_name in str(exc_info.value)


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

    def test_missing_route_returns_not_found_json(self, client: TestClient) -> None:
        response = client.get("/missing-route")

        assert response.status_code == 404
        assert response.json() == {"detail": "Not Found"}

    def test_search_returns_not_implemented_json(self, client: TestClient) -> None:
        response = client.post("/search", json={"query": "λόγος"})

        assert response.status_code == 501
        assert response.json()["detail"].startswith(
            "Phonological search is not implemented yet."
        )


class TestHealthEndpoint:
    def test_api_health(self, client: TestClient) -> None:
        """Health endpoint returns 200 ok."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestSearchValidation:
    @pytest.mark.parametrize("max_results", [-1, 0, 101])
    def test_search_rejects_invalid_max_results(
        self, client: TestClient, max_results: int
    ) -> None:
        """Pydantic validation rejects negative, zero, and oversized max_results."""
        response = client.post(
            "/search",
            json={"query": "λόγος", "dialect": "attic", "max_results": max_results},
        )

        assert response.status_code == 422
        assert "max_results" in response.text

    @pytest.mark.parametrize("max_results", [1, 100])
    def test_search_accepts_boundary_max_results(
        self, client: TestClient, max_results: int
    ) -> None:
        """Boundary-valid max_results values reach the handler instead of validation errors."""
        response = client.post(
            "/search",
            json={"query": "λόγος", "dialect": "attic", "max_results": max_results},
        )

        assert response.status_code == 501
        assert "max_results" not in response.text
