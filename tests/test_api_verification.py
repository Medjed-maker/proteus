"""Tests for search verification URL and request echo metadata."""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

from api._models import SearchRequest
from api._request_context import build_verification_url
from tests.conftest import assert_uuid4_hex, mock_search_dependencies


def _search_payload(
    client: TestClient, body: dict[str, object]
) -> dict[str, object]:
    response = client.post("/search", json=body)
    assert response.status_code == 200
    return response.json()


def _verification_query(payload: dict[str, object]) -> dict[str, list[str]]:
    meta = payload["meta"]
    assert isinstance(meta, dict), "payload meta must be a dictionary"
    verification_url = meta.get("verification_url")
    assert isinstance(
        verification_url, str
    ) and verification_url, "meta must contain a valid verification_url string"
    parsed = urlparse(verification_url)
    assert (
        parsed.scheme and parsed.netloc
    ), f"verification_url must be an absolute URL: {verification_url!r}"
    return parse_qs(parsed.query)


def test_verification_url_uses_public_base_url_env(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_search_dependencies(monkeypatch)
    monkeypatch.setenv("PROTEUS_PUBLIC_BASE_URL", "https://proteus.example/")

    payload = _search_payload(client, {"query_form": "λόγος"})

    assert payload["meta"]["verification_url"].startswith("https://proteus.example/?")


def test_verification_url_uses_request_base_url_when_env_unset(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_search_dependencies(monkeypatch)
    monkeypatch.delenv("PROTEUS_PUBLIC_BASE_URL", raising=False)

    payload = _search_payload(client, {"query_form": "λόγος"})

    assert payload["meta"]["verification_url"].startswith("http://testserver/?")


def test_verification_url_includes_query_form_urlencoded(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_search_dependencies(monkeypatch)

    payload = _search_payload(client, {"query_form": "λόγος"})

    verification_url = payload["meta"]["verification_url"]
    assert "%CE%BB" in verification_url
    assert _verification_query(payload)["q"] == ["λόγος"]


def test_verification_url_includes_language_and_dialect(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_search_dependencies(monkeypatch)

    payload = _search_payload(
        client,
        {
            "query_form": "λόγος",
            "language": "ancient_greek",
            "dialect_hint": "koine",
            "max_candidates": 7,
            "response_language": "ja",
        },
    )

    query = _verification_query(payload)
    assert query["language"] == ["ancient_greek"]
    assert query["dialect"] == ["koine"]
    assert query["max_candidates"] == ["7"]
    assert query["response_language"] == ["ja"]


def test_verification_url_excludes_orthography_hint_deprecated(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_search_dependencies(monkeypatch)

    payload = _search_payload(
        client,
        {"query_form": "λόγος", "orthography_hint": "pre_403_2_attic"},
    )

    assert "orthography_hint" not in _verification_query(payload)
    assert "orthography_hint" not in payload["meta"]["request_echo"]


def test_verification_url_deterministic_across_calls(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_search_dependencies(monkeypatch)
    body = {"query_form": "λόγος", "dialect_hint": "attic", "max_candidates": 3}

    payloads = [_search_payload(client, body) for _ in range(10)]
    verification_urls = {
        payload["meta"]["verification_url"] for payload in payloads
    }
    request_ids = {payload["meta"]["request_id"] for payload in payloads}

    assert verification_urls == {payloads[0]["meta"]["verification_url"]}
    assert len(request_ids) == len(payloads)


def test_request_echo_mirrors_validated_request_fields(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_search_dependencies(monkeypatch)

    payload = _search_payload(client, {"query_form": "λόγος"})

    assert payload["meta"]["request_echo"] == {
        "query_form": "λόγος",
        "language": "ancient_greek",
        "dialect_hint": "attic",
        "max_candidates": 20,
        "response_language": "en",
    }


def test_request_echo_excludes_internal_legacy_flag(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_search_dependencies(monkeypatch)

    payload = _search_payload(client, {"query_form": "λόγος", "language": "ja"})

    request_echo = payload["meta"]["request_echo"]
    assert request_echo["language"] == "ancient_greek"
    assert request_echo["response_language"] == "ja"
    assert "legacy_language_alias_used" not in request_echo


def test_request_id_is_uuid4_hex_lowercase(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_search_dependencies(monkeypatch)

    payload = _search_payload(client, {"query_form": "λόγος"})

    assert_uuid4_hex(payload["meta"]["request_id"])


@pytest.mark.parametrize(
    "invalid_base",
    ["", "   ", "foo", "/api", "example.com/api"],
)
def test_build_verification_url_rejects_relative_base(invalid_base: str) -> None:
    """Misconfigured base URLs (relative, schemeless) must fail at the boundary."""
    request = SearchRequest(query_form="λόγος")

    with pytest.raises(ValueError):
        build_verification_url(invalid_base, request)


def test_build_verification_url_accepts_absolute_https_base() -> None:
    request = SearchRequest(query_form="λόγος")

    result = build_verification_url("https://proteus.example/", request)

    assert result.startswith("https://proteus.example/?")


@pytest.mark.parametrize(
    "invalid_env",
    [
        "https://example.com?foo=bar",
        "https://example.com#frag",
        "foo",
        "/api",
    ],
)
def test_startup_fails_when_public_base_url_is_invalid(
    monkeypatch: pytest.MonkeyPatch, invalid_env: str
) -> None:
    """Misconfigured ``PROTEUS_PUBLIC_BASE_URL`` must abort startup, not 500 per request."""
    from api import main as api_main

    monkeypatch.setenv("PROTEUS_PUBLIC_BASE_URL", invalid_env)

    with pytest.raises(RuntimeError) as excinfo:
        with TestClient(api_main.app):
            pass

    assert "PROTEUS_PUBLIC_BASE_URL" in str(excinfo.value)


def test_startup_succeeds_when_public_base_url_is_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from api import main as api_main

    monkeypatch.setenv("PROTEUS_PUBLIC_BASE_URL", "https://proteus.example/")
    monkeypatch.setenv("PROTEUS_DISABLE_STARTUP_WARMUP", "1")

    with TestClient(api_main.app) as test_client:
        response = test_client.get("/health")

    assert response.status_code == 200


def test_startup_succeeds_when_public_base_url_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from api import main as api_main

    monkeypatch.delenv("PROTEUS_PUBLIC_BASE_URL", raising=False)
    monkeypatch.setenv("PROTEUS_DISABLE_STARTUP_WARMUP", "1")

    with TestClient(api_main.app) as test_client:
        response = test_client.get("/health")

    assert response.status_code == 200
