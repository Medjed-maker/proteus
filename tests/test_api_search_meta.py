"""Tests for /search response metadata."""

from __future__ import annotations

from datetime import datetime
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api import main as api_main
from api._constants import API_VERSION, SCHEMA_VERSION
from api._models import DataVersions, ResponseMeta
from tests.conftest import assert_uuid4_hex, mock_search_dependencies


def _post_search(
    client: TestClient,
    *,
    query_form: str = "λόγος",
    dialect_hint: str = "attic",
    max_candidates: int = 3,
    response_language: str = "ja",
) -> dict[str, object]:
    response = client.post(
        "/search",
        json={
            "query_form": query_form,
            "dialect_hint": dialect_hint,
            "max_candidates": max_candidates,
            "response_language": response_language,
        },
    )
    assert response.status_code == 200
    return response.json()


def test_search_response_includes_meta_envelope(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_search_dependencies(monkeypatch)
    monkeypatch.setattr(
        api_main,
        "_build_ruleset_versions",
        lambda _language: {"ancient_greek": "1.2.3"},
    )

    meta = _post_search(client)["meta"]

    assert meta["engine_version"] == api_main._APP_VERSION
    assert meta["api_version"] == API_VERSION
    assert meta["schema_version"] == SCHEMA_VERSION


def test_search_response_meta_includes_request_id_uuid4_hex(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_search_dependencies(monkeypatch)

    meta = _post_search(client)["meta"]

    assert_uuid4_hex(meta["request_id"])


def test_search_response_meta_includes_verification_url(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_search_dependencies(monkeypatch)

    meta = _post_search(client)["meta"]

    assert "q=%CE%BB" in meta["verification_url"]
    assert "language=ancient_greek" in meta["verification_url"]
    assert "dialect=attic" in meta["verification_url"]


def test_search_response_meta_includes_timestamp_iso8601(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_search_dependencies(monkeypatch)

    meta = _post_search(client)["meta"]

    parsed = datetime.fromisoformat(meta["timestamp"])
    assert isinstance(parsed, datetime)
    assert parsed.isoformat() == meta["timestamp"]


def test_search_response_meta_includes_ruleset_versions_for_language(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_search_dependencies(monkeypatch)
    real_data_versions = api_main._build_data_versions("ancient_greek")
    original_load_search_dependencies = api_main._load_search_dependencies

    def load_search_dependencies_with_real_versions(
        language: str,
    ) -> api_main.SearchDependencies:
        return original_load_search_dependencies(language)._replace(
            data_versions=real_data_versions,
        )

    monkeypatch.setattr(
        api_main,
        "_load_search_dependencies",
        load_search_dependencies_with_real_versions,
    )

    payload = _post_search(client)

    assert payload["meta"]["ruleset_versions"]["ancient_greek"] == payload[
        "data_versions"
    ]["rules"]


def test_search_response_meta_data_versions_mirrors_top_level(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_search_dependencies(monkeypatch)

    payload = _post_search(client)

    assert payload["meta"]["data_versions"] == payload["data_versions"]


def test_search_response_meta_includes_request_echo(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_search_dependencies(monkeypatch)

    meta = _post_search(client)["meta"]

    assert meta["request_echo"] == {
        "query_form": "λόγος",
        "language": "ancient_greek",
        "dialect_hint": "attic",
        "max_candidates": 3,
        "response_language": "ja",
    }


def test_search_response_meta_engine_version_matches_app_version(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_search_dependencies(monkeypatch)

    meta = _post_search(client)["meta"]

    assert meta["engine_version"] == api_main._APP_VERSION


def test_response_meta_rejects_empty_timestamp() -> None:
    """ResponseMeta.timestamp must be non-empty per public contract."""
    with pytest.raises(ValidationError) as excinfo:
        ResponseMeta(
            api_version=API_VERSION,
            schema_version=SCHEMA_VERSION,
            engine_version="test",
            data_versions=DataVersions(),
            request_id="abcdef1234567890",
            timestamp="",
        )

    assert "timestamp must not be empty" in str(excinfo.value)
