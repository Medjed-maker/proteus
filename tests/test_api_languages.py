"""Tests for the /languages API endpoint."""

from __future__ import annotations

from collections.abc import Callable
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api import main as api_main
from api import _models as api_models
from phonology.core.ports.profiles import LanguageProfile, register_language_profile


def _language_by_id(payload: dict[str, object], language_id: str) -> dict[str, object]:
    """Return the language object with the requested id from a response payload.

    Args:
        payload: API payload shaped like ``{"languages": [{...}]}``.
        language_id: Stable language identifier to find.

    Returns:
        The matched language dictionary.

    Raises:
        AssertionError: If ``payload["languages"]`` is not a list, a language
            item is not a dict, or no matching language is present.
    """
    languages = payload["languages"]
    assert isinstance(languages, list)
    for language in languages:
        assert isinstance(language, dict)
        if language["language_id"] == language_id:
            return language
    raise AssertionError(f"missing language {language_id!r}")


def test_languages_endpoint_lists_ancient_greek(client: TestClient) -> None:
    response = client.get("/languages")

    assert response.status_code == 200
    language = _language_by_id(response.json(), "ancient_greek")
    assert language["display_name"] == "Ancient Greek"
    assert language["status"] == "pilot"


def test_public_language_models_are_exported() -> None:
    assert "VersionInfo" in api_models.__all__
    assert "RequestEcho" in api_models.__all__
    assert "ResponseMeta" in api_models.__all__
    assert "LanguageInfo" in api_models.__all__
    assert "LanguagesResponse" in api_models.__all__


def test_languages_endpoint_returns_supported_dialects(client: TestClient) -> None:
    response = client.get("/languages")

    assert response.status_code == 200
    language = _language_by_id(response.json(), "ancient_greek")
    assert {"attic", "koine"}.issubset(set(language["supported_dialects"]))


def test_languages_endpoint_returns_ruleset_version(client: TestClient) -> None:
    response = client.get("/languages")

    assert response.status_code == 200
    language = _language_by_id(response.json(), "ancient_greek")
    assert re.fullmatch(r"\d+\.\d+\.\d+", str(language["ruleset_version"]))


def test_languages_endpoint_includes_toy_language_when_registered(
    isolated_language_registry: None,
    tmp_path: Path,
    client: TestClient,
    build_toy_profile: Callable[[Path, str], LanguageProfile],
) -> None:
    register_language_profile(build_toy_profile(tmp_path, "toy_language"))

    response = client.get("/languages")

    assert response.status_code == 200
    language = _language_by_id(response.json(), "toy_language")
    assert language["supported_dialects"] == ["toy"]
    assert language["status"] == "experimental"
    assert language["ruleset_version"] == "unknown"
    assert language["lexicon_schema_version"] == "unknown"
    assert language["matrix_version"] == "unknown"
    assert language["description"] == ""


def test_languages_endpoint_sorted_by_language_id(
    isolated_language_registry: None,
    tmp_path: Path,
    client: TestClient,
    build_toy_profile: Callable[[Path, str], LanguageProfile],
) -> None:
    register_language_profile(build_toy_profile(tmp_path, "toy_beta"))
    register_language_profile(build_toy_profile(tmp_path, "toy_alpha"))

    response = client.get("/languages")

    assert response.status_code == 200
    language_ids = [language["language_id"] for language in response.json()["languages"]]
    assert language_ids == sorted(language_ids)


def test_languages_endpoint_includes_response_meta(client: TestClient) -> None:
    languages_response = client.get("/languages")
    version_response = client.get("/version")

    assert languages_response.status_code == 200
    assert version_response.status_code == 200
    languages_meta = languages_response.json()["meta"]
    version_payload = version_response.json()
    assert languages_meta["engine_version"] == version_payload["engine_version"]
    assert languages_meta["api_version"] == version_payload["api_version"]
    assert languages_meta["schema_version"] == version_payload["schema_version"]


def test_languages_endpoint_returns_unknown_on_missing_assets(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_rules_version(_rules_dir: Path) -> dict[str, str]:
        raise OSError("missing rules")

    monkeypatch.setattr(api_main, "get_rules_version", _raise_rules_version)

    response = client.get("/languages")

    assert response.status_code == 200
    language = _language_by_id(response.json(), "ancient_greek")
    assert language["ruleset_version"] == "unknown"


def test_languages_endpoint_head_returns_204(client: TestClient) -> None:
    response = client.head("/languages")

    assert response.status_code == 204
    assert response.content == b""
