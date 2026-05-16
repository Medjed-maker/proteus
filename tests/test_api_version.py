"""Tests for the /version API endpoint."""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from api import main as api_main
from api._constants import API_VERSION, SCHEMA_VERSION


@pytest.mark.parametrize(
    ("field_name", "expected_value"),
    (
        ("engine_version", api_main._APP_VERSION),
        ("api_version", API_VERSION),
    ),
)
def test_version_endpoint_returns_expected_value(
    client: TestClient, field_name: str, expected_value: str
) -> None:
    response = client.get("/version")

    assert response.status_code == 200
    assert response.json()[field_name] == expected_value


def test_version_endpoint_returns_schema_version(client: TestClient) -> None:
    response = client.get("/version")

    assert response.status_code == 200
    schema_version = response.json()["schema_version"]
    assert schema_version == SCHEMA_VERSION
    assert re.fullmatch(r"\d+\.\d+\.\d+", schema_version)


def test_version_endpoint_returns_rule_schema_id(client: TestClient) -> None:
    response = client.get("/version")

    assert response.status_code == 200
    rule_schema_version = response.json()["rule_schema_version"]
    assert rule_schema_version.startswith("https://")
    assert rule_schema_version.endswith(".schema.json")


def test_version_endpoint_python_version_format(client: TestClient) -> None:
    response = client.get("/version")

    assert response.status_code == 200
    assert re.fullmatch(r"\d+\.\d+\.\d+", response.json()["python_version"])


def test_version_endpoint_build_timestamp_env_override(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        api_main._BUILD_TIMESTAMP_ENV_VAR,
        "2026-05-15T00:00:00Z",
    )

    response = client.get("/version")

    assert response.status_code == 200
    assert response.json()["build_timestamp"] == "2026-05-15T00:00:00Z"


def test_version_endpoint_git_sha_env_override(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(api_main._GIT_SHA_ENV_VAR, "abcdef12")

    response = client.get("/version")

    assert response.status_code == 200
    assert response.json()["git_sha"] == "abcdef12"


def test_version_endpoint_git_sha_is_truncated_to_max_length(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    full_sha = "abc123def456789012345678901234567890abcd"
    monkeypatch.setenv(api_main._GIT_SHA_ENV_VAR, full_sha)

    response = client.get("/version")

    assert response.status_code == 200
    git_sha = response.json()["git_sha"]
    assert git_sha == full_sha[: api_main._GIT_SHA_MAX_LENGTH]
    assert len(git_sha) == api_main._GIT_SHA_MAX_LENGTH


def test_version_endpoint_head_returns_204(client: TestClient) -> None:
    response = client.head("/version")

    assert response.status_code == 204
    assert response.content == b""
