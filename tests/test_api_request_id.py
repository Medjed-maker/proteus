"""Tests for API request id propagation."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api import main as api_main
from api._request_context import is_valid_request_id
from tests.conftest import assert_uuid4_hex, mock_search_dependencies


def test_x_request_id_header_set_when_absent(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert_uuid4_hex(response.headers["X-Request-ID"])


def test_x_request_id_header_respected_when_valid(client: TestClient) -> None:
    request_id = "ABCDEF1234567890abcdef1234567890"

    response = client.get("/health", headers={"X-Request-ID": request_id})

    assert response.status_code == 200
    # Middleware lowercases client-supplied IDs so logs and traces always see
    # a single canonical form regardless of casing chosen by the caller.
    assert response.headers["X-Request-ID"] == request_id.lower()


@pytest.mark.parametrize(
    "request_id",
    [
        "",
        "not-hex",
        "f" * 7,
        "f" * 65,
        "!@#$%^&*()",
        "\x00abc",
        # ``int(value, 16)`` would silently accept these but they are not a
        # canonical hex identifier — the regex-based validator must reject.
        "0x" + "f" * 30,
        "0X" + "f" * 30,
        "+" + "f" * 30,
        "-" + "f" * 30,
    ],
)
def test_x_request_id_header_replaced_when_invalid(
    client: TestClient, request_id: str
) -> None:
    response = client.get("/health", headers={"X-Request-ID": request_id})

    assert response.status_code == 200
    replacement = response.headers["X-Request-ID"]
    assert replacement != request_id
    assert_uuid4_hex(replacement)


def test_is_valid_request_id_rejects_hex_prefix() -> None:
    """``0x`` prefixed strings must not be treated as canonical hex IDs."""
    assert is_valid_request_id("0x" + "f" * 30) is False
    assert is_valid_request_id("0X" + "f" * 30) is False
    # Plain hex within bounds is still accepted.
    assert is_valid_request_id("f" * 30) is True


def test_request_id_present_on_400_error_responses(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_search_dependencies(monkeypatch)
    monkeypatch.setattr(
        api_main.phonology_search,
        "search_execution",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad query")),
    )

    response = client.post("/search", json={"query_form": "λόγος"})

    assert response.status_code == 400
    assert_uuid4_hex(response.headers["X-Request-ID"])


def test_request_id_present_on_500_error_responses(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OSError is treated as an internal error; X-Request-ID must still be set."""
    mock_search_dependencies(monkeypatch)
    monkeypatch.setattr(
        api_main.phonology_search,
        "search_execution",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("unexpected internal failure")
        ),
    )

    response = client.post("/search", json={"query_form": "λόγος"})

    assert response.status_code == 500
    assert_uuid4_hex(response.headers["X-Request-ID"])


def test_request_id_present_on_503_error_responses(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def raise_not_ready(_language: str) -> api_main.SearchDependencies:
        raise api_main.SearchDependenciesNotReadyError("not ready")

    monkeypatch.setattr(api_main, "_load_search_dependencies", raise_not_ready)

    response = client.post("/search", json={"query_form": "λόγος"})

    assert response.status_code == 503
    assert_uuid4_hex(response.headers["X-Request-ID"])


def test_request_id_present_when_handler_raises_unhandled_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare ``RuntimeError`` from a handler must still emit ``X-Request-ID``.

    Unlike ``OSError`` (translated to 500 by an explicit handler), a
    ``RuntimeError`` escapes FastAPI's built-in handlers and would hit
    Starlette's outermost ``ServerErrorMiddleware`` without the correlation
    id unless the dedicated ``Exception`` handler attaches it.
    """
    mock_search_dependencies(monkeypatch)
    monkeypatch.setattr(
        api_main.phonology_search,
        "search_execution",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("simulated unhandled failure")
        ),
    )

    # FastAPI's TestClient re-raises server exceptions by default; disable that
    # so the 500 response is observable.
    client_no_raise = TestClient(api_main.app, raise_server_exceptions=False)
    response = client_no_raise.post("/search", json={"query_form": "λόγος"})

    assert response.status_code == 500
    assert_uuid4_hex(response.headers["X-Request-ID"])
