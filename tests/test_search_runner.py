"""Tests for the FastAPI-agnostic search runner in :mod:`api._search_runner`.

These tests exercise the pure-exception interface that both the REST
endpoint (``api.main.search``) and the MCP adapter rely on.
"""

from __future__ import annotations

import pytest

from api import _search_runner, main as api_main
from api._models import SearchRequest
from api._request_context import generate_request_id
from api._search_runner import (
    InvalidDialectError,
    InvalidSearchQueryError,
    SearchExecutionError,
    UnsupportedLanguageError,
    run_search,
)
from phonology import search as phonology_search

from tests.conftest import mock_search_dependencies


def _request(**overrides: object) -> SearchRequest:
    """Build a SearchRequest with sensible defaults."""
    fields: dict[str, object] = {"query_form": "λόγος"}
    fields.update(overrides)
    return SearchRequest(**fields)  # type: ignore[arg-type]


def _runner_kwargs(deps: object) -> dict[str, object]:
    return {
        "deps": deps,
        "request_id": generate_request_id(),
        "base_url": None,
        "engine_version": "test-engine",
        "ruleset_versions": {"ancient_greek": "test-ruleset"},
    }


def test_run_search_returns_outcome_with_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The happy path returns a populated SearchExecutionOutcome."""
    mock_search_dependencies(monkeypatch)
    deps = api_main._load_search_dependencies("ancient_greek")

    outcome = run_search(_request(), **_runner_kwargs(deps))

    assert outcome.response.query == "λόγος"
    assert outcome.query_ipa  # non-empty IPA transcription
    assert outcome.response.hits


def test_run_search_raises_invalid_dialect_when_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An out-of-profile dialect should produce :class:`InvalidDialectError`."""
    mock_search_dependencies(monkeypatch)
    deps = api_main._load_search_dependencies("ancient_greek")

    # Bypass the SearchRequest validator (which already rejects unsupported
    # dialects) by mutating the immutable model with model_copy so we can
    # verify the runner's own guard surfaces the same error format.
    request = _request().model_copy(update={"dialect_hint": "not_a_dialect"})

    with pytest.raises(InvalidDialectError, match="Invalid dialect_hint"):
        run_search(request, **_runner_kwargs(deps))


def test_run_search_raises_invalid_query_when_engine_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ValueError from the phonology engine should map to InvalidSearchQueryError."""
    mock_search_dependencies(monkeypatch)
    deps = api_main._load_search_dependencies("ancient_greek")

    def boom(*args: object, **kwargs: object) -> object:
        raise ValueError("bad query syntax")

    monkeypatch.setattr(phonology_search, "search_execution", boom)

    with pytest.raises(InvalidSearchQueryError, match="bad query syntax"):
        run_search(_request(), **_runner_kwargs(deps))


def test_run_search_raises_execution_error_on_oserror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OSError from the phonology engine should map to SearchExecutionError."""
    mock_search_dependencies(monkeypatch)
    deps = api_main._load_search_dependencies("ancient_greek")

    def boom(*args: object, **kwargs: object) -> object:
        raise OSError("disk gone")

    monkeypatch.setattr(phonology_search, "search_execution", boom)

    with pytest.raises(SearchExecutionError, match="internal error"):
        run_search(_request(), **_runner_kwargs(deps))


def test_load_search_dependencies_raises_unsupported_language() -> None:
    """Unknown languages should yield UnsupportedLanguageError (a ValueError)."""
    with pytest.raises(UnsupportedLanguageError):
        api_main._load_search_dependencies("definitely-not-a-language")


def test_summarize_query_for_logs_redacts_payload() -> None:
    """The log summary should not echo the raw query string."""
    summary = _search_runner._summarize_query_for_logs("λόγος")

    assert "λόγος" not in summary
    assert summary.startswith("len=")
    assert "sha256=" in summary
