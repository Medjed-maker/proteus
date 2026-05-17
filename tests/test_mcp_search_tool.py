"""Tests for the Ancient Phonology MCP search tool."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import anyio
import pytest

from api import main as api_main
from mcp_server._search_adapter import _run_search_for_mcp
from mcp_server.server import app
from mcp_server.tools.search import McpSearchInput
from phonology import search as search_module
from phonology.explainer import RuleApplication
from phonology.profiles import get_default_language_profile
from phonology.search import SearchResult
from tests.conftest import assert_uuid4_hex, mock_search_dependencies


def _run_tool_call(arguments: dict[str, Any]) -> dict[str, Any]:
    """Call the MCP tool in process and return its structured payload."""

    async def call() -> dict[str, Any]:
        result = await app.call_tool("ancient_phonology.search", arguments)
        assert isinstance(result, tuple)
        _, structured = result
        assert isinstance(structured, dict)
        return structured

    return anyio.run(call)


def test_mcp_server_lists_ancient_phonology_search_tool() -> None:
    """The MCP server should expose the public Ancient Phonology search tool."""

    async def list_tool_names() -> list[str]:
        tools = await app.list_tools()
        return [tool.name for tool in tools]

    assert "ancient_phonology.search" in anyio.run(list_tool_names)


def test_mcp_search_tool_returns_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    """A known query should return at least one structured candidate."""
    mock_search_dependencies(monkeypatch)

    output = _run_search_for_mcp(McpSearchInput(query_form="λόγος"))

    assert output.query == "λόγος"
    assert output.candidates
    assert output.candidates[0].headword == "λόγος"


def test_mcp_search_tool_call_returns_structured_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FastMCP in-process calls should expose the same structured payload."""
    mock_search_dependencies(monkeypatch)

    payload = _run_tool_call(
        {"request": {"query_form": "λόγος", "max_candidates": 2}}
    )

    assert payload["query"] == "λόγος"
    assert payload["candidates"][0]["headword"] == "λόγος"


def test_mcp_search_tool_returns_confidence_and_distance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Candidates should include both confidence and normalized distance."""
    mock_search_dependencies(monkeypatch)

    output = _run_search_for_mcp(McpSearchInput(query_form="λόγος"))
    candidate = output.candidates[0]

    assert candidate.confidence == 0.75
    assert candidate.distance == pytest.approx(0.25)


def test_mcp_search_tool_returns_applied_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Koine searches should expose applied rule steps."""
    captured = mock_search_dependencies(monkeypatch)
    monkeypatch.setattr(
        api_main,
        "_load_rules_registry",
        lambda _language: {
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
    def mock_search_execution(*args: object, **kwargs: object) -> object:
        captured["dialect"] = kwargs.get("dialect")
        return search_module.SearchExecutionResult(
            results=[
                SearchResult(
                    lemma="λόγος",
                    confidence=0.8,
                    dialect_attribution=(
                        "lemma dialect: attic; query-compatible dialects: koine"
                    ),
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
            query_ipa="loɣos",
            query_mode="Full-form",
            truncated=False,
        )

    monkeypatch.setattr(
        api_main.phonology_search,
        "search_execution",
        mock_search_execution,
    )

    output = _run_search_for_mcp(
        McpSearchInput(query_form="λόγος", dialect_hint="koine")
    )

    assert captured["dialect"] == "koine"
    assert output.candidates[0].rules_applied[0].rule_id == "CCH-009"
    assert output.candidates[0].rule_support is True


def test_mcp_search_tool_returns_meta_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MCP output should include the shared metadata envelope."""
    mock_search_dependencies(monkeypatch)

    output = _run_search_for_mcp(McpSearchInput(query_form="λόγος"))

    assert output.meta.engine_version == api_main._APP_VERSION
    assert "ancient_greek" in output.meta.ruleset_versions
    assert_uuid4_hex(output.meta.request_id)
    assert output.meta.request_echo is not None
    assert output.meta.request_echo.query_form == "λόγος"


def test_mcp_search_tool_respects_max_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The MCP adapter should pass max_candidates to the shared search engine."""
    captured = mock_search_dependencies(monkeypatch)

    _run_search_for_mcp(McpSearchInput(query_form="λόγος", max_candidates=2))

    assert captured["max_results"] == 2


def test_mcp_search_tool_validates_unsupported_language() -> None:
    """Unknown source languages should be rejected before search execution."""
    with pytest.raises(ValueError, match="Invalid MCP search input"):
        _run_search_for_mcp(
            McpSearchInput(query_form="λόγος", source_language="unknown_language")
        )


def test_mcp_search_tool_validates_dialect_hint() -> None:
    """Unsupported dialects should be rejected by SearchRequest validation."""
    with pytest.raises(ValueError, match="Invalid MCP search input"):
        _run_search_for_mcp(
            McpSearchInput(query_form="λόγος", dialect_hint="not_a_dialect")
        )


def test_mcp_search_tool_uses_default_dialect_when_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitting dialect_hint should use the language profile default."""
    captured = mock_search_dependencies(monkeypatch)

    output = _run_search_for_mcp(McpSearchInput(query_form="λόγος"))

    assert captured["dialect"] == get_default_language_profile().default_dialect
    assert output.meta.request_echo is not None
    assert output.meta.request_echo.dialect_hint == "attic"


def test_mcp_search_tool_response_language_ja(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Japanese response_language should localize candidate support prose."""
    mock_search_dependencies(monkeypatch)

    output = _run_search_for_mcp(
        McpSearchInput(query_form="λόγος", response_language="ja")
    )

    assert output.meta.request_echo is not None
    assert output.meta.request_echo.response_language == "ja"
    assert any("音韻" in line for line in output.candidates[0].why_candidate)


def test_mcp_search_tool_verification_url_uses_env_when_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MCP verification URLs should use PROTEUS_PUBLIC_BASE_URL when set."""
    mock_search_dependencies(monkeypatch)
    monkeypatch.setenv("PROTEUS_PUBLIC_BASE_URL", "https://proteus.example/")

    output = _run_search_for_mcp(McpSearchInput(query_form="λόγος"))

    assert output.meta.verification_url.startswith("https://proteus.example/?")
    assert "language=ancient_greek" in output.meta.verification_url


def test_mcp_search_tool_verification_url_empty_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """MCP verification URL should be empty without a request base URL."""
    mock_search_dependencies(monkeypatch)
    monkeypatch.delenv("PROTEUS_PUBLIC_BASE_URL", raising=False)
    caplog.set_level("WARNING", logger="proteus.mcp")

    output = _run_search_for_mcp(McpSearchInput(query_form="λόγος"))

    assert output.meta.verification_url == ""
    assert output.meta.request_echo is not None
    assert "PROTEUS_PUBLIC_BASE_URL is unset" in caplog.text


def test_mcp_search_tool_uses_profile_orthographic_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The MCP adapter should pass profile services through to hit formatting."""
    mock_search_dependencies(monkeypatch)
    called: dict[str, object] = {}

    def fake_builder(**kwargs: object) -> list[object]:
        called.update(kwargs)
        return []

    profile = replace(
        get_default_language_profile(),
        orthographic_note_builder=fake_builder,
    )
    original_loader = api_main.load_search_dependencies

    def load_with_builder(language: str) -> api_main.SearchDependencies:
        deps = original_loader(language)
        return deps._replace(profile=profile)

    monkeypatch.setattr(api_main, "load_search_dependencies", load_with_builder)

    _run_search_for_mcp(McpSearchInput(query_form="λόγος"))

    assert called["query_form"] == "λόγος"
