"""Tests for Buck reference MCP tools."""

from __future__ import annotations

from typing import Any
import unicodedata

import anyio
import pytest

from mcp_server.server import app
from mcp_server.tools.buck import (
    McpBuckDialectInput,
    McpBuckGlossarySearchInput,
    McpBuckRuleSearchInput,
    get_buck_dialect_for_mcp,
    search_buck_glossary_for_mcp,
    search_buck_rules_for_mcp,
)
from phonology.languages.ancient_greek import buck as buck_module
from phonology.languages.ancient_greek.buck_service import (
    clear_buck_reference_index_cache,
)


@pytest.fixture(autouse=True)
def reset_buck_loader_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep Buck MCP tests independent from loader override/cache tests."""
    monkeypatch.delenv("PROTEUS_TRUSTED_BUCK_DIR", raising=False)
    buck_module.clear_buck_data_cache()
    clear_buck_reference_index_cache()


def _run_tool_call(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Call an MCP tool in process and return its structured payload."""

    async def call() -> dict[str, Any]:
        result = await app.call_tool(tool_name, arguments)
        assert isinstance(result, tuple)
        _, structured = result
        assert isinstance(structured, dict)
        return structured

    return anyio.run(call)


def test_mcp_server_lists_buck_reference_tools() -> None:
    """The MCP server should expose Buck reference tools."""

    async def list_tool_names() -> list[str]:
        tools = await app.list_tools()
        return [tool.name for tool in tools]

    tool_names = anyio.run(list_tool_names)

    assert "ancient_phonology.search_buck_rules" in tool_names
    assert "ancient_phonology.get_buck_dialect" in tool_names
    assert "ancient_phonology.search_buck_glossary" in tool_names


def test_search_buck_rules_by_numeric_section_returns_review_metadata() -> None:
    """Buck rule lookup should canonicalize sections and expose review metadata."""
    output = search_buck_rules_for_mcp(McpBuckRuleSearchInput(section=41.4))

    assert output.count >= 1
    assert output.metadata.status == "provisional"
    assert output.metadata.review_status == "not_expert_reviewed"
    assert output.metadata.citation_ready is False
    assert "not expert-reviewed" in output.metadata.review_note
    rule = next(rule for rule in output.rules if rule.id == "grc_phon_41_4")
    assert rule.buck_section == "41.4"
    assert rule.review_status == "not_expert_reviewed"
    assert rule.citation_ready is False


def test_get_buck_dialect_returns_dialect_and_inherited_rule_details() -> None:
    """Dialect lookup should return the dialect plus catalog-linked rules."""
    output = get_buck_dialect_for_mcp(McpBuckDialectInput(dialect_id="attic"))

    assert output.dialect.id == "attic"
    assert output.dialect.review_status == "not_expert_reviewed"
    assert output.dialect.citation_ready is False
    assert any(rule.id == "grc_phon_41_4" for rule in output.rules)
    assert all(rule.review_status == "not_expert_reviewed" for rule in output.rules)


def test_search_buck_glossary_by_word_and_nfd_standard_form() -> None:
    """Glossary lookup should use service-level NFC strict matching."""
    by_word = search_buck_glossary_for_mcp(
        McpBuckGlossarySearchInput(word="λεώς")
    )
    by_standard_form = search_buck_glossary_for_mcp(
        McpBuckGlossarySearchInput(
            standard_form=unicodedata.normalize("NFD", "λαός")
        )
    )

    assert [entry.word for entry in by_word.entries] == ["λεώς"]
    assert [entry.word for entry in by_standard_form.entries] == ["λεώς"]
    entry = by_word.entries[0]
    assert entry.buck_ref is not None
    assert entry.buck_ref.section == "41.4"
    assert entry.buck_ref.page == 130
    assert entry.review_status == "not_expert_reviewed"
    assert entry.citation_ready is False


def test_buck_reference_tools_reject_unsupported_language() -> None:
    """Buck reference tools currently support Ancient Greek only."""
    with pytest.raises(ValueError, match="Unsupported Buck reference language"):
        search_buck_rules_for_mcp(
            McpBuckRuleSearchInput(source_language="old_english")
        )


def test_get_buck_dialect_rejects_unknown_dialect() -> None:
    """Unknown Buck dialect ids should fail clearly."""
    with pytest.raises(ValueError, match="Unknown Buck dialect"):
        get_buck_dialect_for_mcp(McpBuckDialectInput(dialect_id="unknown"))


def test_buck_tool_call_returns_structured_payload() -> None:
    """FastMCP in-process calls should expose JSON structured Buck payloads."""
    payload = _run_tool_call(
        "ancient_phonology.search_buck_rules",
        {"request": {"section": 41.4, "max_results": 5}},
    )

    assert payload["count"] >= 1
    assert payload["metadata"]["citation_ready"] is False
    assert any(rule["id"] == "grc_phon_41_4" for rule in payload["rules"])


def test_search_buck_rules_requires_rule_id_and_section_to_match() -> None:
    """Rule id and section filters should compose as an AND condition."""
    output = search_buck_rules_for_mcp(
        McpBuckRuleSearchInput(rule_id="grc_phon_41_4", section=8)
    )

    assert output.count == 0
    assert output.rules == []


def test_search_buck_rules_exposes_typed_transformation() -> None:
    """Rule transformations should be returned as a typed object keyed by 'from'."""
    output = search_buck_rules_for_mcp(McpBuckRuleSearchInput(section=8))

    rule = next(rule for rule in output.rules if rule.id == "grc_phon_8")
    assert rule.transformation is not None
    assert rule.transformation.from_ == "ā"
    assert rule.transformation.to == "η"


def test_search_buck_rules_emits_from_alias_in_json_payload() -> None:
    """The JSON payload should serialize the reserved-word field as 'from'."""
    payload = _run_tool_call(
        "ancient_phonology.search_buck_rules",
        {"request": {"section": 8}},
    )

    rule = next(rule for rule in payload["rules"] if rule["id"] == "grc_phon_8")
    assert rule["transformation"]["from"] == "ā"
    assert "from_" not in rule["transformation"]


def test_search_buck_rules_exposes_typed_variants() -> None:
    """Rule variants should be returned as typed form/dialects objects."""
    output = search_buck_rules_for_mcp(McpBuckRuleSearchInput(section="134.1"))

    rule = next(rule for rule in output.rules if rule.id == "grc_morph_134_1")
    assert rule.variants
    forms = {variant.form for variant in rule.variants}
    assert "ei" in forms
    assert all(isinstance(variant.dialects, list) for variant in rule.variants)


def test_search_buck_rules_returns_deterministic_id_order() -> None:
    """Unfiltered rule search should be ordered by id, independent of file order."""
    output = search_buck_rules_for_mcp(McpBuckRuleSearchInput(category="vowels"))

    rule_ids = [rule.id for rule in output.rules]
    assert rule_ids == sorted(rule_ids)


def test_rule_without_transformation_reports_none() -> None:
    """Rules carrying no transformation should serialize transformation as null."""
    payload = _run_tool_call(
        "ancient_phonology.search_buck_rules",
        {"request": {"rule_id": "grc_orth_4_1"}},
    )

    assert payload["count"] == 1
    assert payload["rules"][0]["transformation"] is None
