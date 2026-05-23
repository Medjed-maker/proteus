"""Tests for source references attached to search results."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from api import main as api_main
from api._hit_formatting import _build_search_hit
from api._models import SearchHit
from mcp_server._search_adapter import _run_search_for_mcp
from mcp_server.tools.search import McpSearchInput
from phonology import search as phonology_search
from phonology.corpus import CorpusAdapter, SourceReference
from phonology.distance import MatrixData
from phonology.profiles import get_default_language_profile
from phonology.search import SearchResult
from tests.conftest import (
    _make_fake_search_execution,
    _make_test_dependencies,
    mock_search_dependencies,
)


FIXTURE_ENTRY_ID = "L1"
FIXTURE_LEMMA = "λόγος"
FIXTURE_IPA = "lóɡos"
FIXTURE_DIALECT_ATTRIBUTION = "lemma dialect: attic"
FIXTURE_SOURCE_ID = "Perseus:text:1999.04.0057:entry=lo/gos"
FIXTURE_EXTERNAL_URL = (
    "https://www.perseus.tufts.edu/hopper/text"
    "?doc=Perseus:text:1999.04.0057:entry=lo/gos"
)
FIXTURE_LICENSE = "Perseus Digital Library reference metadata."


def _make_source_reference() -> SourceReference:
    return SourceReference(
        source_id=FIXTURE_SOURCE_ID,
        corpus="Perseus Digital Library LSJ",
        short_citation="LSJ, logos",
        external_url=FIXTURE_EXTERNAL_URL,
        license_note=FIXTURE_LICENSE,
        access_policy="open_metadata",
        citation_ready=False,
    )


def _make_default_result(entry_id: str | None = FIXTURE_ENTRY_ID) -> SearchResult:
    return SearchResult(
        lemma=FIXTURE_LEMMA,
        confidence=0.75,
        dialect_attribution=FIXTURE_DIALECT_ATTRIBUTION,
        ipa=FIXTURE_IPA,
        entry_id=entry_id,
    )


class SingleReferenceAdapter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self.reference = _make_source_reference()

    def lookup(
        self,
        *,
        entry_id: str,
        headword: str,
        language: str,
    ) -> tuple[SourceReference, ...]:
        self.calls.append((entry_id, headword, language))
        if entry_id == FIXTURE_ENTRY_ID:
            return (self.reference,)
        return ()


class FailingAdapter:
    def lookup(
        self,
        *,
        entry_id: str,
        headword: str,
        language: str,
    ) -> tuple[SourceReference, ...]:
        raise RuntimeError("adapter unavailable")


def _install_search_with_adapter(
    monkeypatch: pytest.MonkeyPatch,
    adapter: CorpusAdapter,
    *,
    result: SearchResult | None = None,
) -> CorpusAdapter:
    td = _make_test_dependencies()
    profile = replace(
        get_default_language_profile(),
        converter=lambda text, *, dialect: FIXTURE_IPA,
    )
    selected_result = result or _make_default_result()

    def load_dependencies(_language: str) -> api_main.SearchDependencies:
        return api_main.SearchDependencies(
            lexicon=cast(tuple[dict[str, Any], ...], td["lexicon"]),
            matrix=cast(MatrixData, td["matrix"]),
            rules_registry=cast(dict[str, dict[str, Any]], td["rules_registry"]),
            search_index=cast(phonology_search.KmerIndex, td["search_index"]),
            unigram_index=cast(phonology_search.KmerIndex, td["unigram_index"]),
            lexicon_map=cast(phonology_search.LexiconMap, td["lexicon_map"]),
            ipa_index=cast(phonology_search.IpaIndex, td["ipa_index"]),
            data_versions=api_main.DataVersions(),
            profile=profile,
            corpus_adapter=adapter,
        )

    monkeypatch.setattr(api_main, "_load_search_dependencies", load_dependencies)
    monkeypatch.setattr(api_main, "load_search_dependencies", load_dependencies)
    monkeypatch.setattr(
        api_main.phonology_search,
        "search_execution",
        _make_fake_search_execution({}, results=[selected_result]),
    )
    return adapter


def test_build_search_hit_accepts_source_references_argument() -> None:
    reference = _make_source_reference()
    result = _make_default_result()

    hit = _build_search_hit(
        result,
        query_ipa=FIXTURE_IPA,
        rules_registry={},
        query_mode="Full-form",
        source_references=(reference,),
    )

    assert isinstance(hit, SearchHit)
    assert hit.source_references == [reference]


def test_build_search_hit_defaults_source_references_to_empty_list() -> None:
    hit = _build_search_hit(
        _make_default_result(),
        query_ipa=FIXTURE_IPA,
        rules_registry={},
        query_mode="Full-form",
    )

    assert hit.source_references == []


def test_search_hit_includes_source_references_field(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_search_dependencies(monkeypatch)

    response = client.post("/search", json={"query_form": FIXTURE_LEMMA})

    assert response.status_code == 200
    assert response.json()["hits"][0]["source_references"] == []


def test_search_hit_includes_adapter_source_reference(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = SingleReferenceAdapter()
    _install_search_with_adapter(monkeypatch, adapter)

    response = client.post("/search", json={"query_form": FIXTURE_LEMMA})

    assert response.status_code == 200
    references = response.json()["hits"][0]["source_references"]
    assert references[0]["source_id"] == FIXTURE_SOURCE_ID
    assert references[0]["external_url"].startswith("https://www.perseus.tufts.edu/")
    assert "Perseus" in references[0]["license_note"]
    assert adapter.calls == [(FIXTURE_ENTRY_ID, FIXTURE_LEMMA, "ancient_greek")]


def test_search_hit_source_references_empty_for_non_fixture_entry(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_search_with_adapter(
        monkeypatch,
        SingleReferenceAdapter(),
        result=SearchResult(
            lemma="missing",
            confidence=0.75,
            dialect_attribution=FIXTURE_DIALECT_ATTRIBUTION,
            ipa="m",
            entry_id="missing",
        ),
    )

    response = client.post("/search", json={"query_form": FIXTURE_LEMMA})

    assert response.status_code == 200
    assert response.json()["hits"][0]["source_references"] == []


def test_search_hit_source_references_empty_without_entry_id(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = SingleReferenceAdapter()
    _install_search_with_adapter(
        monkeypatch,
        adapter,
        result=_make_default_result(entry_id=None),
    )

    response = client.post("/search", json={"query_form": FIXTURE_LEMMA})

    assert response.status_code == 200
    assert response.json()["hits"][0]["source_references"] == []
    assert adapter.calls == []


def test_adapter_failure_degrades_to_empty_source_references(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _install_search_with_adapter(monkeypatch, FailingAdapter())

    with caplog.at_level("WARNING", logger="api.main"):
        response = client.post("/search", json={"query_form": FIXTURE_LEMMA})

    assert response.status_code == 200
    assert response.json()["hits"][0]["source_references"] == []
    assert "Corpus adapter lookup failed" in caplog.text


def test_mcp_search_response_includes_source_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_search_with_adapter(monkeypatch, SingleReferenceAdapter())

    output = _run_search_for_mcp(McpSearchInput(query_form=FIXTURE_LEMMA))

    assert output.candidates[0].source_references
    assert output.candidates[0].source_references[0].source_id == FIXTURE_SOURCE_ID
