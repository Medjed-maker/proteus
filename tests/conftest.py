"""Shared pytest fixtures for the test suite."""

from collections.abc import Generator
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import pytest

from fastapi.testclient import TestClient

from api.main import app
from phonology import search as search_module
from phonology.search import SearchResult


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Yield ``TestClient`` while ``app.state.disable_startup_warmup`` skips startup warmup."""
    app.state.disable_startup_warmup = True
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.state.disable_startup_warmup = False


def _clear_pos_overrides(lsx: ModuleType | None) -> None:
    """Clear the module's ``_pos_overrides`` cache attribute.

    Args:
        lsx: The ``phonology.lsj_extractor`` module, or ``None`` if not yet imported.
    """
    if lsx is not None:
        lsx._pos_overrides = None


@pytest.fixture
def reset_pos_overrides_cache() -> Generator[None, None, None]:
    """Reset the lsj_extractor _pos_overrides cache before and after each test.

    Applied via ``pytestmark`` in test modules that exercise POS extraction
    (test_lsj_extractor, test_build_lexicon).
    """
    _clear_pos_overrides(sys.modules.get("phonology.lsj_extractor"))
    yield
    _clear_pos_overrides(sys.modules.get("phonology.lsj_extractor"))


@pytest.fixture(autouse=True)
def clear_rule_cache() -> Generator[None, None, None]:
    """Reset cached rule registry state between tests.

    Autouse scoped at ``tests/`` because ``phonology.search.get_rules_registry``
    and ``_get_tokenized_rules`` are shared module-level ``lru_cache`` wrappers.
    Clearing before every test guarantees test isolation for any suite that
    exercises rule loading, and is a no-op (double-clear is harmless) for
    suites that do not.
    """
    search_module.get_rules_registry.cache_clear()
    search_module._get_tokenized_rules.cache_clear()

    yield

    search_module.get_rules_registry.cache_clear()
    search_module._get_tokenized_rules.cache_clear()


@pytest.fixture
def sample_search_results() -> list[SearchResult]:
    """Return a small unsorted result list for ranking tests."""
    return [
        SearchResult(lemma="γ", confidence=0.5, dialect_attribution="lemma dialect: attic"),
        SearchResult(lemma="α", confidence=0.9, dialect_attribution="lemma dialect: attic"),
        SearchResult(lemma="β", confidence=0.9, dialect_attribution="lemma dialect: attic"),
    ]


@pytest.fixture
def sample_lexicon() -> list[dict[str, str]]:
    """Return a compact lexicon fixture with deterministic ids and IPA."""
    return [
        {"id": "L1", "headword": "πτην", "ipa": "pten", "dialect": "attic"},
        {"id": "L2", "headword": "πτω", "ipa": "pto", "dialect": "attic"},
        {"id": "L3", "headword": "κτην", "ipa": "kten", "dialect": "doric"},
    ]


@pytest.fixture(scope="session")
def translations_data() -> dict[str, Any]:
    """Load and return translations.json data for tests."""
    path = Path(__file__).resolve().parent.parent / "src/web/static/translations.json"
    if not path.exists():
        pytest.fail(f"translations.json not found at {path}")
    with path.open(encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            pytest.fail(f"Failed to parse translations.json: {e}")
