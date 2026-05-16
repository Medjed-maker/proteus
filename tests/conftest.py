"""Shared pytest fixtures for the test suite."""

from collections.abc import Callable, Generator
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import pytest

from fastapi.testclient import TestClient

from api.main import app
from phonology import search as search_module
from phonology.languages.ancient_greek.ipa import get_known_phones
from phonology.profiles import LanguageProfile
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
    """Reset cached rule loading and tokenization state between tests.

    Autouse scoped at ``tests/`` because ``phonology.search._load_rules_cached``,
    ``_get_tokenized_rules``, and ``api.main._get_rules_version_cached``
    are shared module-level ``lru_cache`` wrappers.
    Clearing before every test guarantees test isolation for any suite that
    exercises rule loading, and is a no-op (double-clear is harmless) for
    suites that do not.
    """
    from api import main as api_main

    search_module._load_rules_cached.cache_clear()
    search_module._get_tokenized_rules.cache_clear()
    api_main._get_rules_version_cached.cache_clear()

    yield

    search_module._load_rules_cached.cache_clear()
    search_module._get_tokenized_rules.cache_clear()
    api_main._get_rules_version_cached.cache_clear()


@pytest.fixture(scope="session")
def known_phones() -> tuple[str, ...]:
    """Return the known IPA phone inventory once for search tests."""
    return tuple(get_known_phones())


@pytest.fixture
def isolated_language_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[None, None, None]:
    """Reset the language profile registry to empty before the test, restore after.

    Use in tests that probe the registry's pre-registration state or register
    custom toy profiles. Replaces the manual try/finally pattern that calls
    ``_reset_language_registry_for_tests()`` and ``register_default_profiles()``.

    Also forbids implicit ``phonology.search.to_ipa`` calls during isolation:
    ``_legacy_to_ipa`` (the module-level seam) routes through
    ``get_default_language_profile()``, which silently rebuilds the default
    profile and re-registers it, defeating registry isolation. Tests that need
    IPA conversion must pass ``converter=`` explicitly.
    """
    from phonology.profiles import (
        _reset_language_registry_for_tests,
        register_default_profiles,
    )

    def _forbid_implicit_to_ipa(*_args: object, **_kwargs: object) -> str:
        raise AssertionError(
            "isolated_language_registry: implicit search_module.to_ipa is "
            "forbidden; pass converter=… explicitly."
        )

    _reset_language_registry_for_tests()
    monkeypatch.setattr(search_module, "to_ipa", _forbid_implicit_to_ipa)
    try:
        yield
    finally:
        _reset_language_registry_for_tests()
        register_default_profiles()


def _toy_converter(text: str, *, dialect: str = "toy") -> str:
    """Return toy-language input as compact IPA."""
    if dialect != "toy":
        raise NotImplementedError(f"Unsupported toy dialect: {dialect!r}")
    return text


@pytest.fixture
def build_toy_profile() -> Callable[[Path, str], LanguageProfile]:
    """Return a builder for minimal registered-language profiles."""

    def _build_toy_profile(tmp_path: Path, language_id: str) -> LanguageProfile:
        language_dir = tmp_path / language_id
        rules_dir = language_dir / "rules"
        matrix_dir = language_dir / "matrices"
        lexicon_dir = language_dir / "lexicon"
        rules_dir.mkdir(parents=True)
        matrix_dir.mkdir()
        lexicon_dir.mkdir()
        return LanguageProfile(
            language_id=language_id,
            display_name=language_id.replace("_", " ").title(),
            default_dialect="toy",
            supported_dialects=("toy",),
            converter=_toy_converter,
            phone_inventory=("p", "a"),
            lexicon_path=lexicon_dir / "lemmas.json",
            matrix_path=matrix_dir / "matrix.json",
            rules_dir=rules_dir,
        )

    return _build_toy_profile


@pytest.fixture
def sample_search_results() -> list[SearchResult]:
    """Return a small unsorted result list for ranking tests."""
    return [
        SearchResult(
            lemma="γ", confidence=0.5, dialect_attribution="lemma dialect: attic"
        ),
        SearchResult(
            lemma="α", confidence=0.9, dialect_attribution="lemma dialect: attic"
        ),
        SearchResult(
            lemma="β", confidence=0.9, dialect_attribution="lemma dialect: attic"
        ),
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
