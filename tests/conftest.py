"""Shared pytest fixtures for the test suite."""

import re
from collections.abc import Callable, Generator
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import pytest

from fastapi.testclient import TestClient

from api import main as api_main
from api.main import app
from phonology import search as search_module
from phonology.explainer import RuleApplication
from phonology.languages.ancient_greek.ipa import get_known_phones
from phonology.core.ports.profiles import LanguageProfile, get_default_language_profile
from phonology.search import LexiconRecord, SearchResult


def assert_uuid4_hex(value: str) -> None:
    """Assert that a request id is a server-generated UUID4 hex string."""
    assert re.fullmatch(r"[0-9a-f]{32}", value)


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
        lsx: The ``phonology.languages.ancient_greek.lsj_extractor`` module, or ``None`` if not yet imported.
    """
    if lsx is not None:
        lsx._pos_overrides = None


@pytest.fixture
def reset_pos_overrides_cache() -> Generator[None, None, None]:
    """Reset the lsj_extractor _pos_overrides cache before and after each test.

    Applied via ``pytestmark`` in test modules that exercise POS extraction
    (test_lsj_extractor, test_build_lexicon).
    """
    _clear_pos_overrides(sys.modules.get("phonology.languages.ancient_greek.lsj_extractor"))
    yield
    _clear_pos_overrides(sys.modules.get("phonology.languages.ancient_greek.lsj_extractor"))


@pytest.fixture(autouse=True)
def clear_rule_cache() -> Generator[None, None, None]:
    """Reset cached rule loading and tokenization state between tests.

    Autouse scoped at ``tests/`` because ``phonology.search._load_rules_cached``,
    ``_get_tokenized_rules``, and ``api._dependencies._get_rules_version_cached``
    are shared module-level ``lru_cache`` wrappers.
    Clearing before every test guarantees test isolation for any suite that
    exercises rule loading, and is a no-op (double-clear is harmless) for
    suites that do not. The rules-version cache is cleared on the canonical
    ``api._dependencies`` module so both REST and MCP surfaces are reset.
    """
    from api import _dependencies as api_deps

    search_module._load_rules_cached.cache_clear()
    search_module._get_tokenized_rules.cache_clear()
    api_deps._get_rules_version_cached.cache_clear()

    yield

    search_module._load_rules_cached.cache_clear()
    search_module._get_tokenized_rules.cache_clear()
    api_deps._get_rules_version_cached.cache_clear()


@pytest.fixture
def default_profile_follows_search_to_ipa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep legacy ``search_module.to_ipa`` monkeypatch seams working in tests."""
    base_profile = search_module.get_default_language_profile()
    original_to_ipa = search_module.to_ipa

    def converter(text: str, *, dialect: str) -> str:
        current_to_ipa = search_module.to_ipa
        if current_to_ipa is original_to_ipa:
            return base_profile.converter(text, dialect=dialect)
        return current_to_ipa(text, dialect=dialect)

    monkeypatch.setattr(
        search_module,
        "get_default_language_profile",
        lambda: replace(base_profile, converter=converter),
    )


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
    from phonology.core.ports.profiles import (
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


# ---------------------------------------------------------------------------
# Shared API search-stub helpers.
#
# These helpers were previously defined in ``tests/test_api_main.py`` and
# re-imported by ``test_api_request_id``, ``test_api_search_meta``, and
# ``test_api_verification``. Centralizing them here removes the fragile
# inter-test cross-imports while keeping each test module self-contained.
# ---------------------------------------------------------------------------


def _make_test_dependencies() -> dict[str, object]:
    """Return standard test dependency fixtures shared across API tests.

    Returns:
        A dict with the following keys:
            - "lexicon": A single lexicon entry dict with id, headword, ipa, dialect.
            - "matrix": A distance matrix dict mapping phone pairs to float distances.
            - "rules_registry": A dict mapping rule IDs to rule definitions.
            - "search_index": A dict mapping IPA strings to lexicon entry IDs.
            - "unigram_index": A dict mapping phones to lexicon entry IDs.
            - "lexicon_map": A dict mapping lexicon IDs to LexiconRecord instances.
            - "ipa_index": A dict mapping IPA strings to lexicon entry IDs.

    Example:
        >>> deps = _make_test_dependencies()
        >>> deps["lexicon"]
        {'id': 'L1', 'headword': 'λόγος', 'ipa': 'lóɡos', 'dialect': 'attic'}
        >>> isinstance(deps["lexicon_map"]["L1"], LexiconRecord)
        True
    """
    return {
        "lexicon": (
            {
                "id": "L1",
                "headword": "λόγος",
                "ipa": "lóɡos",
                "dialect": "attic",
            },
        ),
        "matrix": {"l": {"l": 0.0}},
        "rules_registry": {
            "CCH-001": {
                "id": "CCH-001",
                "input": "s",
                "output": "h",
                "dialects": ["attic"],
            }
        },
        "search_index": {"l ɡ": ["L1"]},
        "unigram_index": {"l": ["L1"]},
        "lexicon_map": {
            "L1": LexiconRecord(
                entry={
                    "id": "L1",
                    "headword": "λόγος",
                    "ipa": "lóɡos",
                    "dialect": "attic",
                },
                token_count=4,
            )
        },
        "ipa_index": {"lóɡos": ["L1"]},
    }


def _make_fake_search_execution(
    captured: dict[str, object],
    *,
    results: list[SearchResult] | None = None,
    truncated: bool = False,
) -> Callable[..., search_module.SearchExecutionResult]:
    """Build a fake public phonology search callable that records API arguments.

    Args:
        captured: A dict used to capture input arguments for assertions.
            The callable records query, lexicon, matrix, max_results, dialect,
            index, unigram_index, prebuilt_lexicon_map, language, converter,
            phone_inventory, query_ipa, prebuilt_ipa_index,
            similarity_fallback_limit, and unigram_fallback_limit.
        results: Optional prepopulated SearchResult list to return when provided.
            If None, returns a default single result for "λόγος".
        truncated: Controls the returned SearchExecutionResult.truncated flag.

    Returns:
        A callable that accepts search_execution parameters and returns
        search_module.SearchExecutionResult. The callable records all arguments
        into the captured dict and handles prepared_query/converter logic to
        compute effective query_ipa and query_mode.
    """

    def fake_search_execution(
        query: str,
        lexicon: tuple[dict[str, object], ...],
        matrix: dict[str, dict[str, float]],
        max_results: int,
        dialect: str,
        index: dict[str, list[str]],
        unigram_index: dict[str, list[str]] | None = None,
        prebuilt_lexicon_map: dict[str, object] | None = None,
        language: str = "ancient_greek",
        converter: Callable[..., str] | None = None,
        phone_inventory: tuple[str, ...] | None = None,
        dialect_skeleton_builders: object | None = None,
        query_ipa: str | None = None,
        prepared_query: object | None = None,
        prebuilt_ipa_index: dict[str, list[str]] | None = None,
        similarity_fallback_limit: int | None = None,
        unigram_fallback_limit: int | None = None,
    ) -> search_module.SearchExecutionResult:
        captured["query"] = query
        captured["lexicon"] = lexicon
        captured["matrix"] = matrix
        captured["max_results"] = max_results
        captured["dialect"] = dialect
        captured["index"] = index
        captured["unigram_index"] = unigram_index
        captured["prebuilt_lexicon_map"] = prebuilt_lexicon_map
        captured["language"] = language
        captured["converter"] = converter
        captured["phone_inventory"] = phone_inventory
        effective_query_ipa = query_ipa
        effective_query_mode = "Full-form"
        if prepared_query is not None:
            effective_query_ipa = getattr(prepared_query, "query_ipa", query_ipa)
            effective_query_mode = getattr(prepared_query, "query_mode", "Full-form")
        elif effective_query_ipa is None and converter is not None:
            # Reuse production prepare_query_ipa so query_mode heuristics
            # (Full-form / Short-query / Partial-form) stay in sync with
            # search_execution without hardcoding IPA per call site.
            prepared = search_module.prepare_query_ipa(
                query,
                dialect=dialect,
                converter=converter,
                phone_inventory=phone_inventory,
            )
            effective_query_ipa = prepared.query_ipa
            effective_query_mode = prepared.query_mode
        captured["query_ipa"] = effective_query_ipa
        captured["prebuilt_ipa_index"] = prebuilt_ipa_index
        captured["similarity_fallback_limit"] = similarity_fallback_limit
        captured["unigram_fallback_limit"] = unigram_fallback_limit
        search_results = results
        if search_results is None:
            search_results = [
                SearchResult(
                    lemma="λόγος",
                    confidence=0.75,
                    dialect_attribution="lemma dialect: attic",
                    applied_rules=["CCH-001"],
                    rule_applications=[
                        RuleApplication(
                            rule_id="CCH-001",
                            rule_name="CCH-001",
                            from_phone="s",
                            to_phone="h",
                            position=2,
                        )
                    ],
                    ipa="lóɡos",
                )
            ]
        return search_module.SearchExecutionResult(
            results=search_results,
            query_ipa=effective_query_ipa or "",
            query_mode=effective_query_mode,
            truncated=truncated,
        )

    return fake_search_execution


def mock_search_dependencies(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Stub all search dependencies and return a capture dict.

    Patches:
        - api_main._load_search_dependencies to return a SearchDependencies
          with test fixtures and a fake converter.
        - api_main.phonology_search.search_execution via
          _make_fake_search_execution to record API arguments.

    Returns:
        A capture dict with the following notable key:
            - "converter_queries": A list collecting queries seen by the
              fake_converter for assertion in tests.

    Example:
        >>> captured = mock_search_dependencies(monkeypatch)
        >>> # Make API call...
        >>> captured["converter_queries"]
        ['λόγος']
    """
    td = _make_test_dependencies()
    captured: dict[str, object] = {}
    converter_queries: list[str] = []
    captured["converter_queries"] = converter_queries

    def fake_converter(query: str, dialect: str = "attic") -> str:
        converter_queries.append(query)
        exceptional_conversions = {
            "νυν": "nyn",
            "νῦν": "nyn",
            "ζηταω": "zɛːtaɔ",
        }
        if query in exceptional_conversions:
            return exceptional_conversions[query]

        character_conversions = {
            "α": "a",
            "γ": "ɡ",
            "ι": "i",
            "λ": "l",
            "ο": "o",
            "ό": "o",
            "ς": "s",
            "σ": "s",
        }
        return "".join(
            character_conversions.get(character, character) for character in query
        )

    profile = replace(
        get_default_language_profile(),
        converter=fake_converter,
    )

    monkeypatch.setattr(
        api_main,
        "_load_search_dependencies",
        lambda _language: api_main.SearchDependencies(
            lexicon=td["lexicon"],
            matrix=td["matrix"],
            rules_registry=td["rules_registry"],
            search_index=td["search_index"],
            unigram_index=td["unigram_index"],
            lexicon_map=td["lexicon_map"],
            ipa_index=td["ipa_index"],
            data_versions=api_main.DataVersions(),
            profile=profile,
        ),
    )
    monkeypatch.setattr(
        api_main.phonology_search,
        "search_execution",
        _make_fake_search_execution(captured),
    )
    return captured
