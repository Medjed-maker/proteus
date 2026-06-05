"""Integration tests for language-independent profile registration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
import json
from pathlib import Path
import threading
from typing import Any

import pytest
from pydantic import ValidationError

from phonology.distance import load_matrix
from phonology.core.ports import profiles as profiles_module
from phonology.core.ports.profiles import (
    LanguageProfile,
    get_default_language_profile,
    get_language_profile,
    list_language_profiles,
    register_default_profiles,
    register_language_profile,
)
from phonology.core.ports.orthography_notes import OrthographicNotePayload
from phonology.languages.ancient_greek import build_orthographic_notes

# Import test-only function directly
from phonology.core.ports.profiles import _reset_language_registry_for_tests
from phonology import search as search_module
from phonology.search import search_execution
from api._models import SearchRequest
from tests.conftest import _toy_converter


class _FakeLanguageEntryPoint:
    """Small entry point test double with the subset profiles.py needs."""

    def __init__(
        self,
        name: str,
        builder: Callable[[], LanguageProfile],
    ) -> None:
        self.name = name
        self._builder = builder

    def load(self) -> Callable[[], LanguageProfile]:
        return self._builder


def test_explicit_default_profile_lookup_rediscovers_after_registry_reset(
    isolated_language_registry: None,
) -> None:
    default_language_id = "ancient_greek"
    default_profile = get_language_profile(default_language_id)

    assert default_profile.language_id == default_language_id
    assert get_default_language_profile() == default_profile


def test_register_default_profiles_discovers_entry_points_and_trusts_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    build_toy_profile: Callable[[Path, str], LanguageProfile],
) -> None:
    profile = build_toy_profile(tmp_path, "toy_entry")
    profile.matrix_path.write_text(json.dumps({"p": {"p": 0.0}}), encoding="utf-8")

    _reset_language_registry_for_tests()
    try:
        with monkeypatch.context() as patch_context:
            patch_context.setattr(
                profiles_module,
                "_language_profile_entry_points",
                lambda: (_FakeLanguageEntryPoint("toy_entry", lambda: profile),),
            )

            register_default_profiles()

        assert get_language_profile("toy_entry") == profile
        assert get_default_language_profile() == profile
        assert load_matrix(profile.matrix_path) == {"p": {"p": 0.0}}
    finally:
        _reset_language_registry_for_tests()
        register_default_profiles()


def test_explicit_language_profile_lookup_discovers_entry_points_when_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    build_toy_profile: Callable[[Path, str], LanguageProfile],
) -> None:
    profile = build_toy_profile(tmp_path, "toy_entry")

    _reset_language_registry_for_tests()
    try:
        with monkeypatch.context() as patch_context:
            patch_context.setattr(
                profiles_module,
                "_language_profile_entry_points",
                lambda: (_FakeLanguageEntryPoint("toy_entry", lambda: profile),),
            )

            assert get_language_profile("toy_entry") == profile
            with pytest.raises(ValueError, match="Unsupported language profile"):
                get_language_profile("unknown_toy")
    finally:
        _reset_language_registry_for_tests()
        register_default_profiles()


def test_register_default_profiles_requires_at_least_one_entry_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_language_registry_for_tests()
    try:
        with monkeypatch.context() as patch_context:
            patch_context.setattr(
                profiles_module,
                "_language_profile_entry_points",
                lambda: (),
            )

            with pytest.raises(
                ValueError,
                match="No language profile entry points registered",
            ):
                register_default_profiles()
            with pytest.raises(
                ValueError,
                match="No language profile entry points registered",
            ):
                get_default_language_profile()
    finally:
        _reset_language_registry_for_tests()
        register_default_profiles()


def test_default_language_env_selects_registered_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_language_registry: None,
    build_toy_profile: Callable[[Path, str], LanguageProfile],
) -> None:
    alpha = build_toy_profile(tmp_path, "toy_alpha")
    beta = build_toy_profile(tmp_path, "toy_beta")
    register_language_profile(alpha)
    register_language_profile(beta)
    monkeypatch.setenv("PROTEUS_DEFAULT_LANGUAGE", " Toy_Beta ")

    assert get_default_language_profile() == beta


def test_configured_default_profile_lookup_is_non_recursive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_language_registry: None,
    build_toy_profile: Callable[[Path, str], LanguageProfile],
) -> None:
    profile = build_toy_profile(tmp_path, "toy_default")
    register_language_profile(profile)
    monkeypatch.setenv("PROTEUS_DEFAULT_LANGUAGE", "toy_default")

    def fail_recursive_lookup(_language_id: str | None = None) -> LanguageProfile:
        raise AssertionError("get_default_language_profile must not call get_language_profile")

    monkeypatch.setattr(profiles_module, "get_language_profile", fail_recursive_lookup)

    assert get_default_language_profile() == profile


def test_default_profile_discovery_runs_once_for_concurrent_lazy_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_language_registry: None,
    build_toy_profile: Callable[[Path, str], LanguageProfile],
) -> None:
    profile = build_toy_profile(tmp_path, "toy_concurrent")
    calls = 0
    call_lock = threading.Lock()
    barrier = threading.Barrier(4)
    errors: list[BaseException] = []
    results: list[LanguageProfile] = []

    def fake_register_default_profiles() -> None:
        nonlocal calls
        with call_lock:
            calls += 1
        register_language_profile(profile)

    def worker() -> None:
        try:
            barrier.wait(timeout=5.0)
            results.append(get_language_profile("toy_concurrent"))
        except BaseException as exc:
            errors.append(exc)

    monkeypatch.setattr(
        profiles_module,
        "register_default_profiles",
        fake_register_default_profiles,
    )

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)

    alive_threads = [t for t in threads if t.is_alive()]
    assert not alive_threads, f"Test hung: threads failed to join (deadlocked?): {errors}"
    assert not errors, f"Exceptions occurred in workers: {errors}"
    assert calls == 1
    assert results == [profile, profile, profile, profile]


def test_single_registered_profile_is_default_without_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_language_registry: None,
    build_toy_profile: Callable[[Path, str], LanguageProfile],
) -> None:
    monkeypatch.delenv("PROTEUS_DEFAULT_LANGUAGE", raising=False)
    profile = build_toy_profile(tmp_path, "toy_single")
    register_language_profile(profile)

    assert get_default_language_profile() == profile


def test_multiple_registered_profiles_require_explicit_default_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_language_registry: None,
    build_toy_profile: Callable[[Path, str], LanguageProfile],
) -> None:
    monkeypatch.delenv("PROTEUS_DEFAULT_LANGUAGE", raising=False)
    register_language_profile(build_toy_profile(tmp_path, "toy_alpha"))
    register_language_profile(build_toy_profile(tmp_path, "toy_beta"))

    with pytest.raises(ValueError, match="PROTEUS_DEFAULT_LANGUAGE"):
        get_default_language_profile()


def test_list_language_profiles_snapshot_returns_registered_profiles(
    tmp_path: Path,
    isolated_language_registry: None,
    build_toy_profile: Callable[[Path, str], LanguageProfile],
) -> None:
    first_profile = build_toy_profile(tmp_path, "toy_alpha")
    second_profile = build_toy_profile(tmp_path, "toy_beta")
    register_language_profile(first_profile)

    snapshot = list_language_profiles()
    register_language_profile(second_profile)

    assert snapshot == (first_profile,)
    assert list_language_profiles() == (first_profile, second_profile)


def test_list_language_profiles_sorted_by_language_id(
    tmp_path: Path,
    isolated_language_registry: None,
    build_toy_profile: Callable[[Path, str], LanguageProfile],
) -> None:
    beta_profile = build_toy_profile(tmp_path, "toy_beta")
    alpha_profile = build_toy_profile(tmp_path, "toy_alpha")
    register_language_profile(beta_profile)
    register_language_profile(alpha_profile)

    assert list_language_profiles() == (alpha_profile, beta_profile)


def test_list_language_profiles_returns_tuple(
    tmp_path: Path,
    isolated_language_registry: None,
    build_toy_profile: Callable[[Path, str], LanguageProfile],
) -> None:
    register_language_profile(build_toy_profile(tmp_path, "toy_alpha"))

    assert isinstance(list_language_profiles(), tuple)


def test_default_rules_registry_rediscovers_default_profile_after_registry_reset(
    isolated_language_registry: None,
) -> None:
    """Default search rules remain available through entry point rediscovery."""
    default_language_id = "ancient_greek"
    registry = search_module.get_rules_registry(default_language_id)

    assert registry is not None
    assert len(registry) > 0
    assert get_language_profile(default_language_id).language_id == default_language_id


def test_search_request_uses_default_profile_after_registry_reset(
    isolated_language_registry: None,
) -> None:
    """Direct SearchRequest construction should not require explicit registration."""
    request = SearchRequest(query_form="λόγος")

    assert request.language == "ancient_greek"
    assert request.dialect_hint == "attic"


def test_search_request_still_rejects_unknown_language_after_registry_reset(
    isolated_language_registry: None,
) -> None:
    """Only the built-in default profile gets the lazy fallback."""
    with pytest.raises(ValidationError, match="invalid language profile"):
        SearchRequest(query_form="pa", language="missing_profile")


def test_language_profile_defaults_orthographic_note_builder_to_none(
    tmp_path: Path,
) -> None:
    rules_dir = tmp_path / "rules"
    matrix_dir = tmp_path / "matrices"
    rules_dir.mkdir()
    matrix_dir.mkdir()
    profile = LanguageProfile(
        language_id="toy_no_notes",
        display_name="Toy No Notes",
        default_dialect="toy",
        supported_dialects=("toy",),
        converter=_toy_converter,
        phone_inventory=("p", "a"),
        lexicon_path=tmp_path / "lexicon.json",
        matrix_path=matrix_dir / "matrix.json",
        rules_dir=rules_dir,
    )

    assert profile.orthographic_note_builder is None
    assert profile.always_match_contexts == ()


def test_language_profile_preserves_custom_orthographic_note_builder(
    tmp_path: Path,
    isolated_language_registry: None,
) -> None:
    rules_dir = tmp_path / "rules"
    matrix_dir = tmp_path / "matrices"
    rules_dir.mkdir()
    matrix_dir.mkdir()

    def toy_builder(
        *,
        query_form: str,
        candidate_headword: str,
        candidate_ipa: str,
        query_ipa: str,
        response_language: str,
        orthography_hint: str | None = None,
    ) -> list[OrthographicNotePayload]:
        return [
            OrthographicNotePayload(
                kind="beginner_aid",
                label="Toy note",
                messages=[f"{query_form} -> {candidate_headword}"],
                confidence="low",
            )
        ]

    profile = LanguageProfile(
        language_id="toy_notes",
        display_name="Toy Notes",
        default_dialect="toy",
        supported_dialects=("toy",),
        converter=_toy_converter,
        phone_inventory=("p", "a"),
        lexicon_path=tmp_path / "lexicon.json",
        matrix_path=matrix_dir / "matrix.json",
        rules_dir=rules_dir,
        orthographic_note_builder=toy_builder,
    )
    register_language_profile(profile)

    registered = get_language_profile("toy_notes")

    assert registered.orthographic_note_builder is toy_builder
    assert registered.orthographic_note_builder(
        query_form="pa",
        candidate_headword="ba",
        candidate_ipa="ba",
        query_ipa="pa",
        response_language="en",
    ) == [
        OrthographicNotePayload(
            kind="beginner_aid",
            label="Toy note",
            messages=["pa -> ba"],
            confidence="low",
        )
    ]


def test_default_ancient_greek_profile_sets_orthographic_note_builder() -> None:
    profile = get_default_language_profile()

    assert profile.orthographic_note_builder is build_orthographic_notes


def test_default_ancient_greek_profile_sets_always_match_contexts() -> None:
    profile = get_default_language_profile()

    assert profile.always_match_contexts == (
        "vowel contraction across hiatus",
        "quantitative metathesis environments",
    )


def test_language_profile_description_defaults_to_empty_string(
    tmp_path: Path,
    isolated_language_registry: None,
    build_toy_profile: Callable[[Path, str], LanguageProfile],
) -> None:
    profile = build_toy_profile(tmp_path, "toy_no_description")

    assert profile.description == ""


def test_default_ancient_greek_profile_has_non_empty_description() -> None:
    profile = get_default_language_profile()

    assert profile.description != ""


def test_toy_language_profile_runs_search_execution_without_core_changes(
    tmp_path: Path,
) -> None:
    language_dir = tmp_path / "toy_language"
    lexicon_dir = language_dir / "lexicon"
    matrices_dir = language_dir / "matrices"
    rules_dir = language_dir / "rules"
    lexicon_dir.mkdir(parents=True)
    matrices_dir.mkdir()
    rules_dir.mkdir()

    lexicon_path = lexicon_dir / "toy_lemmas.json"
    matrix_path = matrices_dir / "toy_matrix.json"
    rules_path = rules_dir / "toy_rules.yaml"

    lexicon_document = {
        "schema_version": "toy",
        "lemmas": [
            {
                "id": "toy-1",
                "headword": "ba",
                "ipa": "ba",
                "dialect": "toy",
            }
        ],
    }
    lexicon_path.write_text(
        json.dumps(lexicon_document, ensure_ascii=False),
        encoding="utf-8",
    )
    matrix_path.write_text(
        json.dumps(
            {
                "_meta": {"version": "toy"},
                "p": {"p": 0.0, "b": 0.1, "a": 1.0},
                "b": {"p": 0.1, "b": 0.0, "a": 1.0},
                "a": {"p": 1.0, "b": 1.0, "a": 0.0},
            }
        ),
        encoding="utf-8",
    )
    rules_path.write_text(
        """
schema_version: "1.0.0"
rules:
  - id: TOY-001
    name_en: Toy p to b
    name_ja: Toy p to b
    input: b
    output: p
    context: null
    dialects: [toy]
    period: test
    references: [test]
    examples:
      - standard: pa
        dialect: ba
        meaning: toy example
""".lstrip(),
        encoding="utf-8",
    )

    register_language_profile(
        LanguageProfile(
            language_id="toy_language",
            display_name="Toy Language",
            default_dialect="toy",
            supported_dialects=("toy",),
            converter=_toy_converter,
            phone_inventory=("p", "b", "a"),
            lexicon_path=lexicon_path,
            matrix_path=matrix_path,
            rules_dir=rules_dir,
        )
    )
    profile = get_language_profile("toy_language")
    lexicon = tuple(
        json.loads(profile.lexicon_path.read_text(encoding="utf-8"))["lemmas"]
    )
    matrix_document = json.loads(profile.matrix_path.read_text(encoding="utf-8"))
    matrix = {
        phone: row
        for phone, row in matrix_document.items()
        if not phone.startswith("_")
    }

    execution = search_execution(
        "pa",
        lexicon=lexicon,
        matrix=matrix,
        max_results=1,
        dialect=profile.default_dialect,
        language=profile.language_id,
        converter=profile.converter,
        phone_inventory=profile.phone_inventory,
    )

    assert [result.lemma for result in execution.results] == ["ba"]
    assert execution.results[0].applied_rules == ["TOY-001"]


def test_search_request_accepts_profile_specific_dialect_and_default(
    tmp_path: Path,
    isolated_language_registry: None,
) -> None:
    rules_dir = tmp_path / "rules"
    matrix_dir = tmp_path / "matrices"
    rules_dir.mkdir()
    matrix_dir.mkdir()
    profile = LanguageProfile(
        language_id="toy_request",
        display_name="Toy Request",
        default_dialect="toy",
        supported_dialects=("toy",),
        converter=_toy_converter,
        phone_inventory=("p", "b", "a"),
        lexicon_path=tmp_path / "lexicon.json",
        matrix_path=matrix_dir / "matrix.json",
        rules_dir=rules_dir,
    )
    register_language_profile(profile)

    explicit = SearchRequest(
        query_form="pa",
        language="toy_request",
        dialect_hint=" toy ",
    )
    defaulted = SearchRequest(query_form="pa", language="toy_request")

    assert explicit.language == "toy_request"
    assert explicit.dialect_hint == "toy"
    assert defaulted.dialect_hint == "toy"


def test_multichar_profile_phone_inventory_drives_search_and_rules(
    tmp_path: Path,
    isolated_language_registry: None,
) -> None:
    rules_dir = tmp_path / "rules"
    matrix_dir = tmp_path / "matrices"
    rules_dir.mkdir()
    matrix_dir.mkdir()
    (rules_dir / "rules.yaml").write_text(
        """
schema_version: "1.0.0"
rules:
  - id: TOY-TS
    name_en: Toy ts to p
    name_ja: Toy ts to p
    input: ts
    output: p
    context: null
    dialects: [toy]
    period: test
    references: [test]
""".lstrip(),
        encoding="utf-8",
    )
    profile = LanguageProfile(
        language_id="toy_multichar",
        display_name="Toy Multichar",
        default_dialect="toy",
        supported_dialects=("toy",),
        converter=_toy_converter,
        phone_inventory=("ts", "a", "p"),
        lexicon_path=tmp_path / "lexicon.json",
        matrix_path=matrix_dir / "matrix.json",
        rules_dir=rules_dir,
    )
    lexicon = ({"id": "toy-ts", "headword": "tsa", "ipa": "tsa", "dialect": "toy"},)
    matrix = {
        "ts": {"ts": 0.0, "p": 0.1, "a": 1.0},
        "p": {"ts": 0.1, "p": 0.0, "a": 1.0},
        "a": {"ts": 1.0, "p": 1.0, "a": 0.0},
    }
    register_language_profile(profile)
    execution = search_execution(
        "pa",
        lexicon=lexicon,
        matrix=matrix,
        max_results=1,
        dialect=profile.default_dialect,
        language=profile.language_id,
        converter=profile.converter,
        phone_inventory=profile.phone_inventory,
    )

    assert [result.lemma for result in execution.results] == ["tsa"]
    assert execution.results[0].applied_rules == ["TOY-TS"]


def test_multichar_phone_inventory_tokenizes_rule_lookahead_context(
    tmp_path: Path,
    isolated_language_registry: None,
) -> None:
    rules_dir = tmp_path / "rules"
    matrix_dir = tmp_path / "matrices"
    rules_dir.mkdir()
    matrix_dir.mkdir()
    (rules_dir / "rules.yaml").write_text(
        """
schema_version: "1.0.0"
rules:
  - id: TOY-CTX-TS
    name_en: Toy p to b before later ts
    name_ja: Toy p to b before later ts
    input: p
    output: b
    context: _...ts
    dialects: [toy]
    period: test
    references: [test]
""".lstrip(),
        encoding="utf-8",
    )
    profile = LanguageProfile(
        language_id="toy_multichar_context",
        display_name="Toy Multichar Context",
        default_dialect="toy",
        supported_dialects=("toy",),
        converter=_toy_converter,
        phone_inventory=("ts", "p", "b", "a"),
        lexicon_path=tmp_path / "lexicon.json",
        matrix_path=matrix_dir / "matrix.json",
        rules_dir=rules_dir,
    )
    lexicon = ({"id": "toy-pats", "headword": "pats", "ipa": "pats", "dialect": "toy"},)
    matrix = {
        "ts": {"ts": 0.0, "p": 1.0, "b": 1.0, "a": 1.0},
        "p": {"ts": 1.0, "p": 0.0, "b": 0.1, "a": 1.0},
        "b": {"ts": 1.0, "p": 0.1, "b": 0.0, "a": 1.0},
        "a": {"ts": 1.0, "p": 1.0, "b": 1.0, "a": 0.0},
    }
    register_language_profile(profile)
    execution = search_execution(
        "bats",
        lexicon=lexicon,
        matrix=matrix,
        max_results=1,
        dialect=profile.default_dialect,
        language=profile.language_id,
        converter=profile.converter,
        phone_inventory=profile.phone_inventory,
    )

    assert [result.lemma for result in execution.results] == ["pats"]
    assert execution.results[0].applied_rules == ["TOY-CTX-TS"]


def test_non_default_search_execution_backfills_profile_converter_and_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_language_registry: None,
) -> None:
    rules_dir = tmp_path / "rules"
    matrix_dir = tmp_path / "matrices"
    rules_dir.mkdir()
    matrix_dir.mkdir()
    (rules_dir / "rules.yaml").write_text(
        "schema_version: '1.0.0'\nrules: []\n",
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def converter(text: str, *, dialect: str) -> str:
        captured["text"] = text
        captured["dialect"] = dialect
        return "tsa"

    profile = LanguageProfile(
        language_id="toy_profile_defaults",
        display_name="Toy Profile Defaults",
        default_dialect="toy",
        supported_dialects=("toy",),
        converter=converter,
        phone_inventory=("ts", "a", "p"),
        lexicon_path=tmp_path / "lexicon.json",
        matrix_path=matrix_dir / "matrix.json",
        rules_dir=rules_dir,
    )
    register_language_profile(profile)

    def fake_execute_search(
        query: str, *args: object, **kwargs: Any
    ) -> search_module.SearchExecutionResult:
        captured["phone_inventory"] = tuple(kwargs["phone_inventory"])
        captured["query_ipa"] = kwargs["converter"](
            query,
            dialect=kwargs["dialect"],
        )
        return search_module.SearchExecutionResult(
            results=[],
            query_ipa=str(captured["query_ipa"]),
        )

    monkeypatch.setattr(search_module, "_execute_search", fake_execute_search)
    execution = search_execution(
        "orthographicquery",
        lexicon=(
            {
                "id": "toy-tsa",
                "headword": "tsa",
                "ipa": "tsa",
                "dialect": "toy",
            },
        ),
        matrix={"ts": {"ts": 0.0}, "a": {"a": 0.0}},
        max_results=1,
        dialect=profile.default_dialect,
        language=profile.language_id,
    )

    assert captured["text"] == "orthographicquery"
    assert captured["dialect"] == "toy"
    assert captured["phone_inventory"] == profile.phone_inventory
    assert execution.query_ipa == "tsa"


def test_register_language_profile_trusts_matrix_dir(
    tmp_path: Path,
    isolated_language_registry: None,
) -> None:
    """register_language_profile wires the matrix dir into the trusted set."""
    matrices_dir = tmp_path / "matrices"
    matrices_dir.mkdir()
    matrix_path = matrices_dir / "toy.json"
    matrix_path.write_text(
        json.dumps({"a": {"a": 0.0, "b": 1.0}, "b": {"a": 1.0, "b": 0.0}}),
        encoding="utf-8",
    )
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    (rules_dir / "rules.yaml").write_text(
        "schema_version: '1.0.0'\nrules: []\n", encoding="utf-8"
    )

    profile = LanguageProfile(
        language_id="trust_test",
        display_name="Trust Test",
        default_dialect="test",
        supported_dialects=("test",),
        converter=lambda t, **_: t,
        phone_inventory=("a", "b"),
        lexicon_path=tmp_path / "lex.json",
        matrix_path=matrix_path,
        rules_dir=rules_dir,
    )
    register_language_profile(profile)
    result = load_matrix(matrix_path)
    assert "a" in result


def test_register_language_profile_does_not_partially_register_on_trust_failure(
    tmp_path: Path,
    isolated_language_registry: None,
) -> None:
    """Failed trusted-path setup must not leave the profile registered."""
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    profile = LanguageProfile(
        language_id="broken_trust",
        display_name="Broken Trust",
        default_dialect="test",
        supported_dialects=("test",),
        converter=lambda t, **_: t,
        phone_inventory=("a",),
        lexicon_path=tmp_path / "lex.json",
        matrix_path=tmp_path / "missing_matrices" / "matrix.json",
        rules_dir=rules_dir,
    )

    with pytest.raises(FileNotFoundError):
        register_language_profile(profile)
    with pytest.raises(ValueError, match="Unsupported language profile"):
        get_language_profile("broken_trust")


def test_unregistered_matrix_dir_is_rejected(tmp_path: Path) -> None:
    """Loading a matrix from an unregistered dir raises ValueError."""
    matrices_dir = tmp_path / "unregistered"
    matrices_dir.mkdir()
    matrix_path = matrices_dir / "bad.json"
    matrix_path.write_text(json.dumps({"a": {"a": 0.0}}), encoding="utf-8")

    with pytest.raises(ValueError, match="trusted directory"):
        load_matrix(matrix_path)


def test_reset_clears_trusted_dirs(tmp_path: Path) -> None:
    """_reset_language_registry_for_tests clears both trusted-dir sets."""
    matrices_dir = tmp_path / "matrices"
    matrices_dir.mkdir()
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()

    from phonology.distance import register_trusted_matrices_dir
    from phonology.explainer import register_trusted_rules_dir

    register_trusted_matrices_dir(matrices_dir)
    register_trusted_rules_dir(rules_dir)

    _reset_language_registry_for_tests()

    # After reset the sets should be empty (or contain only re-registered defaults).
    # Verify the unregistered dirs are no longer trusted.
    matrix_path = matrices_dir / "m.json"
    matrix_path.write_text(json.dumps({"a": {"a": 0.0}}), encoding="utf-8")
    with pytest.raises(ValueError, match="trusted directory"):
        load_matrix(matrix_path)

    # Restore default profiles so subsequent tests work.
    register_default_profiles()


def test_custom_profile_does_not_apply_koine_skeleton(
    tmp_path: Path,
    isolated_language_registry: None,
) -> None:
    """Custom profiles with empty dialect_skeleton_builders must not get Koine shifts.

    A toy profile with /b/ in its lexicon must not match a query with /ð/
    (the Koine shift of intervocalic /b/). Empty dialect_skeleton_builders
    suppresses the Koine index augmentation introduced for Ancient Greek.
    """
    rules_dir = tmp_path / "rules"
    matrix_dir = tmp_path / "matrices"
    rules_dir.mkdir()
    matrix_dir.mkdir()
    (rules_dir / "rules.yaml").write_text(
        "schema_version: '1.0.0'\nrules: []\n", encoding="utf-8"
    )
    lexicon = ({"id": "toy-1", "headword": "aba", "ipa": "aba", "dialect": "toy"},)
    profile = LanguageProfile(
        language_id="toy_no_koine",
        display_name="Toy No Koine",
        default_dialect="toy",
        supported_dialects=("toy",),
        converter=_toy_converter,
        phone_inventory=("a", "b", "ð"),
        lexicon_path=tmp_path / "lexicon.json",
        matrix_path=matrix_dir / "matrix.json",
        rules_dir=rules_dir,
        dialect_skeleton_builders=(),
    )
    register_language_profile(profile)
    from phonology.search import build_kmer_index, seed_stage

    index = build_kmer_index(
        lexicon,
        phone_inventory=profile.phone_inventory,
        dialect_skeleton_builders=profile.dialect_skeleton_builders,
    )
    # "aða" contains consonant skeleton ["ð"]; "aba" has skeleton ["b"].
    # With Koine builders, "b" → "ð" would be in the index; without them, it must not be.
    candidates = seed_stage("aða", index, phone_inventory=profile.phone_inventory)
    assert "toy-1" not in candidates, (
        "Custom profile with empty dialect_skeleton_builders must not match "
        "via Koine-shifted skeleton (b → ð)"
    )


def test_default_language_id_can_use_custom_converter_without_api_guard() -> None:
    """Custom converters are passed through the public search boundary."""
    default_profile = get_default_language_profile()
    custom_profile = replace(default_profile, converter=lambda t, **_: "custom")
    result = search_execution(
        "anything",
        lexicon=(),
        matrix={},
        max_results=1,
        converter=custom_profile.converter,
    )

    assert result.query_ipa == "custom"
