"""Integration tests for language-independent profile registration."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from phonology.distance import load_matrix
from phonology.profiles import (
    LanguageProfile,
    get_default_language_profile,
    get_language_profile,
    register_default_profiles,
    register_language_profile,
)

# Import test-only function directly
from phonology.profiles import _reset_language_registry_for_tests
from phonology import search as search_module
from phonology.search import search_execution
from api._models import SearchRequest


def _toy_converter(text: str, *, dialect: str = "toy") -> str:
    """Return toy-language input as compact IPA."""
    if dialect != "toy":
        raise NotImplementedError(f"Unsupported toy dialect: {dialect!r}")
    return text


def test_default_profiles_are_explicitly_registered_and_resettable(
    isolated_language_registry: None,
) -> None:
    default_language_id = "ancient_greek"
    with pytest.raises(ValueError, match="Unsupported language profile"):
        get_language_profile(default_language_id)

    register_default_profiles()
    default_profile = get_default_language_profile()

    assert get_language_profile(default_language_id) == default_profile


def test_default_rules_registry_uses_default_profile_after_registry_reset(
    isolated_language_registry: None,
) -> None:
    """Default search rules remain available without API import side effects."""
    default_language_id = "ancient_greek"
    registry = search_module.get_rules_registry(default_language_id)

    assert registry is not None
    assert len(registry) > 0

    with pytest.raises(ValueError, match="Unsupported language profile"):
        get_language_profile(default_language_id)


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
    lexicon = tuple(json.loads(profile.lexicon_path.read_text(encoding="utf-8"))["lemmas"])
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
        phone_inventory=("ts", "p", "a"),
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
    lexicon = (
        {"id": "toy-1", "headword": "aba", "ipa": "aba", "dialect": "toy"},
    )
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


def test_default_language_id_with_custom_converter_raises() -> None:
    """_get_profile_converter raises when default language id has a custom converter.
    
    NOTE: This test intentionally targets an internal helper function and may break
    on refactor. The test is kept because it validates specific behavior that
    should be preserved through the public API.
    """
    from api import main as api_main

    default_profile = get_default_language_profile()
    custom_profile = dataclasses.replace(
        default_profile, converter=lambda t, **_: "custom"
    )
    with pytest.raises(RuntimeError, match="custom converter"):
        api_main._get_profile_converter(custom_profile)
