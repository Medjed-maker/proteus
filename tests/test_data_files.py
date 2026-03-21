"""Validation tests for committed data files."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator, FormatChecker

from proteus.phonology.matrix_generator import build_attic_doric_matrix


ROOT_DIR = Path(__file__).resolve().parents[1]
LEXICON_PATH = ROOT_DIR / "data" / "lexicon" / "greek_lemmas.json"
LEXICON_SCHEMA_PATH = ROOT_DIR / "data" / "lexicon" / "greek_lemmas.schema.json"
MATRIX_PATH = ROOT_DIR / "data" / "matrices" / "attic_doric.json"
CONSONANT_RULES_PATH = ROOT_DIR / "data" / "rules" / "ancient_greek" / "consonant_changes.yaml"
VOWEL_RULES_PATH = ROOT_DIR / "data" / "rules" / "ancient_greek" / "vowel_shifts.yaml"
PHONOLOGY_RULES_DOC_PATH = ROOT_DIR / "docs" / "phonology_rules.md"


@pytest.fixture
def base_meta() -> dict[str, object]:
    return {
        "source": "LSJ",
        "encoding": "Unicode NFC",
        "ipa_system": "scholarly Ancient Greek IPA",
        "dialect": "attic",
        "version": "1.0.0",
        "last_updated": "2026-03-19T00:00:00Z",
        "license": "MIT",
        "contributors": ["Proteus maintainers"],
        "data_schema_ref": "data/lexicon/greek_lemmas.schema.json",
        "description": "Test fixture",
    }


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _find_rule(rules: list[dict], rule_id: str) -> dict:
    rule = next((candidate for candidate in rules if candidate["id"] == rule_id), None)
    if rule is None:
        raise LookupError(f"Rule {rule_id!r} not found in rule list of length {len(rules)}")
    return rule


def assert_dicts_close(
    expected: object,
    actual: object,
    *,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-12,
    path: str = "root",
) -> None:
    """Recursively compare nested dict/list structures with numeric tolerance.

    Args:
        expected: Expected value, typically a nested dict/list/scalar structure.
        actual: Actual value to compare against ``expected``.
        rel_tol: Relative tolerance passed to ``math.isclose`` for numeric leaves.
        abs_tol: Absolute tolerance passed to ``math.isclose`` for numeric leaves.
        path: Dot/bracket path prefix used in assertion messages during recursion.

    Returns:
        None. The function recurses through nested dicts and lists, checking dict
        key equality and per-key values, list lengths and element pairs, numeric
        closeness for non-bool ints/floats via ``math.isclose``, and plain equality
        for all other leaves. Raises ``AssertionError`` with a path-prefixed
        message on the first mismatch encountered.
    """
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected dict, got {type(actual).__name__}"
        assert set(expected) == set(actual), f"{path}: keys differ"
        for key in expected:
            assert_dicts_close(
                expected[key],
                actual[key],
                rel_tol=rel_tol,
                abs_tol=abs_tol,
                path=f"{path}.{key}",
            )
        return

    if isinstance(expected, list):
        assert isinstance(actual, list), f"{path}: expected list, got {type(actual).__name__}"
        assert len(expected) == len(actual), f"{path}: lengths differ"
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual, strict=True)):
            assert_dicts_close(
                expected_item,
                actual_item,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
                path=f"{path}[{index}]",
            )
        return

    if (
        isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and isinstance(actual, (int, float))
        and not isinstance(actual, bool)
    ):
        assert math.isclose(
            float(expected),
            float(actual),
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        ), f"{path}: {expected!r} != {actual!r}"
        return

    assert expected == actual, f"{path}: {expected!r} != {actual!r}"


def test_lexicon_metadata_and_psykhe_stress() -> None:
    lexicon = _load_json(LEXICON_PATH)
    metadata = lexicon["_meta"]

    assert lexicon["schema_version"] == "2.0.0"
    assert metadata["version"] == "1.0.0"
    assert metadata["last_updated"] == "2026-03-19T00:00:00Z"
    assert metadata["license"] == "MIT"
    assert metadata["contributors"] == ["Proteus maintainers"]
    assert metadata["data_schema_ref"] == "data/lexicon/greek_lemmas.schema.json"
    assert "phonological search experiments" in metadata["description"]

    psykhe = next(lemma for lemma in lexicon["lemmas"] if lemma["headword"] == "ψυχή")
    assert psykhe["ipa"] == "psyːkʰɛ́ː"


def test_lexicon_matches_committed_json_schema() -> None:
    lexicon = _load_json(LEXICON_PATH)
    schema = _load_json(LEXICON_SCHEMA_PATH)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())

    errors = sorted(validator.iter_errors(lexicon), key=lambda error: list(error.path))
    assert not errors, [error.message for error in errors]


def test_lexicon_dialect_is_consistent_across_metadata_and_lemmas() -> None:
    lexicon = _load_json(LEXICON_PATH)
    top_level_dialect = lexicon["_meta"]["dialect"]

    assert top_level_dialect == "attic"
    assert {lemma["dialect"] for lemma in lexicon["lemmas"]} == {top_level_dialect}


def test_lexicon_schema_allows_extension_and_conditional_gender_rules(
    base_meta: dict[str, object],
) -> None:
    schema = _load_json(LEXICON_SCHEMA_PATH)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())

    root_extension = {
        "schema_version": "2.0.0",
        "_meta": base_meta,
        "lemmas": [],
        "future_field": {"status": "reserved"},
    }
    assert not list(validator.iter_errors(root_extension))

    verb_without_gender = {
        "id": "LSJ-123456",
        "headword": "λέγω",
        "transliteration": "legō",
        "ipa": "léɡɔː",
        "pos": "verb",
        "gloss": "say",
        "dialect": "attic",
    }
    assert not list(
        validator.iter_errors(
            {
                "schema_version": "2.0.0",
                "_meta": base_meta,
                "lemmas": [verb_without_gender],
            }
        )
    )

    noun_without_gender = {
        "id": "LSJ-123457",
        "headword": "λόγος",
        "transliteration": "logos",
        "ipa": "lóɡos",
        "pos": "noun",
        "gloss": "word",
        "dialect": "attic",
    }
    errors = list(
        validator.iter_errors(
            {
                "schema_version": "2.0.0",
                "_meta": base_meta,
                "lemmas": [noun_without_gender],
            }
        )
    )
    assert any("gender" in error.message for error in errors)


def test_lexicon_schema_accepts_structured_contributors_and_rejects_extra_keys(
    base_meta: dict[str, object],
) -> None:
    schema = _load_json(LEXICON_SCHEMA_PATH)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())

    structured_meta = dict(base_meta)
    structured_meta["contributors"] = [
        {
            "name": "Proteus Maintainer",
            "email": "maintainer@example.com",
            "role": "editor",
            "affiliation": "Proteus",
            "orcid": "0000-0000-0000-0000",
        }
    ]

    assert not list(
        validator.iter_errors(
            {
                "schema_version": "2.0.0",
                "_meta": structured_meta,
                "lemmas": [],
            }
        )
    )

    invalid_meta = {
        **structured_meta,
        "contributors": [
            {
                "name": "Proteus Maintainer",
                "email": "maintainer@example.com",
                "nickname": "pm",
            }
        ],
    }
    errors = list(
        validator.iter_errors(
            {
                "schema_version": "2.0.0",
                "_meta": invalid_meta,
                "lemmas": [],
            }
        )
    )
    assert any("nickname" in error.message for error in errors)

    invalid_orcid_meta = {
        **structured_meta,
        "contributors": [
            {
                "name": "Proteus Maintainer",
                "email": "maintainer@example.com",
                "role": "editor",
                "affiliation": "Proteus",
                "orcid": "0000-0000-0000",
            }
        ],
    }
    errors = list(
        validator.iter_errors(
            {
                "schema_version": "2.0.0",
                "_meta": invalid_orcid_meta,
                "lemmas": [],
            }
        )
    )
    assert any(
        "does not match" in suberror.message
        for error in errors
        for suberror in error.context
    )


def test_lexicon_schema_restricts_pos_and_supports_longer_ids(
    base_meta: dict[str, object],
) -> None:
    schema = _load_json(LEXICON_SCHEMA_PATH)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())

    valid_long_id = {
        "id": "LSJ-123456",
        "headword": "λέγω",
        "transliteration": "legō",
        "ipa": "léɡɔː",
        "pos": "verb",
        "gloss": "say",
        "dialect": "attic",
    }
    assert not list(
        validator.iter_errors(
            {
                "schema_version": "2.0.0",
                "_meta": base_meta,
                "lemmas": [valid_long_id],
            }
        )
    )

    invalid_pos = {**valid_long_id, "pos": "stem"}
    errors = list(
        validator.iter_errors(
            {
                "schema_version": "2.0.0",
                "_meta": base_meta,
                "lemmas": [invalid_pos],
            }
        )
    )
    assert any("is not one of" in error.message for error in errors)


def test_lexicon_schema_restricts_gender_values(base_meta: dict[str, object]) -> None:
    schema = _load_json(LEXICON_SCHEMA_PATH)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())

    valid_common_gender = {
        "id": "LSJ-123458",
        "headword": "τις",
        "transliteration": "tis",
        "ipa": "tis",
        "pos": "pronoun",
        "gender": "common",
        "gloss": "someone, anyone",
        "dialect": "attic",
    }
    assert not list(
        validator.iter_errors(
            {
                "schema_version": "2.0.0",
                "_meta": base_meta,
                "lemmas": [valid_common_gender],
            }
        )
    )

    invalid_gender = {**valid_common_gender, "gender": "animate"}
    errors = list(
        validator.iter_errors(
            {
                "schema_version": "2.0.0",
                "_meta": base_meta,
                "lemmas": [invalid_gender],
            }
        )
    )
    assert any("is not one of" in error.message for error in errors)


def test_attic_doric_matrix_is_complete_symmetric_and_normalized() -> None:
    matrix = _load_json(MATRIX_PATH)

    assert "ɔː" in matrix["sound_classes"]["vowels"]
    assert "y" in matrix["sound_classes"]["vowels"]
    assert "oi" in matrix["sound_classes"]["vowels"]
    assert "ɡ" in matrix["sound_classes"]["stops"]
    assert "g" not in matrix["sound_classes"]["stops"]

    for class_name in ("vowels", "stops"):
        rows = matrix["sound_classes"][class_name]
        inventory = set(rows)

        for row_phone, row in rows.items():
            assert set(row) == inventory

            for column_phone, distance in row.items():
                assert 0.0 <= distance <= 1.0
                assert distance == rows[column_phone][row_phone]
                if row_phone == column_phone:
                    assert distance == 0.0


def test_attic_doric_matrix_matches_generator_output() -> None:
    committed = _load_json(MATRIX_PATH)

    assert_dicts_close(build_attic_doric_matrix(), committed)


def test_consonant_rule_overlap_is_resolved_for_attic_before_e() -> None:
    rules = _load_yaml(CONSONANT_RULES_PATH)["rules"]
    cch_004 = _find_rule(rules, "CCH-004")
    cch_005 = _find_rule(rules, "CCH-005")

    assert cch_004["context"] == "_{a,o}"
    assert cch_005["context"] == "_{e,i}"
    assert "priority" not in cch_004
    assert "priority" not in cch_005
    assert "Front-vowel environment only" in cch_005["note"]
    assert "/e, i/" in cch_005["note"]
    assert "_{a,o}" in cch_005["note"]


def test_grassmann_rule_notes_define_same_word_span() -> None:
    rules = _load_yaml(CONSONANT_RULES_PATH)["rules"]

    for rule_id, context in (("CCH-001", "_...pʰ"), ("CCH-002", "_...tʰ")):
        rule = _find_rule(rules, rule_id)

        assert rule["context"] == context
        assert "same word" in rule["note"]
        assert "standard same-word span" in rule["note"]


def test_phonology_rules_doc_defines_context_notation_examples() -> None:
    document = PHONOLOGY_RULES_DOC_PATH.read_text(encoding="utf-8")

    assert "`#_V`" in document
    assert "`{p,t,k}_`" in document
    assert "`V...V`" in document
    assert "does not cross `#`" in document
    assert "CCH-001" in document
    assert "CCH-002" in document
    assert "not a special exception" in document


def test_renamed_rules_match_expected_fields() -> None:
    consonant_rules = _load_yaml(CONSONANT_RULES_PATH)["rules"]
    vowel_rules = _load_yaml(VOWEL_RULES_PATH)["rules"]

    cch_006 = _find_rule(consonant_rules, "CCH-006")
    vsh_001 = _find_rule(vowel_rules, "VSH-001")

    assert cch_006["name"] == "Attic sigma+sigma -> tt"
    assert cch_006["to"] == "tt"
    assert cch_006["dialect"] == ["attic"]

    assert vsh_001["name"] == "Attic-Ionic long alpha to eta shift"
    assert vsh_001["from"] == "aː"
    assert vsh_001["to"] == "ɛː"


def test_koine_iotacism_rules_cover_eta_upsilon_oi_and_ei() -> None:
    vowel_rules = _load_yaml(VOWEL_RULES_PATH)["rules"]

    expected_rules = {
        "VSH-005": ("ɛː", "i"),
        "VSH-006": ("y", "i"),
        "VSH-007": ("oi", "i"),
        "VSH-008": ("eː", "i"),
    }

    for rule_id, (from_phone, to_phone) in expected_rules.items():
        rule = _find_rule(vowel_rules, rule_id)

        assert rule["from"] == from_phone
        assert rule["to"] == to_phone
        assert rule["context"] is None
        assert rule["dialect"] == ["koine"]
        assert rule["period"] == "hellenistic"
        assert rule["weight"] == 0.6


def test_vowel_shift_rule_endpoints_match_canonical_vowel_inventory() -> None:
    matrix = _load_json(MATRIX_PATH)
    vowel_rules = _load_yaml(VOWEL_RULES_PATH)["rules"]

    canonical_vowels = set(matrix["sound_classes"]["vowels"])
    allowed_sequences = {"eːo", "eoː"}

    for rule in vowel_rules:
        for endpoint_name in ("from", "to"):
            endpoint = rule[endpoint_name]
            assert endpoint in canonical_vowels or endpoint in allowed_sequences, (
                f"{rule['id']} has non-canonical vowel {endpoint_name}={endpoint!r}"
            )
