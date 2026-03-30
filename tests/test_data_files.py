"""Validation tests for committed data files."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
import unicodedata
from pathlib import Path
from typing import Any

import pytest
import yaml
from jsonschema import Draft202012Validator, FormatChecker

from phonology.ipa_converter import to_ipa
from phonology.matrix_generator import build_attic_doric_matrix


ROOT_DIR = Path(__file__).resolve().parents[1]
LEXICON_PATH = ROOT_DIR / "data" / "lexicon" / "greek_lemmas.json"
LEXICON_SCHEMA_PATH = ROOT_DIR / "data" / "lexicon" / "greek_lemmas.schema.json"
MATRIX_PATH = ROOT_DIR / "data" / "matrices" / "attic_doric.json"
CONSONANT_RULES_PATH = ROOT_DIR / "data" / "rules" / "ancient_greek" / "consonant_changes.yaml"
VOWEL_RULES_PATH = ROOT_DIR / "data" / "rules" / "ancient_greek" / "vowel_shifts.yaml"
PHONOLOGY_RULES_DOC_PATH = ROOT_DIR / "docs" / "phonology_rules.md"
MIN_LEMMAS_COUNT = 100
_skip_no_lexicon = pytest.mark.skipif(
    not LEXICON_PATH.exists(),
    reason="Lexicon not generated; run scripts/extract-lsj.sh",
)


@pytest.fixture
def base_meta() -> dict[str, object]:
    """Base metadata for testing schema validation."""
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


@pytest.fixture
def dummy_lemma() -> dict[str, str]:
    """Dummy lemma for testing schema validation."""
    return {
        "id": "LSJ-99999",
        "headword": "x",
        "transliteration": "x",
        "ipa": "x",
        "pos": "particle",
        "gloss": "x",
        "dialect": "attic",
    }


@pytest.fixture
def vowel_rules() -> list[dict[str, object]]:
    return _load_yaml(VOWEL_RULES_PATH)["rules"]


@pytest.fixture
def consonant_rules() -> list[dict[str, object]]:
    return _load_yaml(CONSONANT_RULES_PATH)["rules"]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _find_rule(rules: list[dict], rule_id: str) -> dict:
    rule = next((candidate for candidate in rules if candidate["id"] == rule_id), None)
    if rule is None:
        raise LookupError(f"Rule {rule_id!r} not found in rule list of length {len(rules)}")
    return rule


def assert_nonempty_str(value: Any, name: str, rule_id: Any) -> None:
    """Assert *value* is a non-empty string; raises AssertionError mentioning *name* and *rule_id*."""
    assert isinstance(value, str) and value, f"rule {rule_id}: '{name}' must be a non-empty string"


def assert_nonempty_list_of_str(value: Any, name: str, rule_id: Any) -> None:
    """Assert *value* is a non-empty list of non-empty strings."""
    assert isinstance(value, list) and value, f"rule {rule_id}: '{name}' must be a non-empty list"
    assert all(isinstance(item, str) and item for item in value), (
        f"rule {rule_id}: all {name} must be non-empty strings"
    )


def _validate_rule_examples(
    examples: list[dict[str, Any]],
    rule_id: str,
    valid_example_keys: set[str],
    required_example_keys: set[str],
    is_koine_rule: bool,
) -> None:
    """Validate rule examples against expected keys and required conditions.

    Args:
        examples: An iterable of example mappings (e.g. list[dict[str, Any]]).
        rule_id: The identifier of the rule being validated.
        valid_example_keys: A set of allowed keys for each example.
        required_example_keys: A set of keys that must be present in each example.
        is_koine_rule: Boolean indicating if the rule applies to Koine Greek.

    Raises:
        AssertionError: If examples is empty, not a list of dicts, contains unknown keys,
            missing required keys, lacks form contrast, or has values that fail assert_nonempty_str.
    """
    assert isinstance(examples, list) and examples, (
        f"rule {rule_id}: 'examples' must be a non-empty list"
    )

    for example in examples:
        assert isinstance(example, dict), (
            f"rule {rule_id}: example must be a dict, got {type(example)}"
        )
        unknown_example_keys = set(example) - valid_example_keys
        assert not unknown_example_keys, (
            f"rule {rule_id}: example contains unknown keys"
            f" {unknown_example_keys}"
        )
        missing_example_keys = required_example_keys - set(example)
        assert not missing_example_keys, (
            f"rule {rule_id}: example missing required keys"
            f" {missing_example_keys}"
        )
        has_form_contrast = (
            "dialect" in example
            or "reconstruction" in example
            or is_koine_rule
        )
        assert has_form_contrast, (
            f"rule {rule_id}: example must have"
            f" 'dialect' or 'reconstruction'"
        )
        assert all(
            isinstance(v, str) and v for v in example.values()
        ), (
            f"rule {rule_id}: all example values"
            f" must be non-empty strings"
        )


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


@_skip_no_lexicon
def test_lexicon_metadata_and_representative_lemma_regressions() -> None:
    lexicon = _load_json(LEXICON_PATH)
    metadata = lexicon["_meta"]
    pos_counts = Counter(lemma["pos"] for lemma in lexicon["lemmas"])

    # Build O(1) headword index
    by_headword: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for lemma in lexicon["lemmas"]:
        by_headword[lemma["headword"]].append(lemma)

    def find(hw: str) -> dict[str, Any] | None:
        entries = by_headword.get(hw)
        return entries[0] if entries else None

    def find_all(hw: str) -> list[dict[str, Any]]:
        return by_headword.get(hw, [])

    assert lexicon["schema_version"] == "2.0.0"
    assert metadata["data_schema_ref"] == "data/lexicon/greek_lemmas.schema.json"
    assert metadata["dialect"] == "attic"
    assert len(lexicon["lemmas"]) > MIN_LEMMAS_COUNT

    anthropos = find("ἄνθρωπος")
    assert anthropos is not None, "Lemma 'ἄνθρωπος' not found in lexicon"
    assert anthropos["ipa"] == "ántʰrɔːpos", (
        f"Expected IPA 'ántʰrɔːpos', got {anthropos['ipa']!r}"
    )
    assert anthropos["gender"] == "common"

    abelteros = find("ἀβέλτερος")
    assert abelteros is not None, "Lemma 'ἀβέλτερος' not found in lexicon"
    assert (
        abelteros["pos"] == "adjective"
    ), f"Expected POS 'adjective' for lemma 'ἀβέλτερος' but got {abelteros['pos']}"

    hosanei = find("ὡσανεί")
    assert hosanei is not None, "Lemma 'ὡσανεί' not found in lexicon"
    assert (
        hosanei["pos"] == "adverb"
    ), f"Expected POS 'adverb' for lemma 'ὡσανεί' but got {hosanei['pos']}"

    hopos = find_all("ὅπως")
    assert hopos, "Lemma 'ὅπως' not found in lexicon"
    assert any(lemma["pos"] == "adverb" for lemma in hopos), (
        "Expected at least one adverb entry for lemma 'ὅπως'"
    )

    for headword in ("ἀκέω", "λέγω", "νέω"):
        lemmas = find_all(headword)
        assert lemmas, f"Lemma {headword!r} not found in lexicon"
        assert any(lemma["pos"] == "verb" for lemma in lemmas), (
            f"Expected at least one verb entry for lemma {headword!r}"
        )

    anelees = find("ἀνελεής")
    assert anelees is not None, "Lemma 'ἀνελεής' not found in lexicon"
    assert (
        anelees["pos"] == "adverb"
    ), f"Expected POS 'adverb' for lemma 'ἀνελεής' but got {anelees['pos']}"

    psykhe = find("ψυχή")
    assert psykhe is not None, "Lemma 'ψυχή' not found in lexicon"
    assert psykhe["ipa"] == "psykʰɛ́ː", (
        f"Expected IPA 'psykʰɛ́ː', got {psykhe['ipa']!r}"
    )

    anaklisis = find("ἀνάκλισις")
    assert anaklisis is not None, "Lemma 'ἀνάκλισις' not found in lexicon"
    assert anaklisis["pos"] == "noun", "Expected POS 'noun' for lemma 'ἀνάκλισις'"
    assert anaklisis["gender"] == "feminine", "Expected gender 'feminine' for lemma 'ἀνάκλισις'"

    apoteikhisma = find("ἀποτείχισμα")
    assert apoteikhisma is not None, "Lemma 'ἀποτείχισμα' not found in lexicon"
    assert apoteikhisma["pos"] == "noun", "Expected POS 'noun' for lemma 'ἀποτείχισμα'"
    assert apoteikhisma["gender"] == "neuter", "Expected gender 'neuter' for lemma 'ἀποτείχισμα'"

    aeropetes = find("ἀεροπέτης")
    assert aeropetes is not None, "Lemma 'ἀεροπέτης' not found in lexicon"
    assert find("ἀεροπέτησ2") is None, (
        "Lemma 'ἀεροπέτησ2' should not exist in lexicon"
    )
    assert find("εως") is None, "Suffix-only lemma 'εως' should not exist"
    assert find("ατος") is None, "Suffix-only lemma 'ατος' should not exist"

    thymiatizo = find("θυμιατίζω")
    assert (
        thymiatizo is None or thymiatizo["pos"] != "adjective"
    ), "Lemma 'θυμιατίζω' should not be classified as an adjective"

    apollymi = find("ἀπόλλυμι")
    assert apollymi is not None, "Lemma 'ἀπόλλυμι' not found in lexicon"
    assert apollymi["pos"] == "verb", "Expected POS 'verb' for lemma 'ἀπόλλυμι'"

    askion = find("ἀσκίον")
    assert askion is not None, "Lemma 'ἀσκίον' not found in lexicon"
    assert askion["pos"] == "noun", "Expected POS 'noun' for lemma 'ἀσκίον'"
    assert askion["gender"] == "neuter", "Expected gender 'neuter' for lemma 'ἀσκίον'"

    epimemptos = find("ἐπίμεμπτος")
    assert epimemptos is not None, "Lemma 'ἐπίμεμπτος' not found in lexicon"
    assert epimemptos["pos"] == "adjective", "Expected POS 'adjective' for lemma 'ἐπίμεμπτος'"

    arkhon = find("ἄρχων")
    assert arkhon is not None, "Lemma 'ἄρχων' not found in lexicon"
    assert arkhon["pos"] == "noun", "Expected POS 'noun' for lemma 'ἄρχων'"

    autos = find("αὐτός")
    assert autos is not None, "Lemma 'αὐτός' not found in lexicon"
    assert autos["pos"] == "pronoun", "Expected POS 'pronoun' for lemma 'αὐτός'"
    assert autos["gender"] == "common", "Expected gender 'common' for lemma 'αὐτός'"

    ego = find("ἐγώ")
    assert ego is not None, "Lemma 'ἐγώ' not found in lexicon"
    assert ego["pos"] == "pronoun", "Expected POS 'pronoun' for lemma 'ἐγώ'"
    assert ego["dialect"] == "attic", "Expected dialect 'attic' for lemma 'ἐγώ'"

    kai = find("καί")
    assert kai is not None, "Lemma 'καί' not found in lexicon"
    assert kai["pos"] == "conjunction", "Expected POS 'conjunction' for lemma 'καί'"

    iou = find("ἰού")
    assert iou is not None, "Lemma 'ἰού' not found in lexicon"
    assert iou["pos"] == "interjection", "Expected POS 'interjection' for lemma 'ἰού'"

    article = find("ὁ")
    assert article is not None, "Lemma 'ὁ' not found in lexicon"
    assert article["pos"] == "article", "Expected POS 'article' for lemma 'ὁ'"
    assert article["gender"] == "common", "Expected gender 'common' for lemma 'ὁ'"

    deka = find("δέκα")
    assert deka is not None, "Lemma 'δέκα' not found in lexicon"
    assert deka["pos"] == "numeral", "Expected POS 'numeral' for lemma 'δέκα'"
    assert deka["gender"] == "common", "Expected gender 'common' for lemma 'δέκα'"

    assert find("ἀγαπάζω") is None, "Non-Attic lemma 'ἀγαπάζω' should not exist"
    assert find("ἄελλα") is None, "Non-Attic lemma 'ἄελλα' should not exist"

    for pos in (
        "pronoun",
        "article",
        "preposition",
        "conjunction",
        "particle",
        "interjection",
        "numeral",
        "participle",
    ):
        assert pos_counts[pos] > 0, f"Expected at least one {pos!r} entry in lexicon"


@_skip_no_lexicon
def test_lexicon_matches_committed_json_schema() -> None:
    lexicon = _load_json(LEXICON_PATH)
    schema = _load_json(LEXICON_SCHEMA_PATH)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())

    errors = sorted(validator.iter_errors(lexicon), key=lambda error: list(error.path))
    assert not errors, [error.message for error in errors]


@_skip_no_lexicon
def test_lexicon_dialect_is_consistent_across_metadata_and_lemmas() -> None:
    lexicon = _load_json(LEXICON_PATH)
    top_level_dialect = lexicon["_meta"]["dialect"]

    assert top_level_dialect == "attic"
    assert {lemma["dialect"] for lemma in lexicon["lemmas"]} == {top_level_dialect}


def test_lexicon_schema_allows_extension_and_conditional_gender_rules(
    base_meta: dict[str, object],
    dummy_lemma: dict[str, str],
) -> None:
    schema = _load_json(LEXICON_SCHEMA_PATH)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())

    root_extension = {
        "schema_version": "2.0.0",
        "_meta": base_meta,
        "lemmas": [dummy_lemma],
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
    dummy_lemma: dict[str, str],
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
                "lemmas": [dummy_lemma],
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
                "lemmas": [dummy_lemma],
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
                "lemmas": [dummy_lemma],
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

    assert "Consonant rules generally use a compact notation in the `context` field:" in document
    assert "Vowel rules may instead use short descriptive English phrases" in document
    assert "`#_V`" in document
    assert "`{p,t,k}_`" in document
    assert "`V...V`" in document
    assert "does not cross `#`" in document
    assert "CCH-001" in document
    assert "CCH-002" in document
    assert "not a special exception" in document


def test_renamed_rules_match_expected_fields(
    vowel_rules: list[dict[str, object]],
    consonant_rules: list[dict[str, object]],
) -> None:

    cch_006 = _find_rule(consonant_rules, "CCH-006")
    vowel_shift = _find_rule(vowel_rules, "VSH-001")

    assert cch_006["name_en"] == "Attic sigma+sigma -> tt"
    assert cch_006["output"] == "tt"
    assert cch_006["dialects"] == ["attic"]

    assert vowel_shift["name_en"] == "Ionic long alpha to eta shift"
    assert vowel_shift["name_ja"] == "イオニア方言の長母音 ā > ē 推移"
    assert vowel_shift["input"] == "aː"
    assert vowel_shift["output"] == "ɛː"
    assert vowel_shift["dialects"] == ["ionic"]


def test_vowel_shift_rules_use_new_schema_and_define_minimum_expected_rules(
    vowel_rules: list[dict[str, object]],
) -> None:
    required_keys = {
        "id",
        "name_en",
        "name_ja",
        "input",
        "output",
        "context",
        "dialects",
        "period",
        "references",
        "examples",
    }
    legacy_keys = {"name", "from", "to", "dialect", "weight", "priority"}

    expected_ids = {f"VSH-{n:03d}" for n in range(1, 23)}
    actual_ids = {rule["id"] for rule in vowel_rules}
    assert actual_ids == expected_ids, (
        f"Expected vowel rule IDs {sorted(expected_ids)}, got {sorted(actual_ids)}"
    )

    for rule in vowel_rules:
        rule_id = rule.get("id", "unknown")
        missing_keys = required_keys - set(rule)
        assert required_keys <= set(rule), f"rule {rule_id}: missing required keys {missing_keys}"
        forbidden_keys = legacy_keys & set(rule)
        assert not forbidden_keys, f"rule {rule_id}: contains legacy keys {forbidden_keys}"
        assert_nonempty_str(rule["id"], "id", rule_id)
        assert_nonempty_str(rule["name_en"], "name_en", rule_id)
        assert_nonempty_str(rule["name_ja"], "name_ja", rule_id)
        assert_nonempty_str(rule["input"], "input", rule_id)
        assert_nonempty_str(rule["output"], "output", rule_id)
        assert_nonempty_str(rule["context"], "context", rule_id)
        assert_nonempty_str(rule["period"], "period", rule_id)
        assert_nonempty_list_of_str(rule["dialects"], "dialects", rule_id)
        assert_nonempty_list_of_str(rule["references"], "references", rule_id)
        if "change_type" in rule:
            assert rule["change_type"] in {"retention"}, (
                f"rule {rule_id}: unknown change_type {rule['change_type']!r}"
            )
        # Koine examples may illustrate a chronological sound value
        # or identical spelling, so they do not always supply a
        # contrasting dialect/reconstruction form.
        _validate_rule_examples(
            rule["examples"],
            rule_id,
            {"standard", "dialect", "meaning", "phonetic", "reconstruction"},
            {"standard", "meaning"},
            is_koine_rule="koine" in rule.get("dialects", []),
        )


def test_representative_vowel_shift_rules_match_expected_content(
    vowel_rules: list[dict[str, object]],
) -> None:
    vowel_shift = _find_rule(vowel_rules, "VSH-001")
    doric_alpha_retention = _find_rule(vowel_rules, "VSH-002")
    nasal_lengthening = _find_rule(vowel_rules, "VSH-003")
    attic_alpha_retention = _find_rule(vowel_rules, "VSH-010")
    attic_omicron_alpha_contraction = _find_rule(vowel_rules, "VSH-017")
    koine_ei_merger = _find_rule(vowel_rules, "VSH-008")
    koine_eta_iotacism = _find_rule(vowel_rules, "VSH-005")
    koine_oi_merger = _find_rule(vowel_rules, "VSH-007")
    koine_long_o_raising = _find_rule(vowel_rules, "VSH-022")

    assert vowel_shift["input"] == "aː"
    assert vowel_shift["output"] == "ɛː"
    assert vowel_shift["context"] == "all environments"
    assert vowel_shift["references"] == ["Buck §9", "Smyth §31"]
    assert vowel_shift["dialects"] == ["ionic"]
    assert "VSH-009" in vowel_shift["note"]
    assert "VSH-010" in vowel_shift["note"]

    assert attic_alpha_retention["input"] == "ɛː"
    assert attic_alpha_retention["output"] == "aː"
    assert attic_alpha_retention["change_type"] == "retention"
    assert attic_alpha_retention["context"] == "after e, i, or r"
    assert attic_alpha_retention["dialects"] == ["attic"]

    assert doric_alpha_retention["change_type"] == "retention"

    assert nasal_lengthening["input"] == "a"
    assert nasal_lengthening["output"] == "aː"
    assert nasal_lengthening["context"] == "_NC"

    assert attic_omicron_alpha_contraction["input"] == "oa"
    assert attic_omicron_alpha_contraction["output"] == "ɔː"

    assert koine_ei_merger["input"] == "eː"
    assert koine_ei_merger["output"] == "i"
    assert koine_ei_merger["dialects"] == ["koine"]
    assert koine_ei_merger["period"] == "3rd century BCE onwards"
    assert "εἰ" in koine_ei_merger["note"]
    assert "[eː]" in koine_ei_merger["note"]

    assert koine_eta_iotacism["input"] == "ɛː"
    assert koine_eta_iotacism["output"] == "i"
    assert koine_eta_iotacism["dialects"] == ["koine"]
    assert koine_eta_iotacism["period"] == "2nd century BCE onwards"
    assert koine_eta_iotacism["examples"][0]["dialect"] == "μίτιρ"

    assert koine_oi_merger["input"] == "oi"
    assert koine_oi_merger["output"] == "i"
    assert koine_oi_merger["dialects"] == ["koine"]
    assert "VSH-021" in koine_oi_merger["note"]
    assert "VSH-007" in koine_oi_merger["note"]
    assert koine_oi_merger["examples"][0]["phonetic"] == "[loipon] → [lipon]"

    assert koine_long_o_raising["input"] == "oː"
    assert koine_long_o_raising["output"] == "u"
    assert koine_long_o_raising["dialects"] == ["koine"]
    assert koine_long_o_raising["references"] == [
        "Allen Vox Graeca 3.2",
        "Horrocks 2010: 169-170",
    ]


def _initial_vowel_with_modifiers(ipa: str) -> str:
    """Extract the initial base character and its trailing modifiers from an IPA string.

    Skips any leading combining characters, then returns the first base character
    along with any immediately following length marks (ː) or combining characters.

    Args:
        ipa: A non-empty IPA transcription string.

    Returns:
        The initial base character concatenated with its trailing modifiers.

    Raises:
        AssertionError: If ipa is empty or contains only combining characters.
    """
    assert ipa, "Expected non-empty IPA string"
    base_index = 0
    while base_index < len(ipa) and unicodedata.combining(ipa[base_index]):
        base_index += 1
    assert base_index < len(ipa), "Expected IPA string containing a base character"

    base = ipa[base_index]
    tail = ""
    for ch in ipa[base_index + 1 :]:
        if ch == "ː" or unicodedata.combining(ch):
            tail += ch
        else:
            break
    return base + tail


def test_koine_vowel_shift_rule_inputs_match_current_ipa_conversion(
    vowel_rules: list[dict[str, object]],
) -> None:
    koine_ei_merger = _find_rule(vowel_rules, "VSH-008")
    koine_long_o_raising = _find_rule(vowel_rules, "VSH-022")

    assert to_ipa("εἰς") == "eːs"
    assert koine_ei_merger["input"] == "eː"

    ouranos_ipa = to_ipa("οὐρανός")

    ouranos_initial_vowel = _initial_vowel_with_modifiers(ouranos_ipa)
    assert ouranos_ipa == "oːranós"
    assert koine_long_o_raising["input"] == ouranos_initial_vowel
    assert _initial_vowel_with_modifiers("\u0301oːranós") == "oː"


def test_attic_omicron_alpha_contraction_example_uses_oa_pair(
    vowel_rules: list[dict[str, object]],
) -> None:
    attic_omicron_alpha_contraction = _find_rule(vowel_rules, "VSH-017")
    example = attic_omicron_alpha_contraction["examples"][0]

    assert example["standard"] == "αἰδῶ"
    assert example["dialect"] == "αἰδόα"


def test_ionic_quantitative_metathesis_note_clarifies_attic_reference_example(
    vowel_rules: list[dict[str, object]],
) -> None:
    ionic_metathesis = _find_rule(vowel_rules, "VSH-004")

    assert "Attic form" in ionic_metathesis["note"]
    assert "Ionic-source" in ionic_metathesis["note"]


def test_consonant_rules_use_new_schema_and_define_expected_rules(
    consonant_rules: list[dict[str, object]],
) -> None:
    required_keys = {
        "id",
        "name_en",
        "name_ja",
        "input",
        "output",
        "context",
        "dialects",
        "period",
        "references",
        "examples",
    }
    legacy_keys = {"name", "from", "to", "dialect", "weight", "priority"}

    expected_ids = {
        "CCH-001",
        "CCH-002",
        "CCH-003",
        "CCH-004",
        "CCH-005",
        "CCH-006",
        "CCH-007",
    }
    actual_ids = {rule["id"] for rule in consonant_rules}
    assert actual_ids == expected_ids, (
        f"Expected consonant rule IDs {sorted(expected_ids)}, got {sorted(actual_ids)}"
    )

    valid_example_keys = {
        "standard", "dialect", "meaning", "phonetic", "reconstruction",
    }
    required_example_keys = {"standard", "meaning"}

    for rule in consonant_rules:
        rule_id = rule.get("id", "unknown")
        missing_keys = required_keys - set(rule)
        assert required_keys <= set(rule), f"rule {rule_id}: missing required keys {missing_keys}"
        forbidden_keys = legacy_keys & set(rule)
        assert not forbidden_keys, f"rule {rule_id}: contains legacy keys {forbidden_keys}"
        assert_nonempty_str(rule["id"], "id", rule_id)
        assert_nonempty_str(rule["name_en"], "name_en", rule_id)
        assert_nonempty_str(rule["name_ja"], "name_ja", rule_id)
        assert_nonempty_str(rule["input"], "input", rule_id)
        # output can be empty string (e.g. CCH-003 sigma deletion)
        assert isinstance(rule["output"], str), f"rule {rule_id}: 'output' must be a string"
        # context can be null for unconditioned rules (e.g. CCH-006)
        assert rule["context"] is None or (isinstance(rule["context"], str) and rule["context"]), (
            f"rule {rule_id}: 'context' must be a non-empty string or null"
        )
        assert_nonempty_str(rule["period"], "period", rule_id)
        assert_nonempty_list_of_str(rule["dialects"], "dialects", rule_id)
        assert_nonempty_list_of_str(rule["references"], "references", rule_id)
        _validate_rule_examples(
            rule["examples"],
            rule_id,
            valid_example_keys,
            required_example_keys,
            is_koine_rule="koine" in rule.get("dialects", []),
        )
