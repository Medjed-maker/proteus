"""Tests for phonology.distance."""

import json
import shutil
from pathlib import Path
from typing import Iterator

import pytest

from phonology import distance as distance_module
from phonology._trusted_paths import TRUSTED_DIR_OVERRIDES_OPT_IN_ENV_VAR
from phonology.distance import (
    DEFAULT_COST,
    MatrixMeta,
    UNKNOWN_SUBSTITUTION_COST,
    load_matrix,
    load_matrix_document,
    normalized_phonological_distance,
    normalized_sequence_distance,
    normalized_word_distance,
    phone_distance,
    phonological_distance,
    sequence_distance,
    word_distance,
)
from phonology._paths import resolve_repo_data_dir
from phonology import ipa_converter as ipa_converter_module
from phonology.ipa_converter import greek_to_ipa, to_ipa


@pytest.fixture(autouse=True)
def enable_trusted_dir_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TRUSTED_DIR_OVERRIDES_OPT_IN_ENV_VAR, "1")


@pytest.fixture
def committed_matrix_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    monkeypatch.delenv("PROTEUS_TRUSTED_MATRICES_DIR", raising=False)
    source_matrix = distance_module._get_trusted_matrices_dir() / "attic_doric.json"
    assert source_matrix.is_file(), f"Expected committed matrix at {source_matrix}"

    trusted_dir = tmp_path / "matrices"
    trusted_dir.mkdir()
    copied_matrix = trusted_dir / source_matrix.name
    shutil.copy2(source_matrix, copied_matrix)
    # Explicitly set opt-in and override to ensure the fixture is self-contained
    # and remains valid even if the enable_trusted_dir_overrides autouse fixture
    # signature or behavior changes in the future.
    monkeypatch.setenv(TRUSTED_DIR_OVERRIDES_OPT_IN_ENV_VAR, "1")
    monkeypatch.setenv("PROTEUS_TRUSTED_MATRICES_DIR", str(trusted_dir))
    return copied_matrix


class TestLoadMatrix:
    def test_loads_valid_json_with_nested_rows(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trusted_dir = tmp_path / "matrices"
        trusted_dir.mkdir()
        monkeypatch.setenv("PROTEUS_TRUSTED_MATRICES_DIR", str(trusted_dir))

        matrix_file = trusted_dir / "sample.json"
        matrix_file.write_text(
            json.dumps(
                {
                    "_meta": {"version": 1},
                    "nested": {
                        "a": {"ɛː": 0.3},
                        "pʰ": {"tʰ": 0.4},
                    },
                }
            ),
            encoding="utf-8",
        )

        matrix = load_matrix(matrix_file)

        assert matrix["a"]["ɛː"] == pytest.approx(0.3)
        assert matrix["pʰ"]["tʰ"] == pytest.approx(0.4)
        assert "_meta" not in matrix

    def test_raises_for_malformed_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trusted_dir = tmp_path / "matrices"
        trusted_dir.mkdir()
        monkeypatch.setenv("PROTEUS_TRUSTED_MATRICES_DIR", str(trusted_dir))

        matrix_file = trusted_dir / "broken.json"
        matrix_file.write_text("{not-json", encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            load_matrix(matrix_file)

    def test_raises_for_nonexistent_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trusted_dir = tmp_path / "matrices"
        trusted_dir.mkdir()
        monkeypatch.setenv("PROTEUS_TRUSTED_MATRICES_DIR", str(trusted_dir))

        with pytest.raises(FileNotFoundError):
            load_matrix(trusted_dir / "missing.json")

    def test_rejects_path_outside_trusted_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trusted_dir = tmp_path / "matrices"
        trusted_dir.mkdir()
        monkeypatch.setenv("PROTEUS_TRUSTED_MATRICES_DIR", str(trusted_dir))

        outside_file = tmp_path / "outside.json"
        outside_file.write_text(json.dumps({"a": {"b": 1.0}}), encoding="utf-8")

        with pytest.raises(ValueError, match="Matrix path must stay within"):
            load_matrix(outside_file)

    def test_accepts_bare_string_relative_to_trusted_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trusted_dir = tmp_path / "matrices"
        trusted_dir.mkdir()
        monkeypatch.setenv("PROTEUS_TRUSTED_MATRICES_DIR", str(trusted_dir))
        monkeypatch.chdir(tmp_path)

        matrix_file = trusted_dir / "sample.json"
        matrix_file.write_text(json.dumps({"a": {"ɛː": 0.3}}), encoding="utf-8")

        matrix = load_matrix("sample.json")

        assert matrix["a"]["ɛː"] == pytest.approx(0.3)

    def test_accepts_bare_path_relative_to_trusted_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trusted_dir = tmp_path / "matrices"
        trusted_dir.mkdir()
        monkeypatch.setenv("PROTEUS_TRUSTED_MATRICES_DIR", str(trusted_dir))
        monkeypatch.chdir(tmp_path)

        matrix_file = trusted_dir / "sample.json"
        matrix_file.write_text(json.dumps({"a": {"ɛː": 0.3}}), encoding="utf-8")

        matrix = load_matrix(Path("sample.json"))

        assert matrix["a"]["ɛː"] == pytest.approx(0.3)

    def test_accepts_legacy_repo_style_relative_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trusted_dir = tmp_path / "matrices"
        trusted_dir.mkdir()
        monkeypatch.setenv("PROTEUS_TRUSTED_MATRICES_DIR", str(trusted_dir))
        monkeypatch.chdir(tmp_path)

        matrix_file = trusted_dir / "sample.json"
        matrix_file.write_text(json.dumps({"a": {"ɛː": 0.3}}), encoding="utf-8")

        matrix = load_matrix("data/matrices/sample.json")

        assert matrix["a"]["ɛː"] == pytest.approx(0.3)

    def test_rejects_symlink_within_trusted_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trusted_dir = tmp_path / "matrices"
        trusted_dir.mkdir()
        monkeypatch.setenv("PROTEUS_TRUSTED_MATRICES_DIR", str(trusted_dir))

        real_dir = trusted_dir / "real"
        real_dir.mkdir()
        matrix_file = real_dir / "sample.json"
        matrix_file.write_text(json.dumps({"a": {"b": 1.0}}), encoding="utf-8")

        symlink_path = trusted_dir / "linked.json"
        try:
            symlink_path.symlink_to(matrix_file)
        except OSError as exc:  # pragma: no cover - platform-dependent
            pytest.skip(f"symlink creation not supported: {exc}")

        with pytest.raises(ValueError, match="must not traverse symlinks"):
            load_matrix(symlink_path)

    def test_rejects_symlink_outside_trusted_directory_even_if_it_points_inside(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trusted_dir = tmp_path / "matrices"
        trusted_dir.mkdir()
        monkeypatch.setenv("PROTEUS_TRUSTED_MATRICES_DIR", str(trusted_dir))

        matrix_file = trusted_dir / "sample.json"
        matrix_file.write_text(json.dumps({"a": {"b": 1.0}}), encoding="utf-8")

        symlink_path = tmp_path / "outside-link.json"
        try:
            symlink_path.symlink_to(matrix_file)
        except OSError as exc:  # pragma: no cover - platform-dependent
            pytest.skip(f"symlink creation not supported: {exc}")

        with pytest.raises(ValueError, match="Matrix path must stay within"):
            load_matrix(symlink_path)

    def test_prefers_environment_override_for_trusted_matrices_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        custom_dir = tmp_path / "custom-matrices"
        custom_dir.mkdir()
        monkeypatch.setenv("PROTEUS_TRUSTED_MATRICES_DIR", str(custom_dir))

        assert distance_module._get_trusted_matrices_dir() == custom_dir.resolve()

    def test_environment_override_raises_for_nonexistent_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PROTEUS_TRUSTED_MATRICES_DIR", str(tmp_path / "missing"))

        with pytest.raises(FileNotFoundError, match="Could not find trusted matrices directory"):
            distance_module._get_trusted_matrices_dir()

    def test_environment_override_rejects_without_opt_in(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trusted_dir = tmp_path / "matrices"
        trusted_dir.mkdir()
        monkeypatch.delenv(TRUSTED_DIR_OVERRIDES_OPT_IN_ENV_VAR, raising=False)
        monkeypatch.setenv("PROTEUS_TRUSTED_MATRICES_DIR", str(trusted_dir))

        with pytest.raises(
            ValueError,
            match="PROTEUS_TRUSTED_MATRICES_DIR requires PROTEUS_ALLOW_TRUSTED_DIR_OVERRIDES=1",
        ):
            distance_module._get_trusted_matrices_dir()

    def test_environment_override_rejects_file_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        matrix_file = tmp_path / "matrix.json"
        matrix_file.write_text("{}", encoding="utf-8")
        monkeypatch.setenv("PROTEUS_TRUSTED_MATRICES_DIR", str(matrix_file))

        with pytest.raises(NotADirectoryError, match="trusted matrices path is not a directory"):
            distance_module._get_trusted_matrices_dir()

    def test_environment_override_rejects_parent_symlink(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_parent = tmp_path / "real-parent"
        real_parent.mkdir()
        link_parent = tmp_path / "link-parent"
        try:
            link_parent.symlink_to(real_parent)
        except OSError as exc:  # pragma: no cover - platform-dependent
            pytest.skip(f"symlink creation not supported: {exc}")
        trusted_dir = link_parent / "matrices"
        monkeypatch.setenv("PROTEUS_TRUSTED_MATRICES_DIR", str(trusted_dir))

        with pytest.raises(ValueError, match="must not contain a symlink"):
            distance_module._get_trusted_matrices_dir()

    def test_falls_back_to_repo_data_directory_when_env_is_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("PROTEUS_TRUSTED_MATRICES_DIR", raising=False)

        resolved_dir = distance_module._get_trusted_matrices_dir()
        expected_dir = resolve_repo_data_dir("matrices")

        assert resolved_dir.name == "matrices"
        assert resolved_dir.is_dir()
        assert resolved_dir.resolve() == expected_dir.resolve()

    def test_uses_package_resources_when_they_are_pathlike(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("PROTEUS_TRUSTED_MATRICES_DIR", raising=False)
        package_root = tmp_path / "package-root"
        resource_dir = package_root / "data" / "languages" / "ancient_greek" / "matrices"
        resource_dir.mkdir(parents=True)

        monkeypatch.setattr(distance_module.resources, "files", lambda _: package_root)

        assert distance_module._get_trusted_matrices_dir() == resource_dir.resolve()


class TestPublicApi:
    def test_declares_explicit_public_api(self) -> None:
        assert distance_module.__all__ == [
            "MatrixData",
            "MatrixMeta",
            "DEFAULT_COST",
            "load_matrix",
            "load_matrix_document",
            "phone_distance",
            "phonological_distance",
            "normalized_phonological_distance",
            "sequence_distance",
            "normalized_sequence_distance",
            "word_distance",
            "normalized_word_distance",
            "register_trusted_matrices_dir",
            "clear_trusted_external_matrix_dirs",
            "UNKNOWN_SUBSTITUTION_COST",
        ]


class TestPhoneDistance:
    def test_identity_is_zero(self) -> None:
        assert phone_distance("a", "a", {}) == 0.0

    def test_accented_known_phone_matches_unaccented_equivalent(self) -> None:
        assert phone_distance("ó", "o", {}) == 0.0

    def test_non_accent_combining_marks_remain_distinct(self) -> None:
        assert phone_distance("n̩", "n", {}) == DEFAULT_COST
        assert phone_distance("ã", "a", {}) == DEFAULT_COST

    def test_direct_matrix_entry_is_used(self) -> None:
        matrix = {"a": {"ɛː": 0.5}}

        assert phone_distance("a", "ɛː", matrix) == pytest.approx(0.5)

    def test_symmetric_fallback_uses_reverse_entry(self) -> None:
        matrix = {"a": {"ɛː": 0.5}}

        assert phone_distance("ɛː", "a", matrix) == pytest.approx(0.5)

    def test_unknown_pair_uses_default_cost(self) -> None:
        assert phone_distance("!", "?", {}) == DEFAULT_COST

    def test_known_unmapped_pair_uses_unknown_substitution_cost(self) -> None:
        assert phone_distance("s", "n", {}) == UNKNOWN_SUBSTITUTION_COST

    def test_accented_known_unmapped_pair_still_uses_unknown_substitution_cost(self) -> None:
        assert phone_distance("ó", "ɛ́ː", {}) == UNKNOWN_SUBSTITUTION_COST


class TestPhonologicalDistance:
    def test_both_empty_sequences_have_zero_cost(self) -> None:
        assert phonological_distance([], [], {}) == 0.0

    def test_identical_sequences_have_zero_cost(self) -> None:
        seq = ["d", "a", "m", "o", "s"]

        assert phonological_distance(seq, seq, {}) == 0.0

    def test_accented_pre_tokenized_converter_output_matches_unaccented_sequence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_greek_to_ipa(text: str) -> list[str]:
            if text == "λόγος":
                return ["l", "ó", "ɡ", "o", "s"]
            if text == "λογος":
                return ["l", "o", "ɡ", "o", "s"]
            pytest.fail(f"Unexpected input to greek_to_ipa stub: {text}")

        monkeypatch.setattr(ipa_converter_module, "greek_to_ipa", fake_greek_to_ipa)

        assert (
            phonological_distance(
                ipa_converter_module.greek_to_ipa("λόγος"),
                ipa_converter_module.greek_to_ipa("λογος"),
                {},
            )
            == 0.0
        )

    def test_completely_different_sequences_use_full_default_cost(self) -> None:
        assert phonological_distance(["x", "q"], ["!", "?"], {}) == pytest.approx(
            DEFAULT_COST * 2
        )

    def test_internal_substitution_accumulates_cost(self) -> None:
        matrix = {"b": {"x": 2.0}, "x": {"b": 2.0}}

        assert phonological_distance(
            ["a", "b", "d"],
            ["a", "x", "d"],
            matrix,
        ) == pytest.approx(2.0)

    def test_accented_matrix_backed_substitution_uses_sequence_normalization(self) -> None:
        matrix = {"o": {"ɛː": 0.5}, "ɛː": {"o": 0.5}}

        assert phonological_distance(["ó"], ["ɛ́ː"], matrix) == pytest.approx(0.5)

    def test_non_accent_combining_marks_are_not_normalized_away(self) -> None:
        assert phonological_distance(["n̩"], ["n"], {}) == pytest.approx(DEFAULT_COST)
        assert phonological_distance(["ã"], ["a"], {}) == pytest.approx(DEFAULT_COST)

    def test_internal_mismatch_uses_default_cost(self) -> None:
        assert phonological_distance(
            ["a", "b", "d"],
            ["a", "!", "d"],
            {},
        ) == pytest.approx(DEFAULT_COST)

    def test_suffix_overhang_is_penalized(self) -> None:
        assert phonological_distance(["a"], ["a", "b"], {}) == pytest.approx(DEFAULT_COST)

    def test_prefix_overhang_is_penalized(self) -> None:
        assert phonological_distance(["b", "a"], ["a"], {}) == pytest.approx(DEFAULT_COST)


class TestSequenceDistance:
    def test_empty_sequences_return_zero(self) -> None:
        assert sequence_distance([], [], {}) == 0.0

    def test_single_phone_sequence_uses_matrix_distance(self) -> None:
        matrix = {"a": {"ɛː": 0.5}, "ɛː": {"a": 0.5}}

        assert sequence_distance(["a"], ["ɛː"], matrix) == pytest.approx(0.5)

    def test_overhang_is_penalized_by_full_word_alignment(self) -> None:
        assert sequence_distance(["a"], ["a", "b"], {}) == pytest.approx(DEFAULT_COST)

    def test_normalized_sequence_distance_is_zero_for_identical_sequences(self) -> None:
        assert normalized_sequence_distance(["a", "b"], ["a", "b"], {}) == pytest.approx(
            0.0
        )

    def test_normalized_sequence_distance_counts_single_insertion(self) -> None:
        assert normalized_sequence_distance(["a"], ["a", "b"], {}) == pytest.approx(0.5)

    def test_sequence_distance_converts_generic_sequences_to_lists(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}

        def fake_phonological_distance(
            seq1: list[str], seq2: list[str], matrix: distance_module.MatrixData
        ) -> float:
            captured["seq1"] = seq1
            captured["seq2"] = seq2
            captured["matrix"] = matrix
            return 1.25

        monkeypatch.setattr(distance_module, "phonological_distance", fake_phonological_distance)

        result = sequence_distance(("a", "b"), ("a", "c"), {"a": {"c": 0.5}})

        assert result == pytest.approx(1.25)
        assert captured["seq1"] == ["a", "b"]
        assert captured["seq2"] == ["a", "c"]
        assert captured["matrix"] == {"a": {"c": 0.5}}

    def test_normalized_sequence_distance_converts_generic_sequences_to_lists(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}

        def fake_normalized_distance(
            seq1: list[str], seq2: list[str], matrix: distance_module.MatrixData
        ) -> float:
            captured["seq1"] = seq1
            captured["seq2"] = seq2
            captured["matrix"] = matrix
            return 0.25

        monkeypatch.setattr(
            distance_module,
            "normalized_phonological_distance",
            fake_normalized_distance,
        )

        result = normalized_sequence_distance(("a",), ("b",), {"a": {"b": 0.5}})

        assert result == pytest.approx(0.25)
        assert captured["seq1"] == ["a"]
        assert captured["seq2"] == ["b"]
        assert captured["matrix"] == {"a": {"b": 0.5}}


class TestWordDistance:
    def test_empty_words_return_zero(self) -> None:
        assert word_distance("", "", {}) == 0.0

    def test_single_phone_words_use_matrix_distance(self) -> None:
        matrix = {"a": {"ɛː": 0.5}, "ɛː": {"a": 0.5}}

        assert word_distance("a", "ɛː", matrix) == pytest.approx(0.5)

    def test_space_separated_ipa_is_tokenized_before_alignment(self) -> None:
        matrix = {"b": {"x": 2.0}, "x": {"b": 2.0}}

        assert word_distance("a b d", "a x d", matrix) == pytest.approx(2.0)

    def test_word_overhang_is_penalized(self) -> None:
        assert word_distance("a", "a b", {}) == pytest.approx(DEFAULT_COST)

    def test_compact_ipa_is_tokenized_before_alignment(self) -> None:
        assert word_distance("dɛːmos", "dɛːmon", {}) == pytest.approx(
            UNKNOWN_SUBSTITUTION_COST
        )

    def test_converter_output_matches_stressed_lexicon_ipa(self) -> None:
        assert word_distance(to_ipa("λόγος"), "lóɡos", {}) == pytest.approx(0.0)

    def test_word_distance_keeps_non_accent_combining_marks_attached_to_phone(self) -> None:
        assert word_distance("n̩", "n", {}) == pytest.approx(sequence_distance(["n̩"], ["n"], {}))
        assert word_distance("ã", "a", {}) == pytest.approx(sequence_distance(["ã"], ["a"], {}))

    def test_normalized_word_distance_is_clamped_to_one(self) -> None:
        assert normalized_word_distance("x q", "! ?", {}) == pytest.approx(1.0)

    def test_normalized_word_distance_matches_sequence_api_for_combining_mark_phones(
        self,
    ) -> None:
        assert normalized_word_distance("n̩", "n", {}) == pytest.approx(
            normalized_sequence_distance(["n̩"], ["n"], {})
        )
        assert normalized_word_distance("ã", "a", {}) == pytest.approx(
            normalized_sequence_distance(["ã"], ["a"], {})
        )

    def test_word_distance_tokenizes_inputs_before_delegating(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tokenize_calls: list[str] = []
        captured: dict[str, object] = {}

        def fake_tokenize_ipa(text: str) -> list[str]:
            tokenize_calls.append(text)
            return [f"<{text}>"]

        def fake_phonological_distance(
            seq1: list[str], seq2: list[str], matrix: distance_module.MatrixData
        ) -> float:
            captured["seq1"] = seq1
            captured["seq2"] = seq2
            captured["matrix"] = matrix
            return 2.5

        monkeypatch.setattr(distance_module, "tokenize_ipa", fake_tokenize_ipa)
        monkeypatch.setattr(distance_module, "phonological_distance", fake_phonological_distance)

        result = word_distance("λόγος", "λογος", {"a": {"b": 0.5}})

        assert result == pytest.approx(2.5)
        assert tokenize_calls == ["λόγος", "λογος"]
        assert captured["seq1"] == ["<λόγος>"]
        assert captured["seq2"] == ["<λογος>"]
        assert captured["matrix"] == {"a": {"b": 0.5}}

    def test_normalized_word_distance_tokenizes_inputs_before_delegating(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tokenize_calls: list[str] = []
        captured: dict[str, object] = {}

        def fake_tokenize_ipa(text: str) -> list[str]:
            tokenize_calls.append(text)
            return [f"<{text}>"]

        def fake_normalized_distance(
            seq1: list[str], seq2: list[str], matrix: distance_module.MatrixData
        ) -> float:
            captured["seq1"] = seq1
            captured["seq2"] = seq2
            captured["matrix"] = matrix
            return 0.4

        monkeypatch.setattr(distance_module, "tokenize_ipa", fake_tokenize_ipa)
        monkeypatch.setattr(
            distance_module,
            "normalized_phonological_distance",
            fake_normalized_distance,
        )

        result = normalized_word_distance("λόγος", "λογος", {"a": {"b": 0.5}})

        assert result == pytest.approx(0.4)
        assert tokenize_calls == ["λόγος", "λογος"]
        assert captured["seq1"] == ["<λόγος>"]
        assert captured["seq2"] == ["<λογος>"]
        assert captured["matrix"] == {"a": {"b": 0.5}}


class TestNormalizedPhonologicalDistance:
    def test_identical_sequences_have_zero_normalized_cost(self) -> None:
        assert normalized_phonological_distance(["a"], ["a"], {}) == pytest.approx(0.0)

    def test_accented_pre_tokenized_converter_output_normalizes_to_zero(self) -> None:
        assert normalized_phonological_distance(
            greek_to_ipa("λόγος"),
            greek_to_ipa("λογος"),
            {},
        ) == pytest.approx(0.0)

    def test_matrix_backed_substitution_keeps_unit_scale_matrix_value(self) -> None:
        matrix = {"a": {"b": 0.5}, "b": {"a": 0.5}}

        assert normalized_phonological_distance(["a"], ["b"], matrix) == pytest.approx(0.5)

    def test_known_unmapped_substitution_uses_full_unit_cost(self) -> None:
        assert normalized_phonological_distance(["s"], ["n"], {}) == pytest.approx(1.0)

    def test_unknown_substitution_uses_full_unit_cost(self) -> None:
        assert normalized_phonological_distance(["x"], ["?"], {}) == pytest.approx(1.0)

    def test_completely_different_sequences_normalize_to_one(self) -> None:
        assert normalized_phonological_distance(
            ["x", "q"], ["!", "?"], {}
        ) == pytest.approx(
            1.0
        )

    @pytest.mark.parametrize(
        ("seq1", "seq2", "matrix"),
        [
            ([], [], {}),
            (["a"], ["a"], {}),
            (["x", "y"], ["a", "b"], {}),
            (["a"], ["b"], {"a": {"b": 0.5}, "b": {"a": 0.5}}),
        ],
    )
    def test_normalized_distance_stays_within_closed_interval(
        self,
        seq1: list[str],
        seq2: list[str],
        matrix: distance_module.MatrixData,
    ) -> None:
        result = normalized_phonological_distance(seq1, seq2, matrix)

        assert 0.0 <= result <= 1.0


class TestRealMatrix:
    def test_committed_attic_doric_matrix_contains_direct_rows_for_completed_inventory(
        self,
        committed_matrix_copy: Path,
    ) -> None:
        matrix = load_matrix(committed_matrix_copy)

        assert matrix["e"]["o"] == pytest.approx(0.4)
        assert matrix["ɡ"]["k"] == pytest.approx(0.2)
        assert matrix["ɣ"]["ɡ"] == pytest.approx(0.1)
        assert matrix["x"]["kʰ"] == pytest.approx(0.1)
        assert matrix["y"]["i"] == pytest.approx(0.3)
        assert matrix["oi"]["i"] == pytest.approx(0.15)
        assert phone_distance("ɡ", "k", matrix) == pytest.approx(0.2)
        assert phone_distance("x", "kʰ", matrix) == pytest.approx(0.1)
        assert phone_distance("y", "i", matrix) == pytest.approx(0.3)
        assert phone_distance("oi", "i", matrix) == pytest.approx(0.15)

    def test_committed_attic_doric_matrix_keeps_normalized_word_distance_reasonable(
        self,
        committed_matrix_copy: Path,
    ) -> None:
        matrix = load_matrix(committed_matrix_copy)

        logos_distance = normalized_word_distance("loɡos", "lokos", matrix)
        diphthong_distance = normalized_word_distance("oi", "i", matrix)
        far_known_distance = normalized_word_distance("loɡos", "tʰyːmos", matrix)

        assert 0.03 <= logos_distance <= 0.05
        assert 0.14 <= diphthong_distance <= 0.16
        assert far_known_distance == pytest.approx(0.6, rel=0.1)

    def test_known_consonant_substitution_no_longer_uses_default_gap_cost(
        self,
        committed_matrix_copy: Path,
    ) -> None:
        matrix = load_matrix(committed_matrix_copy)

        result = normalized_word_distance("logos", "logon", matrix)

        assert 0.15 <= result <= 0.25

    @pytest.mark.parametrize(
        ("koine_phone", "attic_phone"),
        [("f", "pʰ"), ("θ", "tʰ"), ("x", "kʰ"), ("ð", "d"), ("ɣ", "ɡ")],
    )
    def test_committed_matrix_includes_supported_koine_consonants(
        self,
        committed_matrix_copy: Path,
        koine_phone: str,
        attic_phone: str,
    ) -> None:
        matrix = load_matrix(committed_matrix_copy)

        assert koine_phone in matrix
        assert attic_phone in matrix[koine_phone]
        assert phone_distance(koine_phone, attic_phone, matrix) == pytest.approx(0.1)

    def test_koine_short_word_distance_uses_matrix_backed_consonant_mapping(
        self,
        committed_matrix_copy: Path,
    ) -> None:
        matrix = load_matrix(committed_matrix_copy)

        assert normalized_word_distance("xa", "kʰa", matrix) < normalized_word_distance(
            "xa",
            "pa",
            matrix,
        )


class TestMatrixMeta:
    def test_matrix_meta_is_dict_type_alias(self) -> None:
        meta: MatrixMeta = {"version": "1.0.0", "description": "test"}

        assert isinstance(meta, dict)
        assert meta["version"] == "1.0.0"

    def test_matrix_meta_accepts_arbitrary_value_types(self) -> None:
        meta: MatrixMeta = {"version": 1, "nested": {"key": True}, "items": [1, 2]}

        assert meta["version"] == 1
        assert meta["nested"]["key"] is True


class TestLoadMatrixDocument:
    def test_returns_matrix_data_and_meta_dict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trusted_dir = tmp_path / "matrices"
        trusted_dir.mkdir()
        monkeypatch.setenv("PROTEUS_TRUSTED_MATRICES_DIR", str(trusted_dir))

        matrix_file = trusted_dir / "test.json"
        matrix_file.write_text(
            json.dumps(
                {
                    "_meta": {"version": "1.0.0", "generated_at": "2026-04-20T00:00:00+00:00"},
                    "sound_classes": {
                        "vowels": {"a": {"a": 0.0, "e": 0.3}, "e": {"a": 0.3, "e": 0.0}},
                    },
                }
            ),
            encoding="utf-8",
        )

        matrix, meta = load_matrix_document(matrix_file)

        assert isinstance(matrix, dict)
        assert isinstance(meta, dict)
        assert "a" in matrix
        assert matrix["a"]["e"] == pytest.approx(0.3)
        assert meta["version"] == "1.0.0"
        assert meta["generated_at"] == "2026-04-20T00:00:00+00:00"

    def test_returns_empty_meta_when_meta_key_is_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trusted_dir = tmp_path / "matrices"
        trusted_dir.mkdir()
        monkeypatch.setenv("PROTEUS_TRUSTED_MATRICES_DIR", str(trusted_dir))

        matrix_file = trusted_dir / "no_meta.json"
        matrix_file.write_text(
            json.dumps({"vowels": {"a": {"e": 0.5}}}),
            encoding="utf-8",
        )

        matrix, meta = load_matrix_document(matrix_file)

        assert isinstance(matrix, dict)
        assert meta == {}
        assert matrix["a"]["e"] == pytest.approx(0.5)

    def test_returns_empty_meta_when_meta_is_not_a_dict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trusted_dir = tmp_path / "matrices"
        trusted_dir.mkdir()
        monkeypatch.setenv("PROTEUS_TRUSTED_MATRICES_DIR", str(trusted_dir))

        matrix_file = trusted_dir / "bad_meta.json"
        matrix_file.write_text(
            json.dumps({"_meta": "not-a-dict", "a": {"b": 0.5}}),
            encoding="utf-8",
        )

        matrix, meta = load_matrix_document(matrix_file)

        assert meta == {}
        assert matrix == {"a": {"b": 0.5}}

    def test_raises_for_nonexistent_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trusted_dir = tmp_path / "matrices"
        trusted_dir.mkdir()
        monkeypatch.setenv("PROTEUS_TRUSTED_MATRICES_DIR", str(trusted_dir))

        with pytest.raises(FileNotFoundError):
            load_matrix_document(trusted_dir / "missing.json")

    def test_committed_matrix_returns_version_and_generated_at(
        self, committed_matrix_copy: Path
    ) -> None:
        matrix, meta = load_matrix_document(committed_matrix_copy)

        assert isinstance(matrix, dict)
        assert len(matrix) > 0
        assert isinstance(meta, dict)
        assert "version" in meta
        assert "generated_at" in meta


class TestRegisterTrustedMatricesDir:
    @pytest.fixture(autouse=True)
    def restore_trusted_external_matrix_dirs(self) -> Iterator[None]:
        with distance_module._TRUSTED_EXTERNAL_MATRIX_DIRS_LOCK:
            original_dirs = set(distance_module._TRUSTED_EXTERNAL_MATRIX_DIRS)
        try:
            yield
        finally:
            with distance_module._TRUSTED_EXTERNAL_MATRIX_DIRS_LOCK:
                distance_module._TRUSTED_EXTERNAL_MATRIX_DIRS.clear()
                distance_module._TRUSTED_EXTERNAL_MATRIX_DIRS.update(original_dirs)

    def test_registers_valid_directory(self, tmp_path: Path) -> None:
        from phonology.distance import register_trusted_matrices_dir

        matrices_dir = tmp_path / "matrices"
        matrices_dir.mkdir()
        register_trusted_matrices_dir(matrices_dir)

        matrix_file = matrices_dir / "m.json"
        matrix_file.write_text('{"a": {"a": 0.0}}', encoding="utf-8")
        result = load_matrix(matrix_file)
        assert "a" in result

    def test_rejects_symlink_in_path(self, tmp_path: Path) -> None:
        from phonology.distance import register_trusted_matrices_dir

        real_dir = tmp_path / "real"
        real_dir.mkdir()
        link_dir = tmp_path / "link"
        try:
            link_dir.symlink_to(real_dir)
        except OSError as exc:  # pragma: no cover - platform-dependent
            pytest.skip(f"symlink creation not supported: {exc}")

        with pytest.raises(ValueError, match="Symlink detected"):
            register_trusted_matrices_dir(link_dir)

    def test_allow_symlinks_permits_symlinked_path(self, tmp_path: Path) -> None:
        from phonology.distance import register_trusted_matrices_dir

        real_dir = tmp_path / "real"
        real_dir.mkdir()
        link_dir = tmp_path / "link"
        try:
            link_dir.symlink_to(real_dir)
        except OSError as exc:  # pragma: no cover - platform-dependent
            pytest.skip(f"symlink creation not supported: {exc}")

        register_trusted_matrices_dir(link_dir, allow_symlinks=True)

        matrix_file = real_dir / "m.json"
        matrix_file.write_text('{"a": {"a": 0.0}}', encoding="utf-8")
        result = load_matrix(link_dir / "m.json")
        assert "a" in result

    def test_rejects_nonexistent_directory(self, tmp_path: Path) -> None:
        from phonology.distance import register_trusted_matrices_dir

        with pytest.raises((ValueError, FileNotFoundError)):
            register_trusted_matrices_dir(tmp_path / "nonexistent")
