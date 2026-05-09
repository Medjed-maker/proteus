"""Tests for the smaller search helpers and per-stage components.

Covers ``TestFilterStage``, ``TestQueryModeHelpers``, ``TestBuildLexiconMap``,
``TestBuildKmerIndex``, and ``TestSeedStage`` — the small focused classes that
previously lived alongside ``TestSearch`` in ``tests/test_search.py``.
"""

from __future__ import annotations

import unicodedata

import pytest

from phonology.ipa_converter import tokenize_ipa
from phonology.search import (
    SearchResult,
    build_kmer_index,
    build_lexicon_map,
    classify_query_mode,
    filter_stage,
    normalize_query_for_search,
    seed_stage,
)


class TestFilterStage:
    """Verify final filtering sorts by confidence and rejects invalid limits."""

    def test_sorts_by_confidence_desc_then_lemma(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        filtered = filter_stage(sample_search_results, max_results=2)

        assert [(result.lemma, result.confidence) for result in filtered] == [
            ("α", 0.9),
            ("β", 0.9),
        ]

    @pytest.mark.parametrize("max_results", [0, -1])
    def test_rejects_non_positive_max_results(self, max_results: int) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            filter_stage([], max_results=max_results)


class TestQueryModeHelpers:
    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("λόγος", "Full-form"),
            ("νυν", "Short-query"),
            ("ζηταω-", "Partial-form"),
            ("ζητω~", "Partial-form"),
            ("*λόγ", "Partial-form"),
            ("λο*γος", "Partial-form"),
            (unicodedata.normalize("NFD", "νῦν"), "Short-query"),
            (unicodedata.normalize("NFD", "λόγος"), "Full-form"),
            # Unicode dash markers (paste-from-word-processor, locale
            # keyboards, etc.) must classify as Partial-form just like ASCII
            # hyphen/asterisk. Mirrors _PARTIAL_QUERY_MARKERS in search.py.
            ("ζηταω\u2010", "Partial-form"),  # U+2010 HYPHEN
            ("ζηταω\u2011", "Partial-form"),  # U+2011 NON-BREAKING HYPHEN
            ("ζηταω\u2012", "Partial-form"),  # U+2012 FIGURE DASH
            ("ζηταω\u2013", "Partial-form"),  # U+2013 EN DASH
            ("ζηταω\u2014", "Partial-form"),  # U+2014 EM DASH
            ("ζηταω\u2015", "Partial-form"),  # U+2015 HORIZONTAL BAR
            ("ζηταω\u2212", "Partial-form"),  # U+2212 MINUS SIGN
            ("ζηταω\uff0d", "Partial-form"),  # U+FF0D FULLWIDTH HYPHEN-MINUS
        ],
    )
    def test_classify_query_mode(self, query: str, expected: str) -> None:
        assert classify_query_mode(query) == expected

    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            (" ζηταω- ", "ζηταω"),
            (" ζητω~ ", "ζητω"),
            ("", ""),
            ("   ", ""),
            (" *λόγ ", "λόγ"),
            (" λο*γος ", "λογος"),
            ("λόγος", "λόγος"),
            # Unicode dash markers should be stripped the same way as ASCII
            # hyphen, leaving only the fragment to hand to IPA conversion.
            ("ζηταω\u2013", "ζηταω"),  # U+2013 EN DASH
            ("ζηταω\u2014", "ζηταω"),  # U+2014 EM DASH
            ("ζηταω\uff0d", "ζηταω"),  # U+FF0D FULLWIDTH HYPHEN-MINUS
        ],
    )
    def test_normalize_query_for_search(self, query: str, expected: str) -> None:
        assert normalize_query_for_search(query) == expected

    @pytest.mark.parametrize(
        "query",
        [
            "",
            "   ",
            "*",
            "~",
            "-",
            "-*",
            "~*",
            "*-",
            "ζη~τω-",
            "a-*",
            "a*-",
            "a*b*c",
        ],
    )
    def test_classify_query_mode_rejects_invalid_partial_syntax(
        self, query: str
    ) -> None:
        with pytest.raises(ValueError, match=r"wildcard marker|non-empty string"):
            classify_query_mode(query)

    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("*", "*"),
            ("~", "~"),
            ("-", "-"),
            ("-*", "-*"),
            ("~*", "~*"),
            ("*-", "*-"),
            ("ζη~τω-", "ζη~τω-"),
            ("a-*", "a-*"),
            ("a*-", "a*-"),
            ("a*b*c", "a*b*c"),
        ],
    )
    def test_normalize_query_tolerates_invalid_partial_syntax(
        self, query: str, expected: str
    ) -> None:
        assert normalize_query_for_search(query) == expected


class TestBuildLexiconMap:
    """Verify lexicon map construction and duplicate detection."""

    def test_returns_empty_dict_for_empty_lexicon(self) -> None:
        assert build_lexicon_map([]) == {}

    def test_builds_map_with_token_counts(self) -> None:
        lexicon = [
            {"id": "L1", "headword": "λόγος", "ipa": "lóɡos", "dialect": "attic"},
        ]
        result = build_lexicon_map(lexicon)
        assert "L1" in result
        assert result["L1"].entry == lexicon[0]
        assert result["L1"].token_count == 5

    def test_public_default_caches_ancient_greek_multicharacter_phones(
        self,
    ) -> None:
        """Ancient Greek defaults should tokenize multi-character phones.

        build_lexicon_map should cache apʰlas as ("a", "pʰ", "l", "a", "s")
        with token_count 5 for the sample lexicon.
        """
        lexicon = [
            {"id": "L1", "headword": "target", "ipa": "apʰlas", "dialect": "attic"},
        ]

        result = build_lexicon_map(lexicon)

        assert result["L1"].ipa_tokens == ("a", "pʰ", "l", "a", "s")
        assert result["L1"].token_count == 5

    def test_non_default_language_keeps_literal_fallback_tokens(self) -> None:
        """Non-default language tokenization should keep literal fallback tokens.

        For the apʰlas input lexicon with language="test", build_lexicon_map
        should cache ("a", "p", "ʰ", "l", "a", "s") with token_count 6.
        """
        lexicon = [
            {"id": "L1", "headword": "target", "ipa": "apʰlas", "dialect": "test"},
        ]

        result = build_lexicon_map(lexicon, language="test")

        assert result["L1"].ipa_tokens == ("a", "p", "ʰ", "l", "a", "s")
        assert result["L1"].token_count == 6

    def test_trims_entry_ids_for_lookup_keys(self) -> None:
        lexicon = [
            {"id": " L1 ", "headword": "λόγος", "ipa": "lóɡos", "dialect": "attic"},
        ]
        result = build_lexicon_map(lexicon)

        assert list(result) == ["L1"]

    def test_raises_on_duplicate_entry_id(self) -> None:
        lexicon = [
            {"id": "L1", "headword": "αλφα", "ipa": "alfa", "dialect": "attic"},
            {"id": "L1", "headword": "βητα", "ipa": "beta", "dialect": "attic"},
        ]
        with pytest.raises(ValueError, match="Duplicate entry ID"):
            build_lexicon_map(lexicon)

    def test_raises_on_duplicate_entry_id_after_trimming(self) -> None:
        lexicon = [
            {"id": " L1 ", "headword": "αλφα", "ipa": "alfa", "dialect": "attic"},
            {"id": "L1", "headword": "βητα", "ipa": "beta", "dialect": "attic"},
        ]
        with pytest.raises(ValueError, match="Duplicate entry ID"):
            build_lexicon_map(lexicon)

    @pytest.mark.parametrize("ipa_text", ["", "   "])
    def test_raises_on_empty_or_whitespace_only_ipa(self, ipa_text: str) -> None:
        lexicon = [
            {"id": "L1", "headword": "test", "ipa": ipa_text, "dialect": "attic"},
        ]

        with pytest.raises(ValueError, match="non-empty 'ipa'"):
            build_lexicon_map(lexicon)

    @pytest.mark.parametrize("ipa_text", ["!?", "ã"])
    def test_builds_map_for_non_empty_special_character_ipa(
        self, ipa_text: str
    ) -> None:
        lexicon = [
            {"id": "L1", "headword": "test", "ipa": ipa_text, "dialect": "attic"},
        ]

        result = build_lexicon_map(lexicon)

        assert "L1" in result
        assert result["L1"].entry == lexicon[0]
        assert result["L1"].token_count == len(tokenize_ipa(ipa_text))


class TestBuildKmerIndex:
    """Verify consonant-skeleton k-mer index construction and parameter validation."""

    def test_builds_consonant_skeleton_kmer_index(
        self, sample_lexicon: list[dict[str, str]]
    ) -> None:
        index = build_kmer_index(sample_lexicon, k=2)

        assert index["p t"] == ["L1", "L2"]
        assert index["t n"] == ["L1", "L3"]
        assert "p n" not in index

    def test_public_default_kmer_index_adds_ancient_greek_compatibility_kmers(
        self,
    ) -> None:
        """Verify build_kmer_index with k=2 on IPA "apʰlas" generates compatibility kmers.

        The "pʰ l", "f l", and "l s" kmers are generated via the default Ancient Greek
        phone inventory and dialect skeleton builders, all mapping to id "L1".
        """
        index = build_kmer_index(
            [{"id": "L1", "headword": "target", "ipa": "apʰlas", "dialect": "attic"}],
            k=2,
        )

        assert index["pʰ l"] == ["L1"]
        assert index["f l"] == ["L1"]
        assert index["l s"] == ["L1"]

    def test_explicit_empty_builders_do_not_add_dialect_skeletons(self) -> None:
        from phonology.languages.ancient_greek.ipa import get_known_phones

        index = build_kmer_index(
            [{"id": "L1", "headword": "target", "ipa": "apʰlas", "dialect": "attic"}],
            k=2,
            phone_inventory=get_known_phones(),
            dialect_skeleton_builders=(),
        )

        assert index["pʰ l"] == ["L1"]
        assert "f l" not in index
        assert index["l s"] == ["L1"]

    def test_default_language_fills_in_ancient_greek_defaults(self) -> None:
        """Verify build_kmer_index fills in default phone_inventory and dialect_skeleton_builders for DEFAULT_LANGUAGE_ID."""
        index = build_kmer_index(
            [{"id": "L1", "headword": "target", "ipa": "apʰlas", "dialect": "attic"}],
            k=2,
        )

        # "f l" kmer only exists when dialect_skeleton_builders adds compatibility kmers
        assert index["f l"] == ["L1"]
        # "pʰ l" kmer exists via tokenization with phone_inventory
        assert index["pʰ l"] == ["L1"]

    def test_normalized_default_language_fills_in_ancient_greek_defaults(self) -> None:
        """Default-language compatibility accepts the API/profile-normalized ID form."""
        index = build_kmer_index(
            [{"id": "L1", "headword": "target", "ipa": "apʰlas", "dialect": "attic"}],
            k=2,
            language=" Ancient_Greek ",
        )

        assert index["f l"] == ["L1"]
        assert index["pʰ l"] == ["L1"]

    def test_non_default_language_passes_through_supplied_parameters(self) -> None:
        """Verify build_kmer_index passes through supplied phone_inventory and dialect_skeleton_builders for non-default languages unchanged."""
        custom_inventory = ["a", "p", "l", "s"]

        index = build_kmer_index(
            [{"id": "L1", "headword": "target", "ipa": "aplas", "dialect": "generic"}],
            k=2,
            phone_inventory=custom_inventory,
            dialect_skeleton_builders=(),
            language="other_lang",
        )

        # With empty dialect_skeleton_builders, no compatibility kmers are added
        assert index["p l"] == ["L1"]
        # "f l" would only exist if Ancient Greek defaults were applied (they shouldn't be)
        assert "f l" not in index

    def test_explicit_koine_builder_adds_compatible_kmers(self) -> None:
        from phonology.languages.ancient_greek.ipa import (
            apply_koine_consonant_shifts,
            get_known_phones,
        )

        index = build_kmer_index(
            [{"id": "L1", "headword": "target", "ipa": "apʰlas", "dialect": "attic"}],
            k=2,
            phone_inventory=get_known_phones(),
            dialect_skeleton_builders=(apply_koine_consonant_shifts,),
        )

        assert index["pʰ l"] == ["L1"]
        assert index["f l"] == ["L1"]
        assert index["l s"] == ["L1"]

    def test_dialect_skeleton_builders_receive_independent_token_lists(self) -> None:
        """Verify each dialect skeleton builder receives an independent token list."""
        observed_tokens: list[list[str]] = []

        def mutating_builder(tokens: list[str]) -> list[str]:
            tokens.append("t")
            return tokens

        def observing_builder(tokens: list[str]) -> list[str]:
            observed_tokens.append(list(tokens))
            return tokens

        index = build_kmer_index(
            [{"id": "L1", "headword": "target", "ipa": "pa", "dialect": "attic"}],
            k=2,
            dialect_skeleton_builders=(mutating_builder, observing_builder),
        )

        assert index["p t"] == ["L1"]
        assert observed_tokens == [tokenize_ipa("pa")]

    @pytest.mark.parametrize("k", [0, -1])
    def test_rejects_non_positive_k(self, k: int) -> None:
        with pytest.raises(ValueError, match="build_kmer_index.*k"):
            build_kmer_index([], k=k)

    @pytest.mark.parametrize(
        ("entry", "message"),
        [
            ({"ipa": "pten"}, "non-empty 'id' or 'headword'"),
            (
                {"id": "   ", "headword": "   ", "ipa": "pten"},
                "non-empty 'id' or 'headword'",
            ),
            ({"id": "L1", "headword": "πτην"}, "non-empty 'ipa'"),
            ({"id": "L1", "headword": "πτην", "ipa": "   "}, "non-empty 'ipa'"),
        ],
    )
    def test_skips_invalid_lexicon_entries_with_warning(
        self, entry: dict[str, str], message: str
    ) -> None:
        # Invalid entries are skipped with a warning, not a hard error
        with pytest.warns(UserWarning, match=message):
            result = build_kmer_index([entry], k=2)
        assert result == {}


class TestSeedStage:
    """Verify stage-1 seed ranking by shared consonant-skeleton k-mers."""

    def test_ranks_candidates_by_shared_seed_count(
        self, sample_lexicon: list[dict[str, str]]
    ) -> None:
        index = build_kmer_index(sample_lexicon, k=2)

        candidates = seed_stage("pten", index, k=2)

        assert candidates == ["L1", "L2", "L3"]

    def test_returns_empty_list_when_query_has_no_seedable_consonant_skeleton(
        self,
    ) -> None:
        assert seed_stage("aː", {"p t": ["L1"]}, k=2) == []

    def test_k1_finds_candidates_for_single_consonant_query(self) -> None:
        """k=1 unigram index should find entries sharing a single consonant."""
        lexicon = [
            {"id": "L1", "headword": "πτω", "ipa": "pto", "dialect": "attic"},
            {"id": "L2", "headword": "κτω", "ipa": "kto", "dialect": "attic"},
            {"id": "L3", "headword": "ποι", "ipa": "poi", "dialect": "attic"},
        ]
        unigram_idx = build_kmer_index(lexicon, k=1)

        # Query IPA "poieɔː" has consonant skeleton ['p'], so k=1 produces ["p"].
        candidates = seed_stage("poieɔː", unigram_idx, k=1)

        # L1 (pto → skeleton p,t) and L3 (poi → skeleton p) contain "p"
        assert "L1" in candidates
        assert "L2" not in candidates
        assert "L3" in candidates

    def test_k1_returns_empty_for_pure_vowel_query(self) -> None:
        """k=1 should still return empty for a query with zero consonants."""
        unigram_idx: dict[str, list[str]] = {"p": ["L1"]}
        assert seed_stage("aː", unigram_idx, k=1) == []

    def test_default_language_seeds_ancient_greek_multicharacter_phones(
        self,
    ) -> None:
        """Verify seed_stage default fills in Ancient Greek phone inventory.

        With ``language="ancient_greek"`` (default) and no explicit
        ``phone_inventory``, ``seed_stage`` must tokenize ``pʰt`` as a single
        ``pʰ`` phone followed by ``t``, matching the consonant skeleton
        produced by ``build_kmer_index``. Before the public-defaults backfill
        was applied, the query was split into ``p``/``ʰ``/``t`` and seeding
        returned an empty list even though the index contained the entry.
        """
        lexicon = [
            {"id": "L1", "headword": "target", "ipa": "pʰt", "dialect": "attic"},
        ]
        index = build_kmer_index(lexicon, k=2)

        candidates = seed_stage("pʰt", index, k=2)

        assert candidates == ["L1"]

    def test_non_default_language_does_not_inject_ancient_greek_inventory(
        self,
    ) -> None:
        """Non-default language must keep the literal-character fallback."""
        lexicon = [
            {"id": "L1", "headword": "target", "ipa": "pʰt", "dialect": "test"},
        ]
        # Use a non-default language for both index and query so neither side
        # gets the Ancient Greek backfill.
        index = build_kmer_index(lexicon, k=2, language="test")

        candidates = seed_stage("pʰt", index, k=2, language="test")

        # Without the inventory, "pʰt" tokenizes to ["p", "ʰ", "t"] whose
        # consonant skeleton is ["p", "t"], producing the kmer "p t". The
        # index also produced "p t" for the same reason, so seeding succeeds
        # but via single-character matching, not multi-character phones.
        assert candidates == ["L1"]
