"""Tests for partial/wildcard query behavior in phonology.search."""
# Several tests define monkeypatch stubs whose parameter names must mirror the
# real signatures of the functions they replace (e.g. `_score_stage(query_ipa,
# candidates, lexicon_map, matrix)`). Renaming the unused params to `_prefix`
# would either drift from the real signature or force kwargs-only call sites;
# both are worse than a file-level suppression. Keep this scoped to
# reportUnusedVariable only — other checks stay on.
# pyright: reportUnusedVariable=false

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

import pytest

from phonology import search as search_module
from phonology.search import (
    SearchResult,
    build_lexicon_map,
    search,
)


class TestSearchPartial:
    """Tests for partial/wildcard query behavior."""

    def test_search_normalizes_partial_query_before_ipa_conversion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Partial-form markers should be removed before IPA conversion."""
        captured: list[str] = []

        def fake_to_ipa(query: str, dialect: str = "attic") -> str:
            captured.append(query)
            return "pa"

        monkeypatch.setattr(search_module, "to_ipa", fake_to_ipa)
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: [])
        monkeypatch.setattr(search_module, "_score_stage", lambda *args, **kwargs: [])
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda query_ipa, results, lexicon_map, matrix, language="ancient_greek": results,
        )

        search(
            "ζηταω-",
            [{"id": "L1", "headword": "target", "ipa": "pa", "dialect": "attic"}],
            matrix={},
            max_results=1,
            index={},
            unigram_index={},
        )

        assert captured == ["ζηταω"]

    def test_search_tokenizes_infix_partial_fragments_separately(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Infix partial-form queries should not convert across the wildcard."""
        captured: list[str] = []
        scored: dict[str, str] = {}

        def fake_to_ipa(query: str, dialect: str = "attic") -> str:
            captured.append(query)
            return query

        monkeypatch.setattr(search_module, "to_ipa", fake_to_ipa)
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: [])
        def fake_score_stage(query_ipa, candidates, lexicon_map, matrix):
            scored.setdefault("query_ipa", query_ipa)
            return []

        monkeypatch.setattr(search_module, "_score_stage", fake_score_stage)
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda query_ipa, results, lexicon_map, matrix, language="ancient_greek": results,
        )

        search(
            "a*c",
            [{"id": "L1", "headword": "target", "ipa": "a c", "dialect": "attic"}],
            matrix={},
            max_results=1,
            index={},
            unigram_index={},
        )

        assert captured == ["a", "c"]
        assert scored["query_ipa"] == "a c"

    def test_infix_partial_query_preserves_diphthong_boundary(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A wildcard between Greek letters should block cross-boundary diphthongs."""
        captured: dict[str, object] = {}

        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: ["L1"])

        def fake_score_stage(
            query_ipa: str,
            candidates: Iterable[str],
            lexicon_map: object,
            matrix: object,
        ) -> list[SearchResult]:
            captured["query_ipa"] = query_ipa
            captured["candidates"] = list(candidates)
            return []

        monkeypatch.setattr(search_module, "_score_stage", fake_score_stage)

        def fake_select_partial_seed_candidates(
            partial_query: search_module.PartialQueryTokens,
            seed_candidates: Iterable[str],
            lexicon_map: object,
            stage2_limit: int,
        ) -> list[str]:
            captured["partial_query_tokens"] = partial_query
            return list(seed_candidates)

        monkeypatch.setattr(
            search_module,
            "_select_partial_seed_candidates",
            fake_select_partial_seed_candidates,
        )

        search(
            "α*ι",
            [{"id": "L1", "headword": "target", "ipa": "a i", "dialect": "attic"}],
            matrix={},
            max_results=1,
            index={"": ["L1"]},
            unigram_index={},
        )

        assert captured["query_ipa"] == "a i"
        assert captured["partial_query_tokens"] == search_module.PartialQueryTokens(
            "infix",
            ("a",),
            ("i",),
        )

    def test_partial_query_prefers_exact_prefix_over_stronger_confidence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Partial-form final ordering should prefer exact prefix matches."""
        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "pa")
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: ["L1", "L2", "L3"])
        monkeypatch.setattr(
            search_module,
            "_score_stage",
            lambda query_ipa, candidates, lexicon_map, matrix: [
                SearchResult(
                    lemma="strong-prefix",
                    confidence=0.90,
                    dialect_attribution="lemma dialect: attic",
                    entry_id="L2",
                ),
                SearchResult(
                    lemma="exact-prefix",
                    confidence=0.70,
                    dialect_attribution="lemma dialect: attic",
                    entry_id="L1",
                ),
            ],
        )
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda query_ipa, results, lexicon_map, matrix, language="ancient_greek": results,
        )

        lexicon = [
            {"id": "L1", "headword": "exact-prefix", "ipa": "pa", "dialect": "attic"},
            {"id": "L2", "headword": "strong-prefix", "ipa": "pai", "dialect": "attic"},
            {"id": "L3", "headword": "no-prefix", "ipa": "ba", "dialect": "attic"},
        ]

        results = search("ζηταω-", lexicon, matrix={}, max_results=2, index={})

        assert [result.lemma for result in results] == ["exact-prefix", "strong-prefix"]

    def test_partial_query_keeps_exact_prefix_below_confidence_floor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Partial-form should keep longer exact-prefix hits below the confidence floor."""
        threshold = search_module._PARTIAL_QUERY_CONFIDENCE_THRESHOLD
        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "pa")
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: ["L1"])
        monkeypatch.setattr(
            search_module,
            "_score_stage",
            lambda query_ipa, candidates, lexicon_map, matrix: [
                SearchResult(
                    lemma="longer-prefix",
                    confidence=threshold - 0.01,
                    dialect_attribution="lemma dialect: attic",
                    entry_id="L1",
                )
            ],
        )
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda query_ipa, results, lexicon_map, matrix, language="ancient_greek": results,
        )

        lexicon = [
            {"id": "L1", "headword": "longer-prefix", "ipa": "p a i", "dialect": "attic"},
        ]

        results = search("ζηταω-", lexicon, matrix={}, max_results=1, index={})

        assert len(results) == 1
        assert results[0].lemma == "longer-prefix"

    @pytest.mark.parametrize(
        ("tokens", "target_ipa"),
        [
            (search_module.PartialQueryTokens("prefix", ("p", "a"), ()), "p a i"),
            (search_module.PartialQueryTokens("suffix", (), ("b", "c")), "x b c"),
            (search_module.PartialQueryTokens("infix", ("a",), ("c",)), "a x c"),
        ],
        ids=["prefix", "suffix", "infix"],
    )
    def test_partial_query_keeps_late_exact_match_beyond_old_stage2_limit(
        self,
        tokens: search_module.PartialQueryTokens,
        target_ipa: str,
    ) -> None:
        """Partial-form preselection should retain exact matches from the full seed list."""
        lexicon = [
            {
                "id": f"N{index:02d}",
                "headword": f"noise-{index:02d}",
                "ipa": "b a",
                "dialect": "attic",
            }
            for index in range(40)
        ]
        lexicon.append(
            {"id": "TARGET", "headword": "target", "ipa": target_ipa, "dialect": "attic"}
        )
        lexicon_map = build_lexicon_map(lexicon)

        candidate_ids = search_module._select_partial_seed_candidates(
            tokens,
            [f"N{index:02d}" for index in range(40)] + ["TARGET"],
            lexicon_map,
            stage2_limit=25,
        )

        assert "TARGET" in candidate_ids
        assert candidate_ids[0] == "TARGET"

    def test_partial_infix_seeded_search_keeps_wildcard_gap_candidates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Infix partial search should not stop at seed hits spanning the wildcard."""
        captured: dict[str, object] = {}

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": query)

        def fake_score_stage(
            query_ipa: str,
            candidates: Iterable[str],
            lexicon_map: dict[str, object],
            matrix: object,
        ) -> list[SearchResult]:
            captured["candidate_ids"] = list(candidates)
            return []

        monkeypatch.setattr(search_module, "_score_stage", fake_score_stage)
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda query_ipa, results, lexicon_map, matrix, language="ancient_greek": results,
        )

        lexicon = [
            {"id": "ADJACENT", "headword": "adjacent", "ipa": "k n", "dialect": "attic"},
            {"id": "GAPPED", "headword": "gapped", "ipa": "k r n", "dialect": "attic"},
            {"id": "TAIL", "headword": "tail", "ipa": "x y z", "dialect": "attic"},
        ]

        search(
            "k*n",
            lexicon,
            matrix={},
            max_results=2,
            index={"k n": ["ADJACENT"]},
            unigram_index={"k": ["ADJACENT", "GAPPED"], "n": ["ADJACENT", "GAPPED"]},
        )

        assert captured["candidate_ids"] == ["ADJACENT", "GAPPED"]

    def test_partial_query_k2_seed_path_does_not_append_full_lexicon_tail(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Partial-form k=2 seeds should not force wildcard matching over every entry."""
        captured: dict[str, object] = {}

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": query)

        def fake_seed_stage(query_ipa: str, index: object, k: int = 2) -> list[str]:
            return ["SEED"] if k == 2 else []

        def fake_score_stage(
            query_ipa: str,
            candidates: Iterable[str],
            lexicon_map: dict[str, object],
            matrix: object,
        ) -> list[SearchResult]:
            captured["candidate_ids"] = list(candidates)
            return []

        monkeypatch.setattr(search_module, "seed_stage", fake_seed_stage)
        monkeypatch.setattr(search_module, "_score_stage", fake_score_stage)
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda query_ipa, results, lexicon_map, matrix, language="ancient_greek": results,
        )

        lexicon = [
            {"id": "SEED", "headword": "seed", "ipa": "a x c", "dialect": "attic"},
            {"id": "TAIL", "headword": "tail", "ipa": "a y c", "dialect": "attic"},
        ]

        search("a*c", lexicon, matrix={}, max_results=2, index={}, unigram_index={})

        assert captured["candidate_ids"] == ["SEED"]

    def test_partial_query_keeps_zero_prefix_overlap_candidates_until_annotation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Partial-form candidate generation should defer zero-overlap pruning until annotation."""
        captured: dict[str, object] = {}

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "pa")
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: ["L1", "L2"])

        def fake_score_stage(
            query_ipa: str,
            candidates: Iterable[str],
            lexicon_map: dict[str, object],
            matrix: object,
        ) -> list[SearchResult]:
            captured["candidate_ids"] = list(candidates)
            return []

        monkeypatch.setattr(search_module, "_score_stage", fake_score_stage)
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda query_ipa, results, lexicon_map, matrix, language="ancient_greek": results,
        )

        lexicon = [
            {"id": "L1", "headword": "prefix", "ipa": "pa", "dialect": "attic"},
            {"id": "L2", "headword": "non-prefix", "ipa": "ba", "dialect": "attic"},
        ]

        search("ζηταω-", lexicon, matrix={}, max_results=2, index={})

        assert captured["candidate_ids"] == ["L1", "L2"]

    def test_partial_query_preselection_retains_zero_overlap_tail_within_window(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Partial-form preselection should leave room for some zero-overlap tail candidates."""
        captured: dict[str, object] = {}

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "pa")
        monkeypatch.setattr(
            search_module,
            "seed_stage",
            lambda *_args, **_kwargs: ["L1", "L2"] + [f"L{index:03d}" for index in range(3, 121)],
        )

        def fake_score_stage(
            query_ipa: str,
            candidates: Iterable[str],
            lexicon_map: dict[str, object],
            matrix: object,
        ) -> list[SearchResult]:
            captured["candidate_ids"] = list(candidates)
            return []

        monkeypatch.setattr(search_module, "_score_stage", fake_score_stage)
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda query_ipa, results, lexicon_map, matrix, language="ancient_greek": results,
        )

        lexicon = [
            {"id": "L1", "headword": "exact-prefix", "ipa": "pai", "dialect": "attic"},
            {"id": "L2", "headword": "positive-overlap", "ipa": "pi", "dialect": "attic"},
        ]
        lexicon.extend(
            {
                "id": f"L{index:03d}",
                "headword": f"zero-overlap-{index:03d}",
                "ipa": "ba",
                "dialect": "attic",
            }
            for index in range(3, 121)
        )

        search("ζηταω-", lexicon, matrix={}, max_results=1, index={})

        assert len(captured["candidate_ids"]) == search_module._MIN_PARTIAL_STAGE2_CANDIDATES
        expected_candidate_ids = ["L1", "L2"] + [
            f"L{index:03d}" for index in range(3, search_module._MIN_PARTIAL_STAGE2_CANDIDATES + 1)
        ]
        assert captured["candidate_ids"] == expected_candidate_ids

    def test_partial_query_preselection_caps_exact_matches_to_stage2_window(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Partial-form preselection should not exceed the configured stage-2 window."""
        captured: dict[str, object] = {}

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "pa")
        exact_ids = [f"L{index:03d}" for index in range(120)]
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: exact_ids)

        def fake_score_stage(
            query_ipa: str,
            candidates: Iterable[str],
            lexicon_map: dict[str, object],
            matrix: object,
        ) -> list[SearchResult]:
            captured["candidate_ids"] = list(candidates)
            return []

        monkeypatch.setattr(search_module, "_score_stage", fake_score_stage)
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda query_ipa, results, lexicon_map, matrix, language="ancient_greek": results,
        )

        lexicon = [
            {
                "id": candidate_id,
                "headword": f"exact-prefix-{candidate_id}",
                "ipa": "pa",
                "dialect": "attic",
            }
            for candidate_id in exact_ids
        ]

        search("ζηταω-", lexicon, matrix={}, max_results=1, index={})

        assert len(captured["candidate_ids"]) == search_module._MIN_PARTIAL_STAGE2_CANDIDATES
        assert captured["candidate_ids"] == exact_ids[: search_module._MIN_PARTIAL_STAGE2_CANDIDATES]

    def test_partial_query_rule_supported_candidate_survives_confidence_floor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rule-supported partial-form hits should remain when fragment evidence exists."""
        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "pa")
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: ["L1"])
        monkeypatch.setattr(
            search_module,
            "_score_stage",
            lambda query_ipa, candidates, lexicon_map, matrix: [
                SearchResult(
                    lemma="rule-hit",
                    confidence=0.40,
                    dialect_attribution="lemma dialect: attic",
                    entry_id="L1",
                )
            ],
        )

        def fake_annotate(
            query_ipa: str,
            results: list[SearchResult],
            lexicon_map: dict[str, object],
            matrix: object,
            language: str = "ancient_greek",
        ) -> list[SearchResult]:
            new_results = list(results)
            new_results[0] = replace(new_results[0], applied_rules=["RULE-001"])
            return new_results

        monkeypatch.setattr(search_module, "_annotate_search_results", fake_annotate)

        lexicon = [
            {"id": "L1", "headword": "rule-hit", "ipa": "pi", "dialect": "attic"},
        ]

        results = search("ζηταω-", lexicon, matrix={}, max_results=1, index={})

        assert [result.lemma for result in results] == ["rule-hit"]

    def test_partial_query_drops_rule_supported_candidates_without_fragment_overlap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rule support alone should not keep fragment-free partial-form hits."""
        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "pa")
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: ["L1"])
        monkeypatch.setattr(
            search_module,
            "_score_stage",
            lambda query_ipa, candidates, lexicon_map, matrix: [
                SearchResult(
                    lemma="rule-only-no-overlap",
                    confidence=0.95,
                    dialect_attribution="lemma dialect: attic",
                    entry_id="L1",
                )
            ],
        )

        def fake_annotate(
            query_ipa: str,
            results: list[SearchResult],
            lexicon_map: dict[str, object],
            matrix: object,
            language: str = "ancient_greek",
        ) -> list[SearchResult]:
            annotated = list(results)
            annotated[0].applied_rules = ["RULE-001"]
            return annotated

        monkeypatch.setattr(search_module, "_annotate_search_results", fake_annotate)

        lexicon = [
            {"id": "L1", "headword": "rule-only-no-overlap", "ipa": "sa", "dialect": "attic"},
        ]

        results = search("ζηταω-", lexicon, matrix={}, max_results=1, index={})

        assert results == []

    def test_partial_query_returns_empty_when_no_wildcard_candidate_survives(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Partial-form precision mode should prefer no result over bad matches."""
        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "pa")
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: ["L1", "L2"])
        monkeypatch.setattr(
            search_module,
            "_score_stage",
            lambda query_ipa, candidates, lexicon_map, matrix: [],
        )
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda query_ipa, results, lexicon_map, matrix, language="ancient_greek": results,
        )

        lexicon = [
            {"id": "L1", "headword": "alpha", "ipa": "ba", "dialect": "attic"},
            {"id": "L2", "headword": "beta", "ipa": "ta", "dialect": "attic"},
        ]

        results = search("ζηταω-", lexicon, matrix={}, max_results=2, index={})

        assert results == []

    def test_partial_suffix_query_prefers_suffix_match_over_higher_confidence_noise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Suffix partial-form ordering should prefer full suffix matches."""
        monkeypatch.setattr(
            search_module,
            "to_ipa",
            lambda query, dialect="attic": {"bc": "b c", "a": "a"}.get(query, query),
        )
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: ["L1", "L2", "L3"])
        monkeypatch.setattr(
            search_module,
            "_score_stage",
            lambda query_ipa, candidates, lexicon_map, matrix: [
                SearchResult(
                    lemma="full-suffix",
                    confidence=0.70,
                    dialect_attribution="lemma dialect: attic",
                    entry_id="L1",
                ),
                SearchResult(
                    lemma="high-confidence-noise",
                    confidence=0.95,
                    dialect_attribution="lemma dialect: attic",
                    entry_id="L2",
                ),
                SearchResult(
                    lemma="partial-suffix",
                    confidence=0.80,
                    dialect_attribution="lemma dialect: attic",
                    entry_id="L3",
                ),
            ],
        )
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda query_ipa, results, lexicon_map, matrix, language="ancient_greek": results,
        )

        lexicon = [
            {"id": "L1", "headword": "full-suffix", "ipa": "a b c", "dialect": "attic"},
            {"id": "L2", "headword": "high-confidence-noise", "ipa": "b a x", "dialect": "attic"},
            {"id": "L3", "headword": "partial-suffix", "ipa": "x c", "dialect": "attic"},
        ]

        results = search("*bc", lexicon, matrix={}, max_results=2, index={})

        assert [result.lemma for result in results] == ["full-suffix", "partial-suffix"]

    def test_partial_query_prefers_rule_supported_overlap_over_higher_confidence_overlap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Partial-form ordering should rank rule-supported overlap above raw confidence."""
        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "p a")
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: ["L1", "L2"])
        monkeypatch.setattr(
            search_module,
            "_score_stage",
            lambda query_ipa, candidates, lexicon_map, matrix: [
                SearchResult(
                    lemma="high-confidence-overlap",
                    confidence=0.95,
                    dialect_attribution="lemma dialect: attic",
                    entry_id="L2",
                ),
                SearchResult(
                    lemma="rule-supported-overlap",
                    confidence=0.40,
                    dialect_attribution="lemma dialect: attic",
                    entry_id="L1",
                ),
            ],
        )

        def fake_annotate_results(
            query_ipa: str,
            results: list[SearchResult],
            lexicon_map: dict[str, object],
            matrix: object,
            language: str = "ancient_greek",
        ) -> list[SearchResult]:
            annotated = list(results)
            annotated[1].applied_rules = ["RULE-002"]
            return annotated

        monkeypatch.setattr(search_module, "_annotate_search_results", fake_annotate_results)

        lexicon = [
            {"id": "L1", "headword": "rule-supported-overlap", "ipa": "p i", "dialect": "attic"},
            {"id": "L2", "headword": "high-confidence-overlap", "ipa": "p u", "dialect": "attic"},
        ]

        results = search("ζηταω-", lexicon, matrix={}, max_results=2, index={})

        assert [result.entry_id for result in results] == ["L1", "L2"]
        assert results[0].applied_rules == ["RULE-002"]

    def test_partial_query_prefers_stronger_fragment_match_over_rule_supported_noise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Partial-form rule support should not outrank stronger wildcard coverage."""
        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "p a")
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: ["L1", "L2"])
        monkeypatch.setattr(
            search_module,
            "_score_stage",
            lambda query_ipa, candidates, lexicon_map, matrix: [
                SearchResult(
                    lemma="rule-supported-weak-overlap",
                    confidence=0.95,
                    dialect_attribution="lemma dialect: attic",
                    entry_id="L1",
                ),
                SearchResult(
                    lemma="full-prefix-match",
                    confidence=0.70,
                    dialect_attribution="lemma dialect: attic",
                    entry_id="L2",
                ),
            ],
        )

        def fake_annotate_results(
            query_ipa: str,
            results: list[SearchResult],
            lexicon_map: dict[str, object],
            matrix: object,
            language: str = "ancient_greek",
        ) -> list[SearchResult]:
            annotated = list(results)
            annotated[0].applied_rules = ["RULE-001"]
            return annotated

        monkeypatch.setattr(search_module, "_annotate_search_results", fake_annotate_results)

        lexicon = [
            {"id": "L1", "headword": "rule-supported-weak-overlap", "ipa": "p i", "dialect": "attic"},
            {"id": "L2", "headword": "full-prefix-match", "ipa": "p a", "dialect": "attic"},
        ]

        results = search("ζηταω-", lexicon, matrix={}, max_results=2, index={})

        assert [result.entry_id for result in results] == ["L2", "L1"]
        assert results[1].applied_rules == ["RULE-001"]

    def test_partial_infix_query_supports_zero_length_gap_and_rejects_wrong_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Infix partial-form matching should allow zero-gap but preserve fragment order."""
        monkeypatch.setattr(
            search_module,
            "to_ipa",
            lambda query, dialect="attic": {"ac": "a c", "a": "a", "c": "c"}.get(query, query),
        )
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: ["L1", "L2", "L3", "L4"])
        monkeypatch.setattr(
            search_module,
            "_score_stage",
            lambda query_ipa, candidates, lexicon_map, matrix: [
                SearchResult(
                    lemma="zero-gap",
                    confidence=0.80,
                    dialect_attribution="lemma dialect: attic",
                    entry_id="L1",
                ),
                SearchResult(
                    lemma="with-gap",
                    confidence=0.78,
                    dialect_attribution="lemma dialect: attic",
                    entry_id="L2",
                ),
                SearchResult(
                    lemma="wrong-order",
                    confidence=0.99,
                    dialect_attribution="lemma dialect: attic",
                    entry_id="L3",
                ),
                SearchResult(
                    lemma="right-only-fragment",
                    confidence=0.72,
                    dialect_attribution="lemma dialect: attic",
                    entry_id="L4",
                ),
            ],
        )
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda query_ipa, results, lexicon_map, matrix, language="ancient_greek": results,
        )

        lexicon = [
            {"id": "L1", "headword": "zero-gap", "ipa": "a c", "dialect": "attic"},
            {"id": "L2", "headword": "with-gap", "ipa": "a b c", "dialect": "attic"},
            {"id": "L3", "headword": "wrong-order", "ipa": "c a", "dialect": "attic"},
            {"id": "L4", "headword": "right-only-fragment", "ipa": "x c", "dialect": "attic"},
        ]

        results = search("a*c", lexicon, matrix={}, max_results=3, index={})

        assert [result.lemma for result in results] == ["zero-gap", "with-gap", "right-only-fragment"]

    def test_search_rejects_invalid_partial_query_syntax(self) -> None:
        with pytest.raises(ValueError, match="wildcard marker|non-empty string"):
            search("a*b*c", [], matrix={}, index={}, unigram_index={})

    @pytest.mark.parametrize("query", ["", "*", "-", "-*", "a*b*c"])
    def test_classify_query_mode_rejects_invalid_partial_query_syntax(
        self, query: str
    ) -> None:
        with pytest.raises(ValueError, match="wildcard marker|non-empty string"):
            search_module.classify_query_mode(query)
