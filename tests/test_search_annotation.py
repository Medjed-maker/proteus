"""Tests for annotation windows, caching, and extend_stage in phonology.search."""
# Several tests define monkeypatch stubs whose parameter names must mirror the
# real signatures of the functions they replace (e.g. `_score_stage(query_ipa,
# candidates, lexicon_map, matrix)`). Renaming the unused params to `_prefix`
# would either drift from the real signature or force kwargs-only call sites;
# both are worse than a file-level suppression. Keep this scoped to
# reportUnusedVariable only — other checks stay on.
# pyright: reportUnusedVariable=false

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pytest

from phonology import search as search_module
from phonology.search import (
    LexiconRecord,
    SearchResult,
    extend_stage,
    search,
)
from phonology.search import _scoring as scoring_module


class TestSearchAnnotation:
    """Tests for annotation windows, caching, and extend_stage behavior."""

    def test_search_skips_to_ipa_when_query_ipa_provided(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When query_ipa is pre-computed, search() should not call to_ipa for the main query."""
        to_ipa_calls: list[str] = []

        def tracking_to_ipa(query: str, dialect: str = "attic") -> str:
            to_ipa_calls.append(query)
            return query

        monkeypatch.setattr(search_module, "to_ipa", tracking_to_ipa)
        monkeypatch.setattr(search_module, "load_rules", lambda _path: {})
        lexicon = [{"id": "L1", "headword": "target", "ipa": "loɡos", "dialect": "attic"}]

        search("λόγος", lexicon, matrix={}, max_results=1, query_ipa="loɡos")

        assert "λόγος" not in to_ipa_calls

    def test_short_query_unigram_fallback_uses_only_unigram_hits_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Short-query k=1 fallback should rank only unigram hits by default."""
        captured: dict[str, object] = {}

        def fake_score_stage(
            query_ipa: str,
            candidates: Iterable[str],
            lexicon_map: dict[str, object],
            matrix: object,
        ) -> list[SearchResult]:
            captured["candidate_ids"] = list(candidates)
            return []

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "poi")
        seed_calls: list[int] = []

        def fake_seed_stage(query_ipa: str, index: object, k: int = 2, **_kwargs: Any) -> list[str]:
            seed_calls.append(k)
            if k == 2:
                return []
            return ["L05", "L02", "L29"]

        monkeypatch.setattr(search_module, "seed_stage", fake_seed_stage)
        monkeypatch.setattr(search_module, "_score_stage", fake_score_stage)
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda query_ipa, results, lexicon_map, matrix, language="ancient_greek", **_kwargs: results,
        )
        monkeypatch.setattr(search_module, "filter_stage", lambda results, max_results: results)

        lexicon = [
            {
                "id": f"L{index:02d}",
                "headword": f"noise-{index:02d}",
                "ipa": "pa",
                "dialect": "attic",
            }
            for index in range(30)
        ]
        search("ποι", lexicon, matrix={}, max_results=1, index={}, unigram_index={})

        assert seed_calls == [2, 1]
        assert captured["candidate_ids"] == ["L02", "L05", "L29"]

    def test_fullform_search_annotates_only_ranked_results(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Full-form search should defer explanation work until after top-N ranking."""
        captured: dict[str, object] = {}

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "poi")
        monkeypatch.setattr(
            search_module,
            "seed_stage",
            lambda *_args, **_kwargs: [f"L{index:02d}" for index in range(30)],
        )
        monkeypatch.setattr(
            search_module,
            "_score_stage",
            lambda query_ipa, candidates, lexicon_map, matrix: [
                SearchResult(
                    lemma=f"lemma-{index:02d}",
                    confidence=1.0 - (index * 0.01),
                    dialect_attribution="lemma dialect: attic",
                    entry_id=f"L{index:02d}",
                )
                for index, _candidate_id in enumerate(candidates)
            ],
        )

        def fake_annotate_results(
            query_ipa: str,
            results: list[SearchResult],
            lexicon_map: dict[str, object],
            matrix: object,
            language: str = "ancient_greek", **_kwargs: Any,
        ) -> list[SearchResult]:
            captured["annotated_count"] = len(results)
            return results

        monkeypatch.setattr(search_module, "_annotate_search_results", fake_annotate_results)

        lexicon = [
            {"id": f"L{index:02d}", "headword": f"lemma-{index:02d}", "ipa": "poi", "dialect": "attic"}
            for index in range(30)
        ]

        search("λόγος", lexicon, matrix={}, max_results=5)

        assert captured["annotated_count"] == 5

    def test_short_query_search_bounds_annotation_after_full_fallback_scoring(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Short-query annotation should be capped even when fallback scoring scans the full lexicon."""
        captured: dict[str, object] = {}

        def fake_seed_stage(query_ipa: str, index: object, k: int = 2, **_kwargs: Any) -> list[str]:
            return []

        def fake_score_stage(
            query_ipa: str,
            candidates: Iterable[str],
            lexicon_map: dict[str, object],
            matrix: object,
        ) -> list[SearchResult]:
            candidate_ids = list(candidates)
            captured["candidate_count"] = len(candidate_ids)
            return [
                SearchResult(
                    lemma=f"lemma-{index:03d}",
                    confidence=0.9,
                    dialect_attribution="lemma dialect: attic",
                    entry_id=candidate_id,
                )
                for index, candidate_id in enumerate(candidate_ids)
            ]

        def fake_annotate_results(
            query_ipa: str,
            results: list[SearchResult],
            lexicon_map: dict[str, object],
            matrix: object,
            language: str = "ancient_greek", **_kwargs: Any,
        ) -> list[SearchResult]:
            captured["annotated_count"] = len(results)
            return list(results)

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "poi")
        monkeypatch.setattr(search_module, "seed_stage", fake_seed_stage)
        monkeypatch.setattr(search_module, "_score_stage", fake_score_stage)
        monkeypatch.setattr(search_module, "_annotate_search_results", fake_annotate_results)

        lexicon = [
            {
                "id": f"L{index:03d}",
                "headword": f"lemma-{index:03d}",
                "ipa": "pa",
                "dialect": "attic",
            }
            for index in range(150)
        ]

        search("ποι", lexicon, matrix={}, max_results=5, index={}, unigram_index={})

        assert captured["candidate_count"] == len(lexicon)
        assert captured["annotated_count"] == search_module._annotation_candidate_limit(5)

    def test_short_query_annotation_window_keeps_rule_supported_tail_candidate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Short-query annotation cap should still leave room for late rule-supported hits."""
        annotation_limit = search_module._annotation_candidate_limit(5)

        def fake_seed_stage(query_ipa: str, index: object, k: int = 2, **_kwargs: Any) -> list[str]:
            return []

        def fake_score_stage(
            query_ipa: str,
            candidates: Iterable[str],
            lexicon_map: dict[str, object],
            matrix: object,
        ) -> list[SearchResult]:
            candidate_ids = list(candidates)
            return [
                SearchResult(
                    lemma=f"lemma-{index:03d}",
                    confidence=0.2,
                    dialect_attribution="lemma dialect: attic",
                    entry_id=candidate_id,
                )
                for index, candidate_id in enumerate(candidate_ids)
            ]

        def fake_annotate_results(
            query_ipa: str,
            results: list[SearchResult],
            lexicon_map: dict[str, object],
            matrix: object,
            language: str = "ancient_greek", **_kwargs: Any,
        ) -> list[SearchResult]:
            annotated = list(results)
            if annotated and annotated[-1].entry_id == f"L{annotation_limit - 1:03d}":
                annotated[-1].applied_rules = ["RULE-TAIL"]
            return annotated

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "q")
        monkeypatch.setattr(search_module, "seed_stage", fake_seed_stage)
        monkeypatch.setattr(search_module, "_score_stage", fake_score_stage)
        monkeypatch.setattr(search_module, "_annotate_search_results", fake_annotate_results)

        lexicon = [
            {
                "id": f"L{index:03d}",
                "headword": f"lemma-{index:03d}",
                "ipa": "t",
                "dialect": "attic",
            }
            for index in range(150)
        ]

        results = search(
            "q",
            lexicon,
            matrix={},
            max_results=5,
            index={},
            unigram_index={},
        )

        assert [result.entry_id for result in results] == [f"L{annotation_limit - 1:03d}"]
        assert results[0].applied_rules == ["RULE-TAIL"]

    def test_short_query_annotation_does_not_continue_past_bounded_window(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Short-query annotation should not continue past the bounded candidate window."""
        annotation_limit = search_module._annotation_candidate_limit(5)
        batch_sizes: list[int] = []
        target_id = f"L{annotation_limit:03d}"

        def fake_score_stage(
            query_ipa: str,
            candidates: Iterable[str],
            lexicon_map: dict[str, object],
            matrix: object,
        ) -> list[SearchResult]:
            return [
                SearchResult(
                    lemma=f"lemma-{index:03d}",
                    confidence=0.2,
                    dialect_attribution="lemma dialect: attic",
                    entry_id=candidate_id,
                )
                for index, candidate_id in enumerate(candidates)
            ]

        def fake_annotate_results(
            query_ipa: str,
            results: list[SearchResult],
            lexicon_map: dict[str, object],
            matrix: object,
            language: str = "ancient_greek", **_kwargs: Any,
        ) -> list[SearchResult]:
            batch_sizes.append(len(results))
            annotated = list(results)
            for result in annotated:
                if result.entry_id == target_id:
                    result.applied_rules = ["RULE-NEXT-BATCH"]
            return annotated

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "q")
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: [])
        monkeypatch.setattr(search_module, "_score_stage", fake_score_stage)
        monkeypatch.setattr(search_module, "_annotate_search_results", fake_annotate_results)

        lexicon = [
            {
                "id": f"L{index:03d}",
                "headword": f"lemma-{index:03d}",
                "ipa": "t",
                "dialect": "attic",
            }
            for index in range(annotation_limit + 1)
        ]

        results = search("q", lexicon, matrix={}, max_results=5, index={}, unigram_index={})

        assert batch_sizes == [annotation_limit]
        assert results == []

    def test_short_query_annotation_window_keeps_late_exact_match(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Short-query annotation selection should rescue exact matches outside raw top-N."""
        exact_id = "L120"

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "q")
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: [])
        monkeypatch.setattr(
            search_module,
            "_rank_by_token_count_proximity",
            lambda query_ipa, lexicon_map, max_candidates=None, query_token_count=None: [
                f"L{index:03d}" for index in range(150)
            ],
        )
        monkeypatch.setattr(
            search_module,
            "_score_stage",
            lambda query_ipa, candidates, lexicon_map, matrix: [
                SearchResult(
                    lemma=f"lemma-{index:03d}",
                    confidence=1.0 - (index * 0.001),
                    dialect_attribution="lemma dialect: attic",
                    entry_id=candidate_id,
                )
                for index, candidate_id in enumerate(candidates)
            ],
        )
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda query_ipa, results, lexicon_map, matrix, language="ancient_greek", **_kwargs: list(results),
        )

        lexicon = [
            {
                "id": f"L{index:03d}",
                "headword": f"lemma-{index:03d}",
                "ipa": "q" if index == 120 else "t",
                "dialect": "attic",
            }
            for index in range(150)
        ]

        results = search("q", lexicon, matrix={}, max_results=1, index={}, unigram_index={})

        assert [result.entry_id for result in results] == [exact_id]

    def test_short_query_annotation_ranking_prioritizes_exact_and_confident_candidates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Short-query pre-annotation ordering should pull strong candidates into the first batch."""
        annotation_limit = search_module._annotation_candidate_limit(5)
        exact_id = "L120"
        confident_id = "L121"
        captured_batches: list[list[str]] = []

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "q")
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: [])
        monkeypatch.setattr(
            search_module,
            "_rank_by_token_count_proximity",
            lambda query_ipa, lexicon_map, max_candidates=None, query_token_count=None: [
                f"L{index:03d}" for index in range(150)
            ],
        )

        threshold = search_module._SHORT_QUERY_CONFIDENCE_THRESHOLD

        def fake_score_stage(
            query_ipa: str,
            candidates: Iterable[str],
            lexicon_map: dict[str, object],
            matrix: object,
        ) -> list[SearchResult]:
            scored: list[SearchResult] = []
            for index, candidate_id in enumerate(candidates):
                confidence = threshold - 0.20
                if candidate_id == confident_id:
                    confidence = threshold + 0.05
                scored.append(
                    SearchResult(
                        lemma=f"lemma-{index:03d}",
                        confidence=confidence,
                        dialect_attribution="lemma dialect: attic",
                        entry_id=candidate_id,
                    )
                )
            return scored

        def fake_annotate_results(
            query_ipa: str,
            results: list[SearchResult],
            lexicon_map: dict[str, object],
            matrix: object,
            language: str = "ancient_greek", **_kwargs: Any,
        ) -> list[SearchResult]:
            captured_batches.append([str(result.entry_id) for result in results])
            return list(results)

        monkeypatch.setattr(search_module, "_score_stage", fake_score_stage)
        monkeypatch.setattr(search_module, "_annotate_search_results", fake_annotate_results)

        lexicon = [
            {
                "id": f"L{index:03d}",
                "headword": f"lemma-{index:03d}",
                "ipa": "q" if index == 120 else "t",
                "dialect": "attic",
            }
            for index in range(150)
        ]

        search("q", lexicon, matrix={}, max_results=5, index={}, unigram_index={})

        assert captured_batches
        assert exact_id in captured_batches[0]
        assert confident_id in captured_batches[0]
        assert len(captured_batches[0]) == annotation_limit

    def test_short_query_exploratory_reserve_keeps_rule_supported_candidate_with_many_strong_hits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exploratory reserve should keep a low-confidence rule hit reachable within batch caps."""
        target_id = "L320"
        threshold = search_module._SHORT_QUERY_CONFIDENCE_THRESHOLD

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "q")
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: [])
        monkeypatch.setattr(
            search_module,
            "_rank_by_token_count_proximity",
            lambda query_ipa, lexicon_map, max_candidates=None, query_token_count=None: [
                f"L{index:03d}" for index in range(350)
            ],
        )

        def fake_score_stage(
            query_ipa: str,
            candidates: Iterable[str],
            lexicon_map: dict[str, object],
            matrix: object,
        ) -> list[SearchResult]:
            scored: list[SearchResult] = []
            target_num = int(target_id[1:])
            for candidate_id in candidates:
                candidate_num = int(candidate_id[1:])
                confidence = threshold + 0.05 if candidate_num < target_num else threshold - 0.20
                scored.append(
                    SearchResult(
                        lemma=str(candidate_id),
                        confidence=confidence,
                        dialect_attribution="lemma dialect: attic",
                        entry_id=str(candidate_id),
                    )
                )
            return scored

        def fake_annotate_results(
            query_ipa: str,
            results: list[SearchResult],
            lexicon_map: dict[str, object],
            matrix: object,
            language: str = "ancient_greek", **_kwargs: Any,
        ) -> list[SearchResult]:
            annotated = list(results)
            for result in annotated:
                if result.entry_id == target_id:
                    result.applied_rules = ["RULE-RESCUE"]
            return annotated

        monkeypatch.setattr(search_module, "_score_stage", fake_score_stage)
        monkeypatch.setattr(search_module, "_annotate_search_results", fake_annotate_results)

        lexicon = [
            {
                "id": f"L{index:03d}",
                "headword": f"lemma-{index:03d}",
                "ipa": "q" if index == 320 else "t",
                "dialect": "attic",
            }
            for index in range(350)
        ]

        results = search("q", lexicon, matrix={}, max_results=1, index={}, unigram_index={})

        assert [result.entry_id for result in results] == [target_id]
        assert results[0].applied_rules == ["RULE-RESCUE"]

    def test_short_query_annotation_window_keeps_all_primary_hits_when_they_fill_the_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Short-query reserve should not displace in-budget exact or confident candidates."""
        annotation_limit = search_module._annotation_candidate_limit(5)
        captured_batches: list[list[str]] = []
        threshold = search_module._SHORT_QUERY_CONFIDENCE_THRESHOLD

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "q")
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: [])
        monkeypatch.setattr(
            search_module,
            "_rank_by_token_count_proximity",
            lambda query_ipa, lexicon_map, max_candidates=None, query_token_count=None: [
                f"L{index:03d}" for index in range(150)
            ],
        )

        def fake_score_stage(
            query_ipa: str,
            candidates: Iterable[str],
            lexicon_map: dict[str, object],
            matrix: object,
        ) -> list[SearchResult]:
            scored: list[SearchResult] = []
            for index, candidate_id in enumerate(candidates):
                confidence = threshold + 0.05 if index < annotation_limit else threshold - 0.20
                scored.append(
                    SearchResult(
                        lemma=f"lemma-{index:03d}",
                        confidence=confidence,
                        dialect_attribution="lemma dialect: attic",
                        entry_id=candidate_id,
                    )
                )
            return scored

        def fake_annotate_results(
            query_ipa: str,
            results: list[SearchResult],
            lexicon_map: dict[str, object],
            matrix: object,
            language: str = "ancient_greek", **_kwargs: Any,
        ) -> list[SearchResult]:
            captured_batches.append([str(result.entry_id) for result in results])
            return list(results)

        monkeypatch.setattr(search_module, "_score_stage", fake_score_stage)
        monkeypatch.setattr(search_module, "_annotate_search_results", fake_annotate_results)

        lexicon = [
            {
                "id": f"L{index:03d}",
                "headword": f"lemma-{index:03d}",
                "ipa": "q" if index < annotation_limit else "t",
                "dialect": "attic",
            }
            for index in range(150)
        ]

        results = search("q", lexicon, matrix={}, max_results=5, index={}, unigram_index={})

        expected_primary_ids = [f"L{index:03d}" for index in range(annotation_limit)]
        assert captured_batches == [expected_primary_ids]
        assert [result.entry_id for result in results] == expected_primary_ids[:5]

    def test_partial_query_annotates_candidates_beyond_previous_window(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Partial-form results should survive even when support is found after rank 10."""
        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "pa")
        monkeypatch.setattr(
            search_module,
            "seed_stage",
            lambda *_args, **_kwargs: [f"L{index:02d}" for index in range(11)],
        )
        monkeypatch.setattr(
            search_module,
            "_score_stage",
            lambda query_ipa, candidates, lexicon_map, matrix: [
                SearchResult(
                    lemma=f"lemma-{index:02d}",
                    confidence=1.0 - (index * 0.01),
                    dialect_attribution="lemma dialect: attic",
                    entry_id=candidate_id,
                )
                for index, candidate_id in enumerate(candidates)
            ],
        )

        def fake_annotate_results(
            query_ipa: str,
            results: list[SearchResult],
            lexicon_map: dict[str, object],
            matrix: object,
            language: str = "ancient_greek", **_kwargs: Any,
        ) -> list[SearchResult]:
            annotated = list(results)
            annotated[-1].applied_rules = ["RULE-010"]
            return annotated

        monkeypatch.setattr(search_module, "_annotate_search_results", fake_annotate_results)

        lexicon = [
            {"id": f"L{index:02d}", "headword": f"lemma-{index:02d}", "ipa": "pai", "dialect": "attic"}
            for index in range(11)
        ]

        results = search("ζηταω-", lexicon, matrix={}, max_results=3, index={})

        assert results[0].entry_id == "L10"
        assert results[0].applied_rules == ["RULE-010"]

    def test_partial_query_search_bounds_annotation_after_scoring(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Partial-form annotation should be capped even when scored candidates exceed the visible results."""
        captured: dict[str, object] = {}

        def fake_score_stage(
            query_ipa: str,
            candidates: Iterable[str],
            lexicon_map: dict[str, object],
            matrix: object,
        ) -> list[SearchResult]:
            candidate_ids = list(candidates)
            captured["candidate_count"] = len(candidate_ids)
            return [
                SearchResult(
                    lemma=f"lemma-{index:03d}",
                    confidence=1.0 - (index * 0.001),
                    dialect_attribution="lemma dialect: attic",
                    entry_id=candidate_id,
                )
                for index, candidate_id in enumerate(candidate_ids)
            ]

        def fake_annotate_results(
            query_ipa: str,
            results: list[SearchResult],
            lexicon_map: dict[str, object],
            matrix: object,
            language: str = "ancient_greek", **_kwargs: Any,
        ) -> list[SearchResult]:
            captured["annotated_count"] = len(results)
            return list(results)

        def fake_seed_stage(*args: object, **kwargs: object) -> list[str]:
            return [f"L{index:03d}" for index in range(150)]

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": query)
        monkeypatch.setattr(search_module, "seed_stage", fake_seed_stage)
        monkeypatch.setattr(search_module, "_score_stage", fake_score_stage)
        monkeypatch.setattr(search_module, "_annotate_search_results", fake_annotate_results)

        lexicon = [
            {
                "id": f"L{index:03d}",
                "headword": f"lemma-{index:03d}",
                "ipa": "a x c" if index < 120 else "b x d",
                "dialect": "attic",
            }
            for index in range(150)
        ]

        search("a*c", lexicon, matrix={}, max_results=5, index={})

        assert captured["candidate_count"] == search_module._annotation_candidate_limit(5)
        assert captured["annotated_count"] == search_module._annotation_candidate_limit(5)

    def test_partial_query_deduplicates_after_rule_supported_reordering(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Partial-form dedup should keep the post-filter top candidate per headword."""
        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "p a")
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: ["L1", "L2"])
        monkeypatch.setattr(
            search_module,
            "_score_stage",
            lambda query_ipa, candidates, lexicon_map, matrix: [
                SearchResult(
                    lemma="same",
                    confidence=0.60,
                    dialect_attribution="lemma dialect: attic",
                    entry_id="L1",
                ),
                SearchResult(
                    lemma="same",
                    confidence=0.40,
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
            language: str = "ancient_greek", **_kwargs: Any,
        ) -> list[SearchResult]:
            annotated = list(results)
            annotated[1].applied_rules = ["RULE-002"]
            return annotated

        monkeypatch.setattr(search_module, "_annotate_search_results", fake_annotate_results)

        lexicon = [
            {"id": "L1", "headword": "same", "ipa": "p a", "dialect": "attic"},
            {"id": "L2", "headword": "same", "ipa": "p a i", "dialect": "attic"},
        ]

        results = search("ζηταω-", lexicon, matrix={}, max_results=1, index={})

        assert len(results) == 1
        assert results[0].entry_id == "L2"
        assert results[0].applied_rules == ["RULE-002"]

    def test_partial_query_prefers_full_match_before_rule_supported_overlap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Partial-form ordering should keep full matches ahead of rule-supported overlap hits."""
        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "p a")
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: ["L1", "L2"])
        monkeypatch.setattr(
            search_module,
            "_score_stage",
            lambda query_ipa, candidates, lexicon_map, matrix: [
                SearchResult(
                    lemma="full-match",
                    confidence=0.50,
                    dialect_attribution="lemma dialect: attic",
                    entry_id="L1",
                ),
                SearchResult(
                    lemma="rule-supported-overlap",
                    confidence=0.95,
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
            language: str = "ancient_greek", **_kwargs: Any,
        ) -> list[SearchResult]:
            annotated = list(results)
            annotated[1].applied_rules = ["RULE-002"]
            return annotated

        monkeypatch.setattr(search_module, "_annotate_search_results", fake_annotate_results)

        lexicon = [
            {"id": "L1", "headword": "full-match", "ipa": "p a", "dialect": "attic"},
            {"id": "L2", "headword": "rule-supported-overlap", "ipa": "p i", "dialect": "attic"},
        ]

        results = search("ζηταω-", lexicon, matrix={}, max_results=2, index={})

        assert [result.entry_id for result in results] == ["L1", "L2"]

    def test_partial_query_annotation_selection_prioritizes_positive_overlap_candidates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Partial-form pre-annotation selection should prefer overlap candidates over zero-overlap noise."""
        captured: dict[str, object] = {}
        target_id = "L095"

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "a c")
        monkeypatch.setattr(
            search_module,
            "seed_stage",
            lambda *_args, **_kwargs: [f"L{index:03d}" for index in range(150)],
        )
        monkeypatch.setattr(
            search_module,
            "_score_stage",
            lambda query_ipa, candidates, lexicon_map, matrix: [
                SearchResult(
                    lemma=f"lemma-{index:03d}",
                    confidence=0.95,
                    dialect_attribution="lemma dialect: attic",
                    entry_id=candidate_id,
                )
                for index, candidate_id in enumerate(candidates)
            ],
        )

        def fake_annotate_results(
            query_ipa: str,
            results: list[SearchResult],
            lexicon_map: dict[str, object],
            matrix: object,
            language: str = "ancient_greek", **_kwargs: Any,
        ) -> list[SearchResult]:
            captured["annotated_ids"] = [str(result.entry_id) for result in results]
            return list(results)

        monkeypatch.setattr(search_module, "_annotate_search_results", fake_annotate_results)

        lexicon = [
            {
                "id": f"L{index:03d}",
                "headword": f"lemma-{index:03d}",
                "ipa": "a x c" if index == 95 else "t t",
                "dialect": "attic",
            }
            for index in range(150)
        ]

        search("a*c", lexicon, matrix={}, max_results=5, index={}, unigram_index={})

        assert target_id in captured["annotated_ids"]

    def test_partial_query_exploratory_reserve_keeps_zero_overlap_rule_supported_candidate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Partial-form exploratory reserve should still annotate a low-confidence zero-overlap candidate."""
        target_id = "L095"
        captured: dict[str, object] = {}

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "a c")
        monkeypatch.setattr(
            search_module,
            "seed_stage",
            lambda *_args, **_kwargs: [f"L{index:03d}" for index in range(150)],
        )
        monkeypatch.setattr(
            search_module,
            "_score_stage",
            lambda query_ipa, candidates, lexicon_map, matrix: [
                SearchResult(
                    lemma=f"lemma-{index:03d}",
                    confidence=0.95 if candidate_id != target_id else 0.10,
                    dialect_attribution="lemma dialect: attic",
                    entry_id=candidate_id,
                )
                for index, candidate_id in enumerate(candidates)
            ],
        )

        def fake_annotate_results(
            query_ipa: str,
            results: list[SearchResult],
            lexicon_map: dict[str, object],
            matrix: object,
            language: str = "ancient_greek", **_kwargs: Any,
        ) -> list[SearchResult]:
            captured["annotated_ids"] = [str(result.entry_id) for result in results]
            annotated = list(results)
            for result in annotated:
                if result.entry_id == target_id:
                    result.applied_rules = ["RULE-ZERO-OVERLAP"]
            return annotated

        monkeypatch.setattr(search_module, "_annotate_search_results", fake_annotate_results)

        lexicon = [
            {
                "id": f"L{index:03d}",
                "headword": f"lemma-{index:03d}",
                "ipa": "z z" if index == 95 else "a x c",
                "dialect": "attic",
            }
            for index in range(150)
        ]

        results = search("a*c", lexicon, matrix={}, max_results=1, index={}, unigram_index={})

        assert target_id in captured["annotated_ids"]
        assert results == []

    def test_partial_query_annotation_window_keeps_all_primary_hits_when_they_fill_the_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Partial-form reserve should not displace overlap-matching primary candidates."""
        max_results = 5
        annotation_limit = search_module._annotation_candidate_limit(max_results)
        captured: dict[str, object] = {}

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": query.replace("*", ""))
        monkeypatch.setattr(
            search_module,
            "seed_stage",
            lambda *_args, **_kwargs: [f"L{index:03d}" for index in range(150)],
        )
        monkeypatch.setattr(
            search_module,
            "_score_stage",
            lambda query_ipa, candidates, lexicon_map, matrix: [
                SearchResult(
                    lemma=f"lemma-{index:03d}",
                    confidence=0.95 if index < annotation_limit else 0.10,
                    dialect_attribution="lemma dialect: attic",
                    entry_id=candidate_id,
                )
                for index, candidate_id in enumerate(candidates)
            ],
        )

        def fake_annotate_results(
            query_ipa: str,
            results: list[SearchResult],
            lexicon_map: dict[str, object],
            matrix: object,
            language: str = "ancient_greek", **_kwargs: Any,
        ) -> list[SearchResult]:
            captured["annotated_ids"] = [str(result.entry_id) for result in results]
            return list(results)

        monkeypatch.setattr(search_module, "_annotate_search_results", fake_annotate_results)

        lexicon = [
            {
                "id": f"L{index:03d}",
                "headword": f"lemma-{index:03d}",
                "ipa": "a x c" if index < annotation_limit else "z z",
                "dialect": "attic",
            }
            for index in range(150)
        ]

        results = search(
            "a*c",
            lexicon,
            matrix={},
            max_results=max_results,
            index={},
            unigram_index={},
        )

        expected_primary_ids = [f"L{index:03d}" for index in range(annotation_limit)]
        assert captured["annotated_ids"] == expected_primary_ids
        assert [result.entry_id for result in results] == expected_primary_ids[:max_results]

    def test_short_query_annotation_call_counts_stay_bounded_by_annotation_window(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Short-query annotation should bound tokenize/alignment/explainer call counts."""
        annotation_limit = search_module._annotation_candidate_limit(5)
        counts = {
            "tokenize": 0,
            "alignment": 0,
            "explain": 0,
        }

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "q")
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: [])
        monkeypatch.setattr(
            search_module,
            "_rank_by_token_count_proximity",
            lambda query_ipa, lexicon_map, max_candidates=None, query_token_count=None: [
                f"L{index:03d}" for index in range(150)
            ],
        )
        monkeypatch.setattr(search_module, "get_rules_registry", lambda language="ancient_greek": {})
        monkeypatch.setattr(search_module, "tokenize_rules_for_matching", lambda rules: [])

        original_tokenize = search_module.tokenize_ipa

        def tracking_tokenize(ipa_text: str) -> list[str]:
            counts["tokenize"] += 1
            return original_tokenize(ipa_text)

        def fake_alignment(
            query_tokens: list[str],
            lemma_tokens: list[str],
            matrix: object,
        ) -> tuple[float, list[str | None], list[str | None]]:
            counts["alignment"] += 1
            return 2.0, list(query_tokens), list(lemma_tokens)

        def fake_explain(
            *,
            query_tokens: list[str],
            lemma_tokens: list[str],
            alignment: object,
            tokenized_rules: tuple[object, ...],
            lemma_metadata: dict[str, object],
        ) -> list[object]:
            counts["explain"] += 1
            return []

        monkeypatch.setattr("phonology.search._tokenization.tokenize_ipa", tracking_tokenize)
        monkeypatch.setattr(search_module, "explain_with_tokenized_rules", fake_explain)
        monkeypatch.setattr(scoring_module, "_smith_waterman_alignment", fake_alignment)

        lexicon = [
            {
                "id": f"L{index:03d}",
                "headword": f"lemma-{index:03d}",
                "ipa": "q" if index < 5 else "t",
                "dialect": "attic",
            }
            for index in range(150)
        ]

        search("q", lexicon, matrix={}, max_results=5, index={}, unigram_index={})

        extra_tokenize_calls = 3  # Query tokenization for selection, scoring, and annotation.
        assert counts["alignment"] == len(lexicon)
        assert counts["explain"] == annotation_limit
        assert counts["tokenize"] == len(lexicon) + extra_tokenize_calls

    def test_short_query_deduplicates_after_quality_filter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Short-query dedup should happen after weak same-headword hits are dropped."""
        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "pa")
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: ["L1", "L2"])
        monkeypatch.setattr(
            search_module,
            "_score_stage",
            lambda query_ipa, candidates, lexicon_map, matrix: [
                SearchResult(
                    lemma="same",
                    confidence=0.60,
                    dialect_attribution="lemma dialect: attic",
                    entry_id="L1",
                ),
                SearchResult(
                    lemma="same",
                    confidence=0.40,
                    dialect_attribution="lemma dialect: attic",
                    entry_id="L2",
                ),
            ],
        )
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda query_ipa, results, lexicon_map, matrix, language="ancient_greek", **_kwargs: list(results),
        )

        lexicon = [
            {"id": "L1", "headword": "same", "ipa": "pi", "dialect": "attic"},
            {"id": "L2", "headword": "same", "ipa": "pa", "dialect": "attic"},
        ]

        results = search("νυν", lexicon, matrix={}, max_results=1, index={})

        assert len(results) == 1
        assert results[0].entry_id == "L2"

    def test_extend_stage_tokenizes_rules_once_for_all_candidates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Full extend_stage should reuse one tokenized-rules batch for all candidates."""
        tokenize_calls: list[int] = []

        def mock_tokenize_rules(rules: list[dict[str, object]]) -> list[object]:
            """Record call count and return empty list."""
            tokenize_calls.append(len(rules))
            return []

        monkeypatch.setattr(search_module, "get_rules_registry", lambda language="ancient_greek": {})
        monkeypatch.setattr(
            search_module,
            "tokenize_rules_for_matching",
            mock_tokenize_rules,
        )
        lexicon_map = {
            "L1": LexiconRecord(entry={"id": "L1", "headword": "alpha", "ipa": "pa", "dialect": "attic"}, token_count=2, ipa_tokens=("p", "a")),
            "L2": LexiconRecord(entry={"id": "L2", "headword": "beta", "ipa": "pi", "dialect": "attic"}, token_count=2, ipa_tokens=("p", "i")),
            "L3": LexiconRecord(entry={"id": "L3", "headword": "gamma", "ipa": "po", "dialect": "attic"}, token_count=2, ipa_tokens=("p", "o")),
        }

        extend_stage("poi", ["L1", "L2", "L3"], lexicon_map, matrix={})

        assert tokenize_calls == [0]

    def test_single_consonant_query_keeps_alignment_and_rule_applications(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Single-consonant fallback should still annotate the surviving top hit."""
        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "poi")
        monkeypatch.setattr(search_module, "load_rules", lambda _path: {})
        lexicon = [
            {"id": "L1", "headword": "ποι", "ipa": "poi", "dialect": "attic"},
            {"id": "L2", "headword": "πατρι", "ipa": "patri", "dialect": "attic"},
        ]

        results = search("ποι", lexicon, matrix={}, max_results=1)

        assert results[0].lemma == "ποι"
        assert results[0].alignment_visualization

    def test_extend_stage_reuses_cached_alignment_during_annotation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """extend_stage should not recompute alignment during annotation."""
        alignment_calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

        def fake_alignment(
            query_tokens: list[str],
            lemma_tokens: list[str],
            matrix: object,
        ) -> tuple[float, list[str | None], list[str | None]]:
            alignment_calls.append((tuple(query_tokens), tuple(lemma_tokens)))
            return 2.0, list(query_tokens), list(lemma_tokens)

        monkeypatch.setattr(scoring_module, "_smith_waterman_alignment", fake_alignment)
        monkeypatch.setattr(search_module, "get_rules_registry", lambda language="ancient_greek": {})
        monkeypatch.setattr(search_module, "tokenize_rules_for_matching", lambda rules: [])

        lexicon_map = {
            "L1": LexiconRecord(
                entry={"id": "L1", "headword": "alpha", "ipa": "pa", "dialect": "attic"},
                token_count=2,
                ipa_tokens=("p", "a"),
            ),
            "L2": LexiconRecord(
                entry={"id": "L2", "headword": "beta", "ipa": "pi", "dialect": "attic"},
                token_count=2,
                ipa_tokens=("p", "i"),
            ),
        }

        extend_stage("pa", ["L1", "L2"], lexicon_map, matrix={})

        assert alignment_calls == [(("p", "a"), ("p", "a")), (("p", "a"), ("p", "i"))]
