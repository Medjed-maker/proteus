"""Tests for short-query mode filtering and dedup in phonology.search."""
# Several tests define monkeypatch stubs whose parameter names must mirror the
# real signatures of the functions they replace (e.g. `_score_stage(query_ipa,
# candidates, lexicon_map, matrix)`). Renaming the unused params to `_prefix`
# would either drift from the real signature or force kwargs-only call sites;
# both are worse than a file-level suppression. Keep this scoped to
# reportUnusedVariable only — other checks stay on.
# pyright: reportUnusedVariable=false

from __future__ import annotations

import pytest

from phonology import search as search_module
from phonology.search import (
    SearchResult,
    search,
)
from phonology.search._constants import _SHORT_QUERY_MAX_ANNOTATION_BATCHES


class TestSearchShortQuery:
    """Tests for short-query mode filtering and dedup."""

    def test_short_query_low_confidence_distance_only_candidates_are_dropped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Short-query post-filtering should remove weak distance-only hits."""
        threshold = search_module._SHORT_QUERY_CONFIDENCE_THRESHOLD
        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "pa")
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: ["L1", "L2"])
        monkeypatch.setattr(
            search_module,
            "_score_stage",
            lambda query_ipa, candidates, lexicon_map, matrix: [
                SearchResult(
                    lemma="drop-me",
                    confidence=threshold - 0.01,
                    dialect_attribution="lemma dialect: attic",
                    entry_id="L1",
                ),
                SearchResult(
                    lemma="keep-me",
                    confidence=threshold + 0.01,
                    dialect_attribution="lemma dialect: attic",
                    entry_id="L2",
                ),
            ],
        )
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda query_ipa, results, lexicon_map, matrix, language="ancient_greek": results,
        )

        lexicon = [
            {"id": "L1", "headword": "drop-me", "ipa": "pi", "dialect": "attic"},
            {"id": "L2", "headword": "keep-me", "ipa": "pu", "dialect": "attic"},
        ]

        results = search("\u03bd\u03c5\u03bd", lexicon, matrix={}, max_results=2, index={})

        assert [result.lemma for result in results] == ["keep-me"]

    def test_short_query_keeps_exact_match_below_confidence_floor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Short-query exact token matches should bypass the confidence floor."""
        threshold = search_module._SHORT_QUERY_CONFIDENCE_THRESHOLD
        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "pa")
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: ["L1"])
        monkeypatch.setattr(
            search_module,
            "_score_stage",
            lambda query_ipa, candidates, lexicon_map, matrix: [
                SearchResult(
                    lemma="exact-low-confidence",
                    confidence=threshold - 0.20,
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

        # ``to_ipa`` returns "pa" and the lexicon IPA tokenizes to the same
        # phones, so the mocked low-confidence SearchResult is still an exact
        # token match and should bypass the short-query confidence floor.
        lexicon = [
            {"id": "L1", "headword": "exact-low-confidence", "ipa": "p a", "dialect": "attic"},
        ]

        results = search("\u03bd\u03c5\u03bd", lexicon, matrix={}, max_results=1, index={})

        assert [result.lemma for result in results] == ["exact-low-confidence"]

    def test_short_query_keeps_rule_supported_match_below_confidence_floor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Short-query explicit rule support should bypass the confidence floor."""
        threshold = search_module._SHORT_QUERY_CONFIDENCE_THRESHOLD
        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "pa")
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: ["L1"])
        monkeypatch.setattr(
            search_module,
            "_score_stage",
            lambda query_ipa, candidates, lexicon_map, matrix: [
                SearchResult(
                    lemma="rule-low-confidence",
                    confidence=threshold - 0.20,
                    dialect_attribution="lemma dialect: attic",
                    entry_id="L1",
                )
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
            {"id": "L1", "headword": "rule-low-confidence", "ipa": "p i", "dialect": "attic"},
        ]

        results = search("\u03bd\u03c5\u03bd", lexicon, matrix={}, max_results=1, index={})

        assert [result.lemma for result in results] == ["rule-low-confidence"]
        assert results[0].applied_rules == ["RULE-001"]

    def test_short_query_dedup_prefers_exact_match_over_higher_confidence_distance_only_hit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Short-query dedup should keep an exact homograph variant over a weaker heuristic one."""
        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "pa")
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: ["L1", "L2"])
        monkeypatch.setattr(
            search_module,
            "_score_stage",
            lambda query_ipa, candidates, lexicon_map, matrix: [
                SearchResult(
                    lemma="same",
                    confidence=0.95,
                    dialect_attribution="lemma dialect: attic",
                    entry_id="L1",
                ),
                SearchResult(
                    lemma="same",
                    confidence=0.20,
                    dialect_attribution="lemma dialect: attic",
                    entry_id="L2",
                ),
            ],
        )
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda query_ipa, results, lexicon_map, matrix, language="ancient_greek": list(results),
        )

        lexicon = [
            {"id": "L1", "headword": "same", "ipa": "p i", "dialect": "attic"},
            {"id": "L2", "headword": "same", "ipa": "p a", "dialect": "attic"},
        ]

        results = search("\u03bd\u03c5\u03bd", lexicon, matrix={}, max_results=1, index={})

        assert len(results) == 1
        assert results[0].entry_id == "L2"

    def test_short_query_dedup_prefers_higher_confidence_over_rule_supported_noise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Short-query dedup should not let weak rule support bypass confidence."""
        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "pa")
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: ["L1", "L2"])
        monkeypatch.setattr(
            search_module,
            "_score_stage",
            lambda query_ipa, candidates, lexicon_map, matrix: [
                SearchResult(
                    lemma="same",
                    confidence=0.95,
                    dialect_attribution="lemma dialect: attic",
                    entry_id="L1",
                ),
                SearchResult(
                    lemma="same",
                    confidence=0.20,
                    dialect_attribution="lemma dialect: attic",
                    entry_id="L2",
                    applied_rules=["RULE-002"],
                ),
            ],
        )
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda query_ipa, results, lexicon_map, matrix, language="ancient_greek": list(results),
        )

        lexicon = [
            {"id": "L1", "headword": "same", "ipa": "p i", "dialect": "attic"},
            {"id": "L2", "headword": "same", "ipa": "p u", "dialect": "attic"},
        ]

        results = search("\u03bd\u03c5\u03bd", lexicon, matrix={}, max_results=1, index={})

        assert len(results) == 1
        assert results[0].entry_id == "L1"
        assert results[0].applied_rules == []

    def test_short_query_batch_loop_respects_max_batches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Short-query annotation should process at most MAX_BATCHES batches.

        When post-filtering drops every candidate (all below the confidence
        floor with no exact or rule-supported matches), the loop must stop
        after _SHORT_QUERY_MAX_ANNOTATION_BATCHES iterations rather than
        scanning the entire scored set.

        The _score_stage mock returns more candidates than would normally
        survive seed truncation, simulating the worst-case where a fallback
        path feeds a large scored set into the Short-query annotation loop.
        """
        annotation_call_count = 0
        threshold = search_module._SHORT_QUERY_CONFIDENCE_THRESHOLD

        # With max_results=5, annotation_limit = max(100, 5*10) = 100.
        # We need scored_count > max_batches * annotation_limit = 300.
        scored_count = 400
        entry_ids = [f"L{i}" for i in range(scored_count)]

        monkeypatch.setattr(
            search_module, "to_ipa",
            lambda query, dialect="attic": "zz",
        )
        monkeypatch.setattr(
            search_module, "seed_stage",
            lambda *_args, **_kwargs: list(entry_ids[:50]),
        )
        # Return all 400 entries regardless of the candidates passed in,
        # simulating a fallback path that widens the candidate set.
        monkeypatch.setattr(
            search_module,
            "_score_stage",
            lambda query_ipa, candidates, lexicon_map, matrix: [
                SearchResult(
                    lemma=f"word-{i}",
                    confidence=threshold - 0.20,
                    dialect_attribution="lemma dialect: attic",
                    entry_id=entry_id,
                    ipa=f"z{i}",
                )
                for i, entry_id in enumerate(entry_ids)
            ],
        )

        def fake_annotate_counting(
            query_ipa: str,
            results: list[SearchResult],
            lexicon_map: object,
            matrix: object,
            language: str = "ancient_greek",
        ) -> list[SearchResult]:
            nonlocal annotation_call_count
            annotation_call_count += 1
            return list(results)

        monkeypatch.setattr(
            search_module, "_annotate_search_results", fake_annotate_counting,
        )

        lexicon = [
            {"id": eid, "headword": f"word-{i}", "ipa": f"z{i}", "dialect": "attic"}
            for i, eid in enumerate(entry_ids)
        ]

        results = search(
            "\u03bd\u03c5\u03bd", lexicon, matrix={}, max_results=5, index={},
        )

        # The annotation function should have been called at most
        # _SHORT_QUERY_MAX_ANNOTATION_BATCHES times, not ceil(400/100) = 4.
        assert annotation_call_count >= 1, "Batch loop should process at least one batch"
        assert annotation_call_count <= _SHORT_QUERY_MAX_ANNOTATION_BATCHES

    def test_short_query_truncated_flag_set_when_batch_cap_reached(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When annotation batches are capped, truncated flag should be set on results."""
        # Generate enough candidates to exceed batch limit but have most filtered out
        # so we don't reach max_results before the cap
        threshold = search_module._SHORT_QUERY_CONFIDENCE_THRESHOLD
        entry_ids = [f"E{i:04d}" for i in range(400)]

        def fake_score_stage(
            query_ipa: str,
            candidates: list[str],
            lexicon_map: dict[str, object],
            matrix: object,
        ) -> list[SearchResult]:
            # Return SearchResults for ALL entry_ids (400 items), ignoring the
            # candidates passed in (which are limited by stage2_limit=50).
            # This simulates a fallback path that widens the candidate set.
            # E0000 will be kept (high confidence), others dropped.
            results = []
            for cid in entry_ids:
                conf = threshold + 0.1 if cid == "E0000" else threshold - 0.01
                results.append(
                    SearchResult(
                        lemma=f"word-{cid}",
                        confidence=conf,
                        dialect_attribution="lemma dialect: attic",
                        entry_id=cid,
                    )
                )
            return results

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "zz")
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: entry_ids)
        monkeypatch.setattr(search_module, "_score_stage", fake_score_stage)
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda query_ipa, results, lexicon_map, matrix, language="ancient_greek": results,
        )

        lexicon = [
            {"id": eid, "headword": f"word-{i}", "ipa": f"z{i}", "dialect": "attic"}
            for i, eid in enumerate(entry_ids)
        ]

        # Request more results than we'll get due to filtering
        results = search(
            "ααα", lexicon, matrix={}, max_results=5, index={},
        )

        # One candidate survived filtering (E0000)
        assert len(results) == 1
        assert results[0].entry_id == "E0000"
        # The search hit the batch cap (400 > 300), so the result should be marked as truncated
        assert results[0].truncated is True

    def test_short_query_truncated_flag_false_when_not_capped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When results are found without hitting batch cap, truncated flag should be False."""
        threshold = search_module._SHORT_QUERY_CONFIDENCE_THRESHOLD

        def fake_score_stage(
            query_ipa: str,
            candidates: list[str],
            lexicon_map: dict[str, object],
            matrix: object,
        ) -> list[SearchResult]:
            return [
                SearchResult(
                    lemma=f"word-{cid}",
                    confidence=threshold + 0.1,  # Above threshold
                    dialect_attribution="lemma dialect: attic",
                    entry_id=cid,
                )
                for cid in candidates[:3]  # Only return a few high-confidence results
            ]

        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": "zz")
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: ["E1", "E2", "E3"])
        monkeypatch.setattr(search_module, "_score_stage", fake_score_stage)
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda query_ipa, results, lexicon_map, matrix, language="ancient_greek": results,
        )

        lexicon = [
            {"id": "E1", "headword": "word-1", "ipa": "z1", "dialect": "attic"},
            {"id": "E2", "headword": "word-2", "ipa": "z2", "dialect": "attic"},
            {"id": "E3", "headword": "word-3", "ipa": "z3", "dialect": "attic"},
        ]

        results = search(
            "ααα", lexicon, matrix={}, max_results=5, index={},
        )

        # Results found without hitting batch cap
        assert len(results) > 0
        # truncated flag should be False
        assert all(not r.truncated for r in results)
