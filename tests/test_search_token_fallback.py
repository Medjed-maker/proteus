"""Tests for token-count proximity fallback behavior in phonology.search."""
# ruff: noqa: RUF001, RUF003 — IPA length mark ː (U+02D0) triggers false positives.
# Several tests define monkeypatch stubs whose parameter names must mirror the
# real signatures of the functions they replace (e.g. `_score_stage(query_ipa,
# candidates, lexicon_map, matrix)`). Renaming the unused params to `_prefix`
# would either drift from the real signature or force kwargs-only call sites;
# both are worse than a file-level suppression. Keep this scoped to
# reportUnusedVariable only — other checks stay on.
# pyright: reportUnusedVariable=false

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import pytest

from phonology import search as search_module
from phonology.search import (
    LexiconRecord,
    SearchResult,
    search,
)


@pytest.fixture
def mock_load_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub rule loading for deterministic token-fallback tests."""
    monkeypatch.setattr(search_module, "load_rules", lambda _path: {})


@pytest.fixture
def mock_to_ipa_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[str], None]:
    """Return a helper that stubs IPA conversion to a fixed value."""

    def apply(ipa: str) -> None:
        monkeypatch.setattr(search_module, "to_ipa", lambda query, dialect="attic": ipa)

    return apply


def _make_fake_score_stage(
    captured: dict[str, object],
    *,
    return_results: bool = False,
) -> Callable[[str, Iterable[str], dict[str, object], object], list[SearchResult]]:
    """Return a score-stage stub that captures candidate IDs."""

    def get_headword(record: object) -> str:
        if isinstance(record, LexiconRecord):
            return str(record.entry["headword"])
        assert isinstance(record, dict)
        return str(record["headword"])

    def fake_score_stage(
        query_ipa: str,
        candidates: Iterable[str],
        lexicon_map: dict[str, object],
        matrix: object,
        **_kwargs: Any,
    ) -> list[SearchResult]:
        candidate_ids = list(candidates)
        captured["candidate_ids"] = candidate_ids
        if not return_results:
            return []
        return [
            SearchResult(
                lemma=get_headword(lexicon_map[cid]),
                confidence=1.0,
                dialect_attribution="lemma dialect: attic",
                entry_id=str(cid),
            )
            for cid in candidate_ids
        ]

    return fake_score_stage


class TestSearchTokenFallback:
    """Tests for token-count proximity fallback behavior."""

    def test_token_proximity_helper_returns_capped_fullform_selection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The extracted helper should preserve full-form cap and metadata."""
        captured: dict[str, object] = {}
        tokenized_map = {
            "L1": LexiconRecord(
                entry={"id": "L1", "headword": "one", "ipa": "a", "dialect": "attic"},
                token_count=1,
                ipa_tokens=("a",),
            ),
            "L2": LexiconRecord(
                entry={"id": "L2", "headword": "two", "ipa": "a b", "dialect": "attic"},
                token_count=2,
                ipa_tokens=("a", "b"),
            ),
        }
        dependencies = search_module._LazySearchDependencies(
            lexicon=[],
            prebuilt_lexicon_map=tokenized_map,
            prebuilt_ipa_index=None,
        )

        def fake_rank_by_token_count_proximity(
            query_ipa: str,
            lexicon_map: dict[str, LexiconRecord],
            *,
            max_candidates: int | None = None,
            query_token_count: int | None = None,
            **_kwargs: Any,
        ) -> list[str]:
            captured["query_ipa"] = query_ipa
            captured["lexicon_lookup"] = lexicon_map
            captured["max_candidates"] = max_candidates
            captured["query_token_count"] = query_token_count
            return ["L2"]

        monkeypatch.setattr(
            search_module,
            "_rank_by_token_count_proximity",
            fake_rank_by_token_count_proximity,
        )

        query_tokens = ["a", "b"]
        selection = search_module._select_token_proximity_fallback_candidates(
            query_ipa="a b",
            query_log_label="tokens=2 chars=3 sha256=testlabel",
            query_mode="Full-form",
            query_tokens=query_tokens,
            partial_query_tokens=None,
            dependencies=dependencies,
            max_results=1,
            effective_similarity_fallback_limit=7,
        )

        assert captured == {
            "query_ipa": "a b",
            "lexicon_lookup": tokenized_map,
            "max_candidates": 7,
            "query_token_count": 2,
        }
        assert selection.candidate_ids == ["L2"]
        assert selection.lexicon_lookup is tokenized_map
        assert selection.query_mode == "Full-form"
        assert selection.query_tokens is query_tokens
        assert selection.selection_path == "token-proximity-fallback"
        assert selection.seed_candidate_count == 0
        assert selection.unigram_candidate_count == 0
        assert selection.fallback_limit == 7

    def test_token_proximity_helper_routes_partial_form_through_partial_selector(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Partial-form token fallback should keep the partial selector path."""
        captured: dict[str, object] = {}
        tokenized_map = {
            "L1": LexiconRecord(
                entry={"id": "L1", "headword": "one", "ipa": "a b", "dialect": "attic"},
                token_count=2,
                ipa_tokens=("a", "b"),
            ),
        }
        dependencies = search_module._LazySearchDependencies(
            lexicon=[],
            prebuilt_lexicon_map=tokenized_map,
            prebuilt_ipa_index=None,
        )
        partial_query = search_module.PartialQueryTokens(
            shape="suffix",
            left_tokens=(),
            right_tokens=("b",),
        )

        def fake_select_partial_token_fallback_candidates(
            partial_query_arg: search_module.PartialQueryTokens,
            query_ipa: str,
            query_token_count: int,
            lexicon_map: dict[str, LexiconRecord],
            *,
            max_results: int,
            explicit_limit: int,
            **_kwargs: Any,
        ) -> list[str]:
            captured["partial_query"] = partial_query_arg
            captured["query_ipa"] = query_ipa
            captured["query_token_count"] = query_token_count
            captured["lexicon_lookup"] = lexicon_map
            captured["max_results"] = max_results
            captured["explicit_limit"] = explicit_limit
            return ["L1"]

        monkeypatch.setattr(
            search_module,
            "_select_partial_token_fallback_candidates",
            fake_select_partial_token_fallback_candidates,
        )

        query_tokens = ["a", "b"]
        selection = search_module._select_token_proximity_fallback_candidates(
            query_ipa="a b",
            query_log_label="tokens=2 chars=3 sha256=testlabel",
            query_mode="Partial-form",
            query_tokens=query_tokens,
            partial_query_tokens=partial_query,
            dependencies=dependencies,
            max_results=3,
            effective_similarity_fallback_limit=11,
        )

        assert captured == {
            "partial_query": partial_query,
            "query_ipa": "a b",
            "query_token_count": 2,
            "lexicon_lookup": tokenized_map,
            "max_results": 3,
            "explicit_limit": 11,
        }
        assert selection.candidate_ids == ["L1"]
        assert selection.lexicon_lookup is tokenized_map
        assert selection.query_mode == "Partial-form"
        assert selection.query_tokens is query_tokens
        assert selection.selection_path == "partial-token-proximity-fallback"
        assert selection.seed_candidate_count == 0
        assert selection.unigram_candidate_count == 0
        assert selection.fallback_limit == 11

    def test_uses_token_proximity_when_query_has_no_seedable_skeleton(
        self,
        mock_load_rules: None,
        mock_to_ipa_factory: Callable[[str], None],
    ) -> None:
        mock_to_ipa_factory("aː")
        lexicon = [
            {"id": "L1", "headword": "alpha", "ipa": "aː", "dialect": "attic"},
            {"id": "L2", "headword": "iota", "ipa": "i", "dialect": "attic"},
            {"id": "L3", "headword": "upsilon", "ipa": "y", "dialect": "attic"},
        ]

        results = search("ἄ", lexicon, matrix={}, max_results=2)

        assert [result.lemma for result in results] == ["alpha"]

    def test_k2_seed_candidates_respect_stage2_limit(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_to_ipa_factory: Callable[[str], None],
    ) -> None:
        """Only the primary k=2 seed path should cap candidates at stage2_limit."""
        captured: dict[str, object] = {}

        mock_to_ipa_factory("aː")
        seed_calls = iter(
            [
                [f"L{index:02d}" for index in range(30)],
            ]
        )
        monkeypatch.setattr(
            search_module,
            "seed_stage",
            lambda *_args, **_kwargs: next(seed_calls),
        )
        monkeypatch.setattr(
            search_module,
            "_score_stage",
            _make_fake_score_stage(captured, return_results=True),
        )
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda query_ipa,
            results,
            lexicon_map,
            matrix,
            language="ancient_greek",
            **_kwargs: results,
        )
        monkeypatch.setattr(
            search_module,
            "filter_stage",
            lambda results, max_results: results[:max_results],
        )
        lexicon = [
            {
                "id": f"L{index:02d}",
                "headword": f"lemma-{index}",
                "ipa": f"aː{chr(ord('a') + index)}",
                "dialect": "attic",
            }
            for index in range(30)
        ]

        # max_results=1 → stage2_limit = max(25, 1*10) = 25, capping 30 entries.
        # No entry has IPA matching query "aː", so exact-match injection adds nothing.
        search("ἄ", lexicon, matrix={}, max_results=1)

        assert len(captured["candidate_ids"]) == 25

    def test_token_proximity_finds_exact_match_by_length_similarity(
        self,
        mock_load_rules: None,
        mock_to_ipa_factory: Callable[[str], None],
    ) -> None:
        """Token-count proximity fallback should prefer entries with similar IPA length."""
        mock_to_ipa_factory("aː")
        lexicon = [
            {
                "id": f"L{index}",
                "headword": f"lemma-{index}",
                "ipa": "i",
                "dialect": "attic",
            }
            for index in range(29)
        ]
        lexicon.append(
            {"id": "L29", "headword": "target", "ipa": "aː", "dialect": "attic"}
        )

        results = search("ἄ", lexicon, matrix={}, max_results=1)

        # "aː" (1 token) should match "aː" (1 token) over "i" (1 token) by
        # exact phonological similarity via Smith-Waterman. The fallback is
        # capped, so exact IPA matches must still sort into the evaluated set.
        assert [result.lemma for result in results] == ["target"]

    def test_token_proximity_fallback_does_not_drop_exact_match_after_sorting(
        self,
        mock_load_rules: None,
        mock_to_ipa_factory: Callable[[str], None],
    ) -> None:
        """Exact matches must survive fallback when target has the closest token count."""
        mock_to_ipa_factory("aː")
        # 20 noise entries with 3-token IPA, 1 target with 1-token IPA matching the query.
        # Token-count proximity puts the target (diff=0) ahead of noise (diff=2).
        lexicon = [
            {
                "id": f"L{index:02d}",
                "headword": f"noise-{index:02d}",
                "ipa": "poi",
                "dialect": "attic",
            }
            for index in range(20)
        ]
        lexicon.append(
            {"id": "L99", "headword": "target", "ipa": "aː", "dialect": "attic"}
        )

        results = search("ἄ", lexicon, matrix={}, max_results=1)

        assert [result.lemma for result in results] == ["target"]

    def test_token_proximity_fallback_when_no_kmer_matches(
        self,
        mock_load_rules: None,
        mock_to_ipa_factory: Callable[[str], None],
    ) -> None:
        """Short-query precision mode may return no result when no candidate is credible."""
        mock_to_ipa_factory("mn")
        lexicon = [
            {"id": "L1", "headword": "πτω", "ipa": "pto", "dialect": "attic"},
            {"id": "L2", "headword": "κτω", "ipa": "kto", "dialect": "attic"},
        ]

        results = search("μν", lexicon, matrix={}, max_results=2)

        assert results == []

    @pytest.mark.parametrize(
        ("query", "to_ipa_return", "max_results"),
        [
            pytest.param("ἄ", "aː", 1, id="short-query"),
            pytest.param("λόγος", "loɡos", 5, id="fullform"),
            pytest.param("alpha", "aː", 1, id="fullform-consonantless"),
            pytest.param("lo-", "lo", 5, id="partial-form"),
        ],
    )
    def test_token_proximity_fallback_uses_default_cap(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_to_ipa_factory: Callable[[str], None],
        query: str,
        to_ipa_return: str,
        max_results: int,
    ) -> None:
        """Token fallback should not score an uncapped fallback list by default."""
        captured: dict[str, object] = {}

        mock_to_ipa_factory(to_ipa_return)
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: [])
        monkeypatch.setattr(
            search_module, "_score_stage", _make_fake_score_stage(captured)
        )
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda query_ipa,
            results,
            lexicon_map,
            matrix,
            language="ancient_greek",
            **_kwargs: results,
        )
        monkeypatch.setattr(
            search_module, "filter_stage", lambda results, max_results: results
        )

        lexicon = [
            {
                "id": f"L{index:04d}",
                "headword": f"lemma-{index:04d}",
                "ipa": "i",
                "dialect": "attic",
            }
            for index in range(2500)
        ]

        search(
            query,
            lexicon,
            matrix={},
            max_results=max_results,
            index={},
            unigram_index={},
        )

        # For partial-form, the effective limit is _partial_candidate_limit(max_results)
        # which controls the stage-2 window, while _DEFAULT_FALLBACK_CANDIDATE_LIMIT
        # controls the exploration cap only.
        from phonology.search import _partial_candidate_limit

        expected_limit = (
            _partial_candidate_limit(max_results)
            if query.endswith("-") or query.startswith("-") or "-" in query
            else search_module._DEFAULT_FALLBACK_CANDIDATE_LIMIT
        )
        assert len(captured["candidate_ids"]) == expected_limit

    def test_fullform_token_proximity_fallback_applies_default_cap_when_none(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
        mock_to_ipa_factory: Callable[[str], None],
    ) -> None:
        """Full-form token fallback should fall back to the default cap when None is passed."""
        captured: dict[str, object] = {}

        mock_to_ipa_factory("loɡos")
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: [])
        monkeypatch.setattr(
            search_module, "_score_stage", _make_fake_score_stage(captured)
        )
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda query_ipa,
            results,
            lexicon_map,
            matrix,
            language="ancient_greek",
            **_kwargs: results,
        )
        monkeypatch.setattr(
            search_module, "filter_stage", lambda results, max_results: results
        )

        lexicon = [
            {
                "id": f"L{index:04d}",
                "headword": f"lemma-{index:04d}",
                "ipa": "i",
                "dialect": "attic",
            }
            for index in range(2500)
        ]

        caplog.set_level("WARNING", logger="phonology.search")
        search(
            "λόγος",
            lexicon,
            matrix={},
            max_results=5,
            index={},
            unigram_index={},
            similarity_fallback_limit=None,
        )

        assert (
            len(captured["candidate_ids"])
            == search_module._DEFAULT_FALLBACK_CANDIDATE_LIMIT
        )
        expected_label = search_module._summarize_query_ipa_for_logs(
            "loɡos",
            query_token_count=len(search_module.tokenize_ipa("loɡos")),
            debug_enabled=False,
        )
        assert "similarity_fallback_limit=None for query" in caplog.text
        assert expected_label in caplog.text
        assert "query IPA 'loɡos'" not in caplog.text
        assert (
            f"applying default cap {search_module._DEFAULT_FALLBACK_CANDIDATE_LIMIT}."
            in caplog.text
        )

    def test_search_accepts_explicit_similarity_fallback_limit(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_to_ipa_factory: Callable[[str], None],
    ) -> None:
        """Callers can tighten the token-count fallback candidate cap."""
        captured: dict[str, object] = {}

        mock_to_ipa_factory("aː")
        monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: [])
        monkeypatch.setattr(
            search_module, "_score_stage", _make_fake_score_stage(captured)
        )
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results",
            lambda query_ipa,
            results,
            lexicon_map,
            matrix,
            language="ancient_greek",
            **_kwargs: results,
        )
        monkeypatch.setattr(
            search_module, "filter_stage", lambda results, max_results: results
        )

        lexicon = [
            {
                "id": f"L{index:02d}",
                "headword": f"lemma-{index:02d}",
                "ipa": "i",
                "dialect": "attic",
            }
            for index in range(10)
        ]

        search(
            "ἄ",
            lexicon,
            matrix={},
            max_results=1,
            index={},
            unigram_index={},
            similarity_fallback_limit=3,
        )

        assert len(captured["candidate_ids"]) == 3

    def test_short_query_token_proximity_still_keeps_exact_match_with_default_cap(
        self,
        mock_load_rules: None,
        mock_to_ipa_factory: Callable[[str], None],
    ) -> None:
        """Exact same-length targets should still survive short-query fallback."""
        mock_to_ipa_factory("a")
        lexicon = [
            {
                "id": f"L{index:02d}",
                "headword": f"noise-{index:02d}",
                "ipa": "i",
                "dialect": "attic",
            }
            for index in range(30)
        ]
        lexicon.append(
            {"id": "ZZZ", "headword": "target", "ipa": "e", "dialect": "attic"}
        )
        matrix = {"e": {"a": 0.1}, "i": {"a": 0.9}, "a": {"e": 0.1, "i": 0.9}}

        results = search(
            "dummy", lexicon, matrix=matrix, max_results=1, index={}, unigram_index={}
        )

        assert [result.lemma for result in results] == ["target"]
