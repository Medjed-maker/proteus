"""End-to-end regression tests for packaged-data search behavior."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from api.main import _load_ipa_index, _load_lexicon_map, load_lexicon_entries
from phonology.distance import load_matrix
from phonology.search import (
    IpaIndex,
    KmerIndex,
    LexiconMap,
    SearchResult,
    build_kmer_index,
    search,
)

MATRIX_FILE = "attic_doric.json"


@pytest.fixture(scope="module")
def packaged_lexicon() -> tuple[dict[str, Any], ...]:
    """Return the packaged Greek lemma lexicon."""
    return load_lexicon_entries()


@pytest.fixture(scope="module")
def packaged_matrix() -> dict[str, dict[str, float]]:
    """Return the packaged phonological distance matrix."""
    return load_matrix(MATRIX_FILE)


@pytest.fixture(scope="module")
def packaged_kmer_index(
    packaged_lexicon: Sequence[dict[str, Any]],
) -> KmerIndex:
    """Return a prebuilt k=2 search index for the packaged lexicon."""
    return build_kmer_index(packaged_lexicon)


@pytest.fixture(scope="module")
def packaged_lexicon_map() -> LexiconMap:
    """Return the cached packaged lexicon map."""
    return _load_lexicon_map()


@pytest.fixture(scope="module")
def packaged_ipa_index() -> IpaIndex:
    """Return the cached packaged exact-IPA index."""
    return _load_ipa_index()


def _search_packaged(
    query: str,
    *,
    lexicon: Sequence[dict[str, Any]],
    matrix: dict[str, dict[str, float]],
    index: KmerIndex,
    lexicon_map: LexiconMap,
    ipa_index: IpaIndex,
) -> list[SearchResult]:
    """Run full search with the packaged indexes used by the API."""
    return search(
        query,
        lexicon,
        matrix,
        max_results=8,
        dialect="attic",
        index=index,
        prebuilt_lexicon_map=lexicon_map,
        prebuilt_ipa_index=ipa_index,
    )


@pytest.mark.parametrize(
    ("query", "expected_lemma", "expected_rule_id"),
    [
        ("δῶρο", "δῶρον", "MPH-017"),
        ("παιδίο", "παιδίον", "MPH-015"),
        ("μνημεῖο", "μνημεῖον", "MPH-016"),
        ("τέκνο", "τέκνον", "MPH-017"),
    ],
)
def test_search_surfaces_neuter_final_nu_deletion_targets(
    query: str,
    expected_lemma: str,
    expected_rule_id: str,
    packaged_lexicon: Sequence[dict[str, Any]],
    packaged_matrix: dict[str, dict[str, float]],
    packaged_kmer_index: KmerIndex,
    packaged_lexicon_map: LexiconMap,
    packaged_ipa_index: IpaIndex,
) -> None:
    """Nu-dropped forms should surface their standard neuter lemmas in top results."""
    results = _search_packaged(
        query,
        lexicon=packaged_lexicon,
        matrix=packaged_matrix,
        index=packaged_kmer_index,
        lexicon_map=packaged_lexicon_map,
        ipa_index=packaged_ipa_index,
    )

    target = next((result for result in results if result.lemma == expected_lemma), None)

    result_summary = [
        {
            "lemma": result.lemma,
            "confidence": result.confidence,
            "applied_rules": result.applied_rules,
        }
        for result in results
    ]
    assert target is not None, (
        f"Expected lemma {expected_lemma!r} in search results for query {query!r}; "
        f"results={result_summary!r}"
    )
    assert expected_rule_id in target.applied_rules, (
        f"Expected rule {expected_rule_id!r} in applied rules for target "
        f"{target.lemma!r}; applied_rules={target.applied_rules!r}"
    )


def test_search_exact_neuter_final_nu_match_stays_rule_free(
    packaged_lexicon: Sequence[dict[str, Any]],
    packaged_matrix: dict[str, dict[str, float]],
    packaged_kmer_index: KmerIndex,
    packaged_lexicon_map: LexiconMap,
    packaged_ipa_index: IpaIndex,
) -> None:
    """Exact final-nu forms should keep exact-match ranking and no rule support."""
    results = _search_packaged(
        "δῶρον",
        lexicon=packaged_lexicon,
        matrix=packaged_matrix,
        index=packaged_kmer_index,
        lexicon_map=packaged_lexicon_map,
        ipa_index=packaged_ipa_index,
    )

    assert results, "Expected at least one search result for query 'δῶρον'"
    assert results[0].lemma == "δῶρον", (
        f"Expected top result lemma 'δῶρον'; got {results[0].lemma!r}"
    )
    assert results[0].confidence == pytest.approx(1.0), (
        f"Expected top result confidence 1.0; got {results[0].confidence!r}"
    )
    assert results[0].applied_rules == [], (
        f"Expected top result to have no applied rules; got {results[0].applied_rules!r}"
    )


@pytest.mark.parametrize(
    ("query", "expected_behavior"),
    [
        ("gibberish123xyz", "empty_or_low_confidence"),
        ("", "raises_value_err"),
        ("   ", "raises_value_err"),
        ("aeiou", "low_confidence"),
    ],
)
def test_search_pipeline_edge_cases(
    query: str,
    expected_behavior: str,
    packaged_lexicon: Sequence[dict[str, Any]],
    packaged_matrix: dict[str, dict[str, float]],
    packaged_kmer_index: KmerIndex,
    packaged_lexicon_map: LexiconMap,
    packaged_ipa_index: IpaIndex,
) -> None:
    """Search should handle nonexistent, empty, and low-similarity queries gracefully."""
    if expected_behavior == "raises_value_err":
        with pytest.raises(ValueError, match="query must be a non-empty string"):
            _search_packaged(
                query,
                lexicon=packaged_lexicon,
                matrix=packaged_matrix,
                index=packaged_kmer_index,
                lexicon_map=packaged_lexicon_map,
                ipa_index=packaged_ipa_index,
            )
        return

    results = _search_packaged(
        query,
        lexicon=packaged_lexicon,
        matrix=packaged_matrix,
        index=packaged_kmer_index,
        lexicon_map=packaged_lexicon_map,
        ipa_index=packaged_ipa_index,
    )

    if expected_behavior == "empty_or_low_confidence":
        # We expect either no results or results with very low confidence.
        # Gibberish usually yields empty results if k-mers don't match anything.
        for result in results:
            assert result.confidence < 0.5, (
                f"Expected low confidence for gibberish query {query!r}; "
                f"got {result.confidence} for lemma {result.lemma!r}"
            )
    elif expected_behavior == "low_confidence":
        # Pure vowel queries or low-similarity queries should have low confidence
        for result in results:
            assert result.confidence < 0.8, (
                f"Expected low confidence for query {query!r}; "
                f"got {result.confidence} for lemma {result.lemma!r}"
            )
