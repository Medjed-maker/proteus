"""Tests for proteus.phonology.search."""

import pytest

from proteus.phonology.search import SearchConfig, SearchResult, build_kmer_index, filter_stage


@pytest.fixture
def sample_search_results() -> list[SearchResult]:
    return [
        SearchResult(headword="γ", ipa="ɡ", distance=0.3),
        SearchResult(headword="α", ipa="a", distance=0.1),
        SearchResult(headword="β", ipa="b", distance=0.2),
    ]


class TestFilterStage:
    """Verify final filtering sorts by distance and rejects invalid limits."""

    def test_sorts_by_distance_and_truncates_results(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        """Sort three unsorted hits by ascending distance and keep only the best two."""
        filtered = filter_stage(sample_search_results, max_results=2)

        assert [result.headword for result in filtered] == ["α", "β"]

    @pytest.mark.parametrize("max_results", [0, -1])
    def test_rejects_non_positive_max_results(self, max_results: int) -> None:
        """Reject 0 and negative max_results values with a clear validation error."""
        with pytest.raises(ValueError, match="positive integer"):
            filter_stage([], max_results=max_results)


class TestBuildKmerIndex:
    @pytest.mark.parametrize("k", [0, -1])
    def test_rejects_non_positive_k(self, k: int) -> None:
        """Reject non-positive k-mer sizes before build_kmer_index reaches its stub body."""
        with pytest.raises(ValueError, match="build_kmer_index.*k"):
            build_kmer_index([], k=k)


class TestSearchConfig:
    def test_search_config_validates_invalid_values(self) -> None:
        """SearchConfig raises clear ValueError messages for invalid inputs."""
        with pytest.raises(ValueError, match="kmer_size"):
            SearchConfig(kmer_size=0)

        with pytest.raises(ValueError, match="seed_threshold"):
            SearchConfig(seed_threshold=1.5)

        with pytest.raises(ValueError, match="extend_threshold"):
            SearchConfig(extend_threshold=-0.1)

        with pytest.raises(ValueError, match="max_results"):
            SearchConfig(max_results=0)

        with pytest.raises(ValueError, match="dialect"):
            SearchConfig(dialect="   ")
