"""Unit tests for score_stage_mock assertion helpers."""

import pytest

from tests._helpers.score_stage_mock import (
    EXPECTED_SCORE_STAGE_KWARGS,
    assert_only_expected_score_stage_kwargs,
)


class TestAssertOnlyExpectedScoreStageKwargs:
    """Test suite for assert_only_expected_score_stage_kwargs."""

    def test_expected_kwargs_passes(self) -> None:
        """Calling with kwargs from EXPECTED_SCORE_STAGE_KWARGS raises no exception."""
        valid_kwargs = {key: "dummy_value" for key in EXPECTED_SCORE_STAGE_KWARGS}
        # Should not raise
        assert_only_expected_score_stage_kwargs(valid_kwargs)

    def test_empty_dict_passes(self) -> None:
        """Calling with empty dict raises no exception."""
        # Should not raise
        assert_only_expected_score_stage_kwargs({})

    @pytest.mark.parametrize(
        "unexpected_keys",
        [
            {"unexpected_single"},
            {"unexpected_a", "unexpected_b"},
            {"phone_inventory", "extra_key"},  # Mix of expected and unexpected
        ],
    )
    def test_unexpected_keys_raises_assertion_error(
        self, unexpected_keys: set[str]
    ) -> None:
        """Calling with unexpected keys raises AssertionError."""
        kwargs = {key: "dummy_value" for key in unexpected_keys}

        with pytest.raises(AssertionError):
            assert_only_expected_score_stage_kwargs(kwargs)

    @pytest.mark.parametrize(
        "unexpected_keys,expected_sorted",
        [
            ({"z_key", "a_key", "m_key"}, ["a_key", "m_key", "z_key"]),
            ({"single"}, ["single"]),
        ],
    )
    def test_error_message_contains_sorted_unexpected_names(
        self, unexpected_keys: set[str], expected_sorted: list[str]
    ) -> None:
        """AssertionError message lists unexpected names in sorted order."""
        kwargs = {key: "dummy_value" for key in unexpected_keys}

        with pytest.raises(AssertionError) as exc_info:
            assert_only_expected_score_stage_kwargs(kwargs)

        assert str(expected_sorted) in str(exc_info.value)
