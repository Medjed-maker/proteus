"""Tests for the debug logging utilities in phonology.search._debug_logging.

These tests verify that logging helpers are no-ops when DEBUG is disabled
and function correctly when DEBUG is enabled.
"""

from __future__ import annotations

import logging
import re
from unittest.mock import MagicMock

import pytest

import phonology.search._debug_logging as debug_logging
from phonology.search._debug_logging import (
    log_candidate_selection,
    log_finalization,
    log_scoring,
    perf_counter_if_debug,
    summarize_query_ipa_for_logs,
)


class TestSummarizeQueryIpaForLogs:
    """Tests for summarize_query_ipa_for_logs function."""

    def test_returns_stable_12_hex_sha256_prefix(self) -> None:
        """The SHA-256 prefix should be exactly 12 hex characters (48 bits)."""
        result = summarize_query_ipa_for_logs("test_query", query_token_count=3)

        # Extract the sha256 part
        match = re.search(r"sha256=([a-f0-9]+)", result)
        assert match is not None, f"Expected sha256=... in result: {result}"
        sha_value = match.group(1)

        assert len(sha_value) == 12, f"Expected 12 hex chars, got {len(sha_value)}"
        assert all(c in "0123456789abcdef" for c in sha_value), (
            "Should be lowercase hex"
        )

    def test_returns_expected_format(self) -> None:
        """Result should match the expected 'tokens=N chars=M sha256=...' format."""
        query = "abc"
        token_count = 2
        result = summarize_query_ipa_for_logs(query, query_token_count=token_count)

        expected_pattern = (
            rf"tokens={token_count} chars={len(query)} sha256=[a-f0-9]{{12}}"
        )
        assert re.fullmatch(expected_pattern, result), (
            f"Result '{result}' didn't match pattern"
        )

    def test_different_inputs_produce_different_hashes(self) -> None:
        """Different queries should produce different SHA-256 prefixes."""
        result1 = summarize_query_ipa_for_logs("query_one", query_token_count=2)
        result2 = summarize_query_ipa_for_logs("query_two", query_token_count=2)

        m1 = re.search(r"sha256=([a-f0-9]+)", result1)
        m2 = re.search(r"sha256=([a-f0-9]+)", result2)
        assert m1 is not None, f"Expected sha256=... in result: {result1}"
        assert m2 is not None, f"Expected sha256=... in result: {result2}"
        hash1 = m1.group(1)
        hash2 = m2.group(1)

        assert hash1 != hash2, "Different queries should have different hashes"

    def test_same_input_produces_same_hash(self) -> None:
        """Same query should produce identical SHA-256 prefix (deterministic)."""
        query = "consistent_query"
        result1 = summarize_query_ipa_for_logs(query, query_token_count=3)
        result2 = summarize_query_ipa_for_logs(query, query_token_count=3)

        assert result1 == result2, "Same query should produce identical result"

    def test_empty_query_edge_case(self) -> None:
        """Empty query should still produce the redacted debug-log summary format."""
        result = summarize_query_ipa_for_logs("", query_token_count=3)

        assert re.fullmatch(r"tokens=3 chars=0 sha256=[a-f0-9]{12}", result)

    def test_zero_token_count_edge_case(self) -> None:
        """Zero token count should be preserved with deterministic hashing."""
        query = "normal_query"
        result1 = summarize_query_ipa_for_logs(query, query_token_count=0)
        result2 = summarize_query_ipa_for_logs(query, query_token_count=0)

        assert "tokens=0" in result1
        assert re.fullmatch(
            rf"tokens=0 chars={len(query)} sha256=[a-f0-9]{{12}}",
            result1,
        )
        assert result1 == result2

    def test_non_ascii_query_edge_case(self) -> None:
        """Non-ASCII/IPA query text should hash deterministically in lowercase hex."""
        query = "ˈaβγ"
        result1 = summarize_query_ipa_for_logs(query, query_token_count=4)
        result2 = summarize_query_ipa_for_logs(query, query_token_count=4)

        assert re.fullmatch(
            rf"tokens=4 chars={len(query)} sha256=[a-f0-9]{{12}}",
            result1,
        )
        assert result1 == result2

    def test_debug_disabled_skips_hashing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Disabled debug summaries should not compute SHA-256 digests."""

        def fail_sha256(_value: bytes) -> object:
            raise AssertionError("sha256 should not be called when debug is disabled")

        monkeypatch.setattr(debug_logging.hashlib, "sha256", fail_sha256)

        result = summarize_query_ipa_for_logs(
            "expensive_query",
            query_token_count=2,
            debug_enabled=False,
        )

        assert result == "tokens=2 chars=15 sha256=<debug-disabled>"


class TestPerfCounterIfDebug:
    """Tests for perf_counter_if_debug function."""

    def test_returns_zero_when_debug_disabled(self) -> None:
        """When logger is not enabled for DEBUG, should return 0.0."""
        logger = logging.getLogger("test_debug_disabled")
        logger.setLevel(logging.INFO)

        result = perf_counter_if_debug(logger)

        assert result == 0.0, "Should return 0.0 when DEBUG is disabled"

    def test_returns_positive_value_when_debug_enabled(self) -> None:
        """When logger is enabled for DEBUG, should return a positive perf counter value."""
        logger = logging.getLogger("test_debug_enabled")
        logger.setLevel(logging.DEBUG)

        result = perf_counter_if_debug(logger)

        assert result > 0.0, (
            "Should return positive perf_counter value when DEBUG enabled"
        )


class TestLogHelpersWhenDebugDisabled:
    """Tests that log helpers are no-ops when DEBUG is disabled."""

    @pytest.fixture
    def mock_logger_disabled(self) -> MagicMock:
        """Create a mock logger that reports DEBUG as disabled."""
        mock = MagicMock(spec=logging.Logger)
        mock.isEnabledFor.return_value = False
        return mock

    def test_log_candidate_selection_is_noop_when_debug_disabled(
        self, mock_logger_disabled: MagicMock
    ) -> None:
        """log_candidate_selection should not call logger.debug when DEBUG is disabled."""
        log_candidate_selection(
            mock_logger_disabled,
            query_label="test",
            query_mode="full",
            selection_path="test_path",
            seed_candidate_count=10,
            unigram_candidate_count=5,
            selected_count=8,
            fallback_limit=100,
            elapsed_ms=1.5,
        )

        mock_logger_disabled.debug.assert_not_called()

    def test_log_scoring_is_noop_when_debug_disabled(
        self, mock_logger_disabled: MagicMock
    ) -> None:
        """log_scoring should not call logger.debug when DEBUG is disabled."""
        log_scoring(
            mock_logger_disabled,
            query_label="test",
            selected_count=10,
            scored_count=8,
            elapsed_ms=2.0,
        )

        mock_logger_disabled.debug.assert_not_called()

    def test_log_finalization_is_noop_when_debug_disabled(
        self, mock_logger_disabled: MagicMock
    ) -> None:
        """log_finalization should not call logger.debug when DEBUG is disabled."""
        log_finalization(
            mock_logger_disabled,
            query_label="test",
            query_mode="full",
            annotated_count=8,
            returned_count=5,
            elapsed_ms=1.0,
        )

        mock_logger_disabled.debug.assert_not_called()


class TestLogHelpersWhenDebugEnabled:
    """Tests that log helpers call debug when DEBUG is enabled."""

    @pytest.fixture
    def mock_logger_enabled(self) -> MagicMock:
        """Create a mock logger that reports DEBUG as enabled."""
        mock = MagicMock(spec=logging.Logger)
        mock.isEnabledFor.return_value = True
        return mock

    def test_log_candidate_selection_calls_debug_when_enabled(
        self, mock_logger_enabled: MagicMock
    ) -> None:
        """log_candidate_selection should call logger.debug with expected parameters."""
        log_candidate_selection(
            mock_logger_enabled,
            query_label="query_123",
            query_mode="partial",
            selection_path="partial_path",
            seed_candidate_count=20,
            unigram_candidate_count=10,
            selected_count=15,
            fallback_limit=50,
            elapsed_ms=3.5,
        )

        mock_logger_enabled.debug.assert_called_once()
        call_args = mock_logger_enabled.debug.call_args[0]

        message_format = call_args[0]
        assert "candidate selection completed" in message_format
        args = call_args[1:]
        assert args[0] == "query_123"
        assert args[1] == "partial"
        assert args[2] == "partial_path"
        assert args[3] == 20
        assert args[4] == 10
        assert args[5] == 15

    def test_log_scoring_calls_debug_when_enabled(
        self, mock_logger_enabled: MagicMock
    ) -> None:
        """log_scoring should call logger.debug with expected parameters."""
        log_scoring(
            mock_logger_enabled,
            query_label="query_456",
            selected_count=25,
            scored_count=20,
            elapsed_ms=5.0,
        )

        mock_logger_enabled.debug.assert_called_once()
        call_args = mock_logger_enabled.debug.call_args[0]

        message_format = call_args[0]
        assert "scoring completed" in message_format
        args = call_args[1:]
        assert args[0] == "query_456"
        assert args[1] == 25
        assert args[2] == 20

    def test_log_finalization_calls_debug_when_enabled(
        self, mock_logger_enabled: MagicMock
    ) -> None:
        """log_finalization should call logger.debug with expected parameters."""
        log_finalization(
            mock_logger_enabled,
            query_label="query_789",
            query_mode="short",
            annotated_count=12,
            returned_count=8,
            elapsed_ms=2.5,
        )

        mock_logger_enabled.debug.assert_called_once()
        call_args = mock_logger_enabled.debug.call_args[0]

        message_format = call_args[0]
        assert "annotation/final filtering completed" in message_format
        args = call_args[1:]
        assert args[0] == "query_789"
        assert args[1] == "short"
        assert args[2] == 12
        assert args[3] == 8
