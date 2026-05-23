"""Tests for corpus source metadata adapters."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

import phonology.corpus as corpus_pkg
from phonology.corpus import (
    CompositeCorpusAdapter,
    CorpusSourceDataError,
    EMPTY_CORPUS_ADAPTER,
    SourceReference,
    load_static_corpus_adapter,
    safe_lookup,
)


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = (
    ROOT_DIR
    / "data"
    / "languages"
    / "ancient_greek"
    / "corpus_sources"
    / "perseus_scaife_sources.yaml"
)


def test_static_adapter_returns_references_for_entry_id() -> None:
    """Verify the static adapter returns source references for a known entry."""
    adapter = load_static_corpus_adapter(DATA_PATH)

    references = adapter.lookup(
        entry_id="LSJ-063772",
        headword="λόγος",
        language="ancient_greek",
    )

    assert len(references) == 1
    assert references[0].source_id == "Perseus:text:1999.04.0057:entry=lo%2Fgos"
    assert references[0].external_url.startswith("https://www.perseus.tufts.edu/")


def test_static_adapter_returns_empty_tuple_for_unknown_entry() -> None:
    """Verify unknown entry IDs return an empty reference tuple."""
    adapter = load_static_corpus_adapter(DATA_PATH)

    assert (
        adapter.lookup(
            entry_id="missing",
            headword="missing",
            language="ancient_greek",
        )
        == ()
    )


def test_empty_adapter_returns_empty_tuple() -> None:
    """Verify the empty adapter always returns an empty reference tuple."""
    assert (
        EMPTY_CORPUS_ADAPTER.lookup(
            entry_id="LSJ-063772",
            headword="λόγος",
            language="ancient_greek",
        )
        == ()
    )


def test_composite_adapter_preserves_adapter_order() -> None:
    """Verify composite lookup returns adapter references in input order."""
    first = SourceReference(
        source_id="first",
        corpus="Test",
        short_citation="First",
        external_url="https://example.test/first",
        license_note="Test metadata.",
        access_policy="open_metadata",
        citation_ready=False,
    )
    second = SourceReference(
        source_id="second",
        corpus="Test",
        short_citation="Second",
        external_url="https://example.test/second",
        license_note="Test metadata.",
        access_policy="open_metadata",
        citation_ready=False,
    )

    class Adapter:
        def __init__(self, reference: SourceReference) -> None:
            self.reference = reference

        def lookup(
            self,
            *,
            entry_id: str,
            headword: str,
            language: str,
        ) -> tuple[SourceReference, ...]:
            return (self.reference,)

    adapter = CompositeCorpusAdapter([Adapter(first), Adapter(second)])

    assert [
        reference.source_id
        for reference in adapter.lookup(
            entry_id="L1",
            headword="λόγος",
            language="ancient_greek",
        )
    ] == ["first", "second"]


def test_malformed_adapter_data_raises_validation_error(tmp_path: Path) -> None:
    """Verify malformed source data raises a corpus source validation error."""
    data_path = tmp_path / "sources.yaml"
    data_path.write_text(
        """
_meta:
  status: proof_of_concept
  provider: Test
  license_note: Test
  note: Test
entries:
  L1:
    - source_id: L1
      corpus: Test
      short_citation: Test
      external_url: not-a-url
      license_note: Test
      access_policy: open_metadata
      citation_ready: false
""",
        encoding="utf-8",
    )

    with pytest.raises(CorpusSourceDataError, match="external_url"):
        load_static_corpus_adapter(data_path)


def test_prepare_static_corpus_sources_is_not_exported() -> None:
    """Verify private corpus source preparation helpers are not exported."""
    assert "prepare_static_corpus_sources" not in corpus_pkg.__all__
    assert not hasattr(corpus_pkg, "prepare_static_corpus_sources")


def test_load_static_corpus_adapter_normalizes_paths(tmp_path: Path) -> None:
    """Verify path spellings resolve to one cached static adapter instance."""
    data_path = tmp_path / "sources.yaml"
    data_path.write_text(
        """
_meta:
  status: proof_of_concept
  provider: Test
  license_note: Test
  note: Test
entries:
  L1:
    - source_id: L1
      corpus: Test
      short_citation: Test
      external_url: https://example.test/source
      license_note: Test
      access_policy: open_metadata
      citation_ready: false
""",
        encoding="utf-8",
    )

    canonical = load_static_corpus_adapter(data_path)
    str_form = load_static_corpus_adapter(str(data_path))
    resolved_form = load_static_corpus_adapter(data_path.resolve())

    assert canonical is str_form
    assert canonical is resolved_form


def test_load_static_corpus_adapter_rejects_promoted_status(tmp_path: Path) -> None:
    """Verify promoted corpus source statuses are rejected until reviewed."""
    data_path = tmp_path / "promoted.yaml"
    data_path.write_text(
        """
_meta:
  status: stable
  provider: Test
  license_note: Test
  note: Test
entries:
  L1:
    - source_id: L1
      corpus: Test
      short_citation: Test
      external_url: https://example.test/source
      license_note: Test
      access_policy: open_metadata
      citation_ready: false
""",
        encoding="utf-8",
    )

    with pytest.raises(CorpusSourceDataError, match="status"):
        load_static_corpus_adapter(data_path)


def test_load_static_corpus_adapter_rejects_missing_meta(tmp_path: Path) -> None:
    """Verify corpus source data must declare a metadata block."""
    data_path = tmp_path / "missing_meta.yaml"
    data_path.write_text(
        """
entries:
  L1:
    - source_id: L1
      corpus: Test
      short_citation: Test
      external_url: https://example.test/source
      license_note: Test
      access_policy: open_metadata
      citation_ready: false
""",
        encoding="utf-8",
    )

    with pytest.raises(CorpusSourceDataError, match="_meta"):
        load_static_corpus_adapter(data_path)


def test_load_static_corpus_adapter_rejects_missing_meta_status(
    tmp_path: Path,
) -> None:
    """Verify corpus source data must declare an explicit metadata status."""
    data_path = tmp_path / "missing_status.yaml"
    data_path.write_text(
        """
_meta:
  provider: Test
  license_note: Test
  note: Test
entries:
  L1:
    - source_id: L1
      corpus: Test
      short_citation: Test
      external_url: https://example.test/source
      license_note: Test
      access_policy: open_metadata
      citation_ready: false
""",
        encoding="utf-8",
    )

    with pytest.raises(CorpusSourceDataError, match="status"):
        load_static_corpus_adapter(data_path)


def test_load_static_corpus_adapter_rejects_normalized_duplicate_entry_ids(
    tmp_path: Path,
) -> None:
    """Verify normalized duplicate entry IDs raise a data validation error."""
    data_path = tmp_path / "duplicate_entries.yaml"
    data_path.write_text(
        """
_meta:
  status: proof_of_concept
  provider: Test
  license_note: Test
  note: Test
entries:
  L1:
    - source_id: L1
      corpus: Test
      short_citation: Test
      external_url: https://example.test/source
      license_note: Test
      access_policy: open_metadata
      citation_ready: false
  " L1 ":
    - source_id: L1-duplicate
      corpus: Test
      short_citation: Test duplicate
      external_url: https://example.test/source-duplicate
      license_note: Test
      access_policy: open_metadata
      citation_ready: false
""",
        encoding="utf-8",
    )

    with pytest.raises(CorpusSourceDataError, match="Duplicate corpus source entry id"):
        load_static_corpus_adapter(data_path)


def test_source_reference_rejects_overlong_external_url() -> None:
    """Verify SourceReference rejects external URLs that exceed length limits."""
    overlong_url = "https://example.test/" + ("a" * 4000)
    with pytest.raises(ValueError):
        SourceReference(
            source_id="L1",
            corpus="Test",
            short_citation="Test",
            external_url=overlong_url,
            license_note="Test.",
            access_policy="open_metadata",
            citation_ready=False,
        )


def test_source_reference_rejects_overlong_short_citation() -> None:
    """Verify SourceReference rejects short citations over 200 characters."""
    with pytest.raises(ValueError, match="short_citation"):
        SourceReference(
            source_id="L1",
            corpus="Test",
            short_citation="x" * 201,
            external_url="https://example.test/source",
            license_note="Test.",
            access_policy="open_metadata",
            citation_ready=False,
        )


def test_source_reference_rejects_short_citation_over_word_limit() -> None:
    """Verify SourceReference rejects short citations over 40 words."""
    with pytest.raises(ValueError, match="short_citation"):
        SourceReference(
            source_id="L1",
            corpus="Test",
            short_citation=" ".join("x" for _ in range(41)),
            external_url="https://example.test/source",
            license_note="Test.",
            access_policy="open_metadata",
            citation_ready=False,
        )


def test_safe_lookup_returns_empty_tuple_for_none_entry_id() -> None:
    """Verify safe_lookup skips adapter calls when entry_id is None."""
    class _Adapter:
        def lookup(self, *, entry_id, headword, language):
            raise AssertionError("should not be called when entry_id is None")

    references = safe_lookup(
        _Adapter(),
        entry_id=None,
        headword="λόγος",
        language="ancient_greek",
        logger=logging.getLogger("test"),
    )
    assert references == ()


def test_safe_lookup_degrades_adapter_failure_to_empty(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify safe_lookup logs adapter failures and returns an empty tuple."""
    class _Adapter:
        def lookup(self, *, entry_id, headword, language):
            raise RuntimeError("adapter unavailable")

    logger = logging.getLogger("test.safe_lookup")
    with caplog.at_level("WARNING", logger="test.safe_lookup"):
        references = safe_lookup(
            _Adapter(),
            entry_id="L1",
            headword="λόγος",
            language="ancient_greek",
            logger=logger,
        )
    assert references == ()
    assert "Corpus adapter lookup failed" in caplog.text


def test_safe_lookup_returns_adapter_references_on_success() -> None:
    """Verify safe_lookup returns adapter references on successful lookup."""
    reference = SourceReference(
        source_id="L1",
        corpus="Test",
        short_citation="Test",
        external_url="https://example.test/source",
        license_note="Test.",
        access_policy="open_metadata",
        citation_ready=False,
    )

    class _Adapter:
        def lookup(self, *, entry_id, headword, language):
            assert entry_id == "L1"
            return (reference,)

    references = safe_lookup(
        _Adapter(),
        entry_id="L1",
        headword="λόγος",
        language="ancient_greek",
        logger=logging.getLogger("test"),
    )
    assert references == (reference,)


def test_source_reference_rejects_text_fields() -> None:
    """Verify SourceReference rejects embedded source text fields."""
    with pytest.raises(ValueError, match="source text fields"):
        SourceReference.model_validate(
            {
                "source_id": "L1",
                "corpus": "Test",
                "short_citation": "Test",
                "external_url": "https://example.test/source",
                "license_note": "Test metadata.",
                "access_policy": "open_metadata",
                "citation_ready": False,
                "source_text": "not allowed",
            }
        )
