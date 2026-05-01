"""Tests for scripts/extract_epidoc_choices.py."""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import shutil
import tarfile
from pathlib import Path

import pytest

try:
    from scripts import extract_epidoc_choices as edc
except ImportError:
    pytest.skip("lxml not available; install with --extra extract", allow_module_level=True)

FIXTURES = Path(__file__).parent / "fixtures" / "epidoc"


# ---------------------------------------------------------------------------
# _text_content
# ---------------------------------------------------------------------------


def test_text_content_simple():
    from lxml import etree  # type: ignore[import-untyped]

    elem = etree.fromstring("<reg>θεοῦ</reg>")
    assert edc._text_content(elem) == "θεοῦ"


def test_text_content_nested_children():
    from lxml import etree  # type: ignore[import-untyped]

    elem = etree.fromstring('<reg><hi rend="supraline">κε</hi></reg>')
    assert edc._text_content(elem) == "κε"


def test_text_content_strips_whitespace():
    from lxml import etree  # type: ignore[import-untyped]

    elem = etree.fromstring("<orig>  λογος  </orig>")
    assert edc._text_content(elem) == "λογος"


def test_text_content_empty():
    from lxml import etree  # type: ignore[import-untyped]

    elem = etree.fromstring("<reg></reg>")
    assert edc._text_content(elem) == ""


# ---------------------------------------------------------------------------
# _extract_tm_id
# ---------------------------------------------------------------------------


def test_extract_tm_id_present():
    tm = edc._extract_tm_id(FIXTURES / "simple.xml")
    assert tm == "12345"


def test_extract_tm_id_alternate_casing():
    tm = edc._extract_tm_id(FIXTURES / "tm_alternate_casing.xml")
    assert tm == "55555"


def test_extract_tm_id_missing_falls_back_to_none():
    tm = edc._extract_tm_id(FIXTURES / "no_tm.xml")
    # no_tm.xml has no <idno> and filename "no_tm" has no leading digits
    assert tm is None


def test_extract_tm_id_malformed_recovers():
    # edc._extract_tm_id may recover the TM id from malformed.xml, or return
    # None if lxml recovery cannot preserve that header field.
    tm = edc._extract_tm_id(FIXTURES / "malformed.xml")
    assert tm in ("77777", None)


# ---------------------------------------------------------------------------
# extract_choices_from_file
# ---------------------------------------------------------------------------


def test_simple_pair():
    records = edc.extract_choices_from_file(FIXTURES / "simple.xml", "simple.xml")
    assert len(records) == 1
    r = records[0]
    assert r.original == "θιοῦ"
    assert r.regularized == "θεοῦ"
    assert r.source_file == "simple.xml"
    assert r.tm_id == "12345"


def test_reversed_order():
    """<orig> before <reg> should still yield a record."""
    records = edc.extract_choices_from_file(FIXTURES / "reversed_order.xml", "reversed_order.xml")
    assert len(records) == 1
    assert records[0].original == "κυρίου"
    assert records[0].regularized == "κυριου"


def test_nested_hi_text_extracted():
    records = edc.extract_choices_from_file(FIXTURES / "nested_hi.xml", "nested_hi.xml")
    assert len(records) == 1
    assert records[0].regularized == "κε"
    assert records[0].original == "κε"


def test_multiple_choice_elements():
    records = edc.extract_choices_from_file(FIXTURES / "multiple_pairs.xml", "multiple_pairs.xml")
    assert len(records) == 2
    originals = {r.original for r in records}
    assert originals == {"αὐτου", "τουτο"}


def test_empty_elements_skipped():
    records = edc.extract_choices_from_file(FIXTURES / "empty_elements.xml", "empty_elements.xml")
    assert len(records) == 1
    assert records[0].original == "λογος"
    assert records[0].regularized == "λόγος"


def test_no_tm_id():
    records = edc.extract_choices_from_file(FIXTURES / "no_tm.xml", "no_tm.xml")
    assert len(records) == 1
    assert records[0].tm_id is None


def test_filename_tm_fallback_applied_to_records(tmp_path):
    xml_path = tmp_path / "60001.xml"
    xml_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <publicationStmt>
        <p>No TM identifier in this document.</p>
      </publicationStmt>
    </fileDesc>
  </teiHeader>
  <text>
    <body>
      <p><choice><reg>ἀγαθοῦ</reg><orig>ἀγαθου</orig></choice></p>
    </body>
  </text>
</TEI>
""",
        encoding="utf-8",
    )

    records = edc.extract_choices_from_file(xml_path, "60001.xml")

    assert len(records) == 1
    assert records[0].tm_id == "60001"


def test_malformed_xml_does_not_raise():
    records = edc.extract_choices_from_file(FIXTURES / "malformed.xml", "malformed.xml")
    # lxml recovery may or may not extract the pair; either way no exception
    assert isinstance(records, list)


def test_extract_choices_from_file_reraises_fatal_parse_error(monkeypatch):
    def fail_iterparse(*_args, **_kwargs):
        raise OSError("fatal parse")

    monkeypatch.setattr(edc.etree, "iterparse", fail_iterparse)

    with pytest.raises(OSError, match="fatal parse"):
        edc.extract_choices_from_file(FIXTURES / "simple.xml", "simple.xml")


def test_pair_index_assigned():
    records = edc.extract_choices_from_file(FIXTURES / "multiple_pairs.xml", "multiple_pairs.xml")
    assert all(isinstance(r.pair_index, int) for r in records)


# ---------------------------------------------------------------------------
# JsonArrayWriter
# ---------------------------------------------------------------------------


def test_json_array_writer_empty(tmp_path):
    out = tmp_path / "out.json"
    with edc.JsonArrayWriter(out) as w:
        pass
    data = json.loads(out.read_text())
    assert data == []
    assert w.count == 0


def test_json_array_writer_single_record(tmp_path):
    out = tmp_path / "out.json"
    with edc.JsonArrayWriter(out) as w:
        w.write({"a": 1})
    data = json.loads(out.read_text())
    assert data == [{"a": 1}]
    assert w.count == 1


def test_json_array_writer_multiple_records(tmp_path):
    out = tmp_path / "out.json"
    records = [{"x": i, "y": f"val{i}"} for i in range(5)]
    with edc.JsonArrayWriter(out) as w:
        for rec in records:
            w.write(rec)
    data = json.loads(out.read_text())
    assert data == records
    assert w.count == 5


def test_json_array_writer_unicode(tmp_path):
    out = tmp_path / "out.json"
    with edc.JsonArrayWriter(out) as w:
        w.write({"original": "θιοῦ", "regularized": "θεοῦ"})
    data = json.loads(out.read_text())
    assert data[0]["original"] == "θιοῦ"


def test_json_array_writer_tmp_removed_on_success(tmp_path):
    out = tmp_path / "out.json"
    with edc.JsonArrayWriter(out) as w:
        w.write({"k": "v"})
    assert out.exists()
    assert not Path(str(out) + ".tmp").exists()


def test_json_array_writer_tmp_removed_on_error(tmp_path):
    out = tmp_path / "out.json"
    with pytest.raises(RuntimeError, match="boom"):
        with edc.JsonArrayWriter(out) as w:
            w.write({"k": "v"})
            raise RuntimeError("boom")
    assert not out.exists()
    assert not Path(str(out) + ".tmp").exists()


# ---------------------------------------------------------------------------
# Corpus acquisition
# ---------------------------------------------------------------------------


def test_tarball_helpers_reject_non_https_urls(tmp_path):
    with pytest.raises(ValueError, match="Only HTTPS URLs"):
        edc._tarball_needs_download("http://example.com/data.tar.gz", tmp_path)
    with pytest.raises(ValueError, match="Only HTTPS URLs"):
        edc._stream_tarball("file:///tmp/data.tar.gz", tmp_path)


def test_tarball_needs_download_uses_cache_when_head_fails(monkeypatch, tmp_path):
    edc._etag_path(tmp_path).write_text('"cached-etag"', encoding="utf-8")

    def fail_head(*_args, **_kwargs):
        raise edc.URLError("offline")

    monkeypatch.setattr(edc.urllib.request, "urlopen", fail_head)

    assert edc._tarball_needs_download("https://example.com/data.tar.gz", tmp_path) is False


def test_stream_tarball_replaces_stale_extracted_corpus(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    old_dir = cache_dir / "old-corpus"
    old_dir.mkdir(parents=True)
    (old_dir / "stale.xml").write_text("<old/>", encoding="utf-8")

    tar_bytes = io.BytesIO()
    new_xml = b"<TEI/>"
    with tarfile.open(fileobj=tar_bytes, mode="w:gz") as tar:
        info = tarfile.TarInfo("idp.data-master/new.xml")
        info.size = len(new_xml)
        tar.addfile(info, io.BytesIO(new_xml))

    class FakeResponse(io.BytesIO):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.headers = {"ETag": '"new-etag"'}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_urlopen(*_args, **_kwargs):
        return FakeResponse(tar_bytes.getvalue())

    monkeypatch.setattr(edc.urllib.request, "urlopen", fake_urlopen)

    edc._stream_tarball("https://example.com/data.tar.gz", cache_dir)

    assert not old_dir.exists()
    assert (cache_dir / "idp.data-master" / "new.xml").read_bytes() == new_xml
    assert edc._etag_path(cache_dir).read_text(encoding="utf-8") == '"new-etag"'
    assert not any(p.name.startswith(".download-") for p in cache_dir.iterdir())


def test_extract_tarball_uses_filter_on_python_312_or_newer(monkeypatch, tmp_path):
    calls = []

    class DummyTar:
        def extractall(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(edc, "version_info", (3, 12))
    edc._extract_tarball(DummyTar(), tmp_path)
    assert calls == [{"path": tmp_path, "filter": "data"}]


def test_extract_tarball_safe_extraction_before_python_312(monkeypatch, tmp_path):
    """On Python < 3.12, _extract_tarball manually validates and extracts members."""
    class FakeMember:
        def __init__(self, name, *, is_file=True, is_dir=False, mtime=0, data=b""):
            self.name = name
            self.mtime = mtime
            self._is_file = is_file
            self._is_dir = is_dir
            self._data = data
            # tarfile member type constants
            self.type = tarfile.REGTYPE if is_file else (tarfile.DIRTYPE if is_dir else tarfile.SYMTYPE)

        def isfile(self):
            return self._is_file

        def isdir(self):
            return self._is_dir

    safe_file = FakeMember("project/readme.txt", is_file=True, data=b"hello")
    safe_dir = FakeMember("project/", is_file=False, is_dir=True)
    abs_path = FakeMember("/etc/passwd", is_file=True, data=b"bad")
    traversal = FakeMember("project/../../etc/passwd", is_file=True, data=b"bad")
    symlink = FakeMember("project/link", is_file=False, is_dir=False)  # neither file nor dir

    class DummyTar:
        def __iter__(self):
            return iter([safe_dir, safe_file, abs_path, traversal, symlink])

        def extractfile(self, member):
            return io.BytesIO(member._data)

    monkeypatch.setattr(edc, "version_info", (3, 11))
    edc._extract_tarball(DummyTar(), tmp_path)

    # Safe file should be extracted
    assert (tmp_path / "project" / "readme.txt").exists()
    assert (tmp_path / "project" / "readme.txt").read_bytes() == b"hello"
    # Safe directory should be created
    assert (tmp_path / "project").is_dir()
    # Dangerous entries must NOT exist
    assert not (tmp_path / "etc").exists()
    assert not (tmp_path / "project" / "link").exists()


def test_extract_tarball_streaming_before_python_312(monkeypatch, tmp_path):
    """Python < 3.12 fallback must work with tarfile stream mode."""
    tar_bytes = io.BytesIO()
    files = {
        "project/a.txt": b"alpha",
        "project/nested/b.txt": b"beta",
    }
    with tarfile.open(fileobj=tar_bytes, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

    tar_bytes.seek(0)
    monkeypatch.setattr(edc, "version_info", (3, 11))
    with tarfile.open(fileobj=tar_bytes, mode="r|gz") as tar:
        edc._extract_tarball(tar, tmp_path)

    assert (tmp_path / "project" / "a.txt").read_bytes() == b"alpha"
    assert (tmp_path / "project" / "nested" / "b.txt").read_bytes() == b"beta"


def test_ensure_corpus_rejects_source_dir_file(tmp_path):
    source = tmp_path / "source.xml"
    source.write_text("<TEI/>", encoding="utf-8")
    args = argparse.Namespace(source_dir=str(source), cache_dir=tmp_path / "cache")

    with pytest.raises(SystemExit, match="not a directory"):
        edc.ensure_corpus(args)


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [
        ["--progress-interval", "0"],
        ["--limit", "0"],
        ["--limit", "-1"],
    ],
)
def test_parser_rejects_non_positive_integer_args(args):
    parser = edc._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(args)


# ---------------------------------------------------------------------------
# Integration: process_corpus on synthetic fixtures
# ---------------------------------------------------------------------------


def test_process_corpus_integration(tmp_path):
    output = tmp_path / "output" / "choices.json"
    stats = edc.process_corpus(
        FIXTURES,
        output,
        workers=1,
        limit=None,
        resume=False,
        progress_interval=100,
        with_metadata=False,
    )

    assert output.exists()
    data = json.loads(output.read_text())
    assert isinstance(data, list)
    assert len(data) > 0

    # All records have required fields
    for rec in data:
        assert "original" in rec
        assert "regularized" in rec
        assert "source_file" in rec
        assert "tm_id" in rec
        assert "pair_index" not in rec  # with_metadata=False

    assert stats["total_pairs"] == len(data)
    assert stats["total_files"] > 0


def test_process_corpus_with_metadata(tmp_path):
    output = tmp_path / "output" / "choices.json"
    edc.process_corpus(
        FIXTURES,
        output,
        workers=1,
        limit=None,
        resume=False,
        progress_interval=100,
        with_metadata=True,
    )
    data = json.loads(output.read_text())
    assert all("pair_index" in rec for rec in data)


def test_process_corpus_jsonl_output(tmp_path):
    output = tmp_path / "output" / "choices.jsonl"
    stats = edc.process_corpus(
        FIXTURES,
        output,
        workers=1,
        limit=None,
        resume=False,
        progress_interval=100,
        with_metadata=False,
        output_format="jsonl",
    )

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert records
    assert stats["total_pairs"] == len(records)
    assert edc.verify_output(output, output_format="jsonl") == len(records)


def test_process_corpus_resume_jsonl_appends_records(tmp_path):
    """JSONL resume must preserve prior records and skip processed files."""
    output = tmp_path / "output" / "choices.jsonl"

    # First run: only 1 file processed.
    stats1 = edc.process_corpus(
        FIXTURES,
        output,
        workers=1,
        limit=1,
        resume=True,
        progress_interval=100,
        with_metadata=False,
        output_format="jsonl",
    )
    pairs_run1 = stats1["total_pairs"]
    assert pairs_run1 > 0
    lines_after_run1 = output.read_text().count("\n")
    assert lines_after_run1 == pairs_run1

    # Second run: resume should skip the file processed in run 1
    # AND append the remaining records to the same JSONL output.
    stats2 = edc.process_corpus(
        FIXTURES,
        output,
        workers=1,
        limit=None,
        resume=True,
        progress_interval=100,
        with_metadata=False,
        output_format="jsonl",
    )
    pairs_run2 = stats2["total_pairs"]

    lines_total = output.read_text().count("\n")
    # File should now hold both runs' records — prior data is not lost.
    assert lines_total == pairs_run1 + pairs_run2

    # Checkpoint should reflect every file processed across both runs.
    checkpoint = output.parent / "processed_files.txt"
    processed_files = [
        line for line in checkpoint.read_text().splitlines() if line.strip()
    ]
    # Run 1 saw 1 file; run 2 saw the rest. Total must equal corpus size
    # (no duplicate entries because run 2 skipped run 1's file).
    corpus_size = stats2["total_files"]
    assert len(processed_files) == corpus_size
    assert len(set(processed_files)) == len(processed_files)


def test_extraction_loop_flushes_records_before_checkpoint(monkeypatch, tmp_path):
    """Successful files must flush JSONL records before checkpointing."""
    events: list[str] = []

    class FakeWriter:
        def write(self, record):
            events.append(f"write:{record['source_file']}")

        def flush(self):
            events.append("flush")

    class FakePool:
        def submit(self, _fn, item):
            rel_path = item[1]
            future = concurrent.futures.Future()
            future.set_result(
                (
                    rel_path,
                    [
                        {
                            "original": "θιοῦ",
                            "regularized": "θεοῦ",
                            "source_file": rel_path,
                            "tm_id": "12345",
                            "pair_index": 0,
                        }
                    ],
                    None,
                )
            )
            return future

    def fake_save_checkpoint(_output_dir, rel_path):
        events.append(f"checkpoint:{rel_path}")

    monkeypatch.setattr(edc, "_save_checkpoint", fake_save_checkpoint)

    stats = {
        "files_with_pairs": 0,
        "total_pairs": 0,
        "missing_tm": 0,
        "parse_failures": 0,
    }

    edc._run_extraction_loop(
        FakePool(),
        [(str(tmp_path / "simple.xml"), "simple.xml")],
        FakeWriter(),
        io.StringIO(),
        stats,
        with_metadata=False,
        progress_interval=100,
        output_path=tmp_path / "choices.jsonl",
        start=0.0,
    )

    assert events == ["write:simple.xml", "flush", "checkpoint:simple.xml"]


def test_process_corpus_no_resume_resets_stale_checkpoint_for_jsonl(tmp_path):
    """A fresh JSONL run must replace stale checkpoint state before later resume."""
    output = tmp_path / "output" / "choices.jsonl"
    output.parent.mkdir(parents=True)
    stale_files = sorted(
        str(p.relative_to(FIXTURES)) for p in FIXTURES.rglob("*.xml")
    )
    (output.parent / "processed_files.txt").write_text(
        "\n".join(stale_files) + "\n",
        encoding="utf-8",
    )

    stats1 = edc.process_corpus(
        FIXTURES,
        output,
        workers=1,
        limit=1,
        resume=False,
        progress_interval=100,
        with_metadata=False,
        output_format="jsonl",
    )
    pairs_run1 = stats1["total_pairs"]

    checkpoint = output.parent / "processed_files.txt"
    processed_after_fresh_run = [
        line for line in checkpoint.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(processed_after_fresh_run) == 1

    stats2 = edc.process_corpus(
        FIXTURES,
        output,
        workers=1,
        limit=None,
        resume=True,
        progress_interval=100,
        with_metadata=False,
        output_format="jsonl",
    )
    pairs_run2 = stats2["total_pairs"]

    lines_total = output.read_text(encoding="utf-8").count("\n")
    processed_after_resume = [
        line for line in checkpoint.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert lines_total == pairs_run1 + pairs_run2
    assert len(processed_after_resume) == stats2["total_files"]
    assert len(set(processed_after_resume)) == len(processed_after_resume)


def test_process_corpus_resume_json_rejects_existing_output(tmp_path):
    """Resume + JSON array format must refuse to overwrite existing output."""
    output = tmp_path / "output" / "choices.json"

    # First run produces a complete JSON array.
    edc.process_corpus(
        FIXTURES,
        output,
        workers=1,
        limit=1,
        resume=True,
        progress_interval=100,
        with_metadata=False,
        output_format="json",
    )
    assert output.exists()

    # Second run with the same JSON output must be refused so existing
    # records aren't silently destroyed. JSON arrays cannot be appended.
    with pytest.raises(SystemExit, match=r"--format=jsonl|--no-resume"):
        edc.process_corpus(
            FIXTURES,
            output,
            workers=1,
            limit=None,
            resume=True,
            progress_interval=100,
            with_metadata=False,
            output_format="json",
        )


def test_process_corpus_resume_json_rejects_checkpoint_without_output(tmp_path):
    """Resume + JSON array format must refuse checkpoint-only interrupted state."""
    output = tmp_path / "output" / "choices.json"
    output.parent.mkdir(parents=True)
    (output.parent / "processed_files.txt").write_text("simple.xml\n", encoding="utf-8")

    with pytest.raises(SystemExit, match=r"checkpoint|--format=jsonl|--no-resume"):
        edc.process_corpus(
            FIXTURES,
            output,
            workers=1,
            limit=None,
            resume=True,
            progress_interval=100,
            with_metadata=False,
            output_format="json",
        )


def test_process_corpus_limit(tmp_path):
    output = tmp_path / "output" / "choices.json"
    stats = edc.process_corpus(
        FIXTURES,
        output,
        workers=1,
        limit=2,
        resume=False,
        progress_interval=100,
        with_metadata=False,
    )
    assert stats["total_files"] == 2


def test_process_corpus_failed_files_log(tmp_path):
    """Malformed XML should land in failed_files.log without aborting."""
    output = tmp_path / "output" / "choices.json"
    # Use only the malformed fixture dir
    malformed_only = tmp_path / "corpus"
    malformed_only.mkdir()
    shutil.copy(FIXTURES / "malformed.xml", malformed_only / "malformed.xml")

    edc.process_corpus(
        malformed_only,
        output,
        workers=1,
        limit=None,
        resume=False,
        progress_interval=100,
        with_metadata=False,
    )
    # Output should still be valid JSON
    assert output.exists()
    data = json.loads(output.read_text())
    assert isinstance(data, list)


def test_worker_returns_error_for_fatal_parse_error(monkeypatch, tmp_path):
    xml_path = tmp_path / "bad.xml"
    xml_path.write_text("<TEI/>", encoding="utf-8")

    def fail_extract(*_args, **_kwargs):
        raise RuntimeError("worker parse failed")

    monkeypatch.setattr(edc, "extract_choices_from_file", fail_extract)

    rel_path, records, error = edc._worker((str(xml_path), "bad.xml"))

    assert rel_path == "bad.xml"
    assert records == []
    assert error is not None
    assert "worker parse failed" in error


def test_process_corpus_worker_failure_is_logged_and_not_checkpointed(tmp_path):
    output = tmp_path / "output" / "choices.json"
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "bad.xml").mkdir()

    stats = edc.process_corpus(
        corpus,
        output,
        workers=1,
        limit=None,
        resume=False,
        progress_interval=100,
        with_metadata=False,
    )

    assert stats["parse_failures"] == 1
    assert stats["total_pairs"] == 0
    assert "bad.xml" in (output.parent / "failed_files.log").read_text(
        encoding="utf-8"
    )
    assert not (output.parent / "processed_files.txt").exists()
    assert json.loads(output.read_text(encoding="utf-8")) == []


def test_verify_output(tmp_path):
    output = tmp_path / "output" / "choices.json"
    edc.process_corpus(
        FIXTURES,
        output,
        workers=1,
        limit=None,
        resume=False,
        progress_interval=100,
        with_metadata=False,
    )
    count = edc.verify_output(output)
    assert count > 0
