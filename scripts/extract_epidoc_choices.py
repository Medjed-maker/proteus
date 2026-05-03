#!/usr/bin/env python3
"""Extract <choice><reg>/<orig> pairs from papyri.info EpiDoc XML files.

Downloads the papyri/idp.data tarball (or uses an existing local checkout),
streams through all EpiDoc XML files, and writes a JSON array of spelling-
variant pairs suitable as training data for Proteus phonological distance matrices.

The generated records are training data, not runtime orthographic-note data.
Do not load this output directly in the note builder: promote candidate
correspondences to ``data/languages/ancient_greek/orthography/*.yaml`` only
after source review.

Usage:

    # Download corpus and extract (first run)
    uv run --extra extract python scripts/extract_epidoc_choices.py

    # Use existing local checkout
    uv run --extra extract python scripts/extract_epidoc_choices.py \\
        --source-dir /path/to/idp.data

    # Smoke test (first 100 files only)
    uv run --extra extract python scripts/extract_epidoc_choices.py --limit 100

Output schema (per record):

    {
        "original":     str,   # text content of <orig>
        "regularized":  str,   # text content of <reg>
        "source_file":  str,   # relative path within corpus
        "tm_id":        str | null
    }
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from sys import version_info
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse

try:
    from lxml import etree  # type: ignore[import-untyped]
except ImportError as exc:
    raise SystemExit(
        "lxml is required. Install with: uv run --extra extract python scripts/extract_epidoc_choices.py"
    ) from exc

if __package__ in {None, ""}:
    # Allow `python scripts/extract_epidoc_choices.py` to resolve the scripts package.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._cli_utils import positive_int as _positive_int

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEI_NS = "http://www.tei-c.org/ns/1.0"
_CHOICE = f"{{{TEI_NS}}}choice"
_REG = f"{{{TEI_NS}}}reg"
_ORIG = f"{{{TEI_NS}}}orig"
_IDNO = f"{{{TEI_NS}}}idno"
_TEI_HEADER = f"{{{TEI_NS}}}teiHeader"

_TM_TYPE_VALUES = {"tm", "trismegistos"}

TARBALL_URL = "https://github.com/papyri/idp.data/archive/refs/heads/master.tar.gz"
DEFAULT_OUTPUT = Path("data/training/epidoc_choices.json")
DEFAULT_CACHE_DIR = Path("data/external/idp.data")

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChoiceRecord:
    original: str
    regularized: str
    source_file: str
    tm_id: str | None
    pair_index: int = 0


# ---------------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------------


def _text_content(elem: Any) -> str:
    """Return the concatenated text of *elem* and all its descendants."""
    return "".join(elem.itertext()).strip()


_LEADING_DIGITS_RE = re.compile(r"\d+")


def _filename_tm_fallback(path: Path) -> str | None:
    """Extract the first contiguous run of digits from *path*'s stem.

    Examples: "60001.xml" → "60001"; "abc60001def200.xml" → "60001".
    Returns None when the stem contains no digits.
    """
    m = _LEADING_DIGITS_RE.search(path.stem)
    return m.group(0) if m else None


def _try_tm_from_idno(elem: Any) -> str | None:
    """Return the TM id from a TM-typed <idno> element, or None."""
    type_attr = (elem.get("type") or "").lower()
    if type_attr not in _TM_TYPE_VALUES:
        return None
    value = (elem.text or "").strip()
    return value if value else None


def _extract_choice_records(
    elem: Any,
    source_file: str,
    resolved_tm_id: str | None,
) -> list[ChoiceRecord]:
    """Return ChoiceRecord objects from a single <choice> element."""
    regs: list[str] = []
    origs: list[str] = []
    for child in elem:
        child_tag = child.tag
        if child_tag in (_REG, "reg"):
            text = _text_content(child)
            if text:
                regs.append(text)
        elif child_tag in (_ORIG, "orig"):
            text = _text_content(child)
            if text:
                origs.append(text)

    if len(origs) != len(regs):
        logger.warning(
            "Parse error in %s: Mismatch: %d <orig> vs %d <reg>",
            source_file, len(origs), len(regs),
        )
        return []

    return [
        ChoiceRecord(
            original=orig_text,
            regularized=reg_text,
            source_file=source_file,
            tm_id=resolved_tm_id,
            pair_index=idx,
        )
        for idx, (orig_text, reg_text) in enumerate(zip(origs, regs, strict=True))
    ]


def _scan_xml_once(
    path: Path,
    source_file: str,
) -> tuple[str | None, list[ChoiceRecord]]:
    """Single iterparse pass: extract TM id and all <choice> reg/orig pairs.

    In standard EpiDoc layout the teiHeader precedes the text body, so
    tm_id is resolved before any <choice> element is encountered.
    """
    tm_id: str | None = None
    fallback_tm_id = _filename_tm_fallback(path)
    past_header = False
    records: list[ChoiceRecord] = []

    # Harden iterparse against XXE / billion-laughs / external network
    # attacks. The papyri corpus is a trusted source, but defense in
    # depth costs nothing here.
    context = etree.iterparse(
        str(path),
        events=("start", "end"),
        recover=True,
        huge_tree=False,
        resolve_entities=False,
        load_dtd=False,
        no_network=True,
    )
    choice_depth = 0

    for event, elem in context:
        tag = elem.tag
        is_choice = tag in (_CHOICE, "choice")

        if event == "start":
            if is_choice:
                choice_depth += 1
            continue

        # --- end event ---

        if not past_header:
            if tag in (_IDNO, "idno"):
                tm_id = tm_id or _try_tm_from_idno(elem)
                elem.clear()
                continue
            if tag in (_TEI_HEADER, "teiHeader"):
                past_header = True
                elem.clear()
                continue

        if is_choice:
            choice_depth -= 1
            resolved = tm_id if tm_id is not None else fallback_tm_id
            records.extend(_extract_choice_records(elem, source_file, resolved))
            elem.clear()
        elif choice_depth == 0:
            # Safe to free memory only when outside any <choice>.
            # We deliberately keep this simple (clear() only) — the
            # full lxml getprevious/remove dance was hard to reason
            # about and offered marginal gains for typical EpiDoc files.
            elem.clear()

    if tm_id is None:
        tm_id = fallback_tm_id

    return tm_id, records


def _extract_tm_id(path: Path) -> str | None:
    """Scan *path* for a Trismegistos ID (public API wrapper over _scan_xml_once)."""
    try:
        tm_id, _ = _scan_xml_once(path, path.name)
        return tm_id
    except (OSError, etree.Error, ValueError, TypeError) as exc:
        logger.warning(
            "Failed to extract TM ID from %s; falling back to filename",
            path,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return _filename_tm_fallback(path)


def extract_choices_from_file(path: Path, source_file: str) -> list[ChoiceRecord]:
    """Parse *path* and return all <choice> reg/orig pairs as ChoiceRecord objects."""
    _, records = _scan_xml_once(path, source_file)
    return records


# ---------------------------------------------------------------------------
# Streaming JSON writer
# ---------------------------------------------------------------------------


class JsonArrayWriter:
    """Write a JSON array incrementally to *path*, renaming from .tmp on close."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._tmp = Path(str(path) + ".tmp")
        self._fh: Any = None
        self._first = True
        self._count = 0

    def __enter__(self) -> "JsonArrayWriter":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._tmp.open("w", encoding="utf-8")
        self._fh.write("[\n")
        return self

    def write(self, record: dict[str, Any]) -> None:
        if not self._first:
            self._fh.write(",\n")
        self._fh.write(json.dumps(record, ensure_ascii=False))
        self._first = False
        self._count += 1

    def flush(self) -> None:
        if self._fh:
            self._fh.flush()

    def __exit__(self, exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        if self._fh:
            try:
                if exc_type is None:
                    self._fh.write("\n]\n")
            finally:
                self._fh.close()
        if exc_type is None:
            self._tmp.rename(self._path)
        else:
            try:
                self._tmp.unlink(missing_ok=True)
            except OSError:
                logger.debug(
                    "Failed to remove temporary output file %s",
                    self._tmp,
                    exc_info=True,
                )

    @property
    def count(self) -> int:
        return self._count


class JsonLinesWriter:
    """Write one JSON object per line to *path*.

    By default, writes to a sibling ``.tmp`` and atomically renames on close.
    When *append* is True, writes directly to *path* in append mode — this
    is what enables the ``--resume`` flow to keep records from prior runs.
    """

    def __init__(self, path: Path, *, append: bool = False) -> None:
        self._path = path
        self._append = append
        self._tmp = Path(str(path) + ".tmp")
        self._fh: Any = None
        self._count = 0

    def __enter__(self) -> "JsonLinesWriter":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._append:
            # Append directly to the destination file. No tmp/rename dance:
            # JSONL is line-delimited and an interrupted run leaves the
            # already-written lines parseable.
            self._fh = self._path.open("a", encoding="utf-8")
        else:
            self._fh = self._tmp.open("w", encoding="utf-8")
        return self

    def write(self, record: dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=False))
        self._fh.write("\n")
        self._count += 1

    def flush(self) -> None:
        if self._fh:
            self._fh.flush()

    def __exit__(self, exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        if self._fh:
            self._fh.close()
        if self._append:
            return
        if exc_type is None:
            self._tmp.rename(self._path)
        else:
            try:
                self._tmp.unlink(missing_ok=True)
            except OSError:
                logger.debug(
                    "Failed to remove temporary output file %s",
                    self._tmp,
                    exc_info=True,
                )

    @property
    def count(self) -> int:
        return self._count


# ---------------------------------------------------------------------------
# Corpus acquisition
# ---------------------------------------------------------------------------


def _etag_path(cache_dir: Path) -> Path:
    return cache_dir / ".etag"


def _validate_https_url(url: str) -> None:
    """Raise ValueError unless *url* uses the HTTPS scheme."""
    if urlparse(url).scheme != "https":
        raise ValueError(f"Only HTTPS URLs are supported: {url}")


def _extract_tarball(tar: tarfile.TarFile, cache_dir: Path) -> None:
    """Extract *tar* to *cache_dir* using Python-version-compatible filtering."""
    if version_info >= (3, 12):
        tar.extractall(path=cache_dir, filter="data")
    else:
        # Manual safe extraction for Python < 3.12:
        # Reject absolute paths, path traversal, and non-regular entries.
        resolved_cache = cache_dir.resolve()
        for member in tar:
            # Normalize using PurePosixPath: tar paths always use '/' regardless
            # of the host OS, so os.sep-based splitting would miss traversal
            # attempts on Windows.
            member_path = str(PurePosixPath(member.name))
            # Reject absolute paths
            if PurePosixPath(member_path).is_absolute():
                logger.warning("Skipping absolute path in tarball: %s", member.name)
                continue
            # Reject path traversal via ..
            if ".." in PurePosixPath(member_path).parts:
                logger.warning(
                    "Skipping path traversal entry in tarball: %s", member.name
                )
                continue
            # Ensure the resolved target stays within cache_dir.
            target = (resolved_cache / member_path).resolve()
            try:
                target.relative_to(resolved_cache)
            except ValueError:
                logger.warning(
                    "Skipping entry escaping cache dir: %s", member.name
                )
                continue
            # Allow only regular files and directories
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is not None:
                    with source, open(target, "wb") as fh:
                        while chunk := source.read(65536):
                            fh.write(chunk)
                    # Preserve modification time
                    os.utime(target, (member.mtime, member.mtime))
            else:
                # Skip symlinks, hardlinks, device files, and other specials
                logger.warning(
                    "Skipping non-regular tarball entry (%s): %s",
                    member.type,
                    member.name,
                )


def _tarball_needs_download(url: str, cache_dir: Path) -> bool:
    """Return True if the HTTPS remote tarball is newer than the cached copy."""
    _validate_https_url(url)
    etag_file = _etag_path(cache_dir)
    if not etag_file.exists():
        return True
    cached_etag = etag_file.read_text().strip()
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as resp:
            remote_etag = resp.headers.get("ETag", "")
            return remote_etag != cached_etag
    except URLError:
        logger.warning("HEAD request failed; using cached corpus")
        return False


def _extracted_corpus_dirs(cache_dir: Path) -> list[Path]:
    """Return visible top-level corpus directories in deterministic order."""
    return sorted(
        d for d in cache_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


def _stream_tarball(url: str, cache_dir: Path) -> None:
    """Download HTTPS *url* as a streaming tarball and extract to *cache_dir*."""
    _validate_https_url(url)
    logger.info("Downloading corpus from %s …", url)
    try:
        with urllib.request.urlopen(url, timeout=300) as resp:
            etag = resp.headers.get("ETag", "")
            with tempfile.TemporaryDirectory(prefix=".download-", dir=cache_dir) as tmp:
                tmp_dir = Path(tmp)
                with tarfile.open(fileobj=resp, mode="r|gz") as tar:
                    _extract_tarball(tar, tmp_dir)
                extracted_dirs = _extracted_corpus_dirs(tmp_dir)
                if not extracted_dirs:
                    raise SystemExit("Tarball extraction produced no directories")

                for existing_dir in _extracted_corpus_dirs(cache_dir):
                    shutil.rmtree(existing_dir)
                for extracted_dir in extracted_dirs:
                    shutil.move(str(extracted_dir), str(cache_dir / extracted_dir.name))
        if etag:
            _etag_path(cache_dir).write_text(etag)
        logger.info("Corpus extracted to %s", cache_dir)
    except URLError as exc:
        raise SystemExit(f"Download failed: {exc}") from exc


def ensure_corpus(args: argparse.Namespace) -> Path:
    """Return the path to the local corpus directory."""
    if args.source_dir:
        source = Path(args.source_dir)
        if not source.exists():
            raise SystemExit(f"--source-dir not found: {source}")
        if not source.is_dir():
            raise SystemExit(f"--source-dir is not a directory: {source}")
        return source

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Find extracted directory (GitHub tarballs add a top-level directory)
    extracted_dirs = _extracted_corpus_dirs(cache_dir)
    already_extracted = bool(extracted_dirs)

    if already_extracted and not _tarball_needs_download(TARBALL_URL, cache_dir):
        logger.info("Using cached corpus at %s", extracted_dirs[0])
        return extracted_dirs[0]

    _stream_tarball(TARBALL_URL, cache_dir)

    extracted_dirs = _extracted_corpus_dirs(cache_dir)
    if not extracted_dirs:
        raise SystemExit("Tarball extraction produced no directories")
    return extracted_dirs[0]


# ---------------------------------------------------------------------------
# Resume checkpoint
# ---------------------------------------------------------------------------


def _load_checkpoint(output_dir: Path) -> set[str]:
    checkpoint = output_dir / "processed_files.txt"
    if not checkpoint.exists():
        return set()
    return set(checkpoint.read_text(encoding="utf-8").splitlines())


def _save_checkpoint(output_dir: Path, rel_path: str) -> None:
    checkpoint = output_dir / "processed_files.txt"
    with checkpoint.open("a", encoding="utf-8") as fh:
        fh.write(rel_path + "\n")


# ---------------------------------------------------------------------------
# Worker (runs in subprocess)
# ---------------------------------------------------------------------------


def _worker(args: tuple[str, str]) -> tuple[str, list[dict[str, Any]], str | None]:
    """Process a single XML file. Returns (rel_path, records, error_msg)."""
    abs_path_str, rel_path = args
    try:
        records = extract_choices_from_file(Path(abs_path_str), rel_path)
        return rel_path, [asdict(r) for r in records], None
    except Exception as exc:
        logger.debug("Parse error in %s", rel_path, exc_info=True)
        return rel_path, [], str(exc)


# ---------------------------------------------------------------------------
# Main extraction driver helpers
# ---------------------------------------------------------------------------


def _prepare_pending_files(
    corpus_dir: Path,
    output_path: Path,
    output_format: str,
    resume: bool,
    limit: int | None,
) -> tuple[list[tuple[str, str]], dict[str, Any], Path]:
    """Validate args, enumerate XML files, return (pending, initial_stats, failed_log).

    Raises SystemExit for unsupported format or unsafe JSON-resume combinations.
    """
    if output_format not in {"json", "jsonl"}:
        raise ValueError(f"Unsupported output format: {output_format}")

    checkpoint_path = output_path.parent / "processed_files.txt"

    # JSON array output cannot be safely appended to. Refuse a resume that
    # would silently destroy the prior run's records — direct the user to
    # JSONL instead.
    if (
        resume
        and output_format == "json"
        and (output_path.exists() or checkpoint_path.exists())
    ):
        raise SystemExit(
            "Resume with --format=json cannot safely use existing output "
            f"or checkpoint state ({output_path}); JSON arrays cannot be appended. "
            "Re-run with --format=jsonl to keep prior records, "
            "or pass --no-resume to start over."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    failed_log = output_path.parent / "failed_files.log"
    if not resume:
        checkpoint_path.unlink(missing_ok=True)

    done = _load_checkpoint(output_path.parent) if resume else set()

    xml_files: list[Path] = sorted(corpus_dir.rglob("*.xml"))
    if limit is not None:
        xml_files = xml_files[:limit]

    pending = [
        (str(p), str(p.relative_to(corpus_dir)))
        for p in xml_files
        if str(p.relative_to(corpus_dir)) not in done
    ]

    total = len(xml_files)
    skipped = total - len(pending)
    logger.info("Files: %d total, %d pending, %d skipped (resume)", total, len(pending), skipped)

    stats: dict[str, Any] = {
        "total_files": total,
        "files_with_pairs": 0,
        "total_pairs": 0,
        "missing_tm": 0,
        "parse_failures": 0,
    }
    return pending, stats, failed_log


def _make_writer(output_format: str, output_path: Path, resume: bool) -> Any:
    """Return the appropriate writer context manager for *output_format*."""
    if output_format == "jsonl":
        # JSONL resume opens destination in append mode to preserve prior records.
        return JsonLinesWriter(output_path, append=resume and output_path.exists())
    return JsonArrayWriter(output_path)


def _run_extraction_loop(
    pool: concurrent.futures.ProcessPoolExecutor,
    pending: list[tuple[str, str]],
    writer: Any,
    fail_fh: Any,
    stats: dict[str, Any],
    *,
    with_metadata: bool,
    progress_interval: int,
    output_path: Path,
    start: float,
) -> None:
    """Submit workers, collect futures, write records, and update *stats* in place."""
    futures = {pool.submit(_worker, item): item[1] for item in pending}
    processed = 0

    for future in concurrent.futures.as_completed(futures):
        rel_path, records, error = future.result()
        processed += 1

        if error:
            stats["parse_failures"] += 1
            fail_fh.write(f"{rel_path}\t{error}\n")
            logger.debug("Failed: %s — %s", rel_path, error)
        else:
            if records:
                stats["files_with_pairs"] += 1
            for rec in records:
                if rec["tm_id"] is None:
                    stats["missing_tm"] += 1
                if not with_metadata:
                    rec.pop("pair_index", None)
                writer.write(rec)
            writer.flush()
            _save_checkpoint(output_path.parent, rel_path)

        if processed % progress_interval == 0:
            elapsed = time.monotonic() - start
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = (len(pending) - processed) / rate if rate > 0 else 0
            logger.info(
                "Progress: %d/%d files (%.0f/s, ETA %.0fs)",
                processed, len(pending), rate, remaining,
            )


# ---------------------------------------------------------------------------
# Main extraction driver
# ---------------------------------------------------------------------------


def process_corpus(
    corpus_dir: Path,
    output_path: Path,
    *,
    workers: int,
    limit: int | None,
    resume: bool,
    progress_interval: int,
    with_metadata: bool,
    output_format: str = "json",
) -> dict[str, Any]:
    """Walk *corpus_dir*, extract all choice pairs, stream to *output_path*."""
    pending, stats, failed_log = _prepare_pending_files(
        corpus_dir, output_path, output_format, resume, limit
    )

    start = time.monotonic()
    with (
        _make_writer(output_format, output_path, resume) as writer,
        failed_log.open("a", encoding="utf-8") as fail_fh,
        concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool,
    ):
        # Mark a fresh run boundary in failed_files.log so that
        # operators can correlate failures with a specific extraction run.
        fail_fh.write(f"# run started {datetime.now(timezone.utc).isoformat()}\n")
        _run_extraction_loop(
            pool, pending, writer, fail_fh, stats,
            with_metadata=with_metadata,
            progress_interval=progress_interval,
            output_path=output_path,
            start=start,
        )

    stats["runtime_seconds"] = round(time.monotonic() - start, 2)
    stats["total_pairs"] = writer.count
    return stats


# ---------------------------------------------------------------------------
# Summary and verify
# ---------------------------------------------------------------------------


def write_summary(output_path: Path, stats: dict[str, Any]) -> None:
    summary_path = output_path.parent / (output_path.stem + ".summary.json")
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    logger.info("Summary written to %s", summary_path)


def verify_output(output_path: Path, output_format: str = "json") -> int:
    """Re-parse *output_path* and return the number of records."""
    if output_format not in {"json", "jsonl"}:
        raise ValueError(f"Unsupported output format: {output_format}")

    logger.info("Verifying %s …", output_path)
    if output_format == "json":
        with output_path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        count = len(data)
    else:
        count = 0
        with output_path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    json.loads(line)
                    count += 1
    logger.info("Verified: %d records, JSON is well-formed", count)
    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract EpiDoc <choice> reg/orig pairs from papyri.info corpus",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                   help="Output JSON file path")
    p.add_argument("--source-dir", metavar="PATH",
                   help="Path to existing idp.data checkout (skips download)")
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                   help="Directory for downloaded corpus cache")
    p.add_argument("--workers", type=_positive_int,
                   default=max(1, (os.cpu_count() or 2) // 2),
                   help="Parallel worker processes (>= 1)")
    p.add_argument("--format", choices=["json", "jsonl"], default="json",
                   dest="output_format", help="Output format")
    p.add_argument("--with-metadata", action="store_true",
                   help="Include pair_index in output records")
    p.add_argument("--resume", action="store_true", default=True,
                   help="Skip already-processed files (default: on)")
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="Ignore resume checkpoint; process all files")
    p.add_argument("--progress-interval", type=_positive_int, default=1000,
                   help="Log progress every N files")
    p.add_argument("--limit", type=_positive_int, default=None, metavar="N",
                   help="Process only first N files (for smoke tests)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Logging level")
    p.add_argument("--verify-only", action="store_true",
                   help="Re-verify existing output without re-extracting")
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    output_path = Path(args.output)

    if args.verify_only:
        if not output_path.exists():
            raise SystemExit(f"Output file not found: {output_path}")
        count = verify_output(output_path, output_format=args.output_format)
        logger.info("OK: %d records", count)
        return

    corpus_dir = ensure_corpus(args)
    logger.info("Corpus directory: %s", corpus_dir)

    stats = process_corpus(
        corpus_dir,
        output_path,
        workers=args.workers,
        limit=args.limit,
        resume=args.resume,
        progress_interval=args.progress_interval,
        with_metadata=args.with_metadata,
        output_format=args.output_format,
    )

    write_summary(output_path, stats)

    logger.info(
        "Done. %d pairs from %d/%d files. Failures: %d. Missing TM: %d. Time: %.1fs",
        stats["total_pairs"],
        stats["files_with_pairs"],
        stats["total_files"],
        stats["parse_failures"],
        stats["missing_tm"],
        stats["runtime_seconds"],
    )


if __name__ == "__main__":
    main()
