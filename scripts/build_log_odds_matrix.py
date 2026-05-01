"""Build a BLOSUM-style phonological log-odds matrix from variant pair training data.

Reads JSON or JSONL (original, regularized) Greek word pairs produced by
``scripts/extract_epidoc_choices.py``, aligns each pair in IPA space with
Needleman-Wunsch, accumulates aligned-column frequencies, and computes a
BLOSUM-style log-odds substitution score matrix.

Output schema (``scores`` key distinguishes this from the distance matrix):

.. code-block:: json

    {
      "_meta": { "description": "...", "method": "...", "alphabet": [...], ... },
      "scores": { "phone_i": { "phone_j": float } }
    }

Usage::

    uv run python scripts/build_log_odds_matrix.py \\
        --input data/training/epidoc_choices.json \\
        --output data/languages/ancient_greek/matrices/log_odds_epidoc_choices.json

    # Quick test with a subset
    uv run python scripts/build_log_odds_matrix.py --limit 100 --log-level DEBUG
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterator

if __package__ in {None, ""}:
    # Allow `python scripts/build_log_odds_matrix.py` to resolve the scripts package.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phonology.explainer import Alignment
from phonology.ipa_converter import greek_to_ipa
from phonology.log_odds import (
    NWParams,
    accumulate_counts,
    build_matrix_document,
    compute_log_odds,
    needleman_wunsch,
)
from scripts._cli_utils import (
    finite_float as _finite_float,
    nonneg_float as _nonneg_float,
    nonneg_int as _nonneg_int,
    positive_float as _positive_float,
    positive_int as _positive_int,
)

logger = logging.getLogger(__name__)

DEFAULT_INPUT = Path("data/training/epidoc_choices.json")
DEFAULT_OUTPUT = Path("data/languages/ancient_greek/matrices/log_odds_epidoc_choices.json")
MAX_PHONES = 200


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        metavar="FILE",
        help="Input JSON/JSONL file ({original, regularized, ...} records)",
    )
    parser.add_argument(
        "--input-format",
        choices=["auto", "json", "jsonl"],
        default="auto",
        help="Input record format",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        metavar="FILE",
        help="Output JSON matrix file",
    )
    parser.add_argument(
        "--smoothing",
        choices=["laplace", "lidstone", "floor"],
        default="laplace",
        help="Smoothing strategy for zero-count pairs",
    )
    parser.add_argument(
        "--lidstone-alpha",
        type=_nonneg_float,
        default=0.5,
        metavar="ALPHA",
        help="Lidstone smoothing parameter (used when --smoothing=lidstone)",
    )
    parser.add_argument(
        "--floor",
        type=_finite_float,
        default=-10.0,
        help="Score floor for undefined or non-finite log-odds values",
    )
    parser.add_argument(
        "--min-count",
        type=_nonneg_int,
        default=0,
        metavar="N",
        help="Minimum raw pair count for a cell to appear in output (0 = keep all)",
    )
    parser.add_argument(
        "--gap-cost",
        type=_positive_float,
        default=1.25,
        metavar="COST",
        help="Gap insertion/deletion cost for Needleman-Wunsch alignment",
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        default=None,
        metavar="N",
        help="Process at most N input pairs (useful for quick tests)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity",
    )
    return parser


def _iter_alignments(
    records: list[dict[str, Any]],
    nw_params: NWParams,
    limit: int | None,
    counters: dict[str, int],
) -> Iterator[Alignment]:
    """Yield NW alignments for each record; update ``counters`` in place."""
    for record in records:
        if limit is not None and counters["seen"] >= limit:
            break
        try:
            orig_phones = greek_to_ipa(record.get("original") or "")
            reg_phones = greek_to_ipa(record.get("regularized") or "")
        except (ValueError, TypeError) as exc:
            logger.warning("IPA conversion failed: %s", exc)
            counters["skipped_empty"] += 1
            continue

        if not orig_phones or not reg_phones:
            counters["skipped_empty"] += 1
            continue

        if len(orig_phones) > MAX_PHONES or len(reg_phones) > MAX_PHONES:
            logger.warning(
                "Skipping oversized pair (%d / %d phones)",
                len(orig_phones),
                len(reg_phones),
            )
            counters["skipped_long"] += 1
            continue

        counters["seen"] += 1
        yield needleman_wunsch(orig_phones, reg_phones, nw_params)


def _load_json_records(input_path: Path) -> list[dict[str, Any]]:
    """Load a JSON array of training records."""
    with input_path.open(encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        logger.error(
            "Expected a JSON array in %s, got %s",
            input_path,
            type(records).__name__,
        )
        sys.exit(1)

    for i, record in enumerate(records):
        if not isinstance(record, dict):
            logger.error(
                "Expected a JSON object in %s at index %d, got %s",
                input_path,
                i,
                type(record).__name__,
            )
            sys.exit(1)

    return records


def _load_jsonl_records(input_path: Path) -> list[dict[str, Any]]:
    """Load newline-delimited JSON training records."""
    records: list[dict[str, Any]] = []
    with input_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.error(
                    "Invalid JSONL in %s at line %d: %s",
                    input_path,
                    line_no,
                    exc,
                )
                sys.exit(1)
            if not isinstance(record, dict):
                logger.error(
                    "Expected a JSON object in %s at line %d, got %s",
                    input_path,
                    line_no,
                    type(record).__name__,
                )
                sys.exit(1)
            records.append(record)
    return records


def _load_records(input_path: Path, input_format: str) -> list[dict[str, Any]]:
    """Load training records from JSON or JSONL according to *input_format*."""
    if input_format == "json":
        try:
            return _load_json_records(input_path)
        except json.JSONDecodeError as exc:
            logger.error("Invalid JSON in %s: %s", input_path, exc, exc_info=True)
            sys.exit(1)
    if input_format == "jsonl":
        return _load_jsonl_records(input_path)

    suffix = input_path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        return _load_jsonl_records(input_path)

    try:
        return _load_json_records(input_path)
    except json.JSONDecodeError as exc:
        if suffix == ".json":
            logger.error("Invalid JSON in %s: %s", input_path, exc, exc_info=True)
            sys.exit(1)
        logger.warning(
            "Failed to parse %s as JSON: %s; falling back to JSONL",
            input_path,
            exc,
            exc_info=True,
        )
        return _load_jsonl_records(input_path)


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    input_path: Path = args.input
    if not input_path.exists():
        logger.error(
            "Input file not found: %s  —  run scripts/extract_epidoc_choices.py first.",
            input_path,
        )
        sys.exit(1)

    logger.info("Loading input from %s", input_path)
    records = _load_records(input_path, args.input_format)

    source_pair_count = len(records)
    logger.info("Loaded %d records", source_pair_count)

    nw_params = NWParams(gap=args.gap_cost)

    counters: dict[str, int] = {"seen": 0, "skipped_empty": 0, "skipped_long": 0}
    counts = accumulate_counts(
        _iter_alignments(records, nw_params, args.limit, counters)
    )
    alignments_used = counters["seen"]

    logger.info(
        "Processed %d pairs (skipped: %d empty/invalid, %d oversized)",
        counters["seen"],
        counters["skipped_empty"],
        counters["skipped_long"],
    )
    logger.info(
        "Aligned %d pairs → %d matched columns, alphabet size %d",
        alignments_used,
        counts.pair_total,
        len(counts.phone_totals),
    )
    logger.debug("Top phones: %s", counts.phone_totals.most_common(10))

    if args.smoothing == "laplace":
        smoothing_params: dict[str, Any] = {}
    elif args.smoothing == "lidstone":
        smoothing_params = {"alpha": args.lidstone_alpha}
    else:
        smoothing_params = {"floor": args.floor}

    scores = compute_log_odds(
        counts,
        smoothing=args.smoothing,
        lidstone_alpha=args.lidstone_alpha,
        floor=args.floor,
        min_count=args.min_count,
    )

    document = build_matrix_document(
        counts,
        scores,
        source_path=str(input_path),
        smoothing=args.smoothing,
        smoothing_params=smoothing_params,
        nw_params=nw_params,
        source_pair_count=source_pair_count,
        alignments_used=alignments_used,
    )

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as out:
            json.dump(document, out, ensure_ascii=False, indent=2, sort_keys=True)
        tmp_path.rename(output_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    logger.info("Wrote matrix to %s", output_path)
    logger.info(
        "Alphabet (%d phones): %s",
        len(document["_meta"]["alphabet"]),
        document["_meta"]["alphabet"][:10],
    )


if __name__ == "__main__":
    main()
