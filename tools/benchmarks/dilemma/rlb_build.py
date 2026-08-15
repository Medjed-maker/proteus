"""Precompute the two key files the ladder searches over.

Kept out of the evaluation run because a 4.8M-row build that dies halfway
should not take a measurement with it. Streamed and resumable for the same
reason: rerun it until it prints "complete", and it picks up where it stopped.

    grc_keys.txt   accent-stripped keys with at least one ancient-Greek form
    b3_keys.txt    the same keys, line-aligned, after the fifteen alternations
"""

import argparse
import time
from pathlib import Path

from rlb_keys import collapse
from rlb_lexicon import Lexicon

ROOT = Path(__file__).parent
SRC = ROOT / "grc_keys.txt"
DST = ROOT / "b3_keys.txt"


def _line_count(p: Path) -> int:
    if not p.exists():
        return 0
    count = 0
    last_byte = b""
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            count += chunk.count(b"\n")
            last_byte = chunk[-1:]
    return count + int(bool(last_byte) and last_byte != b"\n")


def _truncate_incomplete_line(p: Path) -> None:
    """Remove an unterminated suffix before resuming an append-only build."""
    if not p.exists():
        return
    with p.open("rb+") as fh:
        fh.seek(0, 2)
        end = fh.tell()
        if not end:
            return
        fh.seek(end - 1)
        if fh.read(1) == b"\n":
            return
        pos = end
        while pos:
            start = max(0, pos - (1 << 20))
            fh.seek(start)
            chunk = fh.read(pos - start)
            newline = chunk.rfind(b"\n")
            if newline >= 0:
                fh.truncate(start + newline + 1)
                return
            pos = start
        fh.truncate(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=90.0,
                    help="seconds to work before checkpointing out")
    args = ap.parse_args()

    if not SRC.exists():
        Lexicon().grc_keys()
    total = _line_count(SRC)
    _truncate_incomplete_line(DST)
    done = _line_count(DST)
    print(f"{done:,}/{total:,} collapsed", flush=True)
    if done >= total:
        print("complete")
        return

    t0 = time.time()
    with SRC.open(encoding="utf-8") as src, DST.open("a", encoding="utf-8") as dst:
        for _ in range(done):
            src.readline()
        n = done
        for line in src:
            dst.write(collapse(line.rstrip("\n")))
            dst.write("\n")
            n += 1
            if time.time() - t0 >= args.budget:
                break
    print(f"{n:,}/{total:,}  ({time.time() - t0:.0f}s)")
    if n >= total:
        print("complete")


if __name__ == "__main__":
    main()
