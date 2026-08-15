"""Build a token-matched subset of one corpus directory.

Without this, "the in-domain corpus won" and "the bigger corpus won" are the
same observation. C_B' is GLAUx-full cut down to the DDbDP token count, so the
only difference left between it and C_C is domain.

Symlinks, not copies: the corpus is the same bytes, and a copy invites the two
from drifting apart.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parent


def tokens_in(path: Path) -> int:
    return sum(len(line.split()) for line in path.open(encoding="utf-8"))


def _resolve_under_root(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _validate_disjoint(input_dir: Path, out: Path, option: str) -> None:
    if (input_dir == out or input_dir.is_relative_to(out)
            or out.is_relative_to(input_dir)):
        raise SystemExit(f"{option} and --out must be disjoint directories")


def _clear_generated_links(out: Path) -> None:
    entries = list(out.glob("*.txt"))
    regular = [path for path in entries if not path.is_symlink()]
    if regular:
        names = ", ".join(path.name for path in regular[:3])
        raise SystemExit(
            f"refusing to delete regular .txt files from --out: {names}")
    for path in entries:
        path.unlink()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--match", default="",
                    help="directory whose token count to match")
    ap.add_argument("--tokens", type=int, default=0)
    args = ap.parse_args()

    src, out = _resolve_under_root(args.src), _resolve_under_root(args.out)
    if not src.is_dir():
        raise SystemExit(f"--src is not a directory: {src}")
    _validate_disjoint(src, out, "--src")
    source_files = sorted(src.glob("*.txt"))
    if not source_files:
        raise SystemExit(f"--src contains no .txt files: {src}")

    budget = args.tokens
    if args.match:
        match = _resolve_under_root(args.match)
        if not match.is_dir():
            raise SystemExit(f"--match is not a directory: {match}")
        _validate_disjoint(match, out, "--match")
        match_files = list(match.glob("*.txt"))
        if not match_files:
            raise SystemExit(f"--match contains no .txt files: {match}")
        budget = sum(tokens_in(p) for p in match_files)
        print(f"matching {args.match}: {budget:,} tokens")
    if budget <= 0:
        raise SystemExit("provide a positive token budget via --tokens or --match")

    if out.exists():
        if not out.is_dir():
            raise SystemExit(f"--out is not a directory: {out}")
        _clear_generated_links(out)
    out.mkdir(exist_ok=True)

    # Sorted order, not random: the selection has to be reproducible from the
    # corpus alone, the same property that makes the document splits checkable.
    total = n = 0
    for p in source_files:
        if total >= budget:
            break
        (out / p.name).symlink_to(p.resolve())
        total += tokens_in(p)
        n += 1

    meta = {"src": args.src, "out": args.out, "match": args.match,
            "budget_tokens": budget, "files": n, "tokens": total}
    (out.parent / f"{out.name}_manifest.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{n} files, {total:,} tokens -> {args.out} "
          f"({total / budget:.1%} of budget)")


if __name__ == "__main__":
    main()
