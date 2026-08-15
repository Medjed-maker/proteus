"""Paired interval between two LM corpus conditions.

The H2 write-up argued that in-domain data buys nothing by observing that
C_C's interval and C_B-prime's interval "almost completely overlap". That is
the overlapping-CI fallacy: two intervals can overlap substantially while the
paired difference is still reliably non-zero, because both marginals carry the
variance of the shared baseline that the difference cancels out.

Both conditions are measured against the same R0 on the same documents, so the
difference is a paired quantity and belongs inside the resample:
bootstrap(hits_a, hits_b, keys) subtracts within each draw and keeps the
pairing. This recomputes the comparison that way.

The weight is fixed in advance (w=2.0, where both conditions peak) so the
headline is not selected after seeing the differences; the full grid is
reported alongside.
"""

import argparse
import json
import math
from pathlib import Path

from benchmark_provenance import directory_identity, file_identity
from rlb_lm import Trigram, contexts, nfc
from rlb_stats import bootstrap
from run_eval import norm_lenient

ROOT = Path(__file__).parent


def _resolve(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def corpus_identity(label: str, configured_dirs: list[str], lm) -> dict:
    """Describe the exact corpus bytes and trained LM without absolute paths."""
    directories = []
    for configured, source in zip(configured_dirs, lm.sources, strict=True):
        identity = directory_identity(_resolve(configured))
        identity.update({
            "group": source["group"],
            "training_tokens": source["tokens"],
        })
        directories.append(identity)
    return {
        "label": label,
        "directories": directories,
        "lm_tokens": lm.total,
        "lm_types": len(lm.uni),
    }


def hits_for(rows, lm, ctx, w: float, k: int = 5) -> dict:
    """Same scoring path as rlb_lm.cmd_rerank: cost band first, LM within."""
    out = {}
    for r in rows:
        p2, p1 = ctx[r["doc"], r["sent"], r["wid"]]
        gl = norm_lenient(r["gold"])

        def key(c):
            s = lm.score(nfc(c[0]), p1, p2)
            return (c[1], -(math.log1p(c[2]) + w * s), -c[3])

        ranked = sorted(r["cands"], key=key)
        out[r["doc"], r["sent"], r["wid"]] = int(
            any(norm_lenient(c[0]) == gl for c in ranked[:k]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="dump_b3u_feat.jsonl")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--a-dir", nargs="+", default=["ddb_lemmas"])
    ap.add_argument("--a-group", type=int, nargs="+", default=[3])
    ap.add_argument("--a-name", default="C_C (DDbDP, in-domain)")
    ap.add_argument("--b-dir", nargs="+", default=["glaux_matched"])
    ap.add_argument("--b-group", type=int, nargs="+", default=[1])
    ap.add_argument("--b-name", default="C_B' (GLAUx, token-matched)")
    ap.add_argument("--weights", type=float, nargs="+",
                    default=[0.3, 1.0, 2.0, 4.0])
    ap.add_argument("--headline-w", type=float, default=2.0)
    ap.add_argument("--contexts", default="dilemma")
    ap.add_argument(
        "--allow-legacy-context-cache",
        action="store_true",
        help="accept a pre-provenance bare context_lemmas.json mapping",
    )
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default="results_lm_paired.json")
    args = ap.parse_args()
    if args.headline_w not in args.weights:
        ap.error("--headline-w must be included in --weights")

    rows = [json.loads(line)
            for line in (ROOT / args.dump).open(encoding="utf-8")]
    rows = [r for r in rows if args.split == "all" or r["split"] == args.split]
    keys = [(r["doc"], r["sent"], r["wid"]) for r in rows]
    print(f"split={args.split}  tokens={len(keys)}  "
          f"documents={len({k[0] for k in keys})}\n")

    print(f"training A: {args.a_dir}")
    lm_a = Trigram.train(args.a_dir, None, args.a_group)
    print(f"training B: {args.b_dir}")
    lm_b = Trigram.train(args.b_dir, None, args.b_group)

    ctx = contexts(
        rows,
        args.contexts,
        allow_legacy_context_cache=args.allow_legacy_context_cache,
    )
    base = hits_for(rows, lm_a, ctx, 0.0)      # w=0 removes the LM term
    r0 = bootstrap(base, None, keys, args.boot)
    print(f"\nR0 baseline recall@5 = {r0['point']:.1%}")

    # Self-check: a system against itself must give a degenerate interval.
    self_check = bootstrap(base, base, keys, args.boot)
    assert self_check["lo"] == self_check["hi"] == 0.0, self_check
    print("sanity: base vs base -> [+0.0%, +0.0%] OK\n")

    inputs = {"candidate_dump": file_identity(_resolve(args.dump))}
    if args.contexts == "dilemma":
        inputs["context_lemmas"] = file_identity(ROOT / "context_lemmas.json")
    out = {
        "split": args.split,
        # Retain the historical scalar fields for existing readers.
        "a": args.a_name,
        "b": args.b_name,
        "corpora": {
            "a": corpus_identity(args.a_name, list(args.a_dir), lm_a),
            "b": corpus_identity(args.b_name, list(args.b_dir), lm_b),
        },
        "inputs": inputs,
        "parameters": {
            "context_mode": args.contexts,
            "weights": list(args.weights),
            "bootstrap_resamples": args.boot,
        },
        "headline_w": args.headline_w,
        "r0": r0,
        "grid": {},
    }

    print(f"{'w':>5}  {'A vs R0':>18}  {'B vs R0':>18}  "
          f"{'A vs B (PAIRED)':>22}")
    for w in args.weights:
        ha = hits_for(rows, lm_a, ctx, w)
        hb = hits_for(rows, lm_b, ctx, w)
        da = bootstrap(ha, base, keys, args.boot)
        db = bootstrap(hb, base, keys, args.boot)
        dab = bootstrap(ha, hb, keys, args.boot)
        star = "" if (dab["lo"] <= 0 <= dab["hi"]) else " *"
        print(f"{w:>5}  {da['point']:>+7.1%} [{da['lo']:+.1%},{da['hi']:+.1%}]"
              f"  {db['point']:>+7.1%} [{db['lo']:+.1%},{db['hi']:+.1%}]"
              f"  {dab['point']:>+7.1%} [{dab['lo']:+.1%},{dab['hi']:+.1%}]"
              f"{star}")
        out["grid"][str(w)] = {"a_vs_r0": da, "b_vs_r0": db, "a_vs_b": dab}

    h = out["grid"][str(args.headline_w)]["a_vs_b"]
    print(f"\nheadline (w={args.headline_w}, fixed in advance): "
          f"{args.a_name} - {args.b_name} = {h['point']:+.2%} "
          f"[{h['lo']:+.2%}, {h['hi']:+.2%}]")
    print("  interval " + ("EXCLUDES zero -- the difference is real"
                           if not (h["lo"] <= 0 <= h["hi"])
                           else "INCLUDES zero -- no detectable difference"))

    (ROOT / args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
