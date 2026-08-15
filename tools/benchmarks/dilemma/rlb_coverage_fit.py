"""Extrapolate the coverage-gain relationship to its ceiling.

H2 rejected the in-domain corpus by a pre-registered threshold, and a threshold
argument is only as strong as the threshold. The mechanism found alongside it
is stronger: a count-based trigram can only reorder candidates it has already
seen, so the gain tracks how much of the candidate space falls inside the LM's
vocabulary. If that is really the binding constraint, the relationship should
be regular enough to extrapolate -- and the extrapolation says what perfect
coverage would buy, with no threshold involved.

Coverage and gain are both recomputed here in one pass so they come from the
same dump; the numbers first reported were measured before the clean.py markup
fix and no longer match the regenerated results.

The fit is n=5 with 3 degrees of freedom and the target (100%) is outside the
observed range (41-75%). Treat it as an order-of-magnitude ceiling, not an
estimate -- the prediction interval is reported for exactly that reason.
"""

import argparse
import json
import math
from pathlib import Path

from rlb_lm import Trigram, contexts, nfc
from rlb_stats import bootstrap
from run_eval import norm_lenient

ROOT = Path(__file__).parent

# (label, dirs, groups) -- the five conditions from the H2 matrix.
CONDITIONS = [
    ("GLAUx-30", ["glaux_lemmas"], [1]),
    ("GLAUx-matched", ["glaux_matched"], [1]),
    ("DDbDP (in-domain)", ["ddb_lemmas"], [3]),
    ("GLAUx-full", ["glaux_lemmas_full"], [1]),
    ("GLAUx-full + DDbDP", ["glaux_lemmas_full", "ddb_lemmas"], [1, 3]),
]


def coverage(lm, rows, ctx) -> dict:
    vocab = set(lm.uni)
    gold = {nfc(r["gold"]) for r in rows}
    cands = {nfc(c[0]) for r in rows for c in r["cands"][:20]}
    ctxw = {w for k in ctx for w in ctx[k] if w != "<s>"}
    return {
        "gold": len(gold & vocab) / len(gold),
        "cand20": len(cands & vocab) / len(cands),
        "context": len(ctxw & vocab) / len(ctxw),
    }


def hits_for(rows, lm, ctx, w: float, k: int = 5) -> dict:
    out = {}
    for r in rows:
        p2, p1 = ctx[r["doc"], r["sent"], r["wid"]]
        gl = norm_lenient(r["gold"])

        def key(c):
            return (c[1], -(math.log1p(c[2]) + w * lm.score(nfc(c[0]), p1, p2)),
                    -c[3])

        ranked = sorted(r["cands"], key=key)
        out[r["doc"], r["sent"], r["wid"]] = int(
            any(norm_lenient(c[0]) == gl for c in ranked[:k]))
    return out


def ols(xs: list[float], ys: list[float]) -> dict:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    dof = n - 2
    s2 = sum(r * r for r in resid) / dof
    syy = sum((y - my) ** 2 for y in ys)
    return {"slope": slope, "intercept": intercept, "residuals": resid,
            "s": math.sqrt(s2), "dof": dof, "mx": mx, "sxx": sxx, "n": n,
            "r2": 1 - sum(r * r for r in resid) / syy if syy else float("nan")}


def predict(fit: dict, x0: float, t: float = 3.182) -> dict:
    """Prediction interval for a new observation at x0. t = t(.975, dof=3)."""
    point = fit["intercept"] + fit["slope"] * x0
    se = fit["s"] * math.sqrt(1 + 1 / fit["n"]
                              + (x0 - fit["mx"]) ** 2 / fit["sxx"])
    return {"point": point, "lo": point - t * se, "hi": point + t * se,
            "se": se}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="dump_b3u_feat.jsonl")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--weight", type=float, default=2.0,
                    help="preselected LM weight used for every condition")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default="results_coverage_fit.json")
    ap.add_argument(
        "--allow-legacy-context-cache",
        action="store_true",
        help="accept a pre-provenance bare context_lemmas.json mapping",
    )
    args = ap.parse_args()

    rows = [json.loads(line)
            for line in (ROOT / args.dump).open(encoding="utf-8")]
    rows = [r for r in rows if args.split == "all" or r["split"] == args.split]
    keys = [(r["doc"], r["sent"], r["wid"]) for r in rows]
    ctx = contexts(
        rows,
        "dilemma",
        allow_legacy_context_cache=args.allow_legacy_context_cache,
    )
    print(f"split={args.split}  tokens={len(keys)}  "
          f"documents={len({k[0] for k in keys})}\n")

    points, table = [], []
    base = None
    for label, dirs, groups in CONDITIONS:
        lm = Trigram.train(dirs, None, groups, verbose=False)
        cov = coverage(lm, rows, ctx)
        if base is None:
            # At w=0 the LM term drops out of the sort key entirely, so this
            # is R0 regardless of which corpus trained the model -- computing
            # it once from the first condition is not an approximation.
            base = hits_for(rows, lm, ctx, 0.0)
            r0 = bootstrap(base, None, keys, args.boot)
            print(f"R0 baseline recall@5 = {r0['point']:.1%}\n")
        # Apply one preselected weight to every condition so the reported
        # split is evaluated exactly once. The default 2.0 was fixed in H2.
        d = bootstrap(hits_for(rows, lm, ctx, args.weight),
                      base, keys, args.boot)
        gain = d["point"] * 100
        points.append((cov["cand20"] * 100, gain))
        table.append({"label": label, "tokens": lm.total,
                      "coverage": cov, "weight": args.weight,
                      "gain_pt": gain,
                      "gain_ci": [d["lo"] * 100, d["hi"] * 100]})
        print(f"{label:<22}{lm.total:>12,}  gold {cov['gold']:>6.1%}  "
              f"cand@20 {cov['cand20']:>6.1%}  ctx {cov['context']:>6.1%}  "
              f"gain {gain:>+5.2f}pt (w={args.weight})")

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    fit = ols(xs, ys)
    at100 = predict(fit, 100.0)

    print(f"\n=== OLS: gain = {fit['intercept']:.3f} "
          f"+ {fit['slope']:.4f} x coverage")
    print(f"    R^2 = {fit['r2']:.3f}   residual s = {fit['s']:.3f}pt   "
          f"dof = {fit['dof']}")
    print("    residuals: "
          + ", ".join(f"{r:+.2f}" for r in fit["residuals"]))
    print("\n=== extrapolation to 100% candidate coverage")
    print(f"    point estimate      {at100['point']:+.2f}pt")
    print(f"    95% prediction int. [{at100['lo']:+.2f}, {at100['hi']:+.2f}]pt")
    print(f"    observed range      {min(xs):.1f}%-{max(xs):.1f}%  "
          f"-> 100% is OUTSIDE it; linearity is assumed, not tested")

    (ROOT / args.out).write_text(json.dumps(
        {"split": args.split, "weight": args.weight, "conditions": table,
         "fit": {k: v for k, v in fit.items()},
         "extrapolation_100pct": at100,
         "caveats": [
             "n=5, 3 residual degrees of freedom",
             "correlational: coverage covaries with corpus size, though the "
             "C_C vs C_B' pair holds size fixed",
             "100% lies outside the observed 41-75% range; linearity is "
             "assumed and untested, and diminishing returns near saturation "
             "would make this an over-estimate",
             "the prediction interval, not the point estimate, is the result",
         ]}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
