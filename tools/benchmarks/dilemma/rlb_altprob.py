"""Use ③ -- the empirical probability of each alternation -- as a tiebreaker.

The ladder killed one use of a phonological matrix (as a *generator*: the
residual "never generated" bucket is 4.3% and consists of heavy damage, not
systematic correspondence) and kept another (as *pruning and evidence*: the
fifteen rules cut the candidate set fourteenfold). A third use was never
measured, and R1's failure is not evidence against it: R1 added frequency, not
per-alternation probability.

The claim under test: B3u's cost is an integer count of alternations, so the
whole of a 179-candidate set piles into two or three cost bands and the order
inside a band is decided by corpus frequency alone. But an itacism η→ει is
orders of magnitude more common in papyri than a sigma insertion, and a cost
function that charges them the same is throwing away real information.

    cost'(candidate) = Σ over alternation sites a of -log P(a)

P(a) is estimated from PapyGreek documents outside the selected evaluation
split. Every row from an evaluation document is excluded, while documents
without evaluated tokens remain available. No download, and leak-free by
construction rather than by inspection.

Pre-registered before the test split was scored (see the plan file):

    if recall@5 improves by less than +3.0 points, or the document-clustered
    bootstrap interval covers zero, ③ is rejected and the empirical matrix is
    kept only as a scientific artefact, not as a ranking component.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from categorize import labels
from clean import clean
from rlb_keys import b1_key
from rlb_stats import bootstrap
from run_eval import load, norm_lenient

ROOT = Path(__file__).parent
FLOOR = 0.5          # add-k for alternations never seen in the estimation set
BLEND_W = 0.2        # selected on dev by sweep over {0.02 .. 1.0}


def estimate(evaluation_docs: set[str],
             verbose: bool = True) -> tuple[dict[str, float], Counter]:
    """P(alternation) from documents outside the evaluation set."""
    if not evaluation_docs:
        raise ValueError("evaluation_docs must not be empty")
    rows = load(ROOT / "dataset.jsonl")
    estimation_rows = [r for r in rows if r["doc"] not in evaluation_docs]
    if not estimation_rows:
        raise ValueError(
            "no rows remain for estimation after excluding evaluation documents")
    estimation_docs = {r["doc"] for r in estimation_rows}
    if not evaluation_docs.isdisjoint(estimation_docs):
        raise RuntimeError("estimation and evaluation documents overlap")

    ct = Counter()
    n_pairs = 0
    for r in estimation_rows:
        a, b = clean(r["form_orig"]), clean(r["form_reg"])
        if a == b:
            continue
        atoms = labels(a, b)
        if not atoms:
            continue
        n_pairs += 1
        ct.update(atoms)

    total = sum(ct.values())
    if total == 0:
        raise ValueError("estimation data contains no alternation sites")
    probs = {k: (v + FLOOR) / (total + FLOOR * (len(ct) + 1))
             for k, v in ct.items()}
    probs["<unseen>"] = FLOOR / (total + FLOOR * (len(ct) + 1))

    if verbose:
        print(f"estimation set: {len(estimation_rows)} tokens in "
              f"{len(estimation_docs)} non-evaluation documents")
        print(f"  {n_pairs} spelling pairs, {total} alternation sites, "
              f"{len(ct)} distinct types\n")
        print(f"{'alternation':<34}{'count':>7}{'P':>9}{'-logP':>8}")
        for k, v in ct.most_common(18):
            print(f"{k:<34}{v:>7}{probs[k]:>9.4f}{-math.log(probs[k]):>8.2f}")
        print(f"{'<unseen>':<34}{'-':>7}{probs['<unseen>']:>9.4f}"
              f"{-math.log(probs['<unseen>']):>8.2f}")
    return probs, ct


def alt_cost(query_key: str, src_key: str, probs: dict) -> float:
    if query_key == src_key:
        return 0.0
    atoms = labels(query_key, src_key)
    if not atoms:
        return 0.0
    unseen = probs["<unseen>"]
    return sum(-math.log(probs.get(a, unseen)) for a in atoms)


# --------------------------------------------------------------------------


def scorers(probs: dict):
    """Two ways of spending the alternation cost, plus the baseline.

    'tiebreak' is the literal reading of the hypothesis: keep the integer cost
    band, order *within* the band by alternation probability. 'replace' is the
    stronger version: the probability cost supersedes the integer count.
    """
    def r0(row, c):
        return (c[1], -c[2], -c[3])

    def tiebreak(row, c):
        return (c[1], row["_alt"][c[4]], -c[2], -c[3])

    def replace(row, c):
        return (row["_alt"][c[4]], -c[2], -c[3])

    def after_freq(row, c):
        """A4: break ties that survive frequency, rather than pre-empting it."""
        return (c[1], -c[2], row["_alt"][c[4]], -c[3])

    def blend(row, c):
        """A5: a small additive nudge inside the cost band. Dev-selected w."""
        return (c[1], -(math.log1p(c[2]) - BLEND_W * row["_alt"][c[4]]), -c[3])

    return {"R0 (baseline)": r0,
            "A1 tiebreak before frequency": tiebreak,
            "A2 replace integer cost": replace,
            "A4 tiebreak after frequency": after_freq,
            f"A5 additive nudge (w={BLEND_W})": blend}


def evaluate(rows, scorer, keys_filter=None) -> dict:
    hits = {}
    for row in rows:
        gl = norm_lenient(row["gold"])
        ranked = sorted(row["cands"], key=lambda c: scorer(row, c))
        hits[row["doc"], row["sent"], row["wid"]] = int(
            any(norm_lenient(c[0]) == gl for c in ranked[:5]))
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    ap.add_argument("--dump", default="dump_b3u_feat.jsonl")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()

    rows = [json.loads(line)
            for line in (ROOT / args.dump).open(encoding="utf-8")]
    rows = [r for r in rows if args.split == "all" or r["split"] == args.split]
    if not rows:
        ap.error(f"dump contains no rows for split {args.split!r}")
    evaluation_docs = {r["doc"] for r in rows}
    try:
        probs, _ = estimate(evaluation_docs)
    except ValueError as exc:
        ap.error(str(exc))

    # Precompute the alternation cost per distinct source key, per token.
    for row in rows:
        qk = b1_key(row["form"])
        row["_alt"] = {c[4]: alt_cost(qk, c[4], probs)
                       for c in row["cands"]}

    keys = [(r["doc"], r["sent"], r["wid"]) for r in rows]
    print(f"\nsplit={args.split}  tokens={len(keys)}  "
          f"documents={len({k[0] for k in keys})}\n")

    sc = scorers(probs)
    base = evaluate(rows, sc["R0 (baseline)"])
    print(f"{'variant':<32}{'recall@5':>10}{'vs R0':>9}{'95% CI (paired)':>22}")
    out = {}
    for name, fn in sc.items():
        hits = evaluate(rows, fn)
        r = bootstrap(hits, None, keys, args.boot)
        d = bootstrap(hits, base, keys, args.boot)
        out[name] = {"recall5": r["point"], "diff": d}
        ci = f"[{d['lo']:+.1%}, {d['hi']:+.1%}]"
        print(f"{name:<32}{r['point']:>10.1%}{d['point']:>+9.1%}{ci:>22}")

    (ROOT / f"results_altprob_{args.split}.json").write_text(
        json.dumps({"split": args.split,
                    "probs": {k: v for k, v in sorted(probs.items())},
                    "variants": out}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\nwritten to results_altprob_{args.split}.json")


if __name__ == "__main__":
    main()
