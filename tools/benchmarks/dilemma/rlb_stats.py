"""Confidence intervals and paired tests for the ladder, clustered by document.

The test split holds 1,484 tokens but only 222 independent units. Spellings
cluster by scribe -- that is exactly why dev/test was split by document -- and
the same logic applies to the intervals. Treating tokens as independent would
make a 0.9-point difference look like a finding.

Two estimators, deliberately both:

  cluster bootstrap   resample the 222 test *documents* with replacement, 1,000
                      times, recompute the metric. For a comparison, both
                      systems are scored on the *same* resample and the
                      difference is taken inside the loop, so the pairing is
                      preserved. This is the primary number.

  McNemar (exact)     the paired test on discordant tokens. Far more powerful
                      than comparing two independent proportions, but it
                      assumes tokens are independent, which they are not. It is
                      reported as a secondary, optimistic bound: when the
                      bootstrap interval covers zero and McNemar does not, the
                      bootstrap wins.

Nothing here re-runs a search. Every number is recomputed from the per-token
records already on disk.
"""

import argparse
import json
import math
import random
from pathlib import Path

from rlb_rerank import load_attestation, make_scorer
from rlb_splits import tag
from run_eval import load, norm_lenient

ROOT = Path(__file__).parent
N_BOOT = 1000
BOOT_SEED = 20260811

# The comparisons the write-up makes claims about. Anything else is
# exploratory and must be labelled as such.
PRE_SPECIFIED = [
    ("B3ur", "B2u", "15交替は素の距離を上回る"),
    ("B3u", "B3", "union はカスケードを上回る"),
    ("R5r + POS (Dilemma head, orig)", "R0 lexicographic (B3u as run)",
     "実POSタガーは改善する"),
    ("R5 + POS (gold, upper bound)", "R0 lexicographic (B3u as run)",
     "gold POS は改善する"),
    ("R1 additive blend", "R0 lexicographic (B3u as run)",
     "加法スコア化そのものの効果"),
    ("R2 + century prior", "R0 lexicographic (B3u as run)",
     "時代事前分布は改善しない (対R0)"),
    ("R3 + dialect prior", "R0 lexicographic (B3u as run)",
     "方言事前分布は改善しない (対R0)"),
    ("R4 + genre prior", "R0 lexicographic (B3u as run)",
     "ジャンル事前分布は改善しない (対R0)"),
    ("R6 all priors, no POS", "R0 lexicographic (B3u as run)",
     "事前分布すべてでも改善しない (対R0)"),
    # R2..R6 are all R1 plus a bonus term, so measuring them against R0
    # confounds the prior with the switch to an additive score. Against R1
    # the prior's own contribution is isolated.
    ("R2 + century prior", "R1 additive blend", "時代事前分布のみの寄与"),
    ("R3 + dialect prior", "R1 additive blend", "方言事前分布のみの寄与"),
    ("R4 + genre prior", "R1 additive blend", "ジャンル事前分布のみの寄与"),
    ("R6 all priors, no POS", "R1 additive blend", "事前分布すべての寄与"),
]


# --------------------------------------------------------------------------
# Loading: every system becomes {(doc, sent, wid): hit_at_5}
# --------------------------------------------------------------------------


def _variant_rows() -> list[dict]:
    return tag([r for r in load(ROOT / "dataset.jsonl")
                if r["stratum"] == "variant_ortho"])


def _validated_detail_rows(stage: str, detail: list[dict],
                           rows: list[dict]) -> list[tuple[dict, dict]]:
    """Validate a positional detail/dataset join before returning its pairs."""
    if len(detail) != len(rows):
        raise SystemExit(
            f"{stage}: {len(detail)} rows vs {len(rows)} expected")
    for i in range(len(rows)):
        d, r = detail[i], rows[i]
        if (d["form"] != r["input"] or d["gold"] != r["lemma_gold"]
                or d["split"] != r["split"]):
            raise SystemExit(
                f"{stage} row {i}: positional join failed "
                f"({d['form']!r}/{d['gold']!r} vs "
                f"{r['input']!r}/{r['lemma_gold']!r})")
    return list(zip(detail, rows))


def ladder_systems(k: int = 5) -> dict[str, dict]:
    """Recover document ids for the ladder stages by positional join.

    results_ladder.json predates the decision to record ids, but every stage
    holds all 2,160 tokens in dataset order (the slices were disjoint and
    concatenated by offset). The join is therefore positional -- and verified
    on every row before anything is computed, because a silent misalignment
    here would corrupt every interval downstream.
    """
    rows = _variant_rows()
    results = json.loads((ROOT / "results_ladder.json").read_text())
    out = {}
    for res in results:
        pairs = _validated_detail_rows(res["stage"], res["detail"], rows)
        out[res["stage"]] = {
            (r["doc"], r["sent"], r["wid"]):
                int(d["rank"] is not None and d["rank"] < k)
            for d, r in pairs
        }
    return out


def rerank_systems(k: int = 5) -> dict[str, dict]:
    """Re-ranking variants, scored from the candidate dump.

    The dump carries doc/sent/wid, so no positional join is needed here.
    """
    dump = [json.loads(line) for line in
            (ROOT / "dump_b3u.jsonl").open(encoding="utf-8")]
    att = load_attestation()
    names = [
        "R0 lexicographic (B3u as run)", "R1 additive blend",
        "R2 + century prior", "R3 + dialect prior", "R4 + genre prior",
        "R6 all priors, no POS", "R5r + POS (Dilemma head, orig)",
        "R5rg + POS (Dilemma head, reg)", "R5n + POS (10%-error tagger)",
        "R5 + POS (gold, upper bound)", "R7 all priors + POS (upper bound)",
    ]
    out = {}
    for name in names:
        scorer = make_scorer(name, att)
        hits = {}
        for row in dump:
            gl = norm_lenient(row["gold"])
            ranked = sorted(row["cands"], key=lambda c: scorer(row, c))
            hits[row["doc"], row["sent"], row["wid"]] = int(
                any(norm_lenient(c[0]) == gl for c in ranked[:k]))
        out[name] = hits
    return out


# --------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------


def _by_doc(hits: dict, keys: list) -> dict[str, list[int]]:
    per = {}
    for key in keys:
        per.setdefault(key[0], []).append(hits[key])
    return per


def bootstrap(hits_a: dict, hits_b: dict | None, keys: list,
              n_boot: int = N_BOOT) -> dict:
    """Percentile interval over documents resampled with replacement.

    When hits_b is given the difference a-b is taken inside each resample, so
    the two systems always see the same documents and the pairing survives.
    """
    docs = sorted({k[0] for k in keys})
    a_doc = _by_doc(hits_a, keys)
    b_doc = _by_doc(hits_b, keys) if hits_b else None

    def point(sample_docs):
        num = den = 0
        for d in sample_docs:
            va = a_doc[d]
            num += sum(va)
            den += len(va)
            if b_doc is not None:
                num -= sum(b_doc[d])
        return num / den if den else 0.0

    rnd = random.Random(BOOT_SEED)
    draws = []
    for _ in range(n_boot):
        draws.append(point([docs[rnd.randrange(len(docs))]
                            for _ in range(len(docs))]))
    draws.sort()
    lo = draws[int(0.025 * n_boot)]
    hi = draws[int(0.975 * n_boot) - 1]
    return {"point": point(docs), "lo": lo, "hi": hi,
            "n_docs": len(docs), "n_tokens": len(keys)}


def rank_profile(ranks: dict, keys: list) -> dict:
    """What the reviewer asked for instead of precision@5.

    precision@5 punishes returning extra candidates, which collides with the
    project's own stance that candidates are sourced hypotheses, not verdicts.
    For a research tool the fatal error is the false negative. These three say
    "how much does the researcher have to read", which is the real cost:

      rank_median / rank_p90   position of the gold lemma, unresolved excluded
      recall@N                 at several N, so the tail is visible
      doc_coverage@k           share of DOCUMENTS in which *every* variant
                               token is resolved within k -- a document with
                               one unresolved word is still an unresolved
                               document for the papyrologist reading it
    """
    got = sorted(ranks[key] for key in keys if ranks[key] is not None)
    n = len(keys)
    per_doc: dict[str, list] = {}
    for key in keys:
        per_doc.setdefault(key[0], []).append(ranks[key])
    out = {
        "n": n,
        "unresolved": (n - len(got)) / n if n else 0.0,
        "rank_median": (got[len(got) // 2] + 1) if got else None,
        "rank_p90": (got[int(len(got) * 0.9)] + 1) if got else None,
    }
    for N in (1, 5, 10, 20, 50):
        out[f"recall@{N}"] = (
            sum(1 for r in got if r < N) / n if n else 0.0)
    for k in (5, 20):
        out[f"doc_coverage@{k}"] = (sum(
            1 for rs in per_doc.values()
            if all(r is not None and r < k for r in rs)) / len(per_doc)
            if per_doc else 0.0)
    return out


def mcnemar(hits_a: dict, hits_b: dict, keys: list) -> dict:
    """Exact binomial test on discordant tokens (two-sided)."""
    b = sum(1 for k in keys if hits_a[k] and not hits_b[k])
    c = sum(1 for k in keys if hits_b[k] and not hits_a[k])
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "p": 1.0}
    lo = min(b, c)
    tail = sum(math.comb(n, i) for i in range(lo + 1)) / (2 ** n)
    return {"b": b, "c": c, "p": min(1.0, 2 * tail)}


# --------------------------------------------------------------------------


def _ranksets() -> dict[str, dict]:
    """Gold's rank (not just hit/miss) for the headline systems."""
    rows = _variant_rows()
    results = json.loads((ROOT / "results_ladder.json").read_text())
    out = {}
    for res in results:
        if res["stage"] not in ("B2u", "B3ur", "B3u"):
            continue
        pairs = _validated_detail_rows(res["stage"], res["detail"], rows)
        out[res["stage"]] = {
            (r["doc"], r["sent"], r["wid"]): d["rank"]
            for d, r in pairs}

    dump = [json.loads(line) for line in
            (ROOT / "dump_b3u.jsonl").open(encoding="utf-8")]
    att = load_attestation()
    for name in ("R5r + POS (Dilemma head, orig)",
                 "R5 + POS (gold, upper bound)"):
        scorer = make_scorer(name, att)
        ranks = {}
        for row in dump:
            gl = norm_lenient(row["gold"])
            ranked = sorted(row["cands"], key=lambda c: scorer(row, c))
            ranks[row["doc"], row["sent"], row["wid"]] = next(
                (i for i, c in enumerate(ranked)
                 if norm_lenient(c[0]) == gl), None)
        out[name] = ranks
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    rows = _variant_rows()
    keys = [(r["doc"], r["sent"], r["wid"]) for r in rows
            if args.split == "all" or r["split"] == args.split]
    print(f"split={args.split}  tokens={len(keys)}  "
          f"documents={len({k[0] for k in keys})}\n")

    systems = ladder_systems(args.k)
    print("positional join verified on all rows of all ladder stages")
    systems.update(rerank_systems(args.k))

    print(f"\n=== recall@{args.k}, cluster bootstrap over documents "
          f"({args.boot} resamples)\n")
    print(f"{'system':<36}{'point':>8}{'95% CI':>20}")
    marks = {}
    for name, hits in systems.items():
        r = bootstrap(hits, None, keys, args.boot)
        marks[name] = r
        interval = f"[{r['lo']:.1%}, {r['hi']:.1%}]"
        print(f"{name:<36}{r['point']:>8.1%}"
              f"{interval:>20}")

    print(f"\n=== pre-specified paired comparisons (difference in recall@{args.k})\n")
    print(f"{'comparison':<44}{'diff':>8}{'95% CI':>20}{'McNemar p':>12}  claim")
    out = {"split": args.split, "k": args.k, "n_boot": args.boot,
           "n_tokens": len(keys), "n_docs": len({k[0] for k in keys}),
           "marginal": {n: r for n, r in marks.items()}, "comparisons": []}
    for a, b, claim in PRE_SPECIFIED:
        if a not in systems or b not in systems:
            print(f"  SKIP {a} vs {b} (missing)")
            continue
        d = bootstrap(systems[a], systems[b], keys, args.boot)
        m = mcnemar(systems[a], systems[b], keys)
        sig = "" if (d["lo"] <= 0 <= d["hi"]) else "  *"
        label = f"{a.split(' ')[0]} vs {b.split(' ')[0]}"
        interval = f"[{d['lo']:+.1%}, {d['hi']:+.1%}]"
        print(f"{label:<44}{d['point']:>+8.1%}"
              f"{interval:>20}{m['p']:>12.3f}"
              f"  {claim}{sig}")
        out["comparisons"].append(
            {"a": a, "b": b, "claim": claim, "diff": d, "mcnemar": m,
             "interval_excludes_zero": not (d["lo"] <= 0 <= d["hi"])})

    # Reader-cost profile for the systems the write-up reports.
    print("\n=== reader cost (what the researcher actually has to scan)\n")
    print(f"{'system':<36}{'med':>6}{'p90':>6}{'@1':>7}{'@5':>7}{'@20':>7}"
          f"{'@50':>7}{'doc@5':>8}{'doc@20':>8}")
    ranksets = _ranksets()
    profiles = {}
    for name, ranks in ranksets.items():
        pr = rank_profile(ranks, keys)
        profiles[name] = pr
        print(f"{name:<36}{str(pr['rank_median']):>6}{str(pr['rank_p90']):>6}"
              + "".join(f"{pr[f'recall@{N}']:>7.1%}" for N in (1, 5, 20, 50))
              + f"{pr['doc_coverage@5']:>8.1%}{pr['doc_coverage@20']:>8.1%}")
    out["reader_cost"] = profiles

    # Sanity: a system against itself must give a degenerate zero interval.
    self_check = bootstrap(systems["B3u"], systems["B3u"], keys, args.boot)
    ok = self_check["lo"] == self_check["hi"] == 0.0
    print(f"\nsanity: B3u vs B3u -> "
          f"[{self_check['lo']:+.1%}, {self_check['hi']:+.1%}] "
          f"{'OK' if ok else 'FAILED'}")
    out["self_check_ok"] = ok

    (ROOT / f"results_stats_{args.split}_k{args.k}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten to results_stats_{args.split}_k{args.k}.json")
    print("\n* = bootstrap interval excludes zero. McNemar assumes token "
          "independence and is optimistic; where the two disagree, the "
          "clustered interval is the one to report.")


if __name__ == "__main__":
    main()
