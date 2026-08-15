"""H5: do a researcher's own constraints cut the candidate list down to size?

§1.2 measured period/dialect/genre as *ranking* signals and found them worth
nothing -- which §2.2 then explained: candidates reached by the same
alternation share the very information a prior would score them on, so adding
points cannot reorder them.

Filtering is a different operation. Adding points reorders a set; excluding
shrinks it. The negation of one is not the negation of the other, and the
faceted-UI hypothesis rests entirely on that distinction. This measures the
second one, using only the frozen dump -- no humans, no interface.

The leverage is an asymmetry already in the data: 92.8% of gold lemmas carry an
attestation entry against 75.5% of candidate slots, because golds are real
words and most of the noise is not. So the interesting question is not whether
a facet helps but what happens to candidates whose attribute is simply
*unknown* -- keeping them protects recall and blunts the filter, dropping them
cuts a quarter of the search space at a cost that has to be measured rather
than assumed. Both policies are reported.
"""

import argparse
import json
from pathlib import Path

from rlb_keys import b1_key
from rlb_rerank import AGDT_POS, KOINE, centuries_for, load_attestation
from rlb_stats import bootstrap, rank_profile
from run_eval import norm_lenient

ROOT = Path(__file__).parent

FACETS = ("pos", "century", "dialect", "initial", "zone")
DEV_HEADLINE = "aggressive|pos"


# -- per-facet predicates ---------------------------------------------------
# Each returns True (keep), False (drop), or None (attribute unknown -- the
# caller decides by policy). Keeping "unknown" as a third value rather than
# folding it into False is the whole point: it is what separates the
# conservative and aggressive policies.


def keep_pos(row, cand, att, ctx) -> bool | None:
    want = AGDT_POS.get((row.get("postag") or "")[:1])
    if not want:
        return None
    a = att.get(cand[0])
    got = (a or {}).get("dominant_pos")
    if not got:
        return None
    return got == want


def keep_century(row, cand, att, ctx) -> bool | None:
    a = att.get(cand[0])
    cent = (a or {}).get("by_century") or {}
    if not cent:
        return None
    return any(cent.get(c, 0) for c in ctx["centuries"])


def keep_dialect(row, cand, att, ctx) -> bool | None:
    a = att.get(cand[0])
    dial = (a or {}).get("by_dialect") or {}
    if not dial:
        return None
    return any(dial.get(d, 0) for d in KOINE)


def keep_initial(row, cand, att, ctx) -> bool | None:
    """The researcher can read the first letter off the papyrus.

    Compared on the accent-stripped key, because breathings and accents are
    editorial and the scribe's first letter is what is actually visible.
    """
    q, c = ctx["query_key"], b1_key(cand[0])
    if not q or not c:
        return None
    return q[0] == c[0]


def keep_zone(row, cand, att, ctx) -> bool | None:
    """Restrict to candidates one of the fifteen alternations can name."""
    from rlb_zones import rule_reachable
    return rule_reachable(row["form"], cand[4], ctx["zone_memo"])


PREDICATES = {"pos": keep_pos, "century": keep_century,
              "dialect": keep_dialect, "initial": keep_initial,
              "zone": keep_zone}


def apply_facets(row, att, facets, policy, zone_memo) -> list:
    ctx = {"centuries": centuries_for(row),
           "query_key": b1_key(row["form"]),
           "zone_memo": zone_memo}
    out = []
    for cand in row["cands"]:
        ok = True
        for f in facets:
            v = PREDICATES[f](row, cand, att, ctx)
            if v is False or (v is None and policy == "aggressive"):
                ok = False
                break
        if ok:
            out.append(cand)
    return out


def gold_rank(cands, gold_l) -> int | None:
    for i, c in enumerate(cands):
        if norm_lenient(c[0]) == gold_l:
            return i
    return None


def measure(rows, att, facets, policy, zone_memo) -> dict:
    keys, ranks, sizes = [], {}, []
    kept_gold = lost_gold = 0
    for row in rows:
        key = (row["doc"], row["sent"], row["wid"])
        keys.append(key)
        gold_l = norm_lenient(row["gold"])
        before = gold_rank(row["cands"], gold_l)
        cands = apply_facets(row, att, facets, policy, zone_memo)
        after = gold_rank(cands, gold_l)
        ranks[key] = after
        sizes.append(len(cands))
        if before is not None:
            if after is None:
                lost_gold += 1
            else:
                kept_gold += 1

    sizes.sort()
    pr = rank_profile(ranks, keys)
    reachable = kept_gold + lost_gold
    return {
        "facets": list(facets), "policy": policy,
        "n_tokens": len(keys), "n_docs": len({k[0] for k in keys}),
        "cand_median": sizes[len(sizes) // 2],
        "cand_p90": sizes[int(len(sizes) * 0.9)],
        "cand_max": sizes[-1],
        "cand_mean": sum(sizes) / len(sizes),
        "empty": sum(1 for s in sizes if s == 0) / len(sizes),
        # recall@inf retained: of the golds the generator DID reach, how many
        # survive the filter. This is the number that decides H5 -- a filter
        # that shrinks the list by killing the answer is worthless.
        "gold_reachable_before": reachable,
        "gold_kept": kept_gold,
        "gold_lost": lost_gold,
        "recall_inf_retained": kept_gold / reachable if reachable else 0.0,
        "recall@1": pr["recall@1"], "recall@5": pr["recall@5"],
        "recall@20": pr["recall@20"],
        "recall_inf": 1.0 - pr["unresolved"],
        "doc_coverage@5": pr["doc_coverage@5"],
        "doc_coverage@20": pr["doc_coverage@20"],
        "rank_median": pr["rank_median"], "rank_p90": pr["rank_p90"],
        "_ranks": ranks, "_keys": keys,
    }


def verdict(res: dict) -> str:
    """H5's pre-registered thresholds."""
    if res["cand_median"] <= 20 and res["recall_inf_retained"] >= 0.85:
        return "ACCEPT"
    if res["recall_inf_retained"] < 0.85 or res["cand_median"] > 50:
        return "REJECT"
    return "INCONCLUSIVE"


def headline_selection(results: dict, split: str) -> dict:
    """Return the recall-maximising dev configuration for every split.

    Test must reuse the choice made on dev rather than selecting on test. The
    assertion makes a changed dev winner visible instead of silently leaving
    the pre-registered headline stale.
    """
    if split == "dev":
        eligible = ((name, result) for name, result in results.items()
                    if name != "none" and "recall@5" in result)
        winner = max(eligible, key=lambda item: (item[1]["recall@5"], item[0]))
        if winner[0] != DEV_HEADLINE:
            raise RuntimeError(
                f"dev recall@5 winner changed from {DEV_HEADLINE!r} "
                f"to {winner[0]!r}; update the pre-registered headline")

    selected = results[DEV_HEADLINE]
    return {
        "name": DEV_HEADLINE,
        "selection_metric": "dev recall@5",
        "selected_on": "dev",
        "recall5_vs_unfiltered": selected["recall5_ci"],
    }


def row_str(label: str, r: dict) -> str:
    ci = r.get("recall5_ci")
    ci_s = f"  [{ci['lo']:+.1f}, {ci['hi']:+.1f}]" if ci else ""
    return (f"{label:<34}{r['cand_median']:>7}{r['cand_p90']:>7}"
            f"{r['recall_inf_retained']:>10.1%}{r['recall@5']:>8.1%}"
            f"{r['doc_coverage@5']:>8.1%}{ci_s}")


def add_ci(r: dict, base: dict, n_boot: int) -> dict:
    """Paired interval on recall@5 against the unfiltered list."""
    hits = {k: int(v is not None and v < 5) for k, v in r["_ranks"].items()}
    ref = {k: int(v is not None and v < 5) for k, v in base["_ranks"].items()}
    d = bootstrap(hits, ref, r["_keys"], n_boot)
    r["recall5_ci"] = {"point": d["point"] * 100,
                       "lo": d["lo"] * 100, "hi": d["hi"] * 100}
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="dump_b3u_feat.jsonl")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = [json.loads(line)
            for line in (ROOT / args.dump).open(encoding="utf-8")]
    rows = [r for r in rows if args.split == "all" or r["split"] == args.split]
    att = load_attestation()
    zone_memo: dict = {}

    print(f"split={args.split}  tokens={len(rows)}  "
          f"documents={len({r['doc'] for r in rows})}\n")
    hdr = (f"{'filter':<34}{'|C|med':>7}{'|C|p90':>7}{'gold kept':>10}"
           f"{'rec@5':>8}{'cov@5':>8}   rec@5 vs none")

    results = {}
    base = measure(rows, att, (), "conservative", zone_memo)
    results["none"] = base

    for policy in ("conservative", "aggressive"):
        print(f"=== {policy} (unknown attribute -> "
              f"{'keep' if policy == 'conservative' else 'drop'})\n")
        print(hdr)
        print(row_str("(no filter)", base))
        # Single facets, then the cumulative stack in a fixed order.
        for f in FACETS:
            r = add_ci(measure(rows, att, (f,), policy, zone_memo),
                       base, args.boot)
            results[f"{policy}|{f}"] = r
            print(row_str(f, r))
        stack = []
        for f in FACETS:
            stack.append(f)
            r = add_ci(measure(rows, att, tuple(stack), policy, zone_memo),
                       base, args.boot)
            results[f"{policy}|+{'+'.join(stack)}"] = r
            if len(stack) > 1:
                print(row_str("+" + "+".join(stack), r))
        print()

    # H5 is decided on the cumulative stacks; single facets are reported
    # alongside because the pre-registered |C| threshold is a round number and
    # a configuration that misses it narrowly should be visible, not hidden.
    print("=== H5 pre-registered: ACCEPT if |C| median <=20 AND "
          "gold-kept >=85%\n")
    best = None
    for name, r in results.items():
        if name == "none":
            continue
        v = verdict(r)
        if "|+" in name and v == "ACCEPT" and (
                best is None or r["cand_median"] < best[1]["cand_median"]):
            best = (name, r)
        print(f"  {name:<46}{v:<14}|C|={r['cand_median']:<5}"
              f"gold={r['recall_inf_retained']:.1%}")
    print(f"\n  -> {'ACCEPT via ' + best[0] if best else 'no stack accepted'}")
    if best:
        results["_preregistered_best"] = {
            "name": best[0],
            "selection_metric": (
                "minimum candidate median among ACCEPT cumulative stacks"),
            "recall5_vs_unfiltered": best[1]["recall5_ci"],
        }
    results["_best"] = headline_selection(results, args.split)

    out = args.out or f"results_facet_{args.split}.json"
    (ROOT / out).write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
         if isinstance(v, dict) and "cand_median" in v else v
         for k, v in results.items()},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
