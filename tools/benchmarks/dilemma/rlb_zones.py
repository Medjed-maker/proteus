"""Split the candidate list into a confident zone and an explore zone.

§1.1 found that every effect flips sign with k, and the pair that matters here
is B3ur vs B2u: the fifteen alternations win at @1 (+3.2*) and lose at @20
(-2.4*). The rules take the head, blind edit distance takes the tail, and B3u
is best at both because it runs them in competition. That is an argument for
*labelling* the two generators in the returned list rather than choosing one:

    confident  reached by one of the fifteen alternations
               -> carries a rule name, an example, a citation. This is the
                  whole of "explainable".
    explore    reached only by blind edit distance
               -> no rule name. Displayed as "variation not described by any
                  known rule", which is an honest label rather than an absent
                  one.

The second use is the interesting one. A gold that lands in the explore zone is
a spelling change the fifteen frozen rules cannot name, so accumulating those
gives a ranked list of candidate sixteenth rules -- and, where the change looks
unmotivated, a list of editorial decisions worth auditing. Classifying the
failures is itself the product feature.

Runs entirely offline against a frozen dump. Nothing here re-runs a search.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from categorize import labels
from rlb_keys import b3_key, b3_variants, collapse
from run_eval import norm_lenient

ROOT = Path(__file__).parent

# Costs the ladder actually emits. 0.5 and 2.0 never occur: blind-at-distance-0
# is the exact key, which already scored 0.0, and rules-at-distance-0 is
# b3_key(word) itself, which b3_variants already yielded at 1.0. Both are
# shadowed by the min() in _from_keys.
COST_EXACT, COST_RULES, COST_RULES1 = 0.0, 1.0, 3.0
COST_BLIND1, COST_BLIND2 = 1.5, 2.5


def rule_reachable(word: str, src_key: str, memo: dict) -> bool:
    """Could the fifteen alternations have produced this source key?

    Not the same question as ``cost < 1.5``. ``_from_keys`` keeps the *minimum*
    tuple per lemma, so a candidate reached both by a rule (3.0) and by blind
    Levenshtein-1 (1.5) is recorded at 1.5 and looks like an explore-zone hit.
    Reading the zone off the cost alone mislabels 87.9% of the cost-1.5 slots.

    Recovering the truth needs no index and no re-search. ``paired.get(ck)``
    returns exactly the b1 keys whose collapse is ``ck``, so membership in it is
    equality of collapsed keys, and ``space3.near(b3_key(word), 1)`` is a
    Levenshtein bound on the same collapsed key. Both become O(1) string tests:

        rules      collapse(src_key) in b3_variants(word)
        rules+1    Lev(collapse(src_key), b3_key(word)) <= 1

    Verified against the frozen dump: every candidate recorded at 0.0, 1.0 or
    3.0 tests positive here, so the reconstruction has no false negatives.
    """
    if word not in memo:
        memo[word] = (frozenset(b3_variants(word)), b3_key(word))
    variants, bkey = memo[word]
    collapsed = collapse(src_key)
    if collapsed in variants:
        return True
    return Levenshtein.distance(collapsed, bkey) <= 1


def zone_of_cost(cost: float) -> str:
    """The naive zone, from the cost alone. Kept only for the contrast."""
    return "explore" if (cost % 1.0) == 0.5 else "confident"


def annotate_row(row: dict, memo: dict) -> dict:
    """Each candidate gains its zone and, in the confident zone, rule names."""
    out = []
    for cand in row["cands"]:
        lemma, cost, ff, lf, src = cand[0], cand[1], cand[2], cand[3], cand[4]
        confident = rule_reachable(row["form"], src, memo)
        rules = labels(row["form"], src) if confident else []
        out.append({
            "lemma": lemma, "cost": cost, "form_freq": ff, "lemma_freq": lf,
            "src_key": src,
            "zone": "confident" if confident else "explore",
            "zone_naive": zone_of_cost(cost),
            "rules": rules,
        })
    return {**row, "cands": out}


def gold_index(row: dict) -> int | None:
    gold = norm_lenient(row["gold"])
    for i, cand in enumerate(row["cands"]):
        lemma = cand["lemma"] if isinstance(cand, dict) else cand[0]
        if norm_lenient(lemma) == gold:
            return i
    return None


def cmd_annotate(args) -> None:
    memo: dict = {}
    n = 0
    slots = Counter()
    out_path = ROOT / args.out
    with out_path.open("w", encoding="utf-8") as fh:
        for line in (ROOT / args.dump).open(encoding="utf-8"):
            row = annotate_row(json.loads(line), memo)
            for cand in row["cands"]:
                slots[(cand["zone"], cand["zone_naive"])] += 1
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"annotated {n} rows -> {args.out}")
    print("candidate slots (true zone, naive zone):")
    for (true, naive), c in sorted(slots.items()):
        flag = "  <- mislabelled by cost alone" if true != naive else ""
        print(f"  {true:9s} / {naive:9s} {c:>7d}{flag}")


def cmd_report(args) -> None:
    rows = [json.loads(line)
            for line in (ROOT / args.inp).open(encoding="utf-8")]
    if args.split != "all":
        rows = [r for r in rows if r["split"] == args.split]

    k = args.k
    gold_zone = Counter()
    hits, keys = {}, []
    # The pre-registered metric is conditioned on a hit, so the bootstrap is
    # restricted to hit tokens too -- resampling over all tokens would put the
    # miss rate into the denominator and estimate a different quantity.
    explore_only, hit_keys = {}, []

    for row in rows:
        key = (row["doc"], row["sent"], row["wid"])
        keys.append(key)
        idx = gold_index(row)
        hit = idx is not None and idx < k
        hits[key] = int(hit)
        if not hit:
            gold_zone["miss"] += 1
            continue
        zone = row["cands"][idx]["zone"]
        gold_zone[zone] += 1
        hit_keys.append(key)
        explore_only[key] = int(zone == "explore")

    n = len(rows)
    n_hit = sum(hits.values())
    n_explore = sum(explore_only.values())

    print(f"split {args.split}  tokens {n}  docs {len({r['doc'] for r in rows})}"
          f"  k={k}")
    print(f"recall@{k} = {n_hit / n:.1%}")
    print("gold arrived via:")
    for z in ("confident", "explore", "miss"):
        print(f"  {z:10s} {gold_zone[z]:5d}  "
              f"{gold_zone[z] / n:6.1%} of tokens" +
              (f"  ({gold_zone[z] / n_hit:.1%} of hits)"
               if z != "miss" and n_hit else ""))

    share = n_explore / n_hit if n_hit else 0.0
    verdict = ("rules materially incomplete" if share > 0.30 else
               "rules do essentially all the work" if share < 0.10 else
               "inconclusive")
    print("\nH4 pre-registered thresholds: <10% safety-net, >30% incomplete")
    print(f"explore-only share of hits = {share:.1%}  ->  {verdict}")

    res = {"split": args.split, "k": k, "n_tokens": n,
           "n_docs": len({r["doc"] for r in rows}),
           f"recall{k}": n_hit / n if n else 0.0,
           "gold_zone": dict(gold_zone),
           "explore_only_share_of_hits": share}

    try:
        from rlb_stats import bootstrap
        res["explore_only_boot"] = bootstrap(explore_only, None, hit_keys,
                                             n_boot=args.boot)
        b = res["explore_only_boot"]
        print(f"bootstrap over hit tokens (documents as clusters, "
              f"n={args.boot}): {b['point']:.1%} [{b['lo']:.1%}, {b['hi']:.1%}]")
    except Exception as exc:                        # noqa: BLE001
        res["explore_only_boot"] = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"bootstrap unavailable: {exc}")

    (ROOT / args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    print(f"wrote {args.out}")


def cmd_undescribed(args) -> None:
    """Explore-zone gold hits, grouped by the alternation they represent.

    These are the spellings the fifteen frozen rules cannot name but blind edit
    distance still recovered -- the candidate sixteenth rules.
    """
    rows = [json.loads(line)
            for line in (ROOT / args.inp).open(encoding="utf-8")]
    if args.split != "all":
        rows = [r for r in rows if r["split"] == args.split]

    groups: dict[str, list] = {}
    for row in rows:
        idx = gold_index(row)
        if idx is None or idx >= args.k:
            continue
        cand = row["cands"][idx]
        if cand["zone"] != "explore":
            continue
        # No rule matched, so name the alternation from the raw diff instead.
        name = " + ".join(sorted(set(labels(row["form"], cand["src_key"]))))
        groups.setdefault(name or "accent/breathing only", []).append(
            {"doc": row["doc"], "form": row["form"], "reg": row.get("reg", ""),
             "gold": row["gold"], "src_key": cand["src_key"],
             "cost": cand["cost"]})

    ranked = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    out = [{"alternation": name, "count": len(items),
            "examples": items[:8]}
           for name, items in ranked if len(items) >= args.min_count]

    print(f"split {args.split}  explore-zone gold hits: "
          f"{sum(len(v) for v in groups.values())} in {len(groups)} groups")
    print(f"groups with count >= {args.min_count}: {len(out)}")
    for g in out[:15]:
        print(f"  {g['count']:4d}  {g['alternation']}")

    (ROOT / args.out).write_text(
        json.dumps({"split": args.split, "k": args.k,
                    "min_count": args.min_count, "groups": out},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("annotate")
    a.add_argument("--dump", default="dump_b3u_feat.jsonl")
    a.add_argument("--out", default="dump_b3u_zoned.jsonl")
    a.set_defaults(fn=cmd_annotate)

    r = sub.add_parser("report")
    r.add_argument("--in", dest="inp", default="dump_b3u_zoned.jsonl")
    r.add_argument("--split", default="dev")
    r.add_argument("--k", type=int, default=5)
    r.add_argument("--boot", type=int, default=1000)
    r.add_argument("--out", default="")
    r.set_defaults(fn=cmd_report)

    u = sub.add_parser("undescribed")
    u.add_argument("--in", dest="inp", default="dump_b3u_zoned.jsonl")
    u.add_argument("--split", default="dev")
    u.add_argument("--k", type=int, default=5)
    u.add_argument("--min-count", type=int, default=3)
    u.add_argument("--out", default="alternations_undescribed.json")
    u.set_defaults(fn=cmd_undescribed)

    args = ap.parse_args()
    if getattr(args, "out", None) == "" and args.cmd == "report":
        args.out = f"results_zones_{args.split}.json"
    args.fn(args)


if __name__ == "__main__":
    main()
