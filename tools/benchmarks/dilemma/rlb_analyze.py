"""Split B3's residual failures into the two kinds that imply opposite decisions.

    (a) the gold lemma IS in the candidate set, but below rank 5.
        The generator found it; the ranker buried it. A phonological distance
        matrix does not fix this -- frequency, context, POS and dialect
        constraints do, and those live in Layer 2.

    (b) the gold lemma is NOT in the candidate set at all.
        The fifteen alternations could not describe the alternation. This is
        the only bucket a richer distance model could reach.

    (c) unreachable: the dictionary never links the regularised form to the
        gold lemma (B0's ceiling). Nothing in Layer 1 or Layer 2 fixes this;
        it is an artefact of lemma-convention mismatch between PapyGreek and
        Wiktionary, and it must be subtracted before (a) and (b) are read.

The go/no-go on B4 is decided by (a) vs (b) among reachable tokens, not by the
headline recall number.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from categorize import label
from clean import clean
from rlb_lexicon import Lexicon
from run_eval import load

ROOT = Path(__file__).parent


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="B3u")
    args = ap.parse_args()
    results = json.loads(
        (ROOT / "results_ladder.json").read_text(encoding="utf-8"))
    by_stage = {r["stage"]: r for r in results}
    stage = args.stage if args.stage in by_stage else results[-1]["stage"]
    detail = by_stage[stage]["detail"]

    # Unreachable means the dictionary has no such lemma at all -- the union
    # generators reach a lemma through any of its forms, not only through the
    # editor's regularised one, so B0's form-linked ceiling is not the bound
    # here and using it would misclassify real generation successes.
    lex = Lexicon()
    reach = {}
    for r in load(ROOT / "dataset.jsonl"):
        if r["stratum"] != "variant_ortho":
            continue
        reach[r["input"], r["lemma_gold"]] = lex.has_lemma(r["lemma_gold"])

    ct = Counter()
    buckets = {"a": [], "b": []}
    for d in detail:
        if d["split"] != "test":
            continue
        ct["n"] += 1
        rank = d["rank"]
        if rank is not None and rank < 5:
            ct["hit@5"] += 1
            continue
        reachable = reach.get((d["form"], d["gold"]), False)
        if not reachable:
            ct["c_unreachable"] += 1
            continue
        ct["reachable_miss"] += 1
        if rank is not None:
            ct["a_ranking"] += 1
            buckets["a"].append(d)
        else:
            ct["b_generation"] += 1
            buckets["b"].append(d)

    n = ct["n"]
    print(f"stage {stage}, test split, n={n}\n")
    for k, lab in [("hit@5", "recall@5 hit"),
                   ("c_unreachable", "(c) unreachable: not in the dictionary"),
                   ("a_ranking", "(a) found but ranked >=5  -> Layer 2"),
                   ("b_generation", "(b) never generated     -> Layer 1")]:
        print(f"{lab:<48} {ct[k]:>5}  {_ratio(ct[k], n):>6.1%}")

    reachable_n = ct["hit@5"] + ct["a_ranking"] + ct["b_generation"]
    print(f"\namong reachable tokens (n={reachable_n}):")
    print(f"  recall@5           {_ratio(ct['hit@5'], reachable_n):>6.1%}")
    if ct["reachable_miss"]:
        print(f"  of the misses, (a) ranking    "
              f"{ct['a_ranking'] / ct['reachable_miss']:>6.1%}")
        print(f"                 (b) generation "
              f"{ct['b_generation'] / ct['reachable_miss']:>6.1%}")

    print("\n(b) by alternation -- what a distance matrix would have to cover:")
    bad = Counter(label(clean(d["form"]), clean(d["reg"])) for d in buckets["b"])
    for lb, c in bad.most_common(12):
        print(f"  {lb:<44} {c:>4}")

    (ROOT / f"results_decision_{stage}.json").write_text(json.dumps(
        {"stage": stage, "counts": dict(ct),
         "b_by_alternation": bad.most_common(),
         "examples_a": buckets["a"][:40], "examples_b": buckets["b"][:40]},
        ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
