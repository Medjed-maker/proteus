"""Run the B3u generator over the DDbDP orig->reg benchmark.

Same search as the PapyGreek ladder, form target instead of lemma target. The
output mirrors dump_b3u.jsonl field for field, so rlb_stats.py, rlb_zones.py
and rlb_lm.py rerank all load it without modification -- which is the only
reason the two benchmarks are comparable at all.

The task: given the papyrus spelling, rank candidate standard forms; gold is
the editor's <reg>. The pre-registered prediction (H3) is that dev recall@5
lands in [80%, 95%], since PapyGreek's B0 puts reg->lemma at 83.3% and the
known dev lemma recall is 74.55%. Outside that band means the extractor or the
form target is wrong, and neither benchmark gets reported until it is found.
"""

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

from benchmark_provenance import (
    benchmark_code_identity,
    dilemma_data_identity,
    file_identity,
)
from rlb_ddb_splits import dev_docs
from rlb_ladder import Ladder, serialize_candidates
from rlb_lexicon import DATA
from run_eval import candidate_ranks

ROOT = Path(__file__).parent


def result_provenance(pairs_path: Path, splits_path: Path) -> dict:
    """Identify the code and inputs that define a DDbDP measurement."""
    return {
        "benchmark_code": benchmark_code_identity(),
        "inputs": {
            "pairs": file_identity(pairs_path),
            "splits": file_identity(splits_path),
            "dilemma_data": dilemma_data_identity(DATA),
        },
    }


def _gold_ranks(candidates: list[str], gold: str) -> tuple[int | None,
                                                           int | None]:
    """Return the first lenient and strict ranks without stopping early."""
    return candidate_ranks(candidates, gold)


def sample(rows: list[dict], per_doc_cap: int, n_docs: int, seed: int) -> list:
    """Stratified-by-document subsample.

    Pairs cluster hard by document -- one letter can carry 1,105 of them -- so
    an unrestricted draw would let a handful of long documents dominate both
    the estimate and its bootstrap. Capping per document and then sampling
    documents keeps the cluster structure that rlb_stats.bootstrap assumes.
    """
    by_doc: dict[str, list] = {}
    for r in rows:
        by_doc.setdefault(r["doc"], []).append(r)
    rnd = random.Random(seed)
    docs = sorted(by_doc)
    if n_docs and n_docs < len(docs):
        docs = rnd.sample(docs, n_docs)
    out = []
    for d in sorted(docs):
        rs = by_doc[d]
        if per_doc_cap and len(rs) > per_doc_cap:
            rs = rnd.sample(rs, per_doc_cap)
        out.extend(sorted(rs, key=lambda r: int(r["wid"])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="ddb_pairs_strata.jsonl")
    ap.add_argument("--stratum", default="ortho",
                    help="ortho | lex | unknown | all. Default ortho, which "
                         "matches PapyGreek's variant_ortho definition")
    ap.add_argument("--splits", default="ddb_splits.json")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--stage", default="B3u")
    ap.add_argument("--per-doc-cap", type=int, default=5)
    ap.add_argument("--n-docs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260812)
    ap.add_argument("--out", default="")
    ap.add_argument("--dump", default="")
    args = ap.parse_args()

    pairs_path = ROOT / args.pairs
    splits_path = ROOT / args.splits
    rows = [json.loads(line) for line in pairs_path.open(encoding="utf-8")]
    dev = dev_docs(splits_path)
    for r in rows:
        r["split"] = "dev" if r["doc"] in dev else "test"
    rows = [r for r in rows if args.split == "all" or r["split"] == args.split]
    if args.stratum != "all":
        before = len(rows)
        rows = [r for r in rows if r.get("stratum") == args.stratum]
        print(f"stratum={args.stratum}: {len(rows)}/{before} pairs")
    rows = sample(rows, args.per_doc_cap, args.n_docs, args.seed)
    print(f"split={args.split}  pairs={len(rows)}  "
          f"documents={len({r['doc'] for r in rows})}")

    ladder = Ladder(target="form")
    dump = open(ROOT / args.dump, "w", encoding="utf-8") if args.dump else None

    t0 = time.time()
    ct = Counter()
    sizes, ranks = [], []
    for i, r in enumerate(rows):
        scored = ladder.scored(r["form"], args.stage)
        cands = Ladder._rank(scored)
        sizes.append(len(cands))

        rank_l, rank_s = _gold_ranks(cands, r["gold"])

        ct["n"] += 1
        if not cands:
            ct["no_cand"] += 1
        for k in (1, 5, 10, 20, 50):
            if rank_l is not None and rank_l < k:
                ct[f"hit{k}"] += 1
        if rank_s is not None and rank_s < 5:
            ct["hit5_strict"] += 1
        if rank_l is not None:
            ct["hit_inf"] += 1
            ranks.append(rank_l)

        if dump is not None:
            dump.write(json.dumps({
                "doc": r["doc"], "sent": r["sent"], "wid": r["wid"],
                "postag": "", "date_nb": "", "date_na": "",
                "form": r["form"], "reg": r["reg"], "gold": r["gold"],
                "split": r["split"], "n_cand": len(cands),
                "series": r["series"], "volume": r["volume"],
                "flags": r.get("flags", []),
                "cands": serialize_candidates(cands, scored),
            }, ensure_ascii=False) + "\n")

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(rows)}  recall@5={ct['hit5'] / ct['n']:.1%}"
                  f"  {time.time() - t0:.0f}s", flush=True)
    if dump:
        dump.close()

    n = ct["n"]
    ranks.sort()
    res = {"split": args.split, "stage": args.stage, "target": "form",
           "stratum": args.stratum,
           "n_pairs": n, "n_docs": len({r["doc"] for r in rows}),
           "per_doc_cap": args.per_doc_cap, "seed": args.seed,
           "recall": {f"@{k}": ct[f"hit{k}"] / n for k in (1, 5, 10, 20, 50)},
           "recall_inf": ct["hit_inf"] / n,
           "recall5_strict": ct["hit5_strict"] / n,
           "no_cand": ct["no_cand"] / n,
           "cand_median": sorted(sizes)[len(sizes) // 2],
           "cand_p90": sorted(sizes)[int(0.9 * len(sizes))],
           "gold_rank_median": ranks[len(ranks) // 2] if ranks else None,
           "gold_rank_p90": ranks[int(0.9 * len(ranks))] if ranks else None,
           "seconds": time.time() - t0,
           "provenance": result_provenance(pairs_path, splits_path)}

    print(f"\n{args.stage} form target, {n} pairs, "
          f"{res['n_docs']} documents, {res['seconds']:.0f}s")
    for k in (1, 5, 10, 20, 50):
        print(f"  recall@{k:<3} {res['recall'][f'@{k}']:.1%}")
    print(f"  recall@inf {res['recall_inf']:.1%}   "
          f"strict@5 {res['recall5_strict']:.1%}   "
          f"no candidates {res['no_cand']:.1%}")
    print(f"  |C| median {res['cand_median']}  p90 {res['cand_p90']}")

    # H3's pre-registered band. Outside it, something is wrong upstream and
    # neither benchmark is reported until it is found.
    r5 = res["recall"]["@5"]
    if args.split == "dev":
        verdict = ("within the pre-registered [80%, 95%] band"
                   if 0.80 <= r5 <= 0.95 else
                   "!! OUTSIDE the pre-registered [80%, 95%] band -- "
                   "diagnose before reporting")
        print(f"\nH3: dev recall@5 = {r5:.1%}, {verdict}")
        print(f"    implied lemma recall = {r5:.1%} x 0.833 = {r5 * 0.833:.1%}"
              f"  (independence assumed; known dev lemma recall 74.6%)")

    out = args.out or f"results_ddb_{args.split}_{args.stage}.json"
    (ROOT / out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
