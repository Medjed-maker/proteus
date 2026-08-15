"""H1 proxy: how often is the buried gold lemma capitalised?

The residual between B3u's recall@5 (72.8%) and its oracle (92.3%) is 19.5
points, of which the context LM took 4.2. The remaining 15.2 has been treated
as a ranking problem because 81.6% of reachable failures are (a) rather than
(b). This asks a different question about the same bucket: documentary papyri
from Egypt carry an unusually high share of personal and place names (Greek,
transliterated Egyptian, Latin), and a proper noun is disadvantaged in every
term of the ranking -- the frequency tables come from literary Greek, the
trigram cannot predict a name, and ``dominant_pos`` collapses on it.

Capitalisation is only a proxy: without an onomasticon match it cannot establish
that a lemma is a proper noun or that an onomasticon is the required remedy.

Read-only against the frozen dump. Recomputes the bucket rather than reading
``results_decision_B3u.json``, which stores only 40 examples -- and records the
reconciliation with ``rlb_analyze.py``'s count, because the two definitions
differ in two ways that nearly cancel.
"""

import argparse
import json
import unicodedata
from collections import Counter
from pathlib import Path

from run_eval import norm_lenient

ROOT = Path(__file__).parent


def is_capitalised(lemma: str) -> bool:
    """First base letter is upper case.

    Decomposed first: a lemma like ``Ἀπύγχις`` starts with a precomposed
    capital-plus-breathing, and testing the composed character directly is
    fragile across the several ways that can be encoded.
    """
    if not lemma:
        return False
    for ch in unicodedata.normalize("NFD", lemma):
        if unicodedata.combining(ch):
            continue
        return ch.upper() == ch and ch.lower() != ch
    return False


def gold_rank(row: dict) -> int | None:
    """Index of the gold lemma in the dumped candidate list, or None."""
    gold = norm_lenient(row["gold"])
    for i, cand in enumerate(row["cands"]):
        if norm_lenient(cand[0]) == gold:
            return i
    return None


def bucket_of(row: dict, k: int) -> str:
    rank = gold_rank(row)
    if rank is None:
        return "absent"
    return "hit" if rank < k else "a"


def load_rows(dump: Path, split: str) -> list[dict]:
    rows = []
    for line in dump.open(encoding="utf-8"):
        row = json.loads(line)
        if split == "all" or row["split"] == split:
            rows.append(row)
    return rows


def analyse(rows: list[dict], k: int) -> dict:
    n = Counter()
    cap = Counter()
    pos = {b: Counter() for b in ("hit", "a", "absent")}
    examples = {b: [] for b in ("hit", "a", "absent")}
    truncated = 0

    for row in rows:
        # Historical dumps were capped at 500 candidates. A gold ranked past
        # that cap is indistinguishable here from one that was never generated,
        # so retain the diagnostic when reading one of those old artefacts.
        if row["n_cand"] > len(row["cands"]):
            truncated += 1

        b = bucket_of(row, k)
        n[b] += 1
        if is_capitalised(row["gold"]):
            cap[b] += 1
            # AGDT postags have no proper-noun tag, so this can only falsify a
            # capitalisation hit (a capitalised gold tagged as a verb would be
            # a bug), never confirm one.
            pos[b][(row.get("postag") or "?")[:1]] += 1
            if len(examples[b]) < 40:
                examples[b].append({"form": row["form"], "gold": row["gold"],
                                    "postag": row.get("postag", ""),
                                    "rank": gold_rank(row)})

    total = sum(n.values())
    total_cap = sum(cap.values())
    return {
        "k": k,
        "n_tokens": total,
        "n_docs": len({r["doc"] for r in rows}),
        "dump_cap_truncated": truncated,
        "base_rate_capitalised": total_cap / total if total else 0.0,
        "buckets": {
            b: {"n": n[b], "capitalised": cap[b],
                "share": cap[b] / n[b] if n[b] else 0.0,
                "postag_head": dict(pos[b].most_common())}
            for b in ("hit", "a", "absent")
        },
        "examples_capitalised": examples,
    }


def reconcile(rows: list[dict], k: int) -> dict:
    """Why this script counts 289 where rlb_analyze.py counts 284.

    rlb_analyze reads the ladder's own rank, which is uncapped, and gates on
    Lexicon.has_lemma; this script reads the capped dump and gates on lenient
    string equality. The two differences run in opposite directions and nearly
    cancel, which is exactly the kind of coincidence that becomes folklore if
    it is not written down.
    """
    try:
        from rlb_lexicon import Lexicon
    except Exception as exc:                       # noqa: BLE001
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    lex = Lexicon()
    lenient_only = []
    for row in rows:
        if bucket_of(row, k) != "a":
            continue
        if not lex.has_lemma(row["gold"]):
            lenient_only.append(row["gold"])

    return {
        "available": True,
        "a_by_lenient_match": sum(1 for r in rows if bucket_of(r, k) == "a"),
        "of_which_has_lemma_false": len(lenient_only),
        "has_lemma_false_lemmas": sorted(set(lenient_only)),
        "note": ("rlb_analyze files has_lemma-false tokens as (c) unreachable, "
                 "so subtract them; it also sees golds ranked beyond the "
                 "dump's 500-candidate cap, which this script cannot, so add "
                 "those back. The residual is the published 284."),
    }


def report(res: dict) -> None:
    print(f"tokens {res['n_tokens']}  docs {res['n_docs']}  "
          f"k={res['k']}  dump-cap truncated {res['dump_cap_truncated']}")
    print(f"{'bucket':10s} {'n':>6s} {'cap':>6s} {'share':>8s}")
    for b in ("hit", "a", "absent"):
        d = res["buckets"][b]
        print(f"{b:10s} {d['n']:6d} {d['capitalised']:6d} {d['share']:7.1%}")
    print(f"{'ALL':10s} {res['n_tokens']:6d} "
          f"{sum(res['buckets'][b]['capitalised'] for b in res['buckets']):6d} "
          f"{res['base_rate_capitalised']:7.1%}")

    share = res["buckets"]["a"]["share"]
    verdict = ("HIGH" if share > 0.50 else
               "LOW" if share < 0.20 else "INCONCLUSIVE")
    print("\nCapitalised-gold thresholds: >50% high, <20% low")
    print(f"(a) bucket capitalised gold share = {share:.1%}  ->  {verdict}")
    print("Capitalisation alone does not establish proper-noun or "
          "lexical-resource coverage.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="dump_b3u.jsonl")
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = load_rows(ROOT / args.dump, args.split)
    res = analyse(rows, args.k)
    res["dump"] = args.dump
    res["split"] = args.split
    res["reconciliation"] = reconcile(rows, args.k)
    report(res)

    out = args.out or f"results_propnoun_{args.split}.json"
    (ROOT / out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
