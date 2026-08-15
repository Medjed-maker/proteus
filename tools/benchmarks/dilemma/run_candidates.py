"""Second UVP axis: can Dilemma *reach* the right headword at all, and does it
say why?

Top-1 accuracy alone cannot tell a competitor apart from a dead end. If the
gold lemma never appears among the returned candidates for a non-standard
spelling, no amount of downstream re-ranking rescues it -- the reverse lookup
simply does not generate that hypothesis. That is the gap a phonological-rule
reverse index is supposed to fill.

Also records the provenance strings Dilemma attaches to each candidate
(``source``/``via``), which is the closest thing it offers to an explanation.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

from run_eval import load, norm_lenient, norm_strict

ROOT = Path(__file__).parent


def main() -> None:
    from dilemma import Dilemma

    rows = load(ROOT / "dataset.jsonl")
    d = Dilemma(lang="grc", resolve_articles=True, normalize=True,
                period="hellenistic")

    per = defaultdict(Counter)
    provenance = defaultdict(Counter)
    ncand = defaultdict(list)
    misses = defaultdict(list)

    for r in rows:
        s = r["stratum"]
        gold = r["lemma_gold"]
        cands = d.lemmatize_verbose(r["input"])
        lemmas = [c.lemma for c in cands]

        per[s]["n"] += 1
        ncand[s].append(len(cands))
        for c in cands:
            provenance[s][f"{c.source}|{c.via}"] += 1

        if lemmas and norm_strict(lemmas[0]) == norm_strict(gold):
            per[s]["top1"] += 1
        if any(norm_lenient(x) == norm_lenient(gold) for x in lemmas):
            per[s]["in_candidates"] += 1
        else:
            misses[s].append((r["input"], r["input_reg"], gold, lemmas[:5]))

    out = {
        "per_stratum": {k: dict(v) for k, v in per.items()},
        "mean_candidates": {k: round(sum(v) / len(v), 2)
                            for k, v in ncand.items()},
        "provenance_top": {k: v.most_common(15)
                           for k, v in provenance.items()},
        "misses": {k: v[:300] for k, v in misses.items()},
    }
    (ROOT / "results_candidates.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{'stratum':<14} {'n':>6} {'top1':>9} {'gold in cands':>15} {'mean #':>8}")
    for s, v in per.items():
        n = v["n"]
        print(f"{s:<14} {n:>6} {v['top1']/n:>8.1%} "
              f"{v['in_candidates']/n:>14.1%} "
              f"{out['mean_candidates'][s]:>8}")


if __name__ == "__main__":
    main()
