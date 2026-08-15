"""B0 -- the reachability ceiling.

Every rung of the ladder ends by naming a lemma that lookup.db already knows,
reached through a form lookup.db already knows. So before measuring any rung,
measure how many of the 2,160 UVP tokens are reachable *at all*:

    (1) is the editor's regularised spelling in the dictionary?
    (2) is the gold lemma in the dictionary?
    (3) does the dictionary link the two?

(3) is the hard ceiling for B1..B5 alike. A recall of 60% means something
completely different against a ceiling of 70% than against a ceiling of 95%,
which is why this runs first.

A fourth number, (3k), relaxes (3) to the accent-stripped key bucket: that is
the ceiling for the key-based rungs specifically, and it can only be higher.
"""

import json
from collections import Counter
from pathlib import Path

from benchmark_provenance import (
    benchmark_code_identity,
    dilemma_data_identity,
    file_identity,
)
from rlb_keys import b1_key
from rlb_lexicon import DATA, Lexicon
from rlb_splits import tag
from run_eval import load, norm_lenient, norm_strict

ROOT = Path(__file__).parent


def result_provenance() -> dict:
    """Identify the code and inputs that define the B0 measurement."""
    return {
        "benchmark_code": benchmark_code_identity(),
        "inputs": {
            "dataset": file_identity(ROOT / "dataset.jsonl"),
            "splits": file_identity(ROOT / "splits.json"),
            "dilemma_data": dilemma_data_identity(DATA),
        },
    }


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> None:
    lex = Lexicon()
    rows = tag([r for r in load(ROOT / "dataset.jsonl")
                if r["stratum"] == "variant_ortho"])

    ct = Counter()
    misses = {"form": [], "lemma": [], "link": []}
    for r in rows:
        reg, gold = r["input_reg"], r["lemma_gold"]
        ct["n"] += 1

        lem_exact = lex.lemmas(reg)
        lem_key = lex.lemmas_for_key(b1_key(reg))

        if lem_exact:
            ct["form_in_dict"] += 1
        elif lem_key:
            ct["form_in_dict_via_key"] += 1
        else:
            misses["form"].append((reg, gold))

        if lex.has_lemma(gold):
            ct["lemma_in_dict"] += 1
        else:
            misses["lemma"].append((reg, gold))

        gs, gl = norm_strict(gold), norm_lenient(gold)
        if any(norm_strict(x) == gs for x in lem_exact):
            ct["link_strict"] += 1
        if any(norm_lenient(x) == gl for x in lem_exact):
            ct["link_lenient"] += 1
        elif lem_exact:
            misses["link"].append((reg, gold, list(lem_exact)[:4]))
        if any(norm_lenient(x) == gl for x in lem_key):
            ct["link_key_lenient"] += 1

        # Same three questions on the *unregularised* spelling, for contrast:
        # this is the share the dictionary happens to attest as written.
        if any(norm_lenient(x) == gl
               for x in lex.lemmas_for_key(b1_key(r["input"]))):
            ct["orig_key_lenient"] += 1

    n = ct["n"]
    print(f"variant_ortho: n={n}\n")
    rows_out = [
        ("(1)  form_reg is an exact dictionary form", ct["form_in_dict"]),
        ("     ...only via its accent-stripped key", ct["form_in_dict_via_key"]),
        ("(2)  lemma_gold is a dictionary lemma", ct["lemma_in_dict"]),
        ("(3)  dictionary links form_reg -> lemma_gold  (strict)",
         ct["link_strict"]),
        ("(3)  ...                                     (lenient)",
         ct["link_lenient"]),
        ("(3k) ... via the accent-stripped key bucket  (lenient)",
         ct["link_key_lenient"]),
        ("     [contrast] same, from the papyrus spelling",
         ct["orig_key_lenient"]),
    ]
    for label, v in rows_out:
        print(f"{label:<58} {v:>6}  {_ratio(v, n):>6.1%}")

    for sp in ("dev", "test"):
        sub = [r for r in rows if r["split"] == sp]
        hit = sum(1 for r in sub
                  if any(norm_lenient(x) == norm_lenient(r["lemma_gold"])
                         for x in lex.lemmas_for_key(b1_key(r["input_reg"]))))
        print(f"\n(3k) on {sp}: {hit}/{len(sub)} = "
              f"{_ratio(hit, len(sub)):.1%}")

    (ROOT / "results_b0.json").write_text(json.dumps(
        {
            "counts": dict(ct),
            "misses": {k: v[:60] for k, v in misses.items()},
            "provenance": result_provenance(),
        },
        ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
