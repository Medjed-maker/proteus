"""Split the DDbDP pairs into orthographic and lexical strata.

PapyGreek's benchmark is the ``variant_ortho`` layer only, defined in
build_dataset.py as: the editor changed the spelling *and both annotation
layers agree on the lexeme*. When the lemma changes too, the editor emended the
word itself, which is a different problem and is excluded.

DDbDP's <choice><reg> carries no such distinction, and the pilot shows the
consequence directly -- alongside real itacism (σφραγείδων -> σφραγίδων) it
contains οἱμοι -> ὁμοίως and ζηκωτάτη -> ζυγοστάτης, which are emendations, and
Μηνᾶς -> Μηνᾶ, which is a case change. Reported undivided, the new benchmark is
measuring a harder and different task than the old one, and the two numbers
cannot be compared.

This reconstructs the distinction the only way available without annotators:
lemmatise both sides and compare. That is *not* the same instrument PapyGreek
used -- it substitutes a tool for a human -- so pairs where either side fails
to resolve are labelled "unknown" rather than guessed at, and the unknown rate
is reported next to every number derived from this split.
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from run_eval import require_batch_result_count

ROOT = Path(__file__).parent


def main() -> None:
    from dilemma import Dilemma

    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="ddb_pairs.jsonl")
    ap.add_argument("--out", default="ddb_pairs_strata.jsonl")
    ap.add_argument("--stats", default="ddb_strata_stats.json")
    ap.add_argument("--batch", type=int, default=100000)
    args = ap.parse_args()

    rows = [json.loads(line)
            for line in (ROOT / args.pairs).open(encoding="utf-8")]
    print(f"{len(rows):,} pairs")
    if not rows:
        raise SystemExit("no pairs to stratify")

    d = Dilemma(lang="grc", resolve_articles=True, normalize=True,
                period="hellenistic")
    words = [w for r in rows for w in (r["form"], r["gold"])]
    preds: list = []
    t0 = time.time()
    for i in range(0, len(words), args.batch):
        batch_words = words[i:i + args.batch]
        batch_preds = d.lemmatize_batch(batch_words, guess=False)
        require_batch_result_count(
            len(batch_words), len(batch_preds), "DDbDP stratum lemmatization")
        preds.extend(batch_preds)
        print(f"  {min(i + args.batch, len(words)):,}/{len(words):,} "
              f"{time.time() - t0:.0f}s", flush=True)
    if len(preds) != len(words):
        raise RuntimeError(
            f"lemmatizer result count mismatch: expected {len(words)}, "
            f"got {len(preds)}")

    ct = Counter()
    with (ROOT / args.out).open("w", encoding="utf-8") as fh:
        for j, r in enumerate(rows):
            lo, lg = preds[2 * j], preds[2 * j + 1]
            if lo is None or lg is None:
                stratum = "unknown"
            elif lo == lg:
                stratum = "ortho"
            else:
                stratum = "lex"
            ct[stratum] += 1
            fh.write(json.dumps({**r, "stratum": stratum,
                                 "lemma_orig": lo, "lemma_reg": lg},
                                ensure_ascii=False) + "\n")

    n = len(rows)
    stats = {"n_pairs": n, "counts": dict(ct),
             "shares": {k: v / n for k, v in ct.items()},
             "note": ("stratum assigned by lemmatising both sides with "
                      "Dilemma guess=False, not by human annotation; "
                      "'unknown' means at least one side did not resolve")}
    (ROOT / args.stats).write_text(json.dumps(stats, ensure_ascii=False,
                                              indent=2), encoding="utf-8")
    for k, v in ct.most_common():
        print(f"  {k:<9} {v:7d}  {v / n:6.1%}")
    print(f"wrote {args.out} and {args.stats}")


if __name__ == "__main__":
    main()
