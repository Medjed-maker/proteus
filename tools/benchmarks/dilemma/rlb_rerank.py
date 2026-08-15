"""How much of the (a) bucket is recoverable by re-ranking alone?

B3u's misses split 81.6% / 18.4% between "the gold lemma was generated but sits
below rank 5" and "it was never generated". This asks what the first 81.6% is
actually worth: the generator is held fixed and only the ordering changes, so
every number here is reachable without touching Layer 1.

The candidate sets come from dump_b3u.jsonl, so each variant is a rescoring of
a fixed set rather than a fresh search -- which is what makes it possible to
run a dozen of them.

Signals, all from tables that ship with Dilemma and none of which is built from
PapyGreek:

    form_freq       corpus_freq.json, 68M tokens
    lemma_freq      lemma_attestation.json, total attestations
    by_century      lemma_attestation.json -- the papyrus is dated, so a lemma
                    attested in the wrong millennium is a worse candidate
    by_dialect      lemma_attestation.json -- documentary papyri are Koine
    by_genre        lemma_attestation.json -- 'epistles' is the nearest genre
    dominant_pos    lemma_attestation.json, against PapyGreek's gold POS

The POS variants use the *gold* tag, so they are an upper bound on what a
tagger could contribute, not an achievable score. Everything else is
achievable as-is.
"""

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

from rlb_lexicon import DATA
from run_eval import norm_lenient

ROOT = Path(__file__).parent

# AGDT/Morpheus postag position 1 -> lemma_attestation's dominant_pos vocabulary.
POS_VALUES = ("noun", "verb", "adjective", "adverb", "article", "particle",
              "conjunction", "preposition", "pronoun", "numeral",
              "interjection")
TAGGER_ERROR = 0.10
_NOISE_SEED = 7

# Penalty for a candidate whose POS contradicts the tag. The gold tag can carry
# a hard penalty because it is right; Dilemma's own head is 56.9% accurate on
# papyrus spellings, so the same penalty actively hurts (68.6% vs 74.6% on dev)
# and 1.5 is what the dev sweep selects.
POS_PENALTY_GOLD = 6.0
POS_PENALTY_REAL = 1.5
POSTAGS = "postags.jsonl"

AGDT_POS = {
    "n": "noun", "v": "verb", "a": "adjective", "d": "adverb",
    "l": "article", "g": "particle", "c": "conjunction", "r": "preposition",
    "p": "pronoun", "m": "numeral", "i": "interjection", "e": "interjection",
}

# Documentary papyri: BCE 300 - CE 700. Centuries are keyed as strings, "-1"
# meaning the first century BCE.
PAPYRUS_CENTURIES = [str(c) for c in range(-3, 8) if c != 0]
KOINE = ("Koine", "Attic/Koine")
NEAR_GENRES = ("epistles", "other", "religion")


_noise_cache: dict = {}


def _noisy_pos(row: dict) -> str | None:
    """Gold POS, wrong TAGGER_ERROR of the time. Deterministic per token."""
    key = (row["doc"], row["sent"], row["wid"])
    if key not in _noise_cache:
        want = AGDT_POS.get((row.get("postag") or "")[:1])
        rnd = random.Random(f"{_NOISE_SEED}|{key}")
        if want is not None and rnd.random() < TAGGER_ERROR:
            want = rnd.choice([p for p in POS_VALUES if p != want])
        _noise_cache[key] = want
    return _noise_cache[key]


_tagger_cache: dict = {}


def load_tagger(path: Path | None = None) -> dict:
    """Dilemma's own POS predictions, from rlb_postag.py."""
    if not _tagger_cache:
        f = path or (ROOT / POSTAGS)
        for line in f.open(encoding="utf-8"):
            r = json.loads(line)
            _tagger_cache[r["doc"], r["sent"], r["wid"]] = r
    return _tagger_cache


def load_attestation() -> dict:
    d = json.load(open(DATA / "lemma_attestation.json", encoding="utf-8"))
    return d["lemmas"]


def centuries_for(row: dict) -> list[str]:
    """The centuries the document could belong to, from its date range."""
    def century(year: str) -> int | None:
        try:
            y = int(year)
        except (TypeError, ValueError):
            return None
        return (y - 1) // 100 + 1 if y > 0 else -((-y - 1) // 100 + 1)

    lo = century(row.get("date_nb", "")) or None
    hi = century(row.get("date_na", "")) or None
    if lo is None and hi is None:
        return PAPYRUS_CENTURIES
    lo = lo if lo is not None else hi
    hi = hi if hi is not None else lo
    out = [str(c) for c in range(min(lo, hi), max(lo, hi) + 1) if c != 0]
    return out or PAPYRUS_CENTURIES


# --------------------------------------------------------------------------
# Scoring variants. Each returns a sort key; lower is better.
# --------------------------------------------------------------------------


def make_scorer(name: str, att: dict):
    def lex(row, c):
        """R0: what B3u actually did -- lexicographic, cost first."""
        lem, cost, ff, lf = c[:4]
        return (cost, -ff, -lf)

    def blend(row, c):
        """R1: one additive score instead of a lexicographic cascade.

        Cost dominating absolutely means a distance-1 candidate attested twice
        outranks an exact-cost candidate attested ten thousand times. Trading
        one unit of cost against a decade of log-frequency lets evidence win.
        """
        lem, cost, ff, lf = c[:4]
        return -(math.log1p(ff) + 0.5 * math.log1p(lf) - 3.0 * cost)

    def _prior(row, lem) -> float:
        a = att.get(lem)
        if not a:
            return 0.0
        total = max(a.get("total", 0), 1)
        cent = a.get("by_century") or {}
        want = centuries_for(row)
        in_range = sum(cent.get(c, 0) for c in want)
        cent_share = in_range / total if cent else 0.0
        dial = a.get("by_dialect") or {}
        koine = sum(dial.get(d, 0) for d in KOINE)
        dial_share = koine / max(sum(dial.values()), 1) if dial else 0.0
        gen = a.get("by_genre") or {}
        near = sum(gen.get(g, 0) for g in NEAR_GENRES)
        gen_share = near / max(sum(gen.values()), 1) if gen else 0.0
        return cent_share, dial_share, gen_share

    def century(row, c):
        lem, cost, ff, lf = c[:4]
        p = _prior(row, lem)
        bonus = 2.0 * (p[0] if p else 0.0)
        return -(math.log1p(ff) + 0.5 * math.log1p(lf) - 3.0 * cost + bonus)

    def dialect(row, c):
        lem, cost, ff, lf = c[:4]
        p = _prior(row, lem)
        bonus = 2.0 * (p[1] if p else 0.0)
        return -(math.log1p(ff) + 0.5 * math.log1p(lf) - 3.0 * cost + bonus)

    def genre(row, c):
        lem, cost, ff, lf = c[:4]
        p = _prior(row, lem)
        bonus = 2.0 * (p[2] if p else 0.0)
        return -(math.log1p(ff) + 0.5 * math.log1p(lf) - 3.0 * cost + bonus)

    def pos(row, c):
        """Gold POS as a hard-ish filter: an upper bound for a tagger."""
        lem, cost, ff, lf = c[:4]
        want = AGDT_POS.get((row.get("postag") or "")[:1])
        a = att.get(lem) or {}
        got = a.get("dominant_pos")
        penalty = 0.0 if (want is None or got is None or want == got) else 6.0
        return -(math.log1p(ff) + 0.5 * math.log1p(lf)
                 - 3.0 * cost - penalty)

    def pos_noisy(row, c):
        """R5 with the gold tag corrupted at the rate a real tagger errs at.

        Included because the gold-POS number is not a deliverable: it says what
        the signal is worth, not what a tagger delivers. The degradation is
        gentle -- see the sweep in the write-up -- so the achievable gain is
        close to the oracle one.
        """
        lem, cost, ff, lf = c[:4]
        want = _noisy_pos(row)
        a = att.get(lem) or {}
        got = a.get("dominant_pos")
        penalty = 0.0 if (want is None or got is None or want == got) else 6.0
        return -(math.log1p(ff) + 0.5 * math.log1p(lf)
                 - 3.0 * cost - penalty)

    def _pos_penalised(row, c, want, pen):
        lem, cost, ff, lf = c[:4]
        got = (att.get(lem) or {}).get("dominant_pos")
        penalty = 0.0 if (want is None or got is None or want == got) else pen
        return -(math.log1p(ff) + 0.5 * math.log1p(lf) - 3.0 * cost - penalty)

    def pos_real(row, c):
        """R5r: Dilemma's POS head on the papyrus spelling. Deployable today."""
        t = load_tagger().get((row["doc"], row["sent"], row["wid"]), {})
        return _pos_penalised(row, c, t.get("pred_orig"), POS_PENALTY_REAL)

    def pos_real_reg(row, c):
        """R5rg: the same tagger on the regularised spelling.

        Not deployable -- it presupposes the answer -- but it separates "the
        tagger is weak" from "the misspelling breaks the tagger".
        """
        t = load_tagger().get((row["doc"], row["sent"], row["wid"]), {})
        return _pos_penalised(row, c, t.get("pred_reg"), POS_PENALTY_REAL)

    def combined(row, c):
        lem, cost, ff, lf = c[:4]
        p = _prior(row, lem)
        bonus = (2.0 * p[0] + 1.0 * p[1] + 1.0 * p[2]) if p else 0.0
        want = AGDT_POS.get((row.get("postag") or "")[:1])
        a = att.get(lem) or {}
        got = a.get("dominant_pos")
        penalty = 0.0 if (want is None or got is None or want == got) else 6.0
        return -(math.log1p(ff) + 0.5 * math.log1p(lf)
                 - 3.0 * cost + bonus - penalty)

    def no_pos(row, c):
        lem, cost, ff, lf = c[:4]
        p = _prior(row, lem)
        bonus = (2.0 * p[0] + 1.0 * p[1] + 1.0 * p[2]) if p else 0.0
        return -(math.log1p(ff) + 0.5 * math.log1p(lf) - 3.0 * cost + bonus)

    return {"R0 lexicographic (B3u as run)": lex,
            "R5r + POS (Dilemma head, orig)": pos_real,
            "R5rg + POS (Dilemma head, reg)": pos_real_reg,
            "R5n + POS (10%-error tagger)": pos_noisy,
            "R1 additive blend": blend,
            "R2 + century prior": century,
            "R3 + dialect prior": dialect,
            "R4 + genre prior": genre,
            "R5 + POS (gold, upper bound)": pos,
            "R6 all priors, no POS": no_pos,
            "R7 all priors + POS (upper bound)": combined}[name]


VARIANTS = ["R0 lexicographic (B3u as run)", "R1 additive blend",
            "R2 + century prior", "R3 + dialect prior", "R4 + genre prior",
            "R6 all priors, no POS",
            "R5r + POS (Dilemma head, orig)",
            "R5rg + POS (Dilemma head, reg)",
            "R5n + POS (10%-error tagger)",
            "R5 + POS (gold, upper bound)",
            "R7 all priors + POS (upper bound)"]


def evaluate(rows, scorer) -> Counter:
    ct = Counter()
    for row in rows:
        gl = norm_lenient(row["gold"])
        ranked = sorted(row["cands"], key=lambda c: scorer(row, c))
        rank = None
        for i, c in enumerate(ranked):
            if norm_lenient(c[0]) == gl:
                rank = i
                break
        ct["n"] += 1
        if rank is not None:
            ct["@inf"] += 1
            for k in (1, 3, 5, 10):
                if rank < k:
                    ct[f"@{k}"] += 1
    return ct


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--dump", default="dump_b3u.jsonl")
    args = ap.parse_args()

    rows = [json.loads(line)
            for line in (ROOT / args.dump).open(encoding="utf-8")]
    rows = [r for r in rows if args.split == "all" or r["split"] == args.split]
    att = load_attestation()
    print(f"{len(rows)} tokens, split={args.split}\n")

    print(f"{'variant':<36}{'@1':>8}{'@3':>8}{'@5':>8}{'@10':>8}{'@inf':>8}")
    out = {}
    base = None
    for name in VARIANTS:
        ct = evaluate(rows, make_scorer(name, att))
        n = ct["n"]
        out[name] = {k: v for k, v in ct.items()}
        d = "" if base is None else f"  ({ct['@5'] / n - base:+.1%})"
        if base is None:
            base = ct["@5"] / n
        print(f"{name:<36}" + "".join(f"{ct[f'@{k}'] / n:>8.1%}"
                                      for k in (1, 3, 5, 10, "inf")) + d)

    # The ceiling: perfect ordering of the same candidate sets.
    print(f"\n{'oracle (perfect ranking)':<36}"
          f"{out[VARIANTS[0]]['@inf'] / len(rows):>8.1%}"
          "   <- recall@inf is what perfect re-ranking would deliver at @1")

    (ROOT / f"results_rerank_{args.split}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
