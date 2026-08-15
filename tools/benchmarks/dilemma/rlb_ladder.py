"""B1 - B3: the reverse-lookup baseline ladder.

Each rung is a cascade: try the cheap exact thing, and only on a miss fall
through to the more permissive thing. Every rung returns a *ranked candidate
set* of lemmas, not a single guess, because the question is recall@N against
Dilemma's effective ceiling of 34.1%, not top-1 accuracy.

    B1   accent-stripped key, exact
    B2   B1, else Levenshtein <=2 over the 4.8M accent-stripped keys
    B3a  B1, else the fifteen alternations collapsed, exact
    B3b  B3a, else Levenshtein <=1 over the collapsed key space
    B3   B3b, else B2's residual Levenshtein <=2

Ranking is by (edit distance, corpus form frequency, lemma attestation count).
Both frequency tables ship with Dilemma and are built from GLAUx/Diorisis/PTA,
none of which is PapyGreek -- so the ordering cannot be leaking the answer.
"""

import argparse
import json

import time
import unicodedata
from collections import Counter
from pathlib import Path

from benchmark_provenance import dilemma_data_identity, file_identity
from rlb_index import KeySpace, PairedIndex
from rlb_keys import b1_key, b3_key, b3_variants
from rlb_lexicon import (
    DATA,
    Lexicon,
    _read_key_cache,
    form_freq,
    lemma_freq,
    strip_for_freq,
)
from rlb_splits import tag
from run_eval import candidate_ranks, load

ROOT = Path(__file__).parent
STAGES = ("B1", "B2", "B3a", "B3b", "B3", "B3ur", "B2u", "B3u")
KEY_CAP = 300


def _accents(form: str) -> int:
    return sum(1 for ch in unicodedata.normalize("NFD", form)
               if unicodedata.combining(ch))


class Ladder:
    def __init__(self, target: str = "lemma"):
        # target="form" ranks surface spellings instead of headwords, for the
        # DDbDP orig->reg normalisation benchmark. Everything else -- the
        # fifteen alternations, KEY_CAP, the cost constants, _union -- stays
        # byte-identical, because the shared generator is the point: the two
        # benchmarks are only comparable if the search is the same search.
        self.target = target
        self.lex = Lexicon()
        print("loading grc key universe...", flush=True)
        self.keys = self.lex.grc_keys()
        print(f"  {len(self.keys):,} accent-stripped keys", flush=True)

        print("loading frequency tables...", flush=True)
        self.ff = form_freq()
        self.lf = lemma_freq()

        self._space1 = None
        self._paired = None
        self._space3 = None

    # -- lazily built search structures ------------------------------------

    @property
    def space1(self) -> KeySpace:
        if self._space1 is None:
            print("building length-bucketed key space (B2)...", flush=True)
            self._space1 = KeySpace(self.keys)
        return self._space1

    @property
    def paired(self) -> PairedIndex:
        if self._paired is None:
            print("loading collapsed keys (B3)...", flush=True)
            coarse = _read_key_cache(ROOT / "b3_keys.txt")
            assert len(coarse) == len(self.keys), "run rlb_build.py first"
            self._paired = PairedIndex(coarse, self.keys)
        return self._paired

    @property
    def space3(self) -> KeySpace:
        if self._space3 is None:
            print("building collapsed key space (B3b)...", flush=True)
            self._space3 = KeySpace(self.paired.distinct_coarse())
        return self._space3

    # -- candidate generation ----------------------------------------------

    def _targets(self, k: str) -> list:
        """What a key expands into: headwords, or surface spellings."""
        if self.target == "lemma":
            return self.lex.lemmas_for_key(k)

        # Every form under one stripped key differs from its siblings only in
        # accentuation, and corpus_freq.json is itself keyed on the stripped
        # form -- so they carry identical frequencies and the ranking tuple is
        # degenerate between them. Elect one representative deterministically:
        # the most fully accented spelling, ties broken lexicographically.
        # Scoring is lenient (accents stripped), so this cannot change a hit
        # into a miss; it only stops the list filling with equivalent entries.
        forms = sorted({f for f, _ in self.lex.forms_for_key(k)})
        if not forms:
            return []
        return [max(forms, key=_accents)]

    def _secondary(self, target: str) -> int:
        """The third ranking term. Lemma attestation, directly or via forms."""
        if self.target == "lemma":
            return self.lf.get(target, 0)
        return max((self.lf.get(lemma, 0)
                    for lemma in self.lex.lemmas(target)),
                   default=0)

    def _from_keys(self, keys, dist: float, out: dict) -> None:
        """Accumulate lemma -> (cost, -form_freq, -lemma_freq, src_key).

        The source key is carried along because the alternation between the
        query and *that* key is what a probabilistic tiebreaker has to score.
        Dropping it, as the first version did, meant every downstream
        experiment that wanted to know "which alternation got us here" had to
        re-run the search. It sorts last, so it only ever breaks exact ties.
        """
        for k in keys:
            fq = self.freq(k)
            for lem in self._targets(k):
                prev = out.get(lem)
                cand = (dist, -fq, -self._secondary(lem), k)
                if prev is None or cand < prev:
                    out[lem] = cand

    def candidates(self, word: str, stage: str) -> list[str]:
        return self._rank(self.scored(word, stage))

    def scored(self, word: str, stage: str) -> dict[str, tuple]:
        """lemma -> (cost, -form_freq, -lemma_freq), unsorted.

        Exposed separately so a run can dump the whole candidate set with its
        features once and every re-ranking experiment can then be replayed
        offline instead of re-searching 4.8M keys.
        """
        out: dict[str, tuple] = {}
        k1 = b1_key(word)

        # Union variants. B3ur and B2u are B3u with one of the two generators
        # switched off, which is what separates "the fifteen rules earned this"
        # from "blind edit distance would have found it anyway".
        if stage in ("B3u", "B3ur", "B2u"):
            self._union(word, k1, out,
                        rules=stage != "B2u", blind=stage != "B3ur")
            return out

        self._from_keys([k1], 0, out)
        if out or stage == "B1":
            return out

        if stage in ("B3a", "B3b", "B3"):
            for ck in b3_variants(word):
                self._from_keys(self.paired.get(ck), 1, out)
            if out or stage == "B3a":
                return out

        if stage in ("B3b", "B3"):
            for ck, d in self.space3.near(b3_key(word), 1):
                self._from_keys(self.paired.get(ck), 1 + d, out)
            if out or stage == "B3b":
                return out

        if stage in ("B2", "B3"):
            # Distance 2 over 4.8M keys routinely matches thousands of them,
            # and every one costs two dictionary queries. Only the closest and
            # commonest KEY_CAP are expanded into lemmas: a candidate ranked
            # below the 300th most frequent near-match is not going to be read
            # by anyone, and keeping the tail would only inflate recall@inf.
            near = self.space1.near(k1, 2)
            near.sort(key=lambda kd: (kd[1], -self.freq(kd[0])))
            for k, d in near[:KEY_CAP]:
                self._from_keys([k], d, out)

        return out

    def _union(self, word: str, k1: str, out: dict, *,
               rules: bool = True, blind: bool = True) -> list[str]:
        """Every generator runs; the results compete on cost.

        The cascades stop at the first generator that produces anything, and
        that is not a neutral choice: a scribe's misspelling is very often a
        different real word (πεδίων for παιδίων is the attested spelling of
        πεδίον 'plain'), so the exact lookup succeeds, returns the wrong lemma,
        and the alternation rules never get to run. Here nothing is pre-empted.

        Costs are ordered by how much linguistic warrant a match has: an
        alternation the fifteen rules name outranks an unexplained letter
        substitution at the same string distance.
        """
        self._from_keys([k1], 0.0, out)                             # exact
        if rules:
            for ck in b3_variants(word):
                self._from_keys(self.paired.get(ck), 1.0, out)      # rules
            for ck, d in self.space3.near(b3_key(word), 1):
                self._from_keys(self.paired.get(ck), 2.0 + d, out)  # rules+1
        if blind:
            near = self.space1.near(k1, 2)
            near.sort(key=lambda kd: (kd[1], -self.freq(kd[0])))
            for k, d in near[:KEY_CAP]:
                self._from_keys([k], 0.5 + d, out)                  # blind
        return out

    def freq(self, key: str) -> int:
        return self.ff.get(strip_for_freq(key), 0)

    @staticmethod
    def _rank(out: dict[str, tuple]) -> list[str]:
        return [lem for lem, _ in sorted(out.items(), key=lambda kv: kv[1])]


# --------------------------------------------------------------------------


def serialize_candidates(candidates: list[str],
                         scored: dict[str, tuple]) -> list[list]:
    """Serialize every generated candidate and its ranking features."""
    return [[candidate, scored[candidate][0], -scored[candidate][1],
             -scored[candidate][2], scored[candidate][3]]
            for candidate in candidates]


def score(rows, ladder: Ladder, stage: str, dump=None) -> dict:
    t0 = time.time()
    per = {sp: Counter() for sp in ("dev", "test", "all")}
    sizes = []
    detail = []
    for r in rows:
        scored = ladder.scored(r["input"], stage)
        cands = Ladder._rank(scored)
        if dump is not None:
            # Features, not just the ranking: every re-ranking experiment
            # replays from this instead of re-searching 4.8M keys.
            dump.write(json.dumps({
                "doc": r["doc"], "sent": r["sent"], "wid": r["wid"],
                "postag": r.get("postag", ""),
                "date_nb": r.get("date_not_before", ""),
                "date_na": r.get("date_not_after", ""),
                "form": r["input"], "reg": r["input_reg"],
                "gold": r["lemma_gold"], "split": r["split"],
                "n_cand": len(cands),
                "cands": serialize_candidates(cands, scored),
            }, ensure_ascii=False) + "\n")
        rank_l, rank_s = candidate_ranks(cands, r["lemma_gold"])
        sizes.append(len(cands))
        detail.append({"doc": r["doc"], "sent": r["sent"], "wid": r["wid"],
                       "form": r["input"], "reg": r["input_reg"],
                       "gold": r["lemma_gold"], "split": r["split"],
                       "n_cand": len(cands), "rank": rank_l,
                       "top5": cands[:5]})
        for sp in (r["split"], "all"):
            c = per[sp]
            c["n"] += 1
            if not cands:
                c["empty"] += 1
            for k in (1, 5, 10):
                if rank_l is not None and rank_l < k:
                    c[f"lenient@{k}"] += 1
                if rank_s is not None and rank_s < k:
                    c[f"strict@{k}"] += 1
            if rank_l is not None:
                c["lenient@inf"] += 1
            if rank_s is not None:
                c["strict@inf"] += 1

    sizes.sort()
    return {
        "stage": stage,
        "seconds": round(time.time() - t0, 1),
        "cand_median": sizes[len(sizes) // 2],
        "cand_p90": sizes[int(len(sizes) * 0.9)],
        "cand_max": sizes[-1],
        "per_split": {k: dict(v) for k, v in per.items()},
        "detail": detail,
    }


def report(res: dict) -> None:
    print(f"\n--- {res['stage']}   ({res['seconds']}s, "
          f"|C| median {res['cand_median']} / p90 {res['cand_p90']} / "
          f"max {res['cand_max']})")
    print(f"{'split':<6} {'n':>5} {'@1':>8} {'@5':>8} {'@10':>8} "
          f"{'@inf':>8} {'empty':>8}")
    for sp in ("dev", "test", "all"):
        v = res["per_split"][sp]
        n = v.get("n", 0)
        if not n:            # a slice can fall entirely on one side of the split
            continue
        print(f"{sp:<6} {n:>5} "
              + " ".join(f"{v.get(f'lenient@{k}', 0) / n:>7.1%}"
                         for k in (1, 5, 10, "inf"))
              + f" {v.get('empty', 0) / n:>7.1%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="*", default=list(STAGES))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--stratum", default="variant_ortho")
    ap.add_argument("--out", default="results_ladder.json")
    ap.add_argument("--dump", default="")
    args = ap.parse_args()

    rows = tag([r for r in load(ROOT / "dataset.jsonl")
                if r["stratum"] == args.stratum])
    if args.offset or args.limit:
        end = args.offset + args.limit if args.limit else None
        rows = rows[args.offset:end]
    print(f"{len(rows)} tokens in {args.stratum}")
    provenance = {
        "dilemma_data": dilemma_data_identity(DATA),
        "dataset": file_identity(ROOT / "dataset.jsonl"),
    }

    ladder = Ladder()
    dump = open(ROOT / args.dump, "w", encoding="utf-8") if args.dump else None
    results = []
    for stage in args.stages:
        res = score(rows, ladder, stage, dump)
        res["provenance"] = provenance
        report(res)
        results.append(res)
        (ROOT / args.out).write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8")
    if dump:
        dump.close()
        print(f"candidates dumped to {ROOT / args.dump}")
    print(f"\nwritten to {ROOT / args.out}")


if __name__ == "__main__":
    main()
