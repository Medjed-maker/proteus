"""Measure Dilemma lemmatization accuracy on PapyGreek, stratified by whether
the papyrus spelling needed editorial regularisation.

Each configuration is run over the same token set so the strata are directly
comparable. What we care about is not the headline number but the gap:

    clean stratum   = spelling already standard      (Dilemma's home turf)
    variant stratum = spelling had to be regularised (the Proteus UVP zone)

Scoring is equivalence-adjusted the same way the Dilemma README describes:
homograph indices are stripped, final sigma and accent conventions are
neutralised in the lenient pass, and the strict pass is exact NFC equality.
"""

import argparse
import json
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from clean import clean, classify, residue

ROOT = Path(__file__).parent

HOMOGRAPH_IDX = re.compile(r"\d+$")


def norm_strict(s: str) -> str:
    return unicodedata.normalize("NFC", (s or "").strip())


def norm_lenient(s: str) -> str:
    """Neutralise lemma-convention differences that are not linguistic claims."""
    s = norm_strict(s)
    s = HOMOGRAPH_IDX.sub("", s)          # ἄν1 -> ἄν, ἄνω2 -> ἄνω
    s = s.replace("ς", "σ")               # final sigma
    nfd = unicodedata.normalize("NFD", s)
    # Drop accents and breathings; keep the letter skeleton + iota subscript.
    nfd = "".join(ch for ch in nfd if ord(ch) not in
                  (0x0300, 0x0301, 0x0342, 0x0313, 0x0314, 0x0308))
    return unicodedata.normalize("NFC", nfd).lower()


def candidate_ranks(candidates: list[str], gold: str) -> tuple[int | None,
                                                               int | None]:
    """Return the first lenient and strict ranks independently."""
    gl, gs = norm_lenient(gold), norm_strict(gold)
    rank_l = rank_s = None
    for index, candidate in enumerate(candidates):
        if rank_s is None and norm_strict(candidate) == gs:
            rank_s = index
        if rank_l is None and norm_lenient(candidate) == gl:
            rank_l = index
        if rank_l is not None and rank_s is not None:
            break
    return rank_l, rank_s


STRATA = ("clean", "variant_ortho", "variant_lex", "abbrev")


def require_batch_result_count(expected: int, actual: int,
                               operation: str) -> None:
    """Enforce the one-result-per-input contract of a batch operation."""
    if actual != expected:
        raise RuntimeError(
            f"{operation} result count mismatch: expected {expected}, "
            f"got {actual}")


def load(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    for r in rows:
        s = classify(r["form_orig"], r["form_reg"])
        if s == "variant":
            # Both annotation layers agreeing on the lexeme means the editor
            # only changed the spelling -- the pure UVP case. When the lemma
            # changes too, the editor emended the word itself (or the reg layer
            # split a crasis into two tokens), which is a different problem.
            s = ("variant_ortho" if r.get("lemma_orig") == r["lemma_gold"]
                 else "variant_lex")
        r["stratum"] = s
        r["input"] = clean(r["form_orig"])
        r["input_reg"] = clean(r["form_reg"])
    kept = [r for r in rows if r["input"]]
    _reject_unstripped_markup(kept)
    return kept


def _reject_unstripped_markup(rows: list[dict]) -> None:
    """Stop the run if clean() let editorial markup through.

    Every rlb_* script that reads dataset.jsonl arrives here, and rlb_ladder
    searches on ``input`` directly, so this is the one place where a gap in
    clean.MARKUP_CHARS can be caught before it silently corrupts a measurement.
    It has happened twice (2026-08-12, 2026-08-14) and both times the damage was
    invisible in the output -- a token searched with a bracket still attached
    just returns nothing, which reads as "the method failed on this word".
    Failing loudly here turns that into a one-line diagnosis; the message names
    the three places the offending character can belong.
    """
    found: dict[str, list[dict]] = {}
    for r in rows:
        for field in ("input", "input_reg"):
            for ch in residue(r[field]):
                found.setdefault(ch, []).append(r)
    if not found:
        return
    lines = [f"{len(found)} unexpected character(s) survived clean() "
             f"in {sum(len(v) for v in found.values())} field(s):"]
    for ch, hits in sorted(found.items(), key=lambda kv: -len(kv[1])):
        name = unicodedata.name(ch, "<unnamed>")
        r = hits[0]
        lines.append(
            f"  U+{ord(ch):04X} {name}  n={len(hits)}  "
            f"e.g. {r['doc']} {r['form_orig']!r} -> {r['input']!r} "
            f"(stratum={r['stratum']})")
    lines.append("Add it to clean.MARKUP_CHARS (editorial markup), "
                 "clean.CONFUSABLES (a source typo for a Greek letter), "
                 "or clean.ALLOWED_RE (genuinely part of the word).")
    raise SystemExit("\n".join(lines))


def evaluate(rows, d, *, use_reg: bool, guess: bool, label: str) -> dict:
    key = "input_reg" if use_reg else "input"
    words = [r[key] for r in rows]

    t0 = time.time()
    preds = d.lemmatize_batch(words, guess=guess)
    elapsed = time.time() - t0
    require_batch_result_count(
        len(rows), len(preds), f"{label} lemmatization")

    per = defaultdict(lambda: Counter())
    errors = defaultdict(list)
    for r, p in zip(rows, preds):
        s = r["stratum"]
        gold = r["lemma_gold"]
        per[s]["n"] += 1
        if p is None:
            per[s]["abstain"] += 1
            errors[s].append((r[key], gold, None))
            continue
        if norm_strict(p) == norm_strict(gold):
            per[s]["strict"] += 1
            per[s]["lenient"] += 1
        elif norm_lenient(p) == norm_lenient(gold):
            per[s]["lenient"] += 1
        else:
            # An echo of the input is indistinguishable from a real identity
            # lemmatization for the caller, so it is tracked separately: it is
            # a silent "I don't know" dressed up as an answer.
            if norm_lenient(p) == norm_lenient(r[key]):
                per[s]["echo"] += 1
            errors[s].append((r[key], gold, p))

    return {
        "label": label,
        "use_reg": use_reg,
        "guess": guess,
        "seconds": round(elapsed, 1),
        "per_stratum": {k: dict(v) for k, v in per.items()},
        "errors": {k: v[:400] for k, v in errors.items()},
    }


def report(res: dict) -> None:
    print(f"\n--- {res['label']}  ({res['seconds']}s)")
    print(f"{'stratum':<14} {'n':>6} {'strict':>9} {'lenient':>9} "
          f"{'echo':>9} {'abstain':>9}")
    for s in STRATA:
        v = res["per_stratum"].get(s)
        if not v:
            continue
        n = v["n"]
        print(f"{s:<14} {n:>6} "
              f"{v.get('strict', 0) / n:>8.1%} "
              f"{v.get('lenient', 0) / n:>8.1%} "
              f"{v.get('echo', 0) / n:>8.1%} "
              f"{v.get('abstain', 0) / n:>8.1%}")
    tot = Counter()
    for v in res["per_stratum"].values():
        tot.update(v)
    n = tot["n"]
    print(f"{'ALL':<14} {n:>6} {tot['strict']/n:>8.1%} {tot['lenient']/n:>8.1%} "
          f"{tot['echo']/n:>8.1%} {tot['abstain']/n:>8.1%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    from dilemma import Dilemma
    import dilemma
    from benchmark_provenance import (
        dilemma_data_identity,
        dilemma_package_identity,
        file_identity,
    )
    from rlb_lexicon import DATA

    rows = load(ROOT / "dataset.jsonl")
    if args.limit:
        rows = rows[:args.limit]
    print(f"dilemma {dilemma.__version__}, {len(rows)} tokens")
    print(Counter(r["stratum"] for r in rows))
    provenance = {
        "dilemma": dilemma_package_identity(require_expected=True),
        "dilemma_data": dilemma_data_identity(DATA),
        "dataset": file_identity(ROOT / "dataset.jsonl"),
    }

    configs = [
        # (label, Dilemma kwargs, use_reg, guess)
        ("A. default, orig spelling",
         dict(lang="grc", resolve_articles=True), False, True),
        ("B. default, editor-regularised spelling",
         dict(lang="grc", resolve_articles=True), True, True),
        ("C. normalize=True, orig spelling",
         dict(lang="grc", resolve_articles=True, normalize=True,
              period="hellenistic"), False, True),
        ("E. normalize=True, guess=False (abstain instead of guessing)",
         dict(lang="grc", resolve_articles=True, normalize=True,
              period="hellenistic"), False, False),
        ("F. convention=lsj, normalize=True, orig spelling",
         dict(lang="grc", resolve_articles=True, normalize=True,
              period="hellenistic", convention="lsj"), False, True),
    ]

    results = []
    for label, kwargs, use_reg, guess in configs:
        print(f"\n>>> loading {label}  {kwargs}", flush=True)
        d = Dilemma(**kwargs)
        res = evaluate(rows, d, use_reg=use_reg, guess=guess, label=label)
        res["kwargs"] = {k: str(v) for k, v in kwargs.items()}
        res["provenance"] = provenance
        report(res)
        results.append(res)
        del d

    (ROOT / args.out).write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten to {ROOT / args.out}")


if __name__ == "__main__":
    main()
