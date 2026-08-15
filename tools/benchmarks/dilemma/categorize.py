"""Group the UVP-zone failures by the phonological/orthographic alternation
that separates the papyrus spelling from the regularised one.

The point is not the taxonomy for its own sake: each bucket names a rule family
that a reverse phonological index would have to cover, ranked by how many real
attested tokens it is worth.
"""

import difflib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from clean import clean
from run_eval import load, norm_lenient, require_batch_result_count

ROOT = Path(__file__).parent


def skeleton(s: str) -> str:
    """Letters only, no accents/breathings -- alternations live at this level."""
    nfd = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in nfd if ch.isalpha() and not unicodedata.combining(ch)).lower()


# (name, orig-side pattern, reg-side pattern) applied to the aligned diff.
RULES = [
    ("itacism ει/ι",        r"^(ει|ι)$",        r"^(ει|ι)$"),
    ("itacism η/ι/ει",      r"^(η|ι|ει|ῃ)$",    r"^(η|ι|ει|ῃ)$"),
    ("itacism οι/υ/ι",      r"^(οι|υ|ι)$",      r"^(οι|υ|ι)$"),
    ("ο/ω quantity",        r"^(ο|ω)$",         r"^(ο|ω)$"),
    ("ε/αι",                r"^(ε|αι)$",        r"^(ε|αι)$"),
    ("nasal assimilation",  r"^(ν|μ|γ)$",       r"^(ν|μ|γ)$"),
    ("voicing κ/γ π/β τ/δ", r"^(κ|γ|π|β|τ|δ)$", r"^(κ|γ|π|β|τ|δ)$"),
    ("σ/ζ",                 r"^(σ|ζ)$",         r"^(σ|ζ)$"),
    ("gemination",          r"^(.)\1?$",        r"^(.)\1?$"),
    ("αυ/α ευ/ε",           r"^(αυ|α|ευ|ε|ου|ο|υ)$", r"^(αυ|α|ευ|ε|ου|ο|υ)$"),
]


def diff_pairs(a: str, b: str) -> list[tuple[str, str]]:
    sm = difflib.SequenceMatcher(None, a, b)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            out.append((a[i1:i2], b[j1:j2]))
    return out


def labels(a: str, b: str) -> list[str]:
    """The alternation atoms separating two spellings, one per diff site.

    ``label`` joins these into a single string for the failure taxonomy. The
    atoms are what a probabilistic tiebreaker needs: it has to charge a
    different price for an itacism than for an epsilon insertion, and joining
    them first throws that away. No cap on the number of sites here -- a
    three-alternation candidate really is three alternations, and the cost
    should say so.
    """
    pairs = diff_pairs(skeleton(a), skeleton(b))
    names = []
    for x, y in pairs:
        hit = None
        for name, px, py in RULES:
            if re.match(px, x) and re.match(py, y):
                hit = name
                break
        if hit is None:
            if not x:
                hit = f"insertion ({y})" if len(y) <= 2 else "insertion (long)"
            elif not y:
                hit = f"deletion ({x})" if len(x) <= 2 else "deletion (long)"
            else:
                hit = "other substitution"
        names.append(hit)
    return names


def label(a: str, b: str) -> str:
    pairs = diff_pairs(skeleton(a), skeleton(b))
    if not pairs:
        return "accent/breathing only"
    if len(pairs) > 2:
        return "multiple alternations"
    names = []
    for x, y in pairs:
        hit = None
        for name, px, py in RULES:
            if re.match(px, x) and re.match(py, y):
                hit = name
                break
        if hit is None:
            if not x:
                hit = f"insertion ({y})" if len(y) <= 2 else "insertion (long)"
            elif not y:
                hit = f"deletion ({x})" if len(x) <= 2 else "deletion (long)"
            else:
                hit = "other substitution"
        names.append(hit)
    return " + ".join(sorted(set(names)))


def main() -> None:
    from dilemma import Dilemma

    rows = [r for r in load(ROOT / "dataset.jsonl")
            if r["stratum"] == "variant_ortho"]

    d = Dilemma(lang="grc", resolve_articles=True, normalize=True,
                period="hellenistic")
    preds = d.lemmatize_batch([r["input"] for r in rows])
    require_batch_result_count(
        len(rows), len(preds), "alternation categorization lemmatization")

    all_ct = Counter()
    bad_ct = Counter()
    examples = {}
    for r, p in zip(rows, preds):
        lb = label(clean(r["form_orig"]), clean(r["form_reg"]))
        all_ct[lb] += 1
        if norm_lenient(p or "") != norm_lenient(r["lemma_gold"]):
            bad_ct[lb] += 1
            examples.setdefault(lb, []).append(
                f"{r['input']} -> {r['input_reg']} ({r['lemma_gold']}) "
                f"[pred {p}]")

    print(f"{'alternation':<40} {'tokens':>7} {'failed':>7} {'rate':>7}  example")
    for lb, n in all_ct.most_common(25):
        ex = examples.get(lb, [""])[0]
        print(f"{lb:<40} {n:>7} {bad_ct[lb]:>7} {bad_ct[lb]/n:>6.0%}  {ex}")

    (ROOT / "categories.json").write_text(json.dumps(
        {"all": all_ct.most_common(), "failed": bad_ct.most_common(),
         "examples": {k: v[:8] for k, v in examples.items()}},
        ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
