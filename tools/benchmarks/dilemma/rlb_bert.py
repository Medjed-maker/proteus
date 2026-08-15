"""H6: can a subword contextual model beat the count-based coverage ceiling?

H2's mechanism was that a trigram can only reorder candidates it has already
seen; H7 extrapolated that constraint to +4.5pt [+3.3, +5.8] even at perfect
candidate coverage. That ceiling is a property of *count-based* models. A
subword model has no OOV floor on its input side -- an unseen lemma is still
decomposed into pieces it knows -- so the ceiling does not transfer, and the
path has to be measured before the context signal can be declared closed.

Model: bowphs/GreBerta (RobertaForMaskedLM, 52k vocab). The commonly cited
pranaydeeps/Ancient-Greek-BERT is NOT usable here -- its published weights
contain no MLM head at all (200 tensors, no cls.predictions.*), so
pseudo-log-likelihood cannot be computed from it.

Two scorers, because the obvious design has a problem the obvious framing
hides:

  A  insert the candidate into the sentence, pseudo-log-likelihood over its
     own subwords, normalised by subword count. Keeps the no-OOV-floor
     property that motivates the whole experiment. But the candidate is a
     LEMMA (a citation form) while the context is inflected running text, so
     the string being scored is off the model's training distribution.

  B  mask the target position, read the model's distribution over surface
     tokens, and map it onto lemmas through the dictionary. No length
     normalisation and no lemma/surface mismatch -- but the output softmax is
     a closed 52k vocabulary, which gives back part of the OOV floor that A
     was chosen to avoid.

B turned out to be unusable, and the tokenizer alone was enough to show it:
a single masked position can only score a word that exists as one token, and
only 18.7% of candidate lemmas do (gold lemmas 39.6%, reg surface forms
46.6%). B cannot address four candidates in five, so A is the measurement and
B is kept only as a diagnostic over the single-token subset.

Two further facts settled from vocab/merges without loading any weights:

  * The vocabulary is NFC (5 of 48,432 Greek tokens carry combining marks)
    and tokenizer.json declares `normalizer: null` -- nothing normalises for
    us, so callers must. Same trap as the GLAUx NFD incident in §3.2.
  * Candidate subword counts inside one list vary by 2.24 on average and by
    up to 7, so a raw sum of log-probs would rank by brevity. Normalising per
    subword is load-bearing, which is why pll() returns a mean.
"""

import argparse
import hashlib
import json
import math
import random
import unicodedata
from pathlib import Path

ROOT = Path(__file__).parent
MODEL = "bowphs/GreBerta"
MODEL_REVISION = "3dce05464f1f429d68acd9b09e117632490c92d4"
TOKENIZER_REVISION = MODEL_REVISION
FORMAL_GATE_N = 200

TOKENIZER_FILES = {
    "added_tokens.json", "merges.txt", "special_tokens_map.json",
    "tokenizer.json", "tokenizer_config.json", "vocab.json",
}


def _content_digest(root: Path, include) -> str:
    """Hash selected checkpoint files with their relative paths."""
    files = sorted(path for path in root.rglob("*")
                   if path.is_file() and include(path))
    if not files:
        raise ValueError(f"no reproducibility files found in {root}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def portable_model_metadata(model_name: str, model_revision: str,
                            tokenizer_revision: str) -> dict[str, str]:
    """Return a portable identity for a public or local checkpoint."""
    local = Path(model_name)
    if not local.is_dir():
        return {"model": model_name,
                "model_revision": model_revision,
                "tokenizer_revision": tokenizer_revision}

    model_hash = _content_digest(
        local,
        lambda path: path.name == "config.json"
        or path.suffix == ".safetensors"
        or path.name.endswith(".bin"),
    )
    tokenizer_hash = _content_digest(
        local, lambda path: path.name in TOKENIZER_FILES)
    return {"model": "local-checkpoint",
            "model_revision": model_hash,
            "tokenizer_revision": tokenizer_hash}


def measurement_conclusion(gate: dict, best_diff: dict) -> tuple[str, str]:
    """Qualify a measurement against the formal gate that authorised it."""
    if gate.get("n", 0) < FORMAL_GATE_N:
        return ("exploratory",
                "EXPLORATORY -- formal 200-token gate incomplete")
    if gate.get("A", {}).get("verdict") != "signal":
        return ("exploratory",
                "EXPLORATORY -- formal gate did not authorise measurement")

    gain = best_diff["point"] * 100
    adopted = (gain >= 6.0
               and not (best_diff["lo"] <= 0 <= best_diff["hi"]))
    return ("preregistered", "ADOPT" if adopted
            else "REJECT -- close the context signal")


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def sentence_index() -> dict:
    """(doc, sent) -> surface tokens in word order, from the full dataset."""
    from run_eval import load
    by: dict = {}
    for r in load(ROOT / "dataset.jsonl"):
        by.setdefault((r["doc"], r["sent"]), []).append(r)
    for v in by.values():
        v.sort(key=lambda r: int(r["wid"]))
    return by


class Scorer:
    def __init__(self, model_name: str = MODEL, device: str = "cpu",
                 model_revision: str = MODEL_REVISION,
                 tokenizer_revision: str = TOKENIZER_REVISION):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        self.torch = torch
        # local_files_only: with a local checkout the loader otherwise tries
        # to reach the hub and blocks for minutes when the network is slow,
        # which is exactly the condition this was written under.
        local = Path(model_name).is_dir()
        tokenizer_kwargs = {"local_files_only": local}
        model_kwargs = {"local_files_only": local}
        if not local:
            tokenizer_kwargs["revision"] = tokenizer_revision
            model_kwargs["revision"] = model_revision
        self.tok = AutoTokenizer.from_pretrained(model_name,
                                                 **tokenizer_kwargs)
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name, **model_kwargs)
        self.model.eval()
        self.device = device
        self.model.to(device)
        self.mask_id = self.tok.mask_token_id
        # The NFC/NFD trap from §3.2, in a new place: if the tokenizer was
        # trained on one normalisation and we feed the other, every Greek word
        # shatters into single characters. Checked rather than assumed.
        probe = "ἄνθρωπος"
        n_nfc = len(self.tok.tokenize(unicodedata.normalize("NFC", probe)))
        n_nfd = len(self.tok.tokenize(unicodedata.normalize("NFD", probe)))
        self.norm = "NFC" if n_nfc <= n_nfd else "NFD"
        self.probe = {"nfc_pieces": n_nfc, "nfd_pieces": n_nfd,
                      "chosen": self.norm}

    def _norm(self, s: str) -> str:
        return unicodedata.normalize(self.norm, s)

    # -- scorer A ----------------------------------------------------------

    def pll(self, left: list[str], cand: str, right: list[str]) -> float:
        """Mean log P over the candidate's own subwords, each masked in turn.

        Mean rather than sum: a raw sum rewards short candidates purely for
        being short, which would rank ὁ above every content word.
        """
        torch = self.torch
        lt = self.tok(" ".join(self._norm(w) for w in left),
                      add_special_tokens=False)["input_ids"] if left else []
        rt = self.tok(" ".join(self._norm(w) for w in right),
                      add_special_tokens=False)["input_ids"] if right else []
        ct = self.tok(" " + self._norm(cand),
                      add_special_tokens=False)["input_ids"]
        if not ct:
            return -20.0

        bos, eos = self.tok.cls_token_id, self.tok.sep_token_id
        rows = []
        for i in range(len(ct)):
            ids = [bos] + lt + ct[:i] + [self.mask_id] + ct[i + 1:] + rt + [eos]
            rows.append(ids)
        width = max(len(r) for r in rows)
        pad = self.tok.pad_token_id
        batch = torch.tensor([r + [pad] * (width - len(r)) for r in rows])
        attn = torch.tensor([[1] * len(r) + [0] * (width - len(r))
                             for r in rows])
        with torch.no_grad():
            logits = self.model(input_ids=batch.to(self.device),
                                attention_mask=attn.to(self.device)).logits
        total = 0.0
        for i in range(len(ct)):
            pos = 1 + len(lt) + i
            lp = torch.log_softmax(logits[i, pos], dim=-1)
            total += lp[ct[i]].item()
        return total / len(ct)

    # -- scorer B ----------------------------------------------------------

    def mask_dist(self, left: list[str], right: list[str], topk: int = 500):
        """Distribution over surface tokens at a single masked position."""
        torch = self.torch
        lt = self.tok(" ".join(self._norm(w) for w in left),
                      add_special_tokens=False)["input_ids"] if left else []
        rt = self.tok(" ".join(self._norm(w) for w in right),
                      add_special_tokens=False)["input_ids"] if right else []
        ids = [self.tok.cls_token_id] + lt + [self.mask_id] + rt \
            + [self.tok.sep_token_id]
        with torch.no_grad():
            logits = self.model(
                input_ids=torch.tensor([ids]).to(self.device)).logits
        lp = torch.log_softmax(logits[0, 1 + len(lt)], dim=-1)
        vals, idx = torch.topk(lp, topk)
        return [(self.tok.decode([i]).strip(), v.item())
                for i, v in zip(idx.tolist(), vals)]


def build_lemma_map(pairs, lex):
    """surface string -> set of lemmas, for scorer B's dictionary mapping."""
    out: dict = {}
    for surf, _ in pairs:
        if not surf:
            continue
        for lem in lex.lemmas(nfc(surf)):
            out.setdefault(nfc(surf), set()).add(lem)
    return out


def pick_form(lemma: str, src_key: str, query: str, lex, memo: dict) -> str:
    """The inflected form of `lemma` that best matches what the scribe wrote.

    Scorer C. H6 showed the model rates the reg surface form 1.7 nats above the
    lemma in the same context -- it was being asked to score citation forms
    that never stand in running Greek. This picks a form that does.

    Only the query spelling may steer the choice; it is the one thing available
    at inference. Selecting against `reg` or `gold` would be choosing the form
    after seeing the answer, so neither is passed in.
    """
    key = (lemma, src_key, query)
    if key in memo:
        return memo[key]
    from rlb_keys import b1_key
    from rapidfuzz.distance import Levenshtein

    forms = [f for f, _ in lex.forms_for_key(src_key)
             if lemma in lex.lemmas(f)]
    if not forms:
        memo[key] = lemma            # fall back to A's behaviour
        return lemma
    q = b1_key(query)
    best = min(forms, key=lambda f: (Levenshtein.distance(b1_key(f), q),
                                     -_accents(f), f))
    memo[key] = best
    return best


def _accents(form: str) -> int:
    return sum(1 for ch in unicodedata.normalize("NFD", form)
               if unicodedata.combining(ch))


def cmd_measure(args) -> None:
    """H6 proper: BERT inside the cost band, recall@5, paired against R0.

    Same shape as rlb_lm.cmd_rerank -- the cost band still dominates and the
    model only reorders within it -- so the number is directly comparable to
    the trigram's +4.1pt rather than to a different experiment.
    """
    from rlb_stats import bootstrap
    from run_eval import norm_lenient

    rows = [json.loads(line)
            for line in (ROOT / args.dump).open(encoding="utf-8")]
    rows = [r for r in rows if r["split"] == args.split]
    if args.limit:
        rows = rows[:args.limit]
    keys = [(r["doc"], r["sent"], r["wid"]) for r in rows]
    print(f"split={args.split}  tokens={len(keys)}  "
          f"documents={len({k[0] for k in keys})}  topc={args.topc}")

    sents = sentence_index()
    sc = Scorer(args.model, model_revision=args.model_revision,
                tokenizer_revision=args.tokenizer_revision)
    print(f"normalisation probe: {sc.probe}")

    lex = memo = None
    diag = {"fallback": 0, "picked": 0, "pick_eq_reg": 0}
    if args.scorer == "C":
        from rlb_lexicon import Lexicon
        lex, memo = Lexicon(), {}
        print("scorer C: scoring each candidate's closest inflected form")
    print()

    # Score once, reuse for every weight.
    cache: dict = {}
    for n, row in enumerate(rows, 1):
        toks = sents.get((row["doc"], row["sent"]), [])
        pos = next((i for i, t in enumerate(toks)
                    if t["wid"] == row["wid"]), None)
        cands = row["cands"][:args.topc]
        if pos is None:
            cache[row["doc"], row["sent"], row["wid"]] = [0.0] * len(cands)
            continue
        left = [t["input"] for t in toks[max(0, pos - 12):pos]]
        right = [t["input"] for t in toks[pos + 1:pos + 13]]
        if args.scorer == "C":
            targets = []
            for c in cands:
                # row["form"] only -- never row["reg"] or row["gold"].
                f = pick_form(c[0], c[4], row["form"], lex, memo)
                targets.append(f)
                diag["fallback" if f == c[0] else "picked"] += 1
                if nfc(f) == nfc(row.get("reg", "")):
                    diag["pick_eq_reg"] += 1
        else:
            targets = [c[0] for c in cands]
        cache[row["doc"], row["sent"], row["wid"]] = [
            sc.pll(left, t, right) for t in targets]
        if n % 25 == 0:
            print(f"  scored {n}/{len(rows)}", flush=True)

    def hits(w: float) -> dict:
        out = {}
        for r in rows:
            key = (r["doc"], r["sent"], r["wid"])
            cands = r["cands"][:args.topc]
            s = cache[key]
            gl = norm_lenient(r["gold"])
            order = sorted(range(len(cands)),
                           key=lambda i: (cands[i][1],
                                          -(math.log1p(cands[i][2])
                                            + w * s[i]),
                                          -cands[i][3]))
            out[key] = int(any(norm_lenient(cands[i][0]) == gl
                               for i in order[:5]))
        return out

    base = hits(0.0)
    r0 = bootstrap(base, None, keys, args.boot)
    print(f"\nR0 recall@5 = {r0['point']:.1%}\n")
    print(f"{'w':>6}{'recall@5':>11}{'vs R0':>9}{'95% CI (paired)':>22}")
    out = {"split": args.split,
           **portable_model_metadata(args.model, args.model_revision,
                                     args.tokenizer_revision),
           "topc": args.topc,
           "scorer": args.scorer, "diagnostics": diag,
           "n_tokens": len(keys), "r0": r0, "variants": {}}
    for w in args.weights:
        h = hits(w)
        rr = bootstrap(h, None, keys, args.boot)
        d = bootstrap(h, base, keys, args.boot)
        ci = f"[{d['lo']:+.1%}, {d['hi']:+.1%}]"
        print(f"{w:>6}{rr['point']:>11.1%}{d['point']:>+9.1%}"
              f"{ci:>22}")
        out["variants"][str(w)] = {"recall5": rr["point"], "diff": d}

    best = max(out["variants"].values(), key=lambda v: v["diff"]["point"])
    gain = best["diff"]["point"] * 100
    gate_path = Path(args.gate_result)
    if not gate_path.is_absolute():
        gate_path = ROOT / gate_path
    gate = json.loads(gate_path.read_text(encoding="utf-8")) \
        if gate_path.exists() else {}
    status, verdict = measurement_conclusion(gate, best["diff"])
    label = "H6 pre-registered" if status == "preregistered" else "H6 exploratory"
    print(f"\n{label}: ADOPT iff dev recall@5 gain >= +6.0pt "
          "and CI excludes zero")
    print(f"best gain {gain:+.2f}pt  ->  {verdict}")
    out["preregistration_status"] = status
    out["verdict"] = verdict

    (ROOT / args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    print(f"wrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="dump_b3u_feat.jsonl")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=100, help="tokens in the gate")
    ap.add_argument("--topc", type=int, default=20,
                    help="candidates rescored per token (R0 order)")
    ap.add_argument("--seed", type=int, default=20260813)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--model-revision", default=MODEL_REVISION)
    ap.add_argument("--tokenizer-revision", default=TOKENIZER_REVISION)
    ap.add_argument("--gate-result", default="results_bert_gate.json")
    ap.add_argument("--out", default="results_bert_gate.json")
    ap.add_argument("--measure", action="store_true",
                    help="run the H6/H8 measurement instead of the gate")
    ap.add_argument("--scorer", choices=("A", "C"), default="A",
                    help="A = score the lemma (H6); C = score the closest "
                         "inflected form (H8)")
    ap.add_argument("--weights", type=float, nargs="+",
                    default=[0.3, 1.0, 2.0, 4.0])
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.measure:
        if args.out == "results_bert_gate.json":
            args.out = f"results_bert_{args.scorer}_{args.split}.json"
        cmd_measure(args)
        return

    from rlb_lexicon import Lexicon
    from run_eval import norm_lenient

    rows = [json.loads(line)
            for line in (ROOT / args.dump).open(encoding="utf-8")]
    rows = [r for r in rows if r["split"] == args.split]
    # The gate only means anything where ranking is possible: gold present and
    # more than a handful of candidates to sort.
    usable = [r for r in rows
              if len(r["cands"]) >= 5
              and any(norm_lenient(c[0]) == norm_lenient(r["gold"])
                      for c in r["cands"][:args.topc])]
    rnd = random.Random(args.seed)
    sample = rnd.sample(usable, min(args.n, len(usable)))
    print(f"split={args.split}  usable={len(usable)}  gate sample={len(sample)}")

    sents = sentence_index()
    lex = Lexicon()
    sc = Scorer(args.model, model_revision=args.model_revision,
                tokenizer_revision=args.tokenizer_revision)
    print(f"tokenizer normalisation probe: {sc.probe}")

    res = {**portable_model_metadata(args.model, args.model_revision,
                                     args.tokenizer_revision),
           "split": args.split, "n": len(sample),
           "topc": args.topc, "seed": args.seed, "probe": sc.probe}
    res["preregistration_status"] = (
        "complete" if len(sample) >= FORMAL_GATE_N else "exploratory")
    stats = {"A": [], "B": [], "B_scoreable": []}

    for n, row in enumerate(sample, 1):
        toks = sents.get((row["doc"], row["sent"]), [])
        pos = next((i for i, t in enumerate(toks)
                    if t["wid"] == row["wid"]), None)
        if pos is None:
            continue
        left = [t["input"] for t in toks[max(0, pos - 12):pos]]
        right = [t["input"] for t in toks[pos + 1:pos + 13]]
        cands = row["cands"][:args.topc]
        gold_l = norm_lenient(row["gold"])

        a_scores = [sc.pll(left, c[0], right) for c in cands]
        dist = sc.mask_dist(left, right)
        lmap = build_lemma_map(dist, lex)
        by_lemma: dict = {}
        for surf, lp in dist:
            for lem in lmap.get(nfc(surf), ()):
                by_lemma[lem] = max(by_lemma.get(lem, -1e9), lp)
        # Candidates the closed vocabulary cannot reach all collapse onto the
        # same floor, and a stable sort then leaves them in R0's order. B's
        # rank for those tokens therefore measures the baseline, not the
        # model, so they are recorded separately rather than averaged in.
        b_scores = [by_lemma.get(c[0], -20.0) for c in cands]
        b_gold_scoreable = norm_lenient(row["gold"]) in {
            norm_lenient(k) for k in by_lemma}

        for tag, scores in (("A", a_scores), ("B", b_scores)):
            order = sorted(range(len(cands)), key=lambda i: -scores[i])
            gi = next((j for j, i in enumerate(order)
                       if norm_lenient(cands[i][0]) == gold_l), None)
            if gi is not None:
                stats[tag].append(gi / max(len(cands) - 1, 1))
                if tag == "B" and b_gold_scoreable:
                    stats["B_scoreable"].append(gi / max(len(cands) - 1, 1))
        if n % 20 == 0:
            print(f"  {n}/{len(sample)}", flush=True)

    print(f"\n{'scorer':<8}{'n':>5}{'mean norm rank':>16}"
          f"{'(random = 0.500)':>20}")
    for tag in ("A", "B", "B_scoreable"):
        v = stats[tag]
        if not v:
            continue
        mean = sum(v) / len(v)
        sd = (sum((x - mean) ** 2 for x in v) / max(len(v) - 1, 1)) ** 0.5
        se = sd / math.sqrt(len(v))
        lo, hi = mean - 1.96 * se, mean + 1.96 * se
        verdict = "signal" if hi < 0.5 else "NO SIGNAL"
        print(f"{tag:<8}{len(v):>5}{mean:>16.3f}"
              f"    [{lo:.3f}, {hi:.3f}]  {verdict}")
        res[tag] = {"n": len(v), "mean_norm_rank": mean,
                    "ci": [lo, hi], "verdict": verdict}

    (ROOT / args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
