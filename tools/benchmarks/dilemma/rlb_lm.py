"""A lemma trigram LM over GLAUx, and context re-ranking of the candidate sets.

Why lemmas and not surface forms: the candidate sets are lemmas, so a form-level
LM would need a second mapping step and would inherit its errors. Dilemma's own
``train_lm.py`` builds a form-level model for an iOS keyboard, exports it to a
binary with a truncated vocabulary, and ships no reader in the pip package --
so reusing it would cost more than counting trigrams here.

Why a subset of GLAUx: the corpus is ~1,400 heavily annotated XML files, some
77MB each, and neither the git archive nor the tarball would transfer reliably
in this environment. Instead each text is streamed from raw.githubusercontent
with a time cap and only the lemma sequence is kept -- a contiguous prefix of
each work, which is still running text. Texts are taken largest-first from the
919 whose date range overlaps the papyri (300 BCE - 700 CE), so the sample is
as close to Koine as GLAUx gets.

What this cannot fix: GLAUx has no documentary material at all (see
rlb_leakcheck). The model meets papyrus vocabulary cold. That is the hypothesis
under test, not a defect of the setup.
"""

import argparse
import json
import math
import re
import subprocess
import time
import unicodedata
from collections import Counter
from pathlib import Path

from benchmark_provenance import (
    dilemma_package_identity,
    file_identity,
)
from run_eval import require_batch_result_count

ROOT = Path(__file__).parent
LEMMA_DIR = ROOT / "glaux_lemmas"
RAW = "https://raw.githubusercontent.com/alekkeersmaekers/glaux/main/xml/{}.xml"

# GLAUx ships NFD; PapyGreek and Dilemma are NFC. Comparing them unnormalised
# puts gold-lemma coverage at 0.4% instead of 84.5% -- a silent encoding
# mismatch that would have read as "context is useless".
def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


SENT = re.compile(r"<sentence\b")
LEMMA = re.compile(r'\blemma="([^"]*)"')
BOS, EOS, UNK = "<s>", "</s>", "<unk>"
CURL_TIMEOUT = 28


def stream_lemmas(tlg_id: str, seconds: int) -> tuple[int, int]:
    """Stream one text, keep the lemma sequence, discard everything else.

    The XML is one element per line, so this is line-oriented rather than a
    real parser. A curl timeout is an expected prefix boundary: sentences
    completed before it are retained, while the in-progress sentence is
    discarded because its closing boundary was not observed.
    """
    out = LEMMA_DIR / f"{tlg_id}.txt"
    if out.exists():
        return 0, 0
    LEMMA_DIR.mkdir(exist_ok=True)
    sents, toks, cur = 0, 0, []
    tmp = out.with_suffix(".part")
    # Popen as a context manager so an exception part-way through a 1,027-text
    # fetch closes the pipe and reaps the child instead of leaking both. A
    # leftover .part is harmless -- resumption keys off {tlg_id}.txt, which only
    # the rename below creates.
    with subprocess.Popen(
            ["curl", "--fail", "-sL", "--max-time", str(seconds),
             RAW.format(tlg_id)],
            stdout=subprocess.PIPE) as proc, \
            tmp.open("w", encoding="utf-8") as fh:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.decode("utf-8", "replace")
            if SENT.search(line):
                if cur:
                    fh.write(" ".join(cur) + "\n")
                    sents += 1
                    toks += len(cur)
                cur = []
                continue
            m = LEMMA.search(line)
            if m and m.group(1):
                cur.append(nfc(m.group(1)))
        returncode = proc.wait()
        if returncode == 0 and cur:
            fh.write(" ".join(cur) + "\n")
            sents += 1
            toks += len(cur)
    if returncode == CURL_TIMEOUT:
        if not sents:
            tmp.unlink(missing_ok=True)
            raise subprocess.TimeoutExpired(proc.args, seconds)
        tmp.rename(out)
        return sents, toks
    if returncode:
        tmp.unlink(missing_ok=True)
        raise subprocess.CalledProcessError(returncode, proc.args)
    tmp.rename(out)
    return sents, toks


def extract_local(path: Path, out: Path) -> tuple[int, int]:
    """Same extraction as stream_lemmas, but from a file already on disk."""
    sents, toks, cur = 0, 0, []
    tmp = out.with_suffix(".part")
    with path.open(encoding="utf-8", errors="replace") as src, \
            tmp.open("w", encoding="utf-8") as fh:
        for line in src:
            if SENT.search(line):
                if cur:
                    fh.write(" ".join(cur) + "\n")
                    sents += 1
                    toks += len(cur)
                cur = []
                continue
            m = LEMMA.search(line)
            if m and m.group(1):
                cur.append(nfc(m.group(1)))
        if cur:
            fh.write(" ".join(cur) + "\n")
            sents += 1
            toks += len(cur)
    tmp.rename(out)
    return sents, toks


def cmd_local(args) -> None:
    """Extract lemma sequences from a local GLAUx checkout.

    Resumable on purpose: the extraction walks a couple of gigabytes and must
    not be able to lose its work halfway.
    """
    src = ROOT / args.src
    out_dir = ROOT / args.out
    out_dir.mkdir(exist_ok=True)
    files = sorted(src.rglob("*.xml"))
    done = {p.stem for p in out_dir.glob("*.txt")}
    todo = [f for f in files if f.stem not in done]
    print(f"{len(files)} xml files, {len(done)} already extracted, "
          f"{len(todo)} to go", flush=True)
    t0 = time.time()
    for i, f in enumerate(todo):
        extract_local(f, out_dir / f"{f.stem}.txt")
        if i % 50 == 0:
            print(f"  {i:>5}/{len(todo)}  {time.time() - t0:>5.0f}s", flush=True)
        if time.time() - t0 > args.budget:
            print("  budget reached; rerun to continue")
            break
    n = len(list(out_dir.glob("*.txt")))
    print(f"extracted {n}/{len(files)}"
          f"{'  COMPLETE' if n == len(files) else ''}")


def cmd_fetch(args) -> None:
    ids = (ROOT / "glaux_fetch_ids.txt").read_text().split()
    done = {p.stem for p in LEMMA_DIR.glob("*.txt")} if LEMMA_DIR.exists() else set()
    todo = [i for i in ids if i not in done][:args.n]
    print(f"{len(done)} already fetched; taking {len(todo)}")
    t0 = time.time()
    for i in todo:
        s, t = stream_lemmas(i, args.seconds)
        print(f"  {i:<12}{s:>7} sentences {t:>9,} lemmas "
              f"({time.time() - t0:.0f}s)", flush=True)
        if time.time() - t0 > args.budget:
            print("  budget reached, stopping")
            break
    total = sum(1 for p in LEMMA_DIR.glob("*.txt") for _ in p.open())
    toks = sum(len(line.split()) for p in LEMMA_DIR.glob("*.txt")
               for line in p.open(encoding="utf-8"))
    print(f"\ncorpus so far: {len(list(LEMMA_DIR.glob('*.txt')))} texts, "
          f"{total:,} sentences, {toks:,} lemmas")


# --------------------------------------------------------------------------


class Trigram:
    """Stupid backoff. No smoothing beyond the backoff factor, which is what
    the technique is: score, not probability, and that is all a ranker needs."""

    ALPHA = 0.4

    def __init__(self, alpha: float = ALPHA):
        self.alpha = alpha
        self.uni, self.bi, self.tri = Counter(), Counter(), Counter()
        self.total = 0
        self.sources: list[dict] = []

    def add_sentence(self, toks: list[str]) -> None:
        seq = [BOS, BOS] + toks + [EOS]
        self.uni.update(seq)
        self.total += len(seq)
        for i in range(1, len(seq)):
            self.bi[seq[i - 1], seq[i]] += 1
        for i in range(2, len(seq)):
            self.tri[seq[i - 2], seq[i - 1], seq[i]] += 1

    def score(self, w: str, prev1: str, prev2: str) -> float:
        """log of the stupid-backoff score of w given the two previous."""
        t = self.tri.get((prev2, prev1, w), 0)
        if t:
            return math.log(t / self.bi[prev2, prev1])
        b = self.bi.get((prev1, w), 0)
        if b:
            return math.log(self.alpha * b / self.uni[prev1])
        u = self.uni.get(w, 0)
        if u:
            return math.log(self.alpha * self.alpha * u / self.total)
        return math.log(self.alpha ** 3 / self.total)

    @classmethod
    def train(cls, lemma_dirs, only_by_dir: dict[Path, set[str]] | None = None,
              groups=None, verbose: bool = True) -> "Trigram":
        """Train over one or more corpus directories.

        ``only_by_dir`` restricts explicitly selected directories to a set of
        file stems. Other directories remain unfiltered, which lets a dated
        GLAUx corpus be combined with DDbDP without guessing namespaces.

        ``groups`` gives the number of consecutive lines to join into one
        sentence, per directory. GLAUx files are already one sentence per line
        (group 1); papyri are one line per papyrus line, which is half a
        clause, so they want group 3. A blank line is always a hard break --
        that is where a lacuna or a section boundary was -- and no sentence
        crosses it.
        """
        if isinstance(lemma_dirs, (str, Path)):
            lemma_dirs = [lemma_dirs]
        dirs = [(Path(d) if Path(d).is_absolute() else ROOT / d).resolve()
                for d in lemma_dirs]
        if groups is None:
            groups = [1] * len(dirs)
        if len(groups) == 1 and len(dirs) > 1:
            groups = groups * len(dirs)
        if len(groups) != len(dirs):
            raise SystemExit(f"--group needs 1 or {len(dirs)} values")
        if any(group <= 0 for group in groups):
            raise SystemExit("--group values must be positive")

        filters = {
            Path(path).resolve(): stems
            for path, stems in (only_by_dir or {}).items()
        }

        lm = cls()
        for d, group in zip(dirs, groups):
            files = sorted(d.glob("*.txt"))
            use = filters.get(d)
            if use is not None and not ({p.stem for p in files} & use):
                raise SystemExit(f"file filter for {d} matches no files")
            before_t, before_f = lm.total, 0
            for p in files:
                if use is not None and p.stem not in use:
                    continue
                before_f += 1
                pending: list[str] = []
                pending_lines = 0
                for line in p.open(encoding="utf-8"):
                    toks = [nfc(t) for t in line.split()]
                    if not toks:                       # hard break
                        if pending:
                            lm.add_sentence(pending)
                            pending = []
                            pending_lines = 0
                        continue
                    pending.extend(toks)
                    pending_lines += 1
                    if pending_lines >= group:
                        lm.add_sentence(pending)
                        pending = []
                        pending_lines = 0
                if pending:
                    lm.add_sentence(pending)
            lm.sources.append({"dir": d.name, "group": group,
                               "files": before_f,
                               "tokens": lm.total - before_t})
            if verbose:
                s = lm.sources[-1]
                print(f"  {d.name}: {s['files']} files, {s['tokens']} tokens "
                      f"(group={group})")

        # An empty LM scores every candidate identically and would read as
        # "context does not help" rather than as the configuration error it is.
        if lm.total == 0:
            raise SystemExit(f"empty LM: no training text in {lemma_dirs}")
        return lm


def window_filters(lemma_dirs, ids: set[str]) -> dict[Path, set[str]]:
    """Build the date-window filter for the first lemma directory."""
    if not lemma_dirs:
        raise SystemExit("--window needs at least one --lemma-dir")
    first = Path(lemma_dirs[0])
    first = (first if first.is_absolute() else ROOT / first).resolve()
    stems = {path.stem for path in first.glob("*.txt")}
    if not (stems & ids):
        raise SystemExit(f"date-window filter matches no files in {first}")
    return {first: ids}


def date_window_ids(lo: int = -300, hi: int = 700) -> set:
    """TLG ids whose date range overlaps the documentary papyri."""
    import csv
    rows = list((ROOT / "glaux_metadata.txt").open(encoding="utf-8"))
    out = set()
    for r in csv.DictReader(rows, delimiter="\t"):
        def num(x):
            x = x.strip()
            return int(x) if x.lstrip("-").isdigit() else None
        s, e = num(r["STARTDATE"]), num(r["ENDDATE"])
        if s is not None and e is not None and e >= lo and s <= hi:
            out.add(r["TLG"].strip())
    return out




# --------------------------------------------------------------------------
# Context re-ranking
# --------------------------------------------------------------------------


def sentence_index() -> dict:
    """(doc, sent) -> tokens in word order, from the full dataset."""
    from run_eval import load
    by = {}
    for r in load(ROOT / "dataset.jsonl"):
        by.setdefault((r["doc"], r["sent"]), []).append(r)
    for v in by.values():
        v.sort(key=lambda r: int(r["wid"]))
    return by


def _context_cache_lemmas(
        *, allow_legacy_context_cache: bool = False) -> dict:
    """Load context lemmas after validating their dataset provenance."""
    cached = json.loads((ROOT / "context_lemmas.json").read_text())
    if not isinstance(cached, dict):
        raise SystemExit("context cache must be a mapping")
    if "lemmas" not in cached:
        if allow_legacy_context_cache:
            return cached
        raise SystemExit(
            "legacy context cache has no provenance; regenerate it with "
            "'rlb_lm.py context' or pass --allow-legacy-context-cache")

    provenance = cached.get("provenance")
    cached_dataset = (
        provenance.get("dataset") if isinstance(provenance, dict) else None)
    current_dataset = file_identity(ROOT / "dataset.jsonl")
    if cached_dataset != current_dataset:
        raise SystemExit(
            "context cache dataset provenance mismatch; regenerate "
            "context_lemmas.json with 'rlb_lm.py context'")

    lemmas = cached["lemmas"]
    if not isinstance(lemmas, dict):
        raise SystemExit("context cache lemmas must be a mapping")
    return lemmas


def contexts(
        rows: list[dict], mode: str, *,
        allow_legacy_context_cache: bool = False) -> dict:
    """The two lemmas preceding each evaluated token.

    mode='gold'      the annotated lemma of the neighbours -- an upper bound
    mode='dilemma'   Dilemma's own lemmatization of the neighbouring *papyrus*
                     spellings, which is what a deployed system would have
    """
    by = sentence_index()
    out = {}
    pred = {}
    if mode == "dilemma":
        pred = _context_cache_lemmas(
            allow_legacy_context_cache=allow_legacy_context_cache)
    for r in rows:
        seq = by.get((r["doc"], r["sent"]), [])
        i = next((i for i, x in enumerate(seq) if x["wid"] == r["wid"]), None)
        prev = seq[max(0, i - 2):i] if i is not None else []
        if mode == "gold":
            lem = [nfc(x["lemma_gold"]) for x in prev]
        else:
            lem = [nfc(pred.get(
                f"{x['doc']}|{x['sent']}|{x['wid']}") or UNK)
                   for x in prev]
        while len(lem) < 2:
            lem.insert(0, BOS)
        out[r["doc"], r["sent"], r["wid"]] = (lem[-2], lem[-1])
    return out


def cmd_context(args) -> None:
    """Lemmatize every context token with Dilemma, once, to a cache."""
    from dilemma import Dilemma
    from clean import clean
    package = dilemma_package_identity(require_expected=True)
    by = sentence_index()
    need = {}
    for seq in by.values():
        for x in seq:
            need[f"{x['doc']}|{x['sent']}|{x['wid']}"] = clean(x["form_orig"])
    keys = [k for k, v in need.items() if v]
    words = [need[k] for k in keys]
    print(f"lemmatizing {len(words):,} context tokens with Dilemma "
          f"(guess=False)...", flush=True)
    d = Dilemma(lang="grc", resolve_articles=True, normalize=True,
                period="hellenistic")
    preds = d.lemmatize_batch(words, guess=False)
    require_batch_result_count(
        len(words), len(preds), "context lemmatization")
    out = {k: p for k, p in zip(keys, preds) if p}
    payload = {
        "provenance": {
            "dilemma": package,
            "dataset": file_identity(ROOT / "dataset.jsonl"),
        },
        "lemmas": out,
    }
    (ROOT / "context_lemmas.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"  {len(out):,}/{len(keys):,} resolved -> context_lemmas.json")


def cmd_ddb(args) -> None:
    """Lemmatise the DDbDP running text into an in-domain LM corpus.

    Output mirrors the input line for line -- one papyrus line per line, blank
    line for a hard break -- rather than pre-joining into sentences. Sentence
    length then becomes a training-time parameter (``Trigram.train(group=)``),
    so the segmentation sensitivity check costs no re-lemmatisation.

    Unresolved tokens become ``<unk>`` rather than their surface form. This
    preserves their position without injecting a surface-form type that no
    candidate lemma can ever match. In particular, the model must not learn a
    trigram by making the neighbours of an unresolved token adjacent.
    """
    from dilemma import Dilemma
    dilemma_package_identity(require_expected=True)

    src, out = ROOT / args.src, ROOT / args.out
    out.mkdir(exist_ok=True)
    files = [p for p in sorted(src.glob("*.txt"))
             if not (out / p.name).exists()]
    print(f"{len(files)} documents to lemmatise "
          f"({len(list(src.glob('*.txt'))) - len(files)} already done)")
    if not files:
        return

    d = Dilemma(lang="grc", resolve_articles=True, normalize=True,
                period="hellenistic")
    t0, done, kept, seen = time.time(), 0, 0, 0
    batch: list[Path] = []
    batch_words = 0

    def flush(paths: list[Path]) -> tuple[int, int]:
        docs = [[line.split() for line in p.read_text(encoding="utf-8")
                 .split("\n")] for p in paths]
        words = [w for doc in docs for line in doc for w in line]
        if not words:
            return 0, 0
        preds = d.lemmatize_batch(words, guess=False)
        require_batch_result_count(
            len(words), len(preds), "DDbDP text lemmatization")
        it = iter(preds)
        n_seen = n_kept = 0
        for path, doc in zip(paths, docs):
            lines = []
            for line in doc:
                lemmas = []
                for _ in line:
                    p = next(it)
                    n_seen += 1
                    if p:
                        lemmas.append(nfc(p))
                        n_kept += 1
                    else:
                        # Preserve the source-token positions. Dropping an
                        # unresolved token would make its neighbours adjacent
                        # and train a bigram/trigram that never occurred.
                        lemmas.append(UNK)
                lines.append(" ".join(lemmas))
            tmp = (out / path.name).with_suffix(".part")
            tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
            tmp.rename(out / path.name)
        return n_seen, n_kept

    for path in files:
        batch.append(path)
        batch_words += path.stat().st_size // 12      # rough token estimate
        if batch_words >= args.batch:
            s, k = flush(batch)
            seen += s
            kept += k
            done += len(batch)
            batch, batch_words = [], 0
            el = time.time() - t0
            print(f"  {done}/{len(files)} docs, {kept:,} lemmas, "
                  f"{seen / el:.0f} tok/s, {el:.0f}s", flush=True)
            if args.budget and el > args.budget:
                print(f"budget reached at {done} docs; rerun to continue")
                return
    if batch:
        s, k = flush(batch)
        seen += s
        kept += k
        done += len(batch)

    rate = kept / seen if seen else 0.0
    print(f"\n{done} documents, {seen:,} tokens, {kept:,} lemmas "
          f"({rate:.1%} resolved) in {time.time() - t0:.0f}s -> {args.out}")


def cmd_rerank(args) -> None:
    from rlb_stats import bootstrap
    from run_eval import norm_lenient

    print("training trigram LM...", flush=True)
    filters = (window_filters(args.lemma_dir, date_window_ids())
               if args.window else None)
    lm = Trigram.train(args.lemma_dir, filters, args.group)
    print(f"  {len(lm.uni):,} types, {lm.total:,} tokens, "
          f"{len(lm.bi):,} bigrams, {len(lm.tri):,} trigrams")

    rows = [json.loads(line) for line in
            (ROOT / args.dump).open(encoding="utf-8")]
    rows = [r for r in rows if args.split == "all" or r["split"] == args.split]
    keys = [(r["doc"], r["sent"], r["wid"]) for r in rows]
    print(f"\nsplit={args.split}  tokens={len(keys)}  "
          f"documents={len({k[0] for k in keys})}")

    def run(ctx, w):
        hits = {}
        for r in rows:
            p2, p1 = ctx[r["doc"], r["sent"], r["wid"]]
            gl = norm_lenient(r["gold"])
            def key(c):
                s = lm.score(nfc(c[0]), p1, p2)
                return (c[1], -(math.log1p(c[2]) + w * s), -c[3])
            ranked = sorted(r["cands"], key=key)
            hits[r["doc"], r["sent"], r["wid"]] = int(
                any(norm_lenient(c[0]) == gl for c in ranked[:5]))
        return hits

    base = run(contexts(rows, "gold"), 0.0)
    b = bootstrap(base, None, keys, args.boot)
    print(f"\nR0 baseline recall@5 = {b['point']:.1%}\n")
    print(f"{'context':<10}{'w':>6}{'recall@5':>11}{'vs R0':>9}"
          f"{'95% CI (paired)':>22}")
    out = {}
    for mode in args.contexts:
        ctx = contexts(
            rows,
            mode,
            allow_legacy_context_cache=args.allow_legacy_context_cache,
        )
        for w in args.weights:
            hits = run(ctx, w)
            r = bootstrap(hits, None, keys, args.boot)
            d = bootstrap(hits, base, keys, args.boot)
            ci = f"[{d['lo']:+.1%}, {d['hi']:+.1%}]"
            print(f"{mode:<10}{w:>6}{r['point']:>11.1%}{d['point']:>+9.1%}{ci:>22}")
            out[f"{mode}|{w}"] = {"recall5": r["point"], "diff": d}

    # Tagged, because a fixed filename means every corpus condition overwrites
    # the previous one. results_lm_dev_fullcorpus.json is byte-identical to the
    # 30-text run for exactly that reason: the artefact was clobbered and only
    # the prose in §3.4 preserved what had been measured.
    name = (f"results_lm_{args.split}_{args.tag}.json" if args.tag
            else f"results_lm_{args.split}.json")
    (ROOT / name).write_text(
        json.dumps({"split": args.split, "tag": args.tag,
                    "dump": args.dump,
                    "lemma_dirs": list(args.lemma_dir),
                    "groups": args.group,
                    "sources": lm.sources,
                    "lm_tokens": lm.total,
                    "lm_types": len(lm.uni), "variants": out},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten to {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--n", type=int, default=8)
    f.add_argument("--seconds", type=int, default=40)
    f.add_argument("--budget", type=int, default=420)
    f.set_defaults(func=cmd_fetch)

    lo = sub.add_parser("local")
    lo.add_argument("--src", default="glaux_partial")
    lo.add_argument("--out", default="glaux_lemmas_full")
    lo.add_argument("--budget", type=int, default=420)
    lo.set_defaults(func=cmd_local)

    c = sub.add_parser("context")
    c.set_defaults(func=cmd_context)

    r = sub.add_parser("rerank")
    r.add_argument("--split", default="dev")
    r.add_argument("--boot", type=int, default=1000)
    r.add_argument("--weights", type=float, nargs="*",
                   default=[0.1, 0.3, 0.6, 1.0, 2.0])
    r.add_argument("--contexts", nargs="*", default=["gold"])
    r.add_argument("--lemma-dir", nargs="+", default=["glaux_lemmas"],
                   help="one or more corpus directories, concatenated")
    r.add_argument("--group", type=int, nargs="+", default=[1],
                   help="lines joined per sentence, one value per --lemma-dir "
                        "(GLAUx 1, papyri 3)")
    r.add_argument("--dump", default="dump_b3u_feat.jsonl")
    r.add_argument("--tag", default="",
                   help="suffix for the results file; without it each corpus "
                        "condition overwrites the last")
    r.add_argument("--window", action="store_true",
                   help="restrict the first --lemma-dir to the papyri date "
                        "range; fail when no TLG filenames match")
    r.add_argument(
        "--allow-legacy-context-cache",
        action="store_true",
        help="accept a pre-provenance bare context_lemmas.json mapping",
    )
    r.set_defaults(func=cmd_rerank)

    d = sub.add_parser("ddb")
    d.add_argument("--src", default="ddb_text")
    d.add_argument("--out", default="ddb_lemmas")
    d.add_argument("--budget", type=float, default=1800)
    d.add_argument("--batch", type=int, default=50000)
    d.set_defaults(func=cmd_ddb)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
