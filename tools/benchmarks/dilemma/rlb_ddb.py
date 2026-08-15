"""Extract running text and orig/reg pairs from the DDbDP EpiDoc corpus.

One walker, two consumers:

  text   surface forms per papyrus line -> lemmatised by Dilemma -> the
         in-domain trigram corpus (rlb_lm.py). §3.4 showed a tenfold increase
         in GLAUx bought +0.5pt on dev, so the bottleneck is domain, not size.
         The failures that motivate this are all documentary formulae:
         ἔρροσσο for ἔρρωσο is a letter's closing, εὐχαριτῶμεν a fixed thanks.

  pairs  <choice><reg>X</reg><orig>Y</orig></choice> -> the orig->reg
         normalisation benchmark. CC BY 3.0 throughout, so it needs no part of
         PapyGreek (CC BY-SA) and its test side is not yet spent.

Files are named by Trismegistos number -- DDbDP/41/41596.xml is TM 41596, which
is PapyGreek's bgu.1.261 -- so leak exclusion is a filename test.

ET.parse rather than regex: the policy here is structural. <rdg> has to be
dropped as a subtree, <choice> needs both children at once, and <lb break="no"/>
has to be read relative to the surrounding text and tail. rlb_leakcheck.py's
regex approach works because it scrapes flat attributes out of a header; this
is the other case.
"""

import argparse
import json
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).parent

GREEK_RE = re.compile(r"[Ͱ-ϡϰ-Ͽἀ-῿]")
COPTIC_RE = re.compile(r"[Ϣ-ϯⲀ-⳿]")

# Kept for their text, tag discarded. <supplied> and <expan>/<ex> are editorial
# restorations, so they are kept for the running text (a whole word is what the
# lemmatiser needs) but flagged, so the pair benchmark can exclude them.
TRANSPARENT = {
    "reg", "orig", "lem", "choice", "app", "ab", "div", "p", "l", "seg", "w",
    "hi", "num", "unclear", "supplied", "expan", "ex", "add", "surplus",
    "persName", "placeName", "geogName", "orgName", "rs", "q", "quote",
    "certainty", "abbr", "am", "subst", "corr", "sic", "g",
}

# Dropped as whole subtrees.
OPAQUE = {"rdg", "note", "del", "milestone", "figure", "handShift",
          "witDetail", "head"}

# Marks a word-internal line break inside a flattened subtree: whitespace on
# either side of it is editorial layout, not a word boundary.
GLUE = "\x00"

# Flag-setting elements: kept, but any pair containing one is marked.
FLAGGING = {"supplied": "supplied", "expan": "ex", "ex": "ex", "abbr": "ex",
            "unclear": "unclear", "add": "add", "surplus": "surplus"}


def _squeeze(lines: list[list[str]]) -> list[list[str]]:
    """Drop leading/trailing blanks and collapse runs of them to one."""
    out: list[list[str]] = []
    for line in lines:
        if line:
            out.append(line)
        elif out and out[-1]:
            out.append([])
    while out and not out[-1]:
        out.pop()
    return out


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def tag_of(el) -> str:
    """TEI tags arrive namespaced: {http://www.tei-c.org/ns/1.0}lb."""
    return el.tag.rpartition("}")[2] if isinstance(el.tag, str) else ""


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def is_greek(s: str) -> bool:
    return bool(s) and bool(GREEK_RE.search(s))


def strip_edges(tok: str) -> str:
    """Drop leading/trailing non-letters: editorial punctuation, not spelling.

    Papyri editions attach ``.``, ``,`` and the high stop ``·`` directly to the
    word. Left in place they make every clause-final word its own LM type and a
    guaranteed lemmatiser miss -- ``χαίρειν.`` is not in any dictionary.
    Interior characters are left alone, so apostrophes marking elision survive.
    """
    start, end = 0, len(tok)
    while start < end and not (tok[start].isalpha()
                               or unicodedata.combining(tok[start])):
        start += 1
    while end > start and not (tok[end - 1].isalpha()
                               or unicodedata.combining(tok[end - 1])):
        end -= 1
    return tok[start:end]


class TokenBuf:
    """Accumulates running text with explicit control over word boundaries.

    <lb break="no"/> means the word continues on the next papyrus line, so the
    two halves must be joined with no space: the sample has ἐνέγ|και and
    καταπλεῦ|σαι. Getting this wrong shatters roughly one word per eight lines,
    and every shattered word is a lemmatiser failure.
    """

    def __init__(self) -> None:
        self.lines: list[list[str]] = [[]]
        self._cur: list[str] = []
        self._glue = False

    def add(self, s: str) -> None:
        if not s:
            return
        parts = s.split()
        if not parts:
            if not self._glue:
                self.close()
            return
        leading_ws = s[:1].isspace()
        for i, part in enumerate(parts):
            if i == 0 and (self._glue or not leading_ws):
                self._cur.append(part)
            else:
                self.close()
                self._cur.append(part)
        self._glue = False
        if s[-1:].isspace():
            self.close()

    def glue(self) -> None:
        """Next add() continues the current token.

        The text before a break="no" line break ends with the newline and
        indentation of the source file, so add() has already closed the token.
        Re-open it: the halves are one word.
        """
        if not self._cur and self.lines[-1]:
            self._cur = [self.lines[-1].pop()]
        self._glue = True

    def close(self) -> None:
        if self._cur:
            self.lines[-1].append("".join(self._cur))
            self._cur = []

    def newline(self) -> None:
        self.close()
        self._glue = False
        if self.lines[-1]:
            self.lines.append([])

    def hard_break(self) -> None:
        """A lacuna or a section boundary: no trigram may cross it."""
        self.newline()
        if self.lines[-1] or len(self.lines) == 1:
            self.lines.append([])
        self.lines.append([])

    def text(self) -> str:
        self.close()
        return " ".join(t for line in self.lines for t in line)

    def finish(self) -> list[list[str]]:
        self.close()
        out = []
        for line in self.lines:
            toks = [nfc(s) for s in (strip_edges(t) for t in line) if s]
            out.append(toks)
        return out


def _subtree_text(el, flags: set) -> str:
    """Flattened text of one side of a <choice>, honouring break="no"."""
    out = []

    def walk(node):
        name = tag_of(node)
        if name in OPAQUE:
            return
        if name in FLAGGING:
            flags.add(FLAGGING[name])
        if name == "lb":
            out.append(" " if node.get("break") != "no" else GLUE)
        elif name == "gap":
            flags.add("gap")
            out.append(" ")
        else:
            if node.text:
                out.append(node.text)
        for child in node:
            walk(child)
            if child.tail:
                out.append(child.tail)

    if el.text:
        out.append(el.text)
    for child in el:
        walk(child)
        if child.tail:
            out.append(child.tail)
    joined = re.sub(rf"\s*{GLUE}\s*", "", "".join(out))
    return nfc(" ".join(s for s in (strip_edges(t) for t in joined.split())
                        if s))


class DocParser:
    def __init__(self, path: Path):
        self.path = path
        self.buf = TokenBuf()
        self.pairs: list[dict] = []
        self.stats: Counter = Counter()
        self.lb = ""

    def parse(self) -> dict | None:
        try:
            root = ET.parse(self.path).getroot()
        except ET.ParseError:
            self.stats["parse_error"] += 1
            return None

        idno = {}
        for el in root.iter():
            if tag_of(el) == "idno" and el.get("type"):
                idno[el.get("type")] = (el.text or "").strip()

        edition = None
        for el in root.iter():
            if tag_of(el) == "div" and el.get("type") == "edition":
                edition = el
                break
        if edition is None:
            self.stats["no_edition"] += 1
            return None

        self._walk(edition)
        lines = self.buf.finish()

        hybrid = idno.get("ddb-hybrid", "")
        series, volume, number = hybrid_parts(hybrid)
        return {
            "tm": idno.get("TM", "") or self.path.stem,
            "hgv": idno.get("HGV", ""),
            "hybrid": hybrid, "series": series, "volume": volume,
            "number": number,
            "path": _rel(self.path),
            # Empty lines are kept: they are the hard breaks that <gap> and
            # section boundaries produced, and Trigram.train reads a blank line
            # as "no sentence crosses here". Dropping them would let the model
            # form trigrams straight across a lacuna -- fabricated evidence,
            # not merely noisy evidence. Leading, trailing and repeated blanks
            # collapse to one.
            "lines": _squeeze(lines),
            "n_lines": sum(1 for line in lines if line),
            "n_tokens": sum(len(line) for line in lines),
            "pairs": self.pairs,
            "stats": dict(self.stats),
        }

    def _walk(self, el, in_lem: bool = False) -> None:
        name = tag_of(el)

        if name in OPAQUE:
            self.stats[f"dropped_{name}"] += 1
            return

        if name == "lb":
            self.lb = el.get("n", "") or self.lb
            if el.get("break") == "no":
                self.buf.glue()
            else:
                self.buf.newline()
            return

        if name == "gap":
            # No placeholder. A <gap> token would enter the LM vocabulary as a
            # type that no candidate lemma can ever match, and would let the
            # trigram bridge a lacuna as if it were continuous text.
            self.stats["gap"] += 1
            self.buf.hard_break()
            return

        if name == "choice":
            self._choice(el, in_lem)
            return

        if name == "app":
            # Take <lem>, drop <rdg>. <choice> nests inside <lem>.
            lem = next((c for c in el if tag_of(c) == "lem"), None)
            self.stats["app"] += 1
            if lem is not None:
                if lem.text:
                    self.buf.add(lem.text)
                for child in lem:
                    self._walk(child, in_lem=True)
                    if child.tail:
                        self.buf.add(child.tail)
            return

        if name == "foreign" and (el.get("{http://www.w3.org/XML/1998/namespace}lang")
                                  or "grc") != "grc":
            self.stats["dropped_foreign"] += 1
            return

        if name in ("div", "ab") and self.buf.lines != [[]]:
            self.buf.hard_break()

        if el.text:
            self.buf.add(el.text)
        for child in el:
            self._walk(child, in_lem)
            if child.tail:
                self.buf.add(child.tail)

    def _choice(self, el, in_lem: bool) -> None:
        kids = {tag_of(c): c for c in el}
        flags: set = set()

        if "reg" in kids and "orig" in kids:
            reg = _subtree_text(kids["reg"], flags)
            orig = _subtree_text(kids["orig"], flags)
            kind = "reg"
        elif "corr" in kids and "sic" in kids:
            # Emendation of the text, not regularisation of the spelling.
            # Mixing the two contaminates the benchmark with editorial
            # conjecture, so it is counted separately and opted into.
            reg = _subtree_text(kids["corr"], flags)
            orig = _subtree_text(kids["sic"], flags)
            kind = "corr"
        else:
            for c in el:
                self._walk(c, in_lem)
            return

        if in_lem:
            flags.add("in_lem")
        self.stats[f"choice_{kind}"] += 1

        # The regularised side enters the running text: Dilemma with
        # guess=False abstains on exactly the non-standard spellings, so an
        # orig-side corpus would be silently biased toward the subset that is
        # already standard.
        self.buf.add(" " + (reg or orig) + " ")

        self.pairs.append({
            "idx": len(self.pairs), "lb": self.lb, "kind": kind,
            "orig": orig, "reg": reg, "flags": sorted(flags),
        })


def hybrid_parts(hybrid: str) -> tuple[str, str, str]:
    """'bgu;1;261' -> ('bgu', '1', '261'). Missing volume stays empty."""
    parts = (hybrid or "").split(";")
    while len(parts) < 3:
        parts.append("")
    return parts[0], parts[1], parts[2]


def pair_drop_reasons(pair: dict, include_corr: bool, strict: bool) -> list[str]:
    reasons = []
    orig, reg = pair["orig"], pair["reg"]
    if pair["kind"] == "corr" and not include_corr:
        reasons.append("corr_sic")
    if not orig or not reg:
        reasons.append("empty")
    if " " in orig or " " in reg:
        reasons.append("multi")
    if not is_greek(orig) or not is_greek(reg):
        # Much of this is Greek vocabulary written in Coptic script
        # (ⲇⲓⲁⲕⲟⲛⲟⲥ -> διάκονος). That is a transliteration task, not an
        # orthographic-variation one, and it is counted separately so the
        # exclusion does not read as generic noise.
        reasons.append("coptic" if COPTIC_RE.search(orig) or
                       COPTIC_RE.search(reg) else "non_greek")
    if orig and reg and nfc(orig) == nfc(reg):
        reasons.append("identical")
    if "ex" in pair["flags"]:
        reasons.append("ex")
    if strict:
        for f in ("supplied", "gap", "in_lem"):
            if f in pair["flags"]:
                reasons.append(f)
    return reasons


def iter_files(src: Path, exclude: set, limit: int = 0):
    n = 0
    for path in sorted(src.rglob("*.xml")):
        if path.stem in exclude:
            continue
        yield path
        n += 1
        if limit and n >= limit:
            return


def papygreek_tms(papygreek: Path) -> set:
    """Every normalized Trismegistos number, as DDbDP file stems."""
    ids = json.loads(papygreek.read_text(encoding="utf-8"))
    return set(ids["all_tm_ids"])


def load_exclusions(policy: str, index_path: Path, papygreek: Path) -> set:
    """TM-only, or TM plus every document sharing a series+volume with one."""
    tms = papygreek_tms(papygreek)
    if policy == "tm":
        return tms
    if not index_path.exists():
        raise FileNotFoundError(
            f"exclusion index not found: {index_path}; run the scan first")

    rows = [json.loads(line) for line in index_path.open(encoding="utf-8")]
    vols = {(r["series"], r["volume"]) for r in rows if r["tm"] in tms}
    vols.discard(("", ""))
    return tms | {r["tm"] for r in rows if (r["series"], r["volume"]) in vols}


def cmd_scan(args) -> None:
    """Cheap measurement pass: metadata and counts, no text written."""
    src = ROOT / args.src
    out = ROOT / args.out
    n = 0
    totals = Counter()
    t0 = time.time()
    with out.open("w", encoding="utf-8") as fh:
        for path in iter_files(src, set(), args.limit):
            doc = DocParser(path).parse()
            if doc is None:
                totals["skipped"] += 1
                continue
            fh.write(json.dumps({
                k: doc[k] for k in
                ("tm", "hgv", "hybrid", "series", "volume", "number", "path",
                 "n_lines", "n_tokens")
            } | {"n_pairs_raw": len(doc["pairs"]),
                 "n_pairs_reg": sum(1 for p in doc["pairs"]
                                    if p["kind"] == "reg")},
                ensure_ascii=False) + "\n")
            totals["docs"] += 1
            totals["tokens"] += doc["n_tokens"]
            totals["lines"] += doc["n_lines"]
            totals["pairs"] += len(doc["pairs"])
            n += 1
            if n % 5000 == 0:
                print(f"  {n} docs, {time.time() - t0:.0f}s", flush=True)
    print(f"scanned {totals['docs']} docs in {time.time() - t0:.0f}s")
    print(f"  tokens {totals['tokens']}  lines {totals['lines']}  "
          f"pairs {totals['pairs']}  skipped {totals['skipped']}")
    print(f"wrote {args.out}")


def cmd_text(args) -> None:
    src, out = ROOT / args.src, ROOT / args.out
    out.mkdir(exist_ok=True)
    exclude = load_exclusions(args.exclude, ROOT / args.index,
                              ROOT / args.papygreek)
    print(f"exclusion policy {args.exclude}: {len(exclude)} documents")
    t0, n = time.time(), 0
    for path in iter_files(src, exclude, args.limit):
        dest = out / f"{path.stem}.txt"
        if dest.exists():
            continue
        doc = DocParser(path).parse()
        if doc is None or not doc["lines"]:
            continue
        tmp = dest.with_suffix(".part")
        tmp.write_text(
            "\n".join(" ".join(line) for line in doc["lines"]) + "\n",
                       encoding="utf-8")
        tmp.rename(dest)
        n += 1
        if n % 2000 == 0:
            print(f"  {n} docs, {time.time() - t0:.0f}s", flush=True)
        if args.budget and time.time() - t0 > args.budget:
            print(f"budget reached at {n} docs; rerun to continue")
            return
    print(f"wrote {n} documents to {args.out} in {time.time() - t0:.0f}s")


def cmd_pairs(args) -> None:
    src = ROOT / args.src
    exclude = load_exclusions(args.exclude, ROOT / args.index,
                              ROOT / args.papygreek)
    print(f"exclusion policy {args.exclude}: {len(exclude)} documents")
    drops = Counter()
    kept = raw = docs = 0
    t0 = time.time()
    with (ROOT / args.out).open("w", encoding="utf-8") as fh:
        for path in iter_files(src, exclude, args.limit):
            doc = DocParser(path).parse()
            if doc is None:
                continue
            docs += 1
            for pair in doc["pairs"]:
                raw += 1
                reasons = pair_drop_reasons(pair, args.include_corr,
                                            args.strict_orig)
                for r in reasons:
                    drops[r] += 1
                if reasons:
                    drops["_any"] += 1
                    continue
                kept += 1
                fh.write(json.dumps({
                    "doc": doc["tm"], "sent": pair["lb"],
                    "wid": str(pair["idx"]),
                    "form": pair["orig"], "reg": pair["reg"],
                    "gold": pair["reg"],
                    "hybrid": doc["hybrid"], "series": doc["series"],
                    "volume": doc["volume"], "hgv": doc["hgv"],
                    "flags": pair["flags"],
                }, ensure_ascii=False) + "\n")
            if docs % 5000 == 0:
                print(f"  {docs} docs, {kept} pairs, "
                      f"{time.time() - t0:.0f}s", flush=True)

    stats = {"policy": args.exclude, "n_docs": docs, "n_pairs_raw": raw,
             "n_pairs_kept": kept,
             "drop_counts": {k: v for k, v in drops.items() if k != "_any"},
             "drop_shares": {k: v / raw for k, v in drops.items()
                             if k != "_any" and raw},
             "any_drop": drops["_any"],
             "any_drop_share": drops["_any"] / raw if raw else 0.0}
    (ROOT / args.stats).write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{docs} docs, {raw} raw pairs, {kept} kept "
          f"({kept / raw:.1%})" if raw else "no pairs")
    for k, v in sorted(drops.items(), key=lambda kv: -kv[1]):
        if k != "_any":
            print(f"  drop {k:12s} {v:7d}  {v / raw:6.1%}")
    # Pre-registered gate: any single reason over 30%, or all of them over 60%,
    # means the policy is re-justified in writing before any recall number is
    # computed from what survives.
    for k, v in drops.items():
        if k != "_any" and raw and v / raw > 0.30:
            print(f"\n!! drop reason {k} exceeds 30% -- policy gate triggered")
    if raw and drops["_any"] / raw > 0.60:
        print("\n!! total drops exceed 60% -- re-scope the pair benchmark")
    print(f"wrote {args.out} and {args.stats}")


def cmd_exclusions(args) -> None:
    """Both policies, costed, from the single scan pass."""
    rows = [json.loads(line)
            for line in (ROOT / args.index).open(encoding="utf-8")]
    tms = papygreek_tms(ROOT / args.papygreek)

    by_tm = {r["tm"]: r for r in rows}
    matched = [r for t, r in by_tm.items() if t in tms]
    vols = {(r["series"], r["volume"]) for r in matched}
    vols.discard(("", ""))
    vol_docs = [r for r in rows if (r["series"], r["volume"]) in vols]

    def cost(subset):
        return {"n_docs": len(subset),
                "n_tokens": sum(r["n_tokens"] for r in subset),
                "n_pairs": sum(r["n_pairs_raw"] for r in subset)}

    total = cost(rows)
    res = {"corpus_total": total,
           "papygreek_tms": len(tms),
           "papygreek_tms_found_in_ddb": len(matched),
           "policy_tm": cost(matched),
           "policy_series_volume": cost(vol_docs),
           "series_volume_pairs": sorted(f"{s};{v}" for s, v in vols)}
    for p in ("policy_tm", "policy_series_volume"):
        res[p]["share_docs"] = (res[p]["n_docs"] / total["n_docs"]
                                if total["n_docs"] else 0.0)
        res[p]["share_tokens"] = (res[p]["n_tokens"] / total["n_tokens"]
                                  if total["n_tokens"] else 0.0)

    print(f"corpus: {total['n_docs']} docs, {total['n_tokens']} tokens, "
          f"{total['n_pairs']} raw pairs")
    print(f"PapyGreek TMs found in DDbDP: {len(matched)}/{len(tms)}")
    for p in ("policy_tm", "policy_series_volume"):
        d = res[p]
        print(f"  {p:22s} excludes {d['n_docs']:6d} docs "
              f"({d['share_docs']:5.1%})  {d['n_tokens']:9d} tokens "
              f"({d['share_tokens']:5.1%})")
    print(f"  series;volume set: {len(vols)} entries")
    (ROOT / args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    print(f"wrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = dict(src="idp.data/DDbDP", index="ddb_index.jsonl",
                  papygreek="papygreek_ids.json")

    s = sub.add_parser("scan")
    s.add_argument("--src", default=common["src"])
    s.add_argument("--out", default="ddb_index.jsonl")
    s.add_argument("--limit", type=int, default=0)
    s.set_defaults(fn=cmd_scan)

    t = sub.add_parser("text")
    t.add_argument("--src", default=common["src"])
    t.add_argument("--out", default="ddb_text")
    t.add_argument("--index", default=common["index"])
    t.add_argument("--papygreek", default=common["papygreek"])
    t.add_argument("--exclude", choices=("tm", "series"), default="series")
    t.add_argument("--limit", type=int, default=0)
    t.add_argument("--budget", type=float, default=0.0)
    t.set_defaults(fn=cmd_text)

    p = sub.add_parser("pairs")
    p.add_argument("--src", default=common["src"])
    p.add_argument("--out", default="ddb_pairs.jsonl")
    p.add_argument("--stats", default="ddb_extract_stats.json")
    p.add_argument("--index", default=common["index"])
    p.add_argument("--papygreek", default=common["papygreek"])
    p.add_argument("--exclude", choices=("tm", "series"), default="tm")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--strict-orig", action="store_true")
    p.add_argument("--include-corr", action="store_true")
    p.set_defaults(fn=cmd_pairs)

    e = sub.add_parser("exclusions")
    e.add_argument("--index", default=common["index"])
    e.add_argument("--papygreek", default=common["papygreek"])
    e.add_argument("--out", default="ddb_exclusions.json")
    e.set_defaults(fn=cmd_exclusions)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
