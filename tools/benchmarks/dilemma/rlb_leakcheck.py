"""Is the evaluation set inside the language model's training corpus?

Secondary sources describe PapyGreek as "incorporated into GLAUx", which would
mean an n-gram LM trained on GLAUx is trained on the very documents it is
evaluated on. This checks that claim against the corpora themselves rather than
against the descriptions.

Result: it does not hold for the published corpus. GLAUx's metadata.txt lists
1,421 texts, 19.5M tokens, and not one papyrological or documentary genre; its
raw texts come from Perseus, First1KGreek and Wikisource, and its treebank
layers from AGDT, PROIEL, Pedalion, Gorman, Harrington and Aphthonius -- no
PapyGreek. The 37 apparent TM-id matches are a namespace collision, not an
overlap (see ``glaux_overlap``).

So there is no leak, and equally no reprieve: GLAUx carries no documentary
material either, so a model trained on it meets papyrus vocabulary cold. The
risk was never contamination; it is domain mismatch.
"""

import argparse
import json
import re
from pathlib import Path

from rlb_splits import dev_docs

ROOT = Path(__file__).parent
PAPYGREEK = ROOT / "papygreek"

META = re.compile(
    r'<document_meta\b[^>]*\bname="(?P<name>[^"]*)"[^>]*', re.S)
ATTR = re.compile(r'(\w+)="([^"]*)"')


def _normalise_tm_ids(value: str) -> list[str]:
    """Split a TM identifier field and remove duplicates in source order."""
    return list(dict.fromkeys(value.split()))


def papygreek_docs() -> list[dict]:
    """One record per PapyGreek treebank file, with its Trismegistos id."""
    if not PAPYGREEK.is_dir():
        raise FileNotFoundError(
            f"PapyGreek source directory does not exist: {PAPYGREEK}")
    out = []
    for path in sorted(PAPYGREEK.rglob("*.xml")):
        head = path.read_text(encoding="utf-8", errors="replace")[:8000]
        m = re.search(r"<document_meta\b([^>]*)>", head)
        if not m:
            continue
        attrs = dict(ATTR.findall(m.group(1)))
        out.append({
            "file": attrs.get("name") or path.name,
            "tm_id": _normalise_tm_ids(attrs.get("tm_id", "")),
            "hgv_id": attrs.get("hgv_id", ""),
            "series": attrs.get("series_name", ""),
            "series_type": attrs.get("series_type", ""),
            "date_not_before": attrs.get("date_not_before", ""),
            "date_not_after": attrs.get("date_not_after", ""),
            "path": str(path.relative_to(PAPYGREEK)),
        })
    return out


def glaux_overlap(meta: Path, pg_docs: list[dict]) -> dict:
    """Cross-check GLAUx's TM_TEXT column against PapyGreek's tm_id.

    The raw id match is NOT the answer. Trismegistos numbers literary *works*
    and documentary *texts* in different spaces, so ids collide across the two:
    TM 705 is Demosthenes' De corona in GLAUx and a Zenon-archive papyrus in
    PapyGreek. Every apparent hit has to be inspected before it is believed,
    which is why the author/title/genre of each is printed rather than counted.
    """
    import csv
    rows = list(csv.DictReader(meta.open(encoding="utf-8"), delimiter="\t"))
    g = {r["TM_TEXT"].strip(): r for r in rows if r.get("TM_TEXT", "").strip()}
    by_tm = {tm_id: d for d in pg_docs for tm_id in d["tm_id"]}
    hits = sorted(set(g) & set(by_tm), key=int)

    genres = {}
    for r in rows:
        genres[r["GENRE_STANDARD"]] = genres.get(r["GENRE_STANDARD"], 0) + 1
    documentary = sum(v for k, v in genres.items()
                      if "papyr" in k.lower() or "document" in k.lower())

    print(f"\nGLAUx: {len(rows)} texts, "
          f"{sum(int(r['TOKENS']) for r in rows if r['TOKENS'].strip().isdigit()):,} tokens")
    print(f"  genres flagged papyrological/documentary: {documentary}")
    print(f"  raw TM id collisions with PapyGreek: {len(hits)}")
    for t in hits[:6]:
        print(f"    TM {t}: GLAUx={g[t]['AUTHOR_STANDARD']}/"
              f"{g[t]['TITLE_STANDARD'][:28]} ({g[t]['GENRE_STANDARD']})"
              f"  vs  PapyGreek={by_tm[t]['file']}")
    return {"n_texts": len(rows), "documentary_genres": documentary,
            "raw_id_collisions": len(hits),
            "collisions": [{"tm": t, "glaux": f"{g[t]['AUTHOR_STANDARD']}/{g[t]['TITLE_STANDARD']}",
                            "glaux_genre": g[t]["GENRE_STANDARD"],
                            "papygreek": by_tm[t]["file"],
                            "role": by_tm[t]["role"]} for t in hits]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="papygreek_ids.json")
    ap.add_argument("--glaux-metadata", default="",
                    help="path to GLAUx metadata.txt (from the GitHub repo)")
    args = ap.parse_args()

    docs = papygreek_docs()
    dev = dev_docs()

    # Which of these actually carry evaluated tokens, and on which side.
    from run_eval import load
    rows = [r for r in load(ROOT / "dataset.jsonl")
            if r["stratum"] == "variant_ortho"]
    evaluated = {r["doc"] for r in rows}
    for d in docs:
        if d["file"] not in evaluated:
            d["role"] = "unused"
        else:
            d["role"] = "dev" if d["file"] in dev else "test"

    by_role = {}
    for d in docs:
        by_role.setdefault(d["role"], []).append(d)

    print(f"PapyGreek treebank files: {len(docs)}")
    for role in ("test", "dev", "unused"):
        sub = by_role.get(role, [])
        with_tm = sum(1 for d in sub if d["tm_id"])
        print(f"  {role:<7} {len(sub):>4} files, {with_tm:>4} with a TM id")
    missing = [d["file"] for d in docs if not d["tm_id"]]
    if missing:
        print(f"  no TM id: {missing[:8]}{' ...' if len(missing) > 8 else ''}")

    types = {}
    for d in docs:
        types[d["series_type"]] = types.get(d["series_type"], 0) + 1
    print(f"  series_type: {types}")

    glaux = None
    if args.glaux_metadata:
        glaux = glaux_overlap(Path(args.glaux_metadata), docs)

    (ROOT / args.out).write_text(
        json.dumps({"docs": docs, "glaux": glaux,
                    "test_tm_ids": sorted({tm_id
                                           for d in by_role.get("test", [])
                                           for tm_id in d["tm_id"]}),
                    "all_tm_ids": sorted({tm_id for d in docs
                                          for tm_id in d["tm_id"]})},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
