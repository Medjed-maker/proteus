"""Document-level dev/test split for the orig->reg normalisation benchmark.

Same hash-rank scheme as rlb_splits.py, new seed, keyed on Trismegistos number
instead of filename. splits.json is untouched: this is a different benchmark on
a different corpus, and mixing the two seeds would make neither reproducible.

Stratified by publication series, because that is the edition tradition and
scribal-practice proxy that actually drives spelling: an unstratified 30% draw
over 30,000 documents can leave a whole series on one side. Volume, HGV id and
century are recorded in the split file for post-hoc balance checking but are
*not* split constraints -- the property that makes rlb_splits.py trustworthy is
that split membership is a pure function of the document id, and every extra
constraint erodes it.
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent

DEV_FRACTION = 0.30
SEED = "proteus-ddb-orig2reg-2026-08-12"
TOP_SERIES = 20


def _doc_rank(tm: str) -> float:
    h = hashlib.sha256((SEED + tm).encode()).hexdigest()
    return int(h[:16], 16) / float(1 << 64)


def build(pairs_path: Path, dev_fraction: float = DEV_FRACTION) -> dict:
    rows = [json.loads(line) for line in pairs_path.open(encoding="utf-8")]
    per_doc = Counter(r["doc"] for r in rows)
    series_of = {r["doc"]: r["series"] for r in rows}

    # Series outside the top N are pooled: a stratum of three documents cannot
    # be split 70/30 in any meaningful sense.
    common = {s for s, _ in Counter(series_of.values()).most_common(TOP_SERIES)}

    def stratum(doc: str) -> str:
        s = series_of[doc]
        return s if s in common else "other"

    by_stratum: dict[str, list[str]] = defaultdict(list)
    for doc in per_doc:
        by_stratum[stratum(doc)].append(doc)

    dev: set[str] = set()
    detail = {}
    for name, docs in by_stratum.items():
        docs = sorted(docs, key=_doc_rank)
        target = dev_fraction * sum(per_doc[d] for d in docs)
        acc = 0
        picked = []
        for d in docs:
            if acc >= target:
                break
            picked.append(d)
            acc += per_doc[d]
        dev.update(picked)
        detail[name] = {
            "n_docs": len(docs), "n_dev_docs": len(picked),
            "n_pairs": sum(per_doc[d] for d in docs), "n_dev_pairs": acc,
        }

    n_pairs = sum(per_doc.values())
    n_dev = sum(per_doc[d] for d in dev)
    return {
        "seed": SEED,
        "dev_fraction": dev_fraction,
        "stratified_by": "series",
        "top_series": sorted(common),
        "dev_docs": sorted(dev),
        "n_docs_total": len(per_doc),
        "n_dev_docs": len(dev),
        "n_pairs_total": n_pairs,
        "n_dev_pairs": n_dev,
        "n_test_pairs": n_pairs - n_dev,
        "per_stratum": detail,
    }


def dev_docs(path: Path | None = None) -> set:
    p = path or ROOT / "ddb_splits.json"
    return set(json.loads(p.read_text(encoding="utf-8"))["dev_docs"])


def tag(rows: list[dict], dev: set) -> list[dict]:
    for r in rows:
        r["split"] = "dev" if r["doc"] in dev else "test"
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="ddb_pairs.jsonl")
    ap.add_argument("--out", default="ddb_splits.json")
    ap.add_argument("--dev-fraction", type=float, default=DEV_FRACTION)
    args = ap.parse_args()

    res = build(ROOT / args.pairs, args.dev_fraction)
    (ROOT / args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    total = res["n_pairs_total"]
    dev_share = res["n_dev_pairs"] / total if total else 0.0
    print(f"documents {res['n_dev_docs']}/{res['n_docs_total']} dev")
    print(f"pairs     {res['n_dev_pairs']}/{total} dev "
          f"({dev_share:.1%}), "
          f"{res['n_test_pairs']} test")
    print(f"{'stratum':<16}{'docs':>7}{'dev':>7}{'pairs':>9}{'dev':>9}{'%':>7}")
    for name, d in sorted(res["per_stratum"].items(),
                          key=lambda kv: -kv[1]["n_pairs"]):
        share = d["n_dev_pairs"] / d["n_pairs"] if d["n_pairs"] else 0
        print(f"{name:<16}{d['n_docs']:>7}{d['n_dev_docs']:>7}"
              f"{d['n_pairs']:>9}{d['n_dev_pairs']:>9}{share:>7.1%}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
