"""Freeze a document-level dev/test split of the UVP zone.

Hand-writing fifteen alternations while looking at all 2,160 variant tokens
would guarantee a good number and destroy its meaning. The split is by
*document*, not by token, because spellings cluster by scribe: two tokens from
the same papyrus are not independent evidence.

Written once to splits.json and read from there afterwards; the assignment is
deterministic given the dataset, so it can be regenerated but not drifted.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

from run_eval import load

ROOT = Path(__file__).parent
DEV_FRACTION = 0.30
SEED = "proteus-b0-b3-2026-08-11"


def _doc_rank(doc: str) -> float:
    h = hashlib.sha256((SEED + doc).encode()).hexdigest()
    return int(h[:16], 16) / float(1 << 64)


def build() -> dict[str, object]:
    """UVP ゾーンの文書単位 dev/test 分割を生成し、splits.json に書き出す。

    Returns:
        seed、dev_fraction、dev_docs、文書数、トークン数を含む分割記述。
    """
    rows = [r for r in load(ROOT / "dataset.jsonl")
            if r["stratum"] == "variant_ortho"]
    per_doc = Counter(r["doc"] for r in rows)
    docs = sorted(per_doc, key=_doc_rank)

    target = DEV_FRACTION * sum(per_doc.values())
    dev, acc = set(), 0
    for d in docs:
        if acc >= target:
            break
        dev.add(d)
        acc += per_doc[d]

    split = {
        "seed": SEED,
        "dev_fraction": DEV_FRACTION,
        "dev_docs": sorted(dev),
        "n_docs_total": len(per_doc),
        "n_dev_tokens": acc,
        "n_test_tokens": sum(per_doc.values()) - acc,
    }
    (ROOT / "splits.json").write_text(
        json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")
    return split


def dev_docs() -> set[str]:
    text = (ROOT / "splits.json").read_text(encoding="utf-8")
    return set(json.loads(text)["dev_docs"])


def tag(rows: list[dict]) -> list[dict]:
    dev = dev_docs()
    for r in rows:
        r["split"] = "dev" if r["doc"] in dev else "test"
    return rows


if __name__ == "__main__":
    s = build()
    print(f"docs: {len(s['dev_docs'])}/{s['n_docs_total']} in dev")
    print(f"tokens: dev {s['n_dev_tokens']}  test {s['n_test_tokens']}")
