"""Merge chunked ladder runs into one result file.

B2 searches 4.8M keys at edit distance 2 and takes about twenty minutes over
the full token set, so it is run in slices; this stitches the slices back
together and recomputes the aggregates from the per-token detail rather than
from the per-slice counters.
"""

import json
import sys
from collections import Counter
from pathlib import Path

from rlb_ladder import report

ROOT = Path(__file__).parent


def merge(paths: list[Path], out: Path) -> dict:
    """Merge disjoint ladder result slices and recompute their aggregates.

    Args:
        paths: JSON result slices from one ladder stage.
        out: Destination for the single merged result.

    Returns:
        The merged stage result with per-occurrence detail and aggregates.
    """
    detail, seconds = [], 0.0
    stage = None
    provenance = None
    provenance_seen = False
    for p in paths:
        for res in json.loads(p.read_text()):
            stage = stage or res["stage"]
            if res["stage"] != stage:
                sys.exit(f"{p} holds stage {res['stage']}, expected {stage}")
            detail.extend(res["detail"])
            seconds += res["seconds"]
            candidate_provenance = res.get("provenance")
            if not provenance_seen:
                provenance = candidate_provenance
                provenance_seen = True
            elif candidate_provenance != provenance:
                raise ValueError(f"{p} has different benchmark provenance")

    seen = set()
    for d in detail:
        try:
            key = (d["doc"], d["sent"], d["wid"])
        except KeyError as exc:
            raise ValueError(
                f"detail entry lacks occurrence identifier {exc.args[0]!r}") \
                from exc
        if key in seen:
            raise ValueError(f"duplicate detail occurrence: {key!r}")
        seen.add(key)

    # The same spelling can recur across papyri; each distinct occurrence
    # remains a separate token in the denominator.
    uniq = detail

    per = {sp: Counter() for sp in ("dev", "test", "all")}
    sizes = []
    for d in uniq:
        sizes.append(d["n_cand"])
        for sp in (d["split"], "all"):
            c = per[sp]
            c["n"] += 1
            if not d["n_cand"]:
                c["empty"] += 1
            if d["rank"] is not None:
                c["lenient@inf"] += 1
                for k in (1, 5, 10):
                    if d["rank"] < k:
                        c[f"lenient@{k}"] += 1

    sizes.sort()
    if not sizes:
        raise ValueError("cannot merge empty detail input")
    res = {"stage": stage, "seconds": round(seconds, 1),
           "cand_median": sizes[len(sizes) // 2],
           "cand_p90": sizes[int(len(sizes) * 0.9)],
           "cand_max": sizes[-1],
           "per_split": {k: dict(v) for k, v in per.items()},
           "detail": uniq}
    if provenance is not None:
        res["provenance"] = provenance
    out.write_text(json.dumps([res], ensure_ascii=False, indent=2),
                   encoding="utf-8")
    return res


if __name__ == "__main__":
    out = Path(sys.argv[1])
    res = merge([Path(p) for p in sys.argv[2:]], out)
    report(res)
    print(f"\nwritten to {out}")
