"""Search structures over the 4.8M accent-stripped ancient-Greek keys.

No clever index: the query set is ~2k tokens, so a length-bucketed brute-force
scan with a C-speed Levenshtein and an early-exit cutoff beats building a
deletion index that would not fit in memory. Bucketing by length is exact --
two strings within distance k cannot differ in length by more than k.
"""

import pickle
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import cdist

ROOT = Path(__file__).parent


class KeySpace:
    """A pool of keys, bucketed by length, searchable by edit distance."""

    def __init__(self, keys: list[str]):
        self.keys = np.array(sorted(set(keys)), dtype=object)
        if len(self.keys) == 0:
            raise ValueError("KeySpace needs at least one key")
        lens = np.array([len(k) for k in self.keys])
        order = np.argsort(lens, kind="stable")
        self.keys = self.keys[order]
        lens = lens[order]
        self._starts = {}
        uniq, idx = np.unique(lens, return_index=True)
        for u, i in zip(uniq.tolist(), idx.tolist()):
            self._starts[u] = i
        self._lens = lens
        self._max_len = int(lens.max())

    def _slice(self, lo: int, hi: int) -> np.ndarray:
        lo = max(lo, int(self._lens.min()))
        hi = min(hi, self._max_len)
        if lo > hi:
            return self.keys[:0]
        a = np.searchsorted(self._lens, lo, side="left")
        b = np.searchsorted(self._lens, hi, side="right")
        return self.keys[a:b]

    def near(self, query: str, max_dist: int) -> list[tuple[str, int]]:
        pool = self._slice(len(query) - max_dist, len(query) + max_dist)
        if len(pool) == 0:
            return []
        d = cdist([query], pool, scorer=Levenshtein.distance,
                  score_cutoff=max_dist, workers=-1, dtype=np.uint8)[0]
        hits = np.nonzero(d <= max_dist)[0]
        return [(pool[i], int(d[i])) for i in hits]


class PairedIndex:
    """collapse-key -> accent-stripped keys, as two sorted parallel arrays.

    A dict over four million keys costs more than a gigabyte of Python objects;
    two sorted arrays plus searchsorted cost a fraction of that and are fast
    enough for a two-thousand-token query set.
    """

    def __init__(self, coarse: list[str], fine: list[str]):
        c = np.array(coarse, dtype=object)
        f = np.array(fine, dtype=object)
        order = np.argsort(c, kind="stable")
        self.coarse = c[order]
        self.fine = f[order]

    def get(self, key: str) -> list[str]:
        a = np.searchsorted(self.coarse, key, side="left")
        b = np.searchsorted(self.coarse, key, side="right")
        return [str(x) for x in self.fine[a:b]]

    def distinct_coarse(self) -> list[str]:
        return [str(x) for x in np.unique(self.coarse)]


def cached(name: str, build):
    p = ROOT / name
    if p.exists():
        with p.open("rb") as fh:
            return pickle.load(fh)
    obj = build()
    with p.open("wb") as fh:
        pickle.dump(obj, fh, protocol=5)
    return obj
