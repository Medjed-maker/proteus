"""The index side of the ladder.

Every rung searches the *same* lexicon Dilemma searches: lookup.db (9.7M
form->lemma entries) and its companion spell_index.db (6.1M accent-stripped
keys). Holding the dictionary fixed is the whole point -- it removes "they just
have more data" as an explanation for any gap that shows up.

Two deliberate differences from Dilemma's own use of these tables:

  * Dilemma's LookupDB.get() ends in ``LIMIT 1``: one lemma per form. Here a
    form maps to the full set, because returning a ranked candidate set rather
    than a single guess is the thing being evaluated.
  * Fuzzy stages search only entries with a ``grc`` source. Matching a papyrus
    spelling against Modern Greek orthography is not a linguistic claim anyone
    would defend, and it is a strict subset, so it can only be conservative.
"""

import json
import os
import sqlite3
import unicodedata
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

DATA = Path(os.environ.get(
    "DILEMMA_DATA_DIR", Path.home() / ".cache/dilemma/data"))
LOOKUP_DB = DATA / "lookup.db"
SPELL_DB = DATA / "spell_index.db"

if not LOOKUP_DB.exists():  # fail here rather than three frames deep in sqlite
    raise FileNotFoundError(
        f"no lookup.db under {DATA}. Run `python -m dilemma download`, "
        f"or set DILEMMA_DATA_DIR to the directory holding it.")


class Lexicon:
    """Read-only access to Dilemma's lookup and spelling databases."""

    def __init__(self, data_dir: Path = DATA) -> None:
        self._lk = sqlite3.connect(
            f"file:{quote(str(data_dir / 'lookup.db'))}?mode=ro", uri=True)
        self._sp = sqlite3.connect(
            f"file:{quote(str(data_dir / 'spell_index.db'))}?mode=ro", uri=True)
        for c in (self._lk, self._sp):
            c.execute("PRAGMA mmap_size=1073741824")
        self._data_dir = data_dir

    # -- exact form -> lemma ------------------------------------------------

    @lru_cache(maxsize=200_000)
    def lemmas(self, form: str) -> tuple[str, ...]:
        """Every lemma the dictionary records for this exact spelling.

        ``lang='grc'`` holds ancient-Greek-specific overrides and ``lang='all'``
        the shared bulk; Dilemma consults the override first and falls through,
        so both are taken here and the override is put first.
        """
        rows = self._lk.execute(
            "SELECT l.text, k.lang FROM lookup k JOIN lemmas l "
            "ON k.lemma_id = l.id WHERE k.form = ? AND k.lang IN ('grc','all')",
            (form,)).fetchall()
        grc = [t for t, lang in rows if lang == "grc"]
        allx = [t for t, lang in rows if lang != "grc"]
        seen, out = set(), []
        for t in grc + allx:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return tuple(out)

    def has_lemma(self, lemma: str) -> bool:
        return self._lk.execute(
            "SELECT 1 FROM lemmas WHERE text = ? LIMIT 1",
            (lemma,)).fetchone() is not None

    # -- accent-stripped key -> forms --------------------------------------

    @lru_cache(maxsize=200_000)
    def forms_for_key(self, key: str, grc_only: bool = True
                      ) -> tuple[tuple[str, str], ...]:
        """(form, src) pairs sharing an accent-stripped key."""
        row = self._sp.execute(
            "SELECT forms FROM spell WHERE stripped = ?", (key,)).fetchone()
        if not row:
            return ()
        out = []
        for line in row[0].split("\n"):
            if "\t" not in line:
                continue
            form, src = line.split("\t", 1)
            if grc_only and src != "grc":
                continue
            out.append((form, src))
        return tuple(out)

    def lemmas_for_key(self, key: str, grc_only: bool = True) -> list[str]:
        seen, out = set(), []
        for form, _ in self.forms_for_key(key, grc_only):
            for lem in self.lemmas(form):
                if lem not in seen:
                    seen.add(lem)
                    out.append(lem)
        return out

    # -- the key universe for fuzzy search ---------------------------------

    def grc_keys(self, cache: Path | None = None) -> list[str]:
        """Every accent-stripped key with at least one ancient-Greek form.

        4.8M strings; the scan takes ~90s, so it is cached to disk.
        """
        cache = cache or (Path(__file__).parent / "grc_keys.txt")
        if cache.exists():
            return _read_key_cache(cache)
        keys = []
        for stripped, forms in self._sp.execute(
                "SELECT stripped, forms FROM spell"):
            if "\tgrc" not in forms:
                continue
            if not _is_greek(stripped):
                continue
            keys.append(stripped)
        _write_key_cache(cache, keys)
        return keys


def _read_key_cache(cache: Path) -> list[str]:
    """Read newline-delimited keys, ignoring blank cache entries."""
    return [line for line in cache.read_text(encoding="utf-8").splitlines()
            if line]


def _write_key_cache(cache: Path, keys: list[str]) -> None:
    """Write one key per line with a final newline when keys are present."""
    cache.write_text("\n".join(keys) + ("\n" if keys else ""), encoding="utf-8")


def _is_greek(s: str) -> bool:
    return bool(s) and all(
        "Ͱ" <= ch <= "Ͽ" or "ἀ" <= ch <= "῿" for ch in s)


# --------------------------------------------------------------------------
# Frequency, for ranking. Both sources are external to PapyGreek, so nothing
# about the evaluation set leaks into the ordering of candidates.
# --------------------------------------------------------------------------


def form_freq(data_dir: Path = DATA) -> dict[str, int]:
    """Accent-stripped form -> token count over 68M words of Greek."""
    d = json.load(open(data_dir / "corpus_freq.json", encoding="utf-8"))
    return {k: v[0] for k, v in d["forms"].items()}


def lemma_freq(data_dir: Path = DATA) -> dict[str, int]:
    d = json.load(open(data_dir / "lemma_attestation.json", encoding="utf-8"))
    return {k: v.get("total", 0) for k, v in d["lemmas"].items()}


def strip_for_freq(s: str) -> str:
    """corpus_freq keys are accent-stripped lowercase with medial sigma."""
    nfd = unicodedata.normalize("NFD", s)
    base = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", base).lower().replace("ς", "σ")
