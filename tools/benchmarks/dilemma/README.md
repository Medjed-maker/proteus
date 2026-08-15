# Dilemma / PapyGreek benchmarks

Two measurements live here.

| Script set | Question | Write-up |
|---|---|---|
| `build_dataset.py`, `run_eval.py`, `run_candidates.py`, `categorize.py` | How much of the reverse-lookup gap does the strongest existing tool leave open? | [dilemma_papygreek_baseline.md](../../../docs/benchmarks/dilemma_papygreek_baseline.md) |
| `rlb_*.py` (B0–B3) | Is a phonological distance matrix necessary to close it? | [reverse_lookup_baseline_ladder.md](../../../docs/benchmarks/reverse_lookup_baseline_ladder.md) |
| `rlb_rerank.py`, `rlb_postag.py` | The misses are mostly ranking failures — what is that worth? | same, §5.1 |
| `rlb_stats.py` | Do the claimed differences survive document-level clustering? | [reranking_and_statistics.md](../../../docs/benchmarks/reranking_and_statistics.md) §1 |
| `rlb_altprob.py` | Does per-alternation probability break ties the integer cost leaves? | same, §2 (rejected) |
| `rlb_leakcheck.py`, `rlb_lm.py` | Does corpus context help, and is the corpus clean? | same, §3 (adopted, +4.2pt) |
| `rlb_propnoun.py` | Is the bucket the ranker buried mostly proper nouns? | `preregistration_3rd.md` H1 (rejected, and inverted) |
| `rlb_zones.py` | Which candidates did the fifteen rules earn, and which did blind distance find? | same, H4 |
| `rlb_ddb*.py` | Same question on documentary papyri, CC BY only | same, H2 and H3 |

The second answers no, and that answer is what redirects Proteus's investment
from Layer 1 (distance and alignment) to Layer 2 (sourced rules, re-ranking,
explanation). The third puts intervals on all of it, and finds the one ranking
signal that actually pays: a lemma trigram model over literary Greek.

## Setup

Neither dataset is committed. Both are regenerated from their sources.

```bash
python -m venv .venv
.venv/bin/pip install numpy rapidfuzz
.venv/bin/pip install \
  "dilemma-nlp[onnx] @ git+https://github.com/open-greek/dilemma.git@f82f15a62ddce5d55c19b299c34a6c89476af5ce"
.venv/bin/python -m dilemma download    # 3.7GB, ~90 min

curl -sL "https://zenodo.org/api/records/5074307/files/ezhenrik/papygreek-treebanks-v1.01.zip/content" \
  -o papygreek.zip && unzip -q papygreek.zip -d papygreek
.venv/bin/python build_dataset.py       # -> dataset.jsonl, 33,615 tokens
```

`rlb_lexicon.py` reads the Dilemma data directory read-only; point `DATA` at it
if `python -m dilemma download` put it somewhere other than
`~/.cache/dilemma/data`.

See `docs/benchmarks/reverse_lookup_baseline_ladder.md` §9 for the run order,
and `DATA_LICENSE.md` §5.1 for the attribution both sources require.

## Notes for whoever runs this next

* `rlb_build.py` is resumable on purpose — the 4.8M-key collapse takes about
  three minutes and should not be able to take a measurement down with it.
  Rerun until it prints `complete`.
* `B2u` and `B3u` take roughly 20 minutes each over the full token set. Run
  them in disjoint `--offset/--limit` slices and stitch with `rlb_merge.py`.
  `rlb_merge.py` recomputes lenient recall only; strict is available from
  unsliced runs.
* `rlb_rerank.py` replays a complete candidate dump
  (`rlb_ladder.py --dump`) rather than re-searching, so a new ranking idea costs
  seconds instead of twenty minutes. Dumps generated before 2026-08-14 were
  capped at 500 candidates and must not be used for new measurements. Select
  variants on `--split dev`; score `--split test` once.
* `rlb_postag.py` is the only script here that needs `dilemma` itself importable
  (it calls the ONNX POS head). It writes `postags.jsonl` once; `rlb_rerank.py`
  reads that file and needs nothing but numpy.
* `rlb_stats.py` is the only place effect sizes should come from. After the
  markup fix, the test split has 1,484 searchable tokens but 222 independent
  documents, and several of the
  differences that look decisive at recall@5 reverse sign at recall@1 or @20.
  Quote an interval, and say which k.
* `rlb_lm.py fetch` streams GLAUx from raw.githubusercontent and keeps only
  lemma sequences; rerun until the corpus is big enough. GLAUx is NFD and
  everything else here is NFC — `nfc()` is not optional, without it lemma
  coverage reads 0.4% instead of 84.5%.
* `splits.json` is frozen. The fifteen alternations in `rlb_keys.py` were
  frozen before the test split was ever scored, and dev/test agree to within
  1.8 points — do not tune them against a test number.
* `rlb_lm.py rerank` writes `results_lm_{split}_{tag}.json`. Pass `--tag`, or
  each corpus condition silently overwrites the previous one. The file once
  committed here as `results_lm_dev_fullcorpus.json` held `lm_tokens:
  1,167,825` — the 30-text corpus, not the 1,027-text one — so it has been
  renamed `results_lm_dev_glaux30.json` to match its contents. The full-corpus
  run itself did happen: its token count in `reranking_and_statistics.md` §3.4
  (14,393,438) matches `glaux_lemmas_full/` exactly. Only the artefact was
  lost, and the `--tag` flag is why it cannot happen again.
* `rlb_lm.py rerank --group N` joins exactly N consecutive non-empty input
  lines; blank lines remain hard corpus breaks. `--window` filters only the
  first `--lemma-dir` (the GLAUx directory) and fails if no TLG filename
  matches, while later directories such as DDbDP remain unfiltered.
* `rlb_downsample.py` requires a positive `--tokens` value or a non-empty
  `--match` corpus. Source and output must be disjoint. Re-running replaces
  only generated `.txt` symlinks and refuses to delete regular files.
* `rlb_zones.py` reads the zone off `src_key`, not off the cost. `_from_keys`
  keeps the minimum tuple per lemma, so a candidate reached both by a rule and
  by blind Levenshtein-1 records the blind cost; judging by cost alone
  mislabels 97,527 of 527,305 candidate slots.
* `clean.py`'s `MARKUP_CHARS` used to miss `⎴⎵ ⸜⸝ ⦑⦒ ⧼` and U+E1C0, letting
  those reach the search (2.4% of dev, 0.6% of test tokens; six generated zero
  candidates for perfectly ordinary words). **Fixed 2026-08-13**, covered by
  `test_clean.py`. Search, re-ranking, statistics, and current LM artefacts were
  regenerated on 2,143 searchable tokens (dev 659 / test 1,484). The retained
  `results_lm_dev_glaux30.json` is explicitly historical: its source corpus is
  no longer local, so it still records the pre-fix 672-token dev run.
  Actual deltas landed under the pre-registered upper bound (dev +1.5pt,
  test +0.3pt): dev R0 recall@5 74.6%→75.3% (+0.71pt), test 72.8%→73.0%
  (+0.13pt). No conclusion in either write-up changed sign or crossed a
  decision threshold — see `preregistration_3rd.md`'s "対応済み" note for the
  full before/after table.
* `rlb_ladder.py` used to stop rank scanning at the first lenient-equivalent
  lemma, which undercounted a later exact strict match. **Fixed 2026-08-13** by
  sharing the independent rank scan with the DDbDP runner. Recomputed @5:
  B3a 52.4%, B3b 58.4%, B3 58.9%; lenient results are unchanged.
* The 2026-08-12 markup fix was found by eye, so on **2026-08-14** the check was
  inverted: `clean.residue()` reports every character *outside* the expected
  Greek repertoire, and `run_eval.load()` — the one function every `rlb_*`
  script goes through — refuses to run if any survives. Sweeping all 33,615
  tokens with it found four more marks, each an ASCII or superscript spelling of
  one already stripped: `…` U+2026 (5 tokens), `⁽⁾` U+207D/207E (3), `-` U+002D
  (3), `~` U+007E (1). Plus one non-markup case: `κυρίoυ` carries U+006F LATIN
  SMALL LETTER O, a source typo, now repaired to omicron via `CONFUSABLES`
  rather than deleted. Residue over the corpus is now empty.

  **No published number changes.** The strata come out identical before and
  after (clean 30,148 / variant_ortho 2,143 / abbrev 1,179 / variant_lex 145;
  zero tokens moved between strata). Of the 12 changed search inputs,
  `variant_ortho` gets **none**, so the ladder, re-ranking and statistics
  results stand as written. One
  `variant_ortho` token changes on the `input_reg` side only (the Latin-o
  repair), which reaches B0's reverse lookup and the `R5rg` variant: 1 of 2,143,
  ≤0.05pt. Verify with:

  ```bash
  python3 -c "
  import json,sys; sys.path.insert(0,'.')
  from clean import clean, residue
  bad=set()
  for l in open('dataset.jsonl',encoding='utf-8'):
      r=json.loads(l); bad |= residue(clean(r['form_orig'])) | residue(clean(r['form_reg']))
  print(sorted(f'U+{ord(c):04X}' for c in bad))"   # -> []
  ```
* Known, not yet investigated: PapyGreek writes the elision apostrophe with
  three different codepoints — counting occurrences across the cleaned
  form_orig and form_reg of every token, U+02BC 1,130, U+2019 12, U+0027 4. All three
  are allowed through as content, but `run_eval.norm_lenient()` does not
  normalise them, so `οὐδʼ` and `οὐδ'` do not compare equal. Whether that costs
  any recall is unmeasured.
* **Measurement validity (2026-08-14):** the committed re-ranking and LM result
  files are historical, not current measurements. Their candidate dumps were
  capped at 500, and predicted context removed unresolved tokens before padding
  with `<s>`, which changed token positions. The code now writes every candidate
  and preserves unresolved positions as `<unk>`. See
  [RESULTS_STATUS.md](RESULTS_STATUS.md) before citing an artefact; regenerate
  the affected results before making a new quantitative claim.
  DDbDP LM corpora generated before this fix also dropped unresolved training
  tokens, fabricating adjacency between their neighbours. Regenerate
  `ddb_lemmas/` as well as its downstream LM results with the pinned Dilemma
  version before treating them as current.
* These scripts are outside the repository's type-checking scope on purpose
  (see the comment above `[tool.mypy]` in `pyproject.toml`). Baseline as of
  2026-08-14: `uv run pyright tools/benchmarks/dilemma` reports 32 errors over
  34 files — 6 are `reportMissingImports` for the optional `dilemma` and
  `rapidfuzz` dependencies, the other 26 are type findings in dict-shaped code
  (9 `reportIndexIssue`, 8 `reportAttributeAccessIssue`, 7 `reportArgumentType`,
  2 `reportReturnType`). Anyone bringing this directory into scope should start
  from those numbers.

## The DDbDP benchmarks

`rlb_ddb.py` parses the Duke Databank EpiDoc corpus (CC BY 3.0, so no
share-alike obligation, unlike PapyGreek) into two things from one walk:
an in-domain lemma corpus for the context LM, and an orig→reg normalisation
benchmark of 121,116 pairs over 26,701 documents — 56× the original 2,160-token
PapyGreek benchmark, which is what takes the pressure off the current
1,484-token test split.

```bash
git clone --filter=blob:none --no-checkout --depth 1 \
    https://github.com/papyri/idp.data.git idp.data
cd idp.data && git sparse-checkout init --cone \
  && git sparse-checkout set DDbDP && git checkout master && cd ..

python rlb_ddb.py scan                          # -> ddb_index.jsonl, ~5 min
python rlb_ddb.py exclusions                    # -> ddb_exclusions.json
python rlb_ddb.py text  --exclude series        # -> ddb_text/,  ~6 min
python rlb_ddb.py pairs --exclude tm            # -> ddb_pairs.jsonl, ~4 min
python rlb_ddb_splits.py                        # -> ddb_splits.json
.venv/bin/python rlb_lm.py ddb                  # -> ddb_lemmas/, ~50 min
```

* DDbDP filenames are Trismegistos numbers, so excluding PapyGreek is a
  filename test. One treebank header carries `tm_id="3420 3420"` — the value is
  doubled at source — so the field must be split or that document leaks.
* Trismegistos archive membership is not obtainable offline, so `--exclude
  series` drops every document sharing a publication volume with a PapyGreek
  text as an archive proxy: 13.7% of documents, 16.8% of tokens.
* `<lb break="no"/>` is a word-internal break and must join with no space
  (`ἐνέγ|και` → `ἐνέγκαι`). `<gap>` is a hard break and must survive into the
  corpus as a blank line — otherwise the trigram forms counts straight across a
  lacuna, which is fabricated evidence rather than noisy evidence.
* The running text takes the `<reg>` side. Dilemma with `guess=False` abstains
  on exactly the non-standard spellings, so an orig-side corpus would be
  silently biased toward the subset that is already standard.
* 16.7% of raw pairs are Greek vocabulary in Coptic script
  (ⲇⲓⲁⲕⲟⲛⲟⲥ → διάκονος). That is transliteration, not orthographic variation,
  and it is counted under its own key rather than as generic noise.
