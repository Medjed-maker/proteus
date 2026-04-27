# Phonology Codemap

**Last Updated:** 2026-04-11  
**Entry Points:** `search.py` (three-stage pipeline), `ipa_converter.py`, `distance.py`

## Architecture

```
USER INPUT (Greek text)
    ↓
[ipa_converter.py: to_ipa(), greek_to_ipa(), tokenize_ipa()]
    ↓ (converts to IPA phones)
[search.py: normalize_query_for_search()]
    ↓
┌───────────────────────────────────────────────────────────┐
│  THREE-STAGE SEARCH PIPELINE                              │
├───────────────────────────────────────────────────────────┤
│ 1. seed_stage()     — K-mer index lookup, initial candidates
│    ↓ (uses KmerIndex from build_kmer_index())
│ 2. extend_stage()   — Distance scoring, dialect attribution
│    ↓ (uses distance.py: phonological_distance(), matrix)
│ 3. filter_stage()   — Confidence filtering, ranking
│    ↓ (uses explainer.py for rule-matching)
│ SEARCH RESULTS (SearchResult list)
└───────────────────────────────────────────────────────────┘
    ↓
[API response mapping in api/main.py]
    ↓
JSON SearchResponse (hits with metadata, alignments, applied rules)
```

## Key Modules

| Module | Purpose | Public Functions/Classes | Dependencies (within phonology/) |
|--------|---------|--------------------------|----------------------------------|
| **_paths.py** | Locate data directories by walking up from `pyproject.toml` | `resolve_repo_data_dir()` | None |
| **_phones.py** | IPA phone inventory constants | `VOWEL_PHONES` (frozenset) | None |
| **ipa_converter.py** | Greek (Unicode polytonic/monotonic) → IPA conversion. Handles diphthongs, diacritics, rough breathing, diaeresis, Koine consonant shifts. | `greek_to_ipa()`, `tokenize_ipa()`, `to_ipa()`, `apply_koine_consonant_shifts()`, `strip_diacritics()`, `get_known_phones()` | `_phones.py` (VOWEL_PHONES) |
| **distance.py** | Weighted edit distance (Needleman-Wunsch) with raw & normalized scoring modes. Matrix loading with TOCTOU-safe symlink checks. | `load_matrix()`, `phone_distance()`, `phonological_distance()`, `normalized_phonological_distance()`, `word_distance()`, `normalized_word_distance()` | `_paths.py`, `ipa_converter.py` |
| **explainer.py** | Load YAML rules from `data/rules/`, match against alignments, generate structured rule-application records. | `load_rules()`, `explain()`, `explain_alignment()`, `to_prose()`, `tokenize_rules_for_matching()`, `RuleApplication`, `Alignment` | `_paths.py`, `_phones.py`, `ipa_converter.py` |
| **search.py** | Three-stage BLAST-like search pipeline with k-mer indexing, distance-scoring, and filtering. | `seed_stage()`, `extend_stage()`, `filter_stage()`, `search()`, `build_kmer_index()`, `build_lexicon_map()`, `classify_query_mode()`, `normalize_query_for_search()`, `SearchResult` | `_phones.py`, `distance.py`, `explainer.py`, `ipa_converter.py` |
| **matrix_generator.py** | Generate/validate canonical Attic-Doric phone distance matrix. Enforces symmetry, completeness, [0.0, 1.0] bounds. | `generate()` (main generation function, run via `python -m phonology.matrix_generator`) | `_paths.py` |
| **betacode.py** | TLG/Perseus Beta Code (ASCII Greek) → Unicode Greek | `beta_to_unicode()` | None (used by lsj_extractor, build_lexicon) |
| **transliterate.py** | Greek Unicode → scholarly Latin transliteration | `transliterate()` | None (used by lsj_extractor, build_lexicon) |
| **lsj_extractor.py** | Parse Perseus LSJ XML, extract headwords/glosses, write JSON lexicon | `extract_entries()` (main function, run as `python -m phonology.lsj_extractor`) | `_paths.py`, `betacode.py`, `ipa_converter.py`, `transliterate.py` |
| **build_lexicon.py** | Clone LSJ repo, run lsj_extractor, compute fingerprint, validate output | `build_lexicon_if_missing()` (main function, run as `python -m phonology.build_lexicon`) | `_paths.py`, `lsj_extractor.py` |
| **buck.py** | Read-only loader for Buck-normalized reference data (grammar rules, dialects, glossary) | `load_buck_data()`, `BuckData` TypedDict | `_paths.py` |

## Data Flow: Full-Form Search Example

```
Query: "λόγος" (logos, "word")
   ↓ greek_to_ipa()
"l ɔː ɡ o s" (IPA phone list)
   ↓ normalize_query_for_search()
"logos" (normalized form)
   ↓ to_ipa(normalized_form, dialect="attic")
query_ipa = "lɔːɡos" (compact form, used for distance calc)
   ↓ classify_query_mode(query_form)
mode = "Full-form"
   ↓ search() pipeline:
   
   SEED STAGE:
   - build_kmer_index(lexicon, k=2)  → {"lɔ": ["logos"], "ɔː": [...], ...}
   - Look up k-mers of query in index → candidates
   ↓
   EXTEND STAGE:
   - For each candidate in lexicon:
     - candidate_ipa = tokenize_ipa(candidate["ipa"])
     - raw_dist = phonological_distance(query_tokens, candidate_ipa, matrix)
     - norm_conf = 1.0 - normalized_phonological_distance(...)
     - Assign dialect_attribution via rule-matching
   ↓
   FILTER STAGE:
   - Sort by confidence descending
   - Truncate to max_results
   - Return SearchResult list
   
   ↓ Each SearchResult includes:
   - lemma, confidence, applied_rules, rule_applications
   - alignment_visualization, dialect_attribution, entry_id
```

## Search Fallback Migration Note

`search()` now bounds fallback exploration by default: omitted
`similarity_fallback_limit` and `unigram_fallback_limit` use
`_DEFAULT_FALLBACK_CANDIDATE_LIMIT` (`2000`). This limits Full-form token-count
fallback and k=1 unigram fallback work unless callers opt out.

Callers that require the previous unlimited fallback behavior must pass
`similarity_fallback_limit=None` and/or `unigram_fallback_limit=None`
explicitly.

## External Dependencies

- **yaml** (PyYAML) — Parse rules files in `data/languages/ancient_greek/rules/`
- **pathlib** (stdlib) — Path manipulation
- **unicodedata** (stdlib) — Unicode normalization for Greek combining marks, diacritics
- **json** (stdlib) — Load lexicon, matrices
- **os, stat** (stdlib) — Secure matrix file loading (TOCTOU checks)
- **functools** (stdlib) — LRU caching for rule loading and matrix access

## Related Areas

- **[api.md](./api.md)** — Integrates all phonology modules; exposes via FastAPI
- **[CLAUDE.md](../CLAUDE.md)** — Architectural decisions (dual data resolution, raw vs normalized distance, matrix flattening)
- **Data files:** `data/languages/ancient_greek/matrices/`, `data/languages/ancient_greek/rules/`, `data/languages/ancient_greek/lexicon/`
