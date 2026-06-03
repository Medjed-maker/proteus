# Phonology Codemap

<!-- Generated: 2026-06-03 | Files scanned: src/phonology/**/*.py (60+) -->

**Last Updated:** 2026-06-03
**Entry Points:** `search/` (three-stage pipeline), `core/ipa.py`, `distance.py`, `explainer.py`, language profiles under `languages/`

> Core-vs-language boundary is enforced by `tests/test_core_language_independence.py`:
> nothing under `src/phonology/` (excluding `languages/`) may reference a specific
> language/dialect. See `docs/MIGRATION_CORE_LANGUAGE_DECOUPLING.md`.

## Layered Layout

```
src/phonology/
├── core/                       # language-agnostic primitives + boundary contracts
│   ├── ipa.py                  # tokenize_ipa(text, *, phone_inventory=...), IPA normalization
│   └── ports/                  # "ports": contracts plugins implement/supply
│       ├── profiles.py         # LanguageProfile, IpaConverter, registry
│       ├── orthography_notes.py# OrthographicNotePayload, OrthographicNoteBuilder protocol
│       └── corpus/             # corpus source-metadata models + CorpusAdapter protocol
├── search/                     # three-stage pipeline package (seed → extend → filter)
├── explain/                    # private rule-explanation implementation
├── explainer.py                # public facade re-exporting explain/*
├── distance.py                 # weighted edit distance over IPA
├── log_odds.py                 # log-odds / likelihood-ratio over alignments
├── _paths.py / _trusted_paths.py
└── languages/ancient_greek/    # the Ancient Greek plugin (all Greek-specific logic)
```

## Three-Stage Search Pipeline

```
QUERY (text or wildcard pattern)
    ↓ profile.converter()  →  IPA
[search/_query.py: prepare_query(), classify_query_mode(), normalize_query_for_search()]
    ↓
┌──────────────────────────────────────────────────────────────┐
│ 1. seed_stage()    consonant-skeleton k-mer lookup (KmerIndex)│
│    ↓                                                          │
│ 2. extend_stage()  Smith-Waterman scoring + rule annotation   │
│    ↓               (distance.py matrix; explain_with_*_rules)  │
│ 3. filter_stage()  sort by confidence, truncate to max_results│
└──────────────────────────────────────────────────────────────┘
    ↓
list[SearchResult]  (lemma, confidence, applied_rules, alignment_visualization, …)
```

Stages and `search()` are exposed via `search/compat.py`; the package
`__init__.py` wires profile defaults (phone inventory, vowel phones, dialect
skeleton builders) at the public boundary. Query modes: `Full-form`,
`Short-query`, `Partial-form` (wildcard `*`/`?`), `Exact-form`.

## Core Modules

| Module | Purpose | Key public symbols |
|--------|---------|--------------------|
| **core/ipa.py** | Language-agnostic IPA helpers. Longest-match tokenizer driven by an injected `phone_inventory`. | `tokenize_ipa()`, `normalize_ipa_for_tokenization()`, `strip_ignored_ipa_combining_marks()`, `sorted_phone_inventory()` |
| **core/ports/profiles.py** | Plugin runtime contract + registry. Plugins self-register via the `proteus.languages` entry-point group. | `LanguageProfile` (frozen dataclass), `IpaConverter` (Protocol), `get_default_language_profile()`, `get_language_profile()`, `list_language_profiles()`, `register_language_profile()`, `register_default_profiles()` |
| **core/ports/orthography_notes.py** | Outward-facing orthographic-note payload + builder protocol. Base note kinds only; plugins add kinds. | `OrthographicNotePayload`, `OrthographicNoteBuilder` (Protocol), `OrthographicNoteKind` |
| **core/ports/corpus/** | Corpus source-metadata models and adapters. | `SourceReference`, `CorpusAdapter` (Protocol), `EmptyCorpusAdapter`, `CompositeCorpusAdapter`, `StaticCorpusAdapter`, `load_static_corpus_adapter()` |
| **distance.py** | Weighted edit distance (Needleman-Wunsch) over IPA. Raw (`phone_distance`, DEFAULT_COST=5.0) and normalized 0.0–1.0 modes. TOCTOU-safe matrix loading. | `load_matrix()`, `phone_distance()`, `phonological_distance()`, `normalized_phonological_distance()`, `word_distance()`, `normalized_word_distance()`, `register_trusted_matrices_dir()` |
| **explainer.py** | Public facade for rule-based explanation; re-exports `explain/*`. Loads YAML rules from a profile-supplied rules dir. | `load_rules()`, `explain()`, `explain_with_tokenized_rules()`, `explain_alignment()`, `tokenize_rules_for_matching()`, `to_prose()`, `get_rules_version()`, `register_trusted_rules_dir()`, `Alignment`, `RuleApplication`, `TokenizedRule`, `Explanation` |
| **log_odds.py** | Log-odds / likelihood-ratio computation over IPA alignments; builds substitution-matrix documents. | `needleman_wunsch()`, `accumulate_counts()`, `compute_log_odds()`, `build_matrix_document()`, `NWParams`, `CountTables` |
| **_paths.py** | Locate repo `data/` by walking up from `pyproject.toml`. | `resolve_repo_data_dir()` |
| **_trusted_paths.py** | Validate trusted runtime directory overrides (`PROTEUS_ALLOW_TRUSTED_DIR_OVERRIDES` + `PROTEUS_TRUSTED_*_DIR`). | trusted-dir validation helpers |

## search/ Package

Public seam: `search/__init__.py` (+ `compat.py`). Internal modules are
prefixed `_` and tests monkeypatch them as `phonology.search.<name>` attributes.

| Module | Responsibility |
|--------|----------------|
| `compat.py` | Public `search()`, `seed_stage()`, `extend_stage()`, `build_kmer_index()`, `build_lexicon_map()`, `prepare_query_ipa()` (profile-default backfill) |
| `_query.py` | Query classification/normalization, partial-query parsing, consonant-skeleton extraction |
| `_indexing.py` | K-mer index construction over consonant skeletons |
| `_lookup.py` | Entry/IPA lookup, IPA index, exact-match injection |
| `_scoring.py` | Smith-Waterman local alignment, stage-2 scoring, rule markers |
| `_annotation.py` | Dialect attribution, alignment markers, ASCII visualization |
| `_selection.py` / `_overlap.py` / `_partial.py` | Candidate selection, fallback caps, partial-query matching |
| `_filtering.py` / `_quality.py` / `_dedup.py` | Mode quality filters, ranking, headword dedup |
| `_orchestration.py` | `_execute_search()` + per-mode finalization |
| `_registry.py` | Tokenized-rules cache, rules registry accessor |
| `_dependencies.py` | `PreparedQueryIpa`, `SearchExecutionResult`, lazy dependency container, fallback limits |
| `_types.py` | `SearchResult`, `KmerIndex`, `LexiconRecord`/`LexiconMap`, `QueryMode`, `PhoneInventory`, partial-query types |
| `_constants.py` | Confidence thresholds, fallback caps (`_DEFAULT_FALLBACK_CANDIDATE_LIMIT=2000`), `OBSERVED_PREFIX` |
| `_tokenization.py` | `tokenize_for_inventory()`, `resolve_entry_tokens()`, `tokenize_ipa` seam |
| `_debug_logging.py` | Debug-gated scoring/finalization logging |

### Fallback caps

`search()` bounds fallback exploration by default. Omitted
`similarity_fallback_limit` / `unigram_fallback_limit` resolve to
`_DEFAULT_FALLBACK_CANDIDATE_LIMIT` (`2000`) with a warning. Pass explicit
`None` to opt back into unbounded fallback.

## explain/ Package

`explainer.py` is a thin facade; implementation is split for cohesion:

| Module | Responsibility |
|--------|----------------|
| `_rule_paths.py` | Trusted-directory registry + rule-path resolution |
| `_rule_loader.py` | YAML rule loading, version-metadata extraction (`load_rules`, `get_rules_version`) |
| `_rule_tokenize.py` | `TokenizedRule`, tokenization/sorting (`tokenize_rules_for_matching`) |
| `_rule_match.py` | `explain()` / `explain_with_tokenized_rules()` mismatch-block matching state machine |
| `_context.py` | Rule-context predicates (vowel/consonant, lookahead) |
| `_types.py` | `Alignment`, `RuleApplication`, `RuleMetadata`, `_MismatchBlock` |
| `_prose.py` | `Explanation`, `to_prose()` |

## languages/ancient_greek/ Plugin

Owns all Greek-specific logic. `profile.py:build_profile()` is the
`proteus.languages` entry point.

| Module | Purpose |
|--------|---------|
| `profile.py` | `build_profile()` — assembles the `LanguageProfile` (converter, phones, paths, note builder, corpus adapter) |
| `ipa.py` | Grapheme→IPA: `greek_to_ipa()`, `to_ipa()`, `tokenize_ipa()`, `apply_koine_consonant_shifts()`, `apply_attic_sigma_sigma_to_tau_tau_shift()`, `strip_diacritics()`, `get_known_phones()` |
| `phones.py` | Greek phone inventory + vowel set |
| `orthography_notes/` | `build_orthographic_notes()` + `AncientGreekNoteKind`, parser, validators, schema, messages |
| `matrix_generator.py` | Generate/validate canonical Attic-Doric matrix (`python -m phonology.languages.ancient_greek.matrix_generator`) |
| `lsj/` + `lsj_extractor.py` | Parse Perseus LSJ TEI XML → JSON lexicon (`extract_entries()`) |
| `build_lexicon.py` | Clone LSJ, run extractor, fingerprint, validate (`build_lexicon_if_missing()`) |
| `betacode.py` | TLG/Perseus Beta Code → Unicode Greek |
| `transliterate.py` | Greek Unicode → scholarly Latin transliteration |
| `buck.py` | Read-only loader for Buck-normalized reference data |

## External Dependencies

- **PyYAML** — rule files under `data/languages/ancient_greek/rules/`
- **lingpy** — phonological alignment support
- **pathlib / unicodedata / json / os, stat / functools** (stdlib) — paths, Unicode normalization, data loading, TOCTOU-safe loading, LRU caching

## Related Areas

- **[api.md](./api.md)** — FastAPI layer integrating these modules
- **[mcp.md](./mcp.md)** — MCP server reusing the shared search path
- **[CLAUDE.md](../../CLAUDE.md)** — architecture rationale
- **Data:** `data/languages/ancient_greek/{matrices,rules,lexicon,orthography,corpus_sources}/`, `data/schemas/`
