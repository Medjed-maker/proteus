# Tests Codemap

<!-- Generated: 2026-06-03 | Files scanned: tests/test_*.py -->

**Last Updated:** 2026-06-03
**Entry Points:** `tests/conftest.py` (shared fixtures), `pyproject.toml [tool.pytest.ini_options]`

## Overview

**61 test files, ~1,520 test functions.** Tests mirror the post-refactor source
layout: core (language-independent) modules, the `search/` and `explain/`
packages, the `api/` and `mcp_server/` layers, the Ancient Greek plugin, plus
data-file validators and meta-tests (packaging, smoke, i18n, OpenAPI).

## Fixtures (tests/conftest.py)

| Fixture | Scope | Purpose |
|---------|-------|---------|
| **client** | function | FastAPI `TestClient`; disables startup warmup for speed |
| **clear_rule_cache** | function (autouse) | Clears rule/registry caches around each test |
| **reset_pos_overrides_cache** | function | Clears `lsj_extractor._pos_overrides` cache (POS-extraction tests) |
| **isolated_language_registry** | function | Resets the language registry; forbids implicit `to_ipa` to catch leaks |
| **build_toy_profile** | function | Factory building a synthetic `LanguageProfile` for core tests |
| **known_phones** | session | Ancient Greek phone inventory |
| **sample_search_results / sample_lexicon** | function | Canned search inputs |
| **translations_data** | session | i18n translation table |
| **mock_search_dependencies** | function | Monkeypatches the search dependency loader with fakes |

## Test Groups

### Core (language-independent)
| File | Target | ~Count |
|------|--------|-------|
| test_core_ipa.py | `core/ipa.py` tokenizer/normalization | 10 |
| test_core_language_independence.py | guard: no language terms in core | 1 |
| test_distance.py | `distance.py` matrix loading, scoring, TOCTOU | 78 |
| test_explainer.py | `explainer.py` / `explain/*` rule loading + matching | 85 |
| test_log_odds.py | `log_odds.py` NW + log-odds computation | 36 |
| test_paths.py / test_trusted_paths.py | `_paths.py`, `_trusted_paths.py` | 6 / 10 |
| test_language_profiles.py | `core/ports/profiles.py` registry | 28 |
| test_orthography_notes.py | orthography-note payload + builder | 44 |
| test_corpus_adapters.py / test_corpus_adapter_schema.py | `core/ports/corpus/` | 18 / 6 |
| test_i18n.py | translation tables / prose locale | 42 |

### search/ package
| File | Focus | ~Count |
|------|-------|-------|
| test_search.py | top-level pipeline behavior | 40 |
| test_search_extend_stage.py | Smith-Waterman scoring + annotation | 38 |
| test_search_small_stages.py | per-stage helpers | 45 |
| test_search_partial.py / test_search_partial_infix.py | wildcard partial queries | 25 / 5 |
| test_search_annotation.py | dialect attribution, visualization | 22 |
| test_search_inject_exact.py / test_search_exact_integration.py | exact-match injection | 17 / 9 |
| test_search_token_fallback.py / test_search_unigram_fallback.py | fallback caps | 13 / 11 |
| test_search_filtering.py / test_search_dedup.py / test_search_short_query.py | ranking, dedup, short-query mode | 9 / 9 / 10 |
| test_search_tokenization.py / test_search_pipeline.py / test_search_runner.py | tokenization, orchestration, shared runner | 4 / 3 / 6 |
| test_score_stage_mock.py / test_debug_logging.py / test_benchmark_search_latency.py | scoring mocks, debug logging, latency | 4 / 16 / 14 |

### API layer (`src/api/`)
| File | Focus | ~Count |
|------|-------|-------|
| test_api_main.py | routes, request/response schema, pipeline | 154 |
| test_api_languages.py / test_api_version.py | `/languages`, `/version` | 9 / 8 |
| test_api_search_meta.py / test_api_search_sources.py | response meta, corpus sources | 9 / 8 |
| test_api_request_id.py / test_api_verification.py | request-id middleware, verification URL | 8 / 14 |
| test_openapi.py | OpenAPI schema drift | 5 |

### MCP layer (`src/mcp_server/`)
| File | Focus | ~Count |
|------|-------|-------|
| test_mcp_search_tool.py | in-process FastMCP tool | 14 |
| test_mcp_server_init.py / test_mcp_schema.py | init, entry point, schema artifact | 3 / 3 |

### Ancient Greek plugin
| File | Target | ~Count |
|------|--------|-------|
| test_ipa_converter.py | `languages/ancient_greek/ipa.py` | 52 |
| test_phones.py | phone inventory | 13 |
| test_betacode.py / test_transliterate.py | Beta Code, transliteration | 17 / 20 |
| test_lsj_extractor.py | LSJ TEI XML → lexicon | 168 |
| test_extract_epidoc_choices.py | EpiDoc `<choice>` extraction | 47 |
| test_build_lexicon.py | lexicon build orchestration | 60 |
| test_matrix_generator.py | Attic-Doric matrix generation | 42 |
| test_buck_loader.py | Buck reference loader | 21 |

### Data / schema validation
| File | Target | ~Count |
|------|--------|-------|
| test_data_files.py | `greek_lemmas.json` vs schema | 34 |
| test_validate_matrix.py | matrix structure, symmetry, bounds | 23 |
| test_validate_rule_files.py | YAML rule file structure | 31 |
| test_hard_query_schema.py | hard-query evaluation schema | 21 |
| test_buck_data_files.py | Buck data files | 7 |
| test_build_log_odds_matrix.py | log-odds matrix build script | 36 |

### Meta
| File | Concern | ~Count |
|------|---------|-------|
| test_packaging.py | wheel build, data bundling, `importlib.resources` | 21 |
| test_web_assets.py | static asset presence | 7 |
| test_smoke.py | cross-module import smoke | 1 |
| test_helpers.py | shared test helpers (no `test_` funcs) | 0 |

## Test Execution

```bash
uv run pytest                              # all
uv run pytest tests/test_distance.py       # one file
uv run pytest tests/test_distance.py::test_name -v
uv run pytest --cov=src                     # coverage
```

Discovery via `pyproject.toml [tool.pytest.ini_options]`:
`testpaths = ["tests"]`, `addopts = "--tb=short -v"`. CI runs Python 3.11 & 3.12.

## Related Areas

- **[phonology.md](./phonology.md)** — modules under unit test
- **[api.md](./api.md)** / **[mcp.md](./mcp.md)** — integration layers
- **[INDEX.md](./INDEX.md)** — codemaps index
