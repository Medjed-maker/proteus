# Tests Codemap

**Last Updated:** 2026-04-11  
**Entry Points:** `conftest.py` (shared fixtures), `pyproject.toml [tool.pytest.ini_options]` (test discovery)

## Overview

Proteus tests follow a modular structure with **18 test files** organized by source module (unit tests), integration layer (API tests), and data validation. Shared fixtures live in `conftest.py`, which provides a FastAPI `TestClient` and cache reset utilities. Each test file mirrors a corresponding source module in `src/phonology/` or `src/api/`, plus dedicated validators for committed data files and a meta-test suite (packaging, smoke tests).

## Fixtures (conftest.py)

| Fixture | Scope | Purpose | Defined At |
|---------|-------|---------|------------|
| **client** | function | FastAPI `TestClient` for integration tests; disables startup warmup to speed up test runs. `app.state.disable_startup_warmup` is set True during test, reset False after. | Line 14–22 |
| **reset_pos_overrides_cache** | function | Clears `phonology.lsj_extractor._pos_overrides` cache before and after each test. Applied via `pytestmark` in test modules exercising POS extraction (test_lsj_extractor, test_build_lexicon). | Line 35–44 |

## Test Files

### Unit Tests (Core Modules)

| Test File | Target Module | Test Count (approx) | Key Areas Covered |
|-----------|----------------|-------------------|-------------------|
| test_distance.py | `phonology/distance.py` | 70 | `load_matrix()` JSON parsing, TOCTOU symlink checks, phone/sequence/word distance calculation, raw vs normalized scoring modes, matrix flattening, edge cases (empty sequences, unknowns) |
| test_ipa_converter.py | `phonology/ipa_converter.py` | 54 | Greek Unicode to IPA conversion (polytonic, monotonic), diphthong handling, diacritics (rough breathing, macron, diaeresis), Koine consonant shifts, IPA tokenization, strip_diacritics |
| test_betacode.py | `phonology/betacode.py` | 18 | Beta Code (TLG/Perseus ASCII Greek) to Unicode conversion |
| test_transliterate.py | `phonology/transliterate.py` | 21 | Greek Unicode to scholarly Latin transliteration |
| test_explainer.py | `phonology/explainer.py` | 48 | YAML rule loading from `data/rules/ancient_greek/`, rule matching, alignment explanation, structured `RuleApplication` generation |
| test_paths.py | `phonology/_paths.py` | 4 | `resolve_repo_data_dir()` path walking, data directory location |
| test_phones.py | `phonology/_phones.py` | 6 | IPA phone inventory constants (`VOWEL_PHONES` frozenset) |

### Integration Tests

| Test File | Target Module | Test Count (approx) | Key Areas Covered |
|-----------|----------------|-------------------|-------------------|
| test_api_main.py | `api/main.py` (FastAPI layer) | 85 | GET `/` (frontend HTML serving), POST `/search` (full pipeline with request validation, response schema), GET `/health` (liveness), GET `/ready` (readiness probe), dependency loading state, error handling, Pydantic model validation (`SearchRequest`, `SearchResponse`, `SearchHit`, `RuleStep`) |
| test_search.py | `phonology/search.py` (three-stage pipeline) | 141 | seed_stage k-mer indexing, extend_stage distance scoring and dialect attribution, filter_stage ranking/truncation, k-mer index construction, lexicon map building, query mode classification (Full-form, Short-query, Partial-form), exact-match injection, headword deduplication, confidence thresholding, rule caching |

### Data Validation Tests

| Test File | Target Module/Data | Test Count (approx) | Key Areas Covered |
|-----------|-------------------|-------------------|-------------------|
| test_data_files.py | `data/lexicon/greek_lemmas.json` + schema | 25 | JSON schema validation against `greek_lemmas.schema.json` (per [project.optional-dependencies].dev), metadata checks, IPA transcription validation, lexicon cardinality (≥100 lemmas), rule file structure |
| test_validate_matrix.py | `data/matrices/attic_doric.json` | 26 | Matrix JSON structure, symmetry enforcement, phonological distance bounds [0.0, 1.0], sound class completeness (vowels, stops, dialect_pairs), matrix flattening correctness |
| test_buck_data_files.py | `data/buck/` reference data | 7 | Buck-normalized dialect/grammar/glossary file validation |
| test_matrix_generator.py | `phonology/matrix_generator.py` | 25 | `generate()` function for Attic-Doric canonical matrix, matrix generation, validation, regeneration workflows |

### Utility Tests

| Test File | Target Module | Test Count (approx) | Key Areas Covered |
|-----------|----------------|-------------------|-------------------|
| test_buck_loader.py | `phonology/buck.py` | 13 | `load_buck_data()` from `data/buck/`, BuckData TypedDict structure, dialect/grammar reference data loading |
| test_lsj_extractor.py | `phonology/lsj_extractor.py` | 109 | Perseus LSJ XML parsing via `extract_entries()`, headword/gloss extraction, Beta Code decoding, IPA transcription, POS tagging with `_pos_overrides` caching (reset via fixture) |
| test_build_lexicon.py | `phonology/build_lexicon.py` | 31 | `build_lexicon_if_missing()` orchestration, LSJ repo cloning, lexicon extraction, fingerprinting, output validation |

### Meta Tests (Packaging & Smoke)

| Test File | Target Module/Concern | Test Count (approx) | Key Areas Covered |
|-----------|----------------------|-------------------|-------------------|
| test_packaging.py | Wheel build & bundling | 18 | Hatchling wheel generation, data file bundling (`[tool.hatch.build.targets.wheel.force-include]`), importlib.resources resolution (installed packages), `_paths.py` data dir fallbacks |
| test_smoke.py | Cross-cutting integration | 1 | Basic import test: verifies `phonology.ipa_converter`, `phonology.distance`, `phonology.search`, `phonology.explainer`, `api.main` are all importable |

## Test Execution

```bash
# Run all tests (from CLAUDE.md)
uv run pytest

# Run a single test file
uv run pytest tests/test_distance.py

# Run a specific test
uv run pytest tests/test_distance.py::TestLoadMatrix::test_loads_valid_json_with_nested_rows -v

# Run tests with coverage (from CLAUDE.md)
uv run pytest --cov=src
```

**Test Discovery:** Configured in `pyproject.toml [tool.pytest.ini_options]`:
- `testpaths = ["tests"]`
- `addopts = "--tb=short -v"`

## Coverage Gaps

All source modules in `src/phonology/` and `src/api/` have corresponding test files:

- ✅ phonology/_paths.py → test_paths.py (4 tests)
- ✅ phonology/_phones.py → test_phones.py (6 tests)
- ✅ phonology/ipa_converter.py → test_ipa_converter.py (54 tests)
- ✅ phonology/distance.py → test_distance.py (70 tests)
- ✅ phonology/search.py → test_search.py (141 tests)
- ✅ phonology/explainer.py → test_explainer.py (48 tests)
- ✅ phonology/betacode.py → test_betacode.py (18 tests)
- ✅ phonology/transliterate.py → test_transliterate.py (21 tests)
- ✅ phonology/buck.py → test_buck_loader.py (13 tests)
- ✅ phonology/build_lexicon.py → test_build_lexicon.py (31 tests)
- ✅ phonology/lsj_extractor.py → test_lsj_extractor.py (109 tests)
- ✅ phonology/matrix_generator.py → test_matrix_generator.py (25 tests)
- ✅ api/main.py → test_api_main.py (85 tests)

**No gaps identified.** Total: **781 test cases** across 18 files.

## Dependencies

Test dependencies (from `[project.optional-dependencies].dev`):
- **pytest** ≥8.0.0 — Test framework
- **pytest-cov** ≥7.0.0 — Coverage reporting
- **httpx** ≥0.28.0 — Async HTTP client (optional for FastAPI testing)
- **jsonschema** ≥4.26.0 — JSON schema validation (test_data_files.py, test_validate_matrix.py)
- **types-PyYAML** ≥6.0.2 — Type stubs for PyYAML

## Related Areas

- **[phonology.md](./phonology.md)** — Modules tested by unit test files
- **[api.md](./api.md)** — FastAPI layer tested by test_api_main.py
- **[INDEX.md](./INDEX.md)** — Codemaps index
- **[CLAUDE.md](../CLAUDE.md)** — Build commands, architecture rationale
- **Data files:** `/Users/nakamuratakahito/proteus/data/matrices/`, `/data/rules/`, `/data/lexicon/`, `/data/buck/`
