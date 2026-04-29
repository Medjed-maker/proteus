# Proteus Codemaps Index

**Last Updated:** 2026-04-11

This index provides an overview of the Proteus codebase architecture. Proteus is a language-independent historical phonology framework with an Ancient Greek pilot, implementing a BLAST-like three-stage search pipeline (Seed → Extend → Filter) over IPA phonological space rather than text.

## Codemaps

### [phonology.md](./phonology.md)
Core computation layer (`src/phonology/`). Implements Greek-to-IPA conversion, phonological distance calculation using weighted edit distance, and the three-stage search pipeline. Includes rule loading, matrix generation, and lexicon building utilities.

### [api.md](./api.md)
FastAPI REST layer (`src/api/`). Exposes the phonological search engine via HTTP with endpoints for search, health checks, readiness probes, and static frontend serving. Manages dependency loading and maps request/response schemas.

### [tests.md](./tests.md)
Test suite organization (`tests/`). 18 test files with 781 test cases covering unit tests (modules), integration tests (FastAPI layer), data validation, and meta-tests (packaging, smoke). Includes shared fixtures and caching strategies.

## Architecture at a Glance

```
REQUEST (JSON)
    ↓
[api/main.py]
    ↓ (delegates to)
[phonology/search.py: seed_stage → extend_stage → filter_stage]
    ↓ (uses)
[phonology/distance.py, ipa_converter.py, explainer.py, ...]
    ↓
RESPONSE (JSON with hits, rules applied, alignments)
```

## Quick Links

- **Entry Points:** `src/api/main.py` (FastAPI app), `src/phonology/search.py` (search pipeline)
- **Data Files:** `data/languages/ancient_greek/matrices/attic_doric.json`, `data/languages/ancient_greek/rules/`, `data/languages/ancient_greek/lexicon/greek_lemmas.json`
- **Key Dependencies:** FastAPI, Pydantic (via FastAPI), uvicorn, PyYAML, lingpy

For detailed architecture rationale, see `/Users/nakamuratakahito/proteus/CLAUDE.md`.
