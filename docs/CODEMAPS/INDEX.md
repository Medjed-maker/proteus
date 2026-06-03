# Proteus Codemaps Index

**Last Updated:** 2026-06-03

This index provides an overview of the Proteus codebase architecture. Proteus is a language-independent historical phonology framework with an Ancient Greek pilot, implementing a BLAST-like three-stage search pipeline (Seed → Extend → Filter) over IPA phonological space rather than text.

## Codemaps

### [phonology.md](./phonology.md)
Core computation layer (`src/phonology/`), split into a language-independent core (`core/`, `core/ports/`, `distance.py`, `search/`, `explain/`, `log_odds.py`) and the Ancient Greek plugin (`languages/ancient_greek/`). Implements the three-stage search pipeline, weighted edit distance over IPA, rule-based explanation, and language profiles wired through a registry.

### [api.md](./api.md)
FastAPI REST layer (`src/api/`). Exposes the phonological search engine via HTTP with endpoints for search, language discovery, version metadata, health checks, readiness probes, and static frontend serving. Manages dependency loading, request metadata, and request/response schemas.

### [mcp.md](./mcp.md)
MCP server layer (`src/mcp_server/`). Exposes the shared search engine through the `proteus-mcp` stdio server and the `ancient_phonology.search` tool, including schema artifacts and in-process MCP tests.

### [tests.md](./tests.md)
Test suite organization (`tests/`). 61 test files with ~1,520 test cases covering core (language-independent) modules, the `search/` and `explain/` packages, FastAPI and MCP integration, the Ancient Greek plugin, data validation, and meta-tests (packaging, smoke, i18n, OpenAPI). Includes shared fixtures and cache-reset strategies.

## Architecture at a Glance

```
REQUEST (JSON)
    ↓
[src/api/main.py or src/mcp_server/server.py]
    ↓ (delegates to shared runner)
[phonology/search: seed_stage → extend_stage → filter_stage]
    ↓ (uses)
[phonology/distance.py, core/ipa.py, explainer.py, language profile]
    ↓
RESPONSE (JSON with hits, rules applied, alignments)
```

## Quick Links

- **Entry Points:** `src/api/main.py` (FastAPI app), `src/mcp_server/server.py` (`proteus-mcp`), `src/phonology/search/` (search pipeline), `src/phonology/languages/ancient_greek/profile.py` (`build_profile()` plugin entry point)
- **Data Files:** `data/languages/ancient_greek/matrices/attic_doric.json`, `data/languages/ancient_greek/rules/`, `data/languages/ancient_greek/lexicon/greek_lemmas.json`
- **Key Dependencies:** FastAPI, Pydantic (via FastAPI), uvicorn, PyYAML, lingpy, FastMCP

For detailed architecture rationale, see `/Users/nakamuratakahito/proteus/CLAUDE.md`.
