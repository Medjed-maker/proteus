# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Proteus is a language-independent historical phonology framework with an Ancient Greek pilot plugin. It finds lexically related word forms across dialects (Attic, Ionic, Doric, Koine) using BLAST-like phonological alignment and implements a three-stage search pipeline (Seed → Extend → Filter) operating over IPA phonological space rather than text.

**Current status**: Early development (v0.1.0). The core phonology modules (IPA conversion, distance calculation, matrix generation, rule loading) and the three-stage search pipeline (`search.py`: seed, extend, filter) are implemented. The `POST /search` API endpoint is functional.

## Build & Development Commands

```bash
# Install all dependencies (including dev)
uv sync --all-extras --dev

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_distance.py

# Run a specific test
uv run pytest tests/test_distance.py::test_function_name -v

# Run tests with coverage
uv run pytest --cov=src

# Build Tailwind CSS (run at least once before starting the dev server; re-run after HTML class changes)
bash scripts/build-css.sh

# Start dev server (localhost:8000, auto-reload)
uv run uvicorn api.main:app --reload

# Regenerate the Attic-Doric distance matrix
uv run python -m phonology.languages.ancient_greek.matrix_generator

# Build wheel
uv build
```

## Architecture

### Source Layout (`src/`)

The package uses `src/` layout with hatchling as the build backend.

**`phonology/`** — Language-independent computation layer. Nothing here may
reference Ancient Greek (enforced by `tests/test_core_language_independence.py`,
which scans `src/phonology/` excluding `languages/` for dialect/language terms).

- **`core/ipa.py`** — Language-agnostic IPA helpers, including the longest-match
  `tokenize_ipa(text, *, phone_inventory=...)` tokenizer. The phone inventory is
  injected by the active language profile; the algorithm itself is generic.
- **`core/ports/`** — Outward-facing core contracts (the "ports") that plugins
  and adapters implement or supply:
  - **`profiles.py`** — `LanguageProfile` / `IpaConverter` plus the language
    registry (`get_default_language_profile()`, `register_default_profiles()`).
    Plugins self-register via the `proteus.languages` entry-point group.
  - **`orthography_notes.py`** — `OrthographicNotePayload` and the
    `OrthographicNoteBuilder` protocol. `OrthographicNoteKind` defines only the
    language-independent base kinds; plugins may supply additional kinds.
  - **`corpus/`** — Corpus source-metadata models and the `CorpusAdapter`
    protocol.
- **`distance.py`** — Weighted edit distance (Needleman-Wunsch) over IPA
  sequences. Two scoring modes: raw (`phone_distance()`, DEFAULT_COST=5.0) and
  normalized 0.0-1.0. Matrix loading includes TOCTOU-safe symlink/path checks.
- **`search/`** — Three-stage search pipeline package (seed → extend → filter)
  with k-mer indexing, scoring, and filtering. Language-dependent behavior
  (converter, phone inventory, dialect skeleton builders) arrives via the
  profile at the public boundary.
- **`explainer.py`** — Public facade for rule-based explanation; re-exports the
  symbols defined across the **`explain/`** subpackage (`_rule_paths`,
  `_rule_loader`, `_rule_tokenize`, `_rule_match`, `_context`, `_types`,
  `_prose`). Loads YAML phonological rules from a profile-supplied rules dir.
- **`log_odds.py`** — Log-odds / likelihood-ratio computation over IPA
  alignments.
- **`_paths.py` / `_trusted_paths.py`** — Shared utilities to locate `data/`
  directories and to validate trusted runtime directory overrides.

**`phonology/languages/ancient_greek/`** — The Ancient Greek plugin. Owns all
Greek-specific logic: the grapheme→IPA converter (`ipa.py`), phone inventory
(`phones.py`), `profile.py` (`build_profile()` entry point), dialect skeleton
builders, the orthography-note builder and its `AncientGreekNoteKind`
vocabulary, and data-prep tooling (`lsj/`, `lsj_extractor.py`,
`build_lexicon.py`, `matrix_generator.py`, `betacode.py`, `buck.py`,
`transliterate.py`).

**`api/`** — FastAPI REST layer:
- **`main.py`** — Endpoints: `GET /` (frontend HTML), `POST /search`,
  `GET /health` (liveness — always OK), `GET /ready` (readiness — 503 until
  search dependencies load). Serves static assets from `web/static/`. Pydantic
  models define the full request/response schema including `RuleStep` and
  `SearchHit`.

### Data Files (`data/`)

Language data lives under `data/languages/<language_id>/` (matrices, rules,
lexicon, orthography, corpus_sources); shared JSON schemas live under
`data/schemas/`. Data files are bundled into the wheel via
`[tool.hatch.build.targets.wheel.force-include]` and resolved at runtime through
`_paths.py` (repo layout) or `importlib.resources` (installed package).

### Key Design Decisions

- **Dual data resolution**: `_paths.py` walks up to find repo-root `data/`, while `distance.py` also tries `importlib.resources` for installed-package paths. `PROTEUS_TRUSTED_MATRICES_DIR` env var overrides matrix location.
- **Normalized vs raw distance**: API consumers use normalized 0.0-1.0 distances; internal computation uses raw costs. These are separate code paths, not a simple division.
- **Matrix flattening**: `load_matrix()` recursively flattens nested JSON structures, skipping `_`-prefixed keys and `dialect_pairs` metadata to extract only phone-distance rows.

## Design Principles for AI Code Generation

AI-generated code in this repository should optimize for maintainability, changeability, and debuggability, not just passing the immediate request. Before implementing non-trivial changes, sketch the responsibilities and boundaries, then implement within the existing module structure.

Apply these principles:

1. **Encapsulation** - Keep data and behavior together. Expose clear methods or functions instead of leaking mutable internal structures. Preserve invariants close to the data they protect.
2. **Separation of concerns** - Keep parsing, validation, phonological/domain logic, persistence or filesystem access, API models, and UI concerns separate. A function or class should have one reason to change.
3. **Design by contract** - Make preconditions, postconditions, and invariants visible through type hints, Pydantic validation, assertions where they clarify internal assumptions, docstrings for public behavior, and targeted tests.
4. **Side-effect isolation** - Prefer pure functions for phonological computation and search logic. Confine I/O, environment access, global state, and external services to narrow adapter layers.
5. **Domain-focused naming** - Use historical phonology terms precisely and consistently. Model core domain concepts carefully; keep peripheral glue straightforward.

For refactoring prompts, use this framing: "Refactor this code according to encapsulation, separation of concerns, design by contract, side-effect isolation, and domain-focused naming. Preserve behavior and add focused tests for any changed boundaries."

Background reference: 『良いコード／悪いコードで学ぶ設計入門』 by MinoDriven. Relevant themes include encapsulation, immutability, responsibility separation, disentangling conditionals, naming through ubiquitous language, and concentrating design effort on the core domain.

## Testing

Tests use `pytest` with a shared `conftest.py` providing a FastAPI `TestClient` fixture. Test files mirror source modules (e.g., `test_distance.py`, `test_ipa_converter.py`). CI runs on Python 3.11 and 3.12.

## Language

Project documentation (`project.md`, CodeRabbit config) and project discussions are in Japanese. Code, docstrings, and commit messages are in English.

## Agent skills

### Issue tracker

Issues and PRDs are tracked as local Markdown under `.scratch/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Triage roles use the repository's Japanese status vocabulary. See `docs/agents/triage-labels.md`.

### Domain docs

This repository uses a single-context domain-document layout. See `docs/agents/domain.md`.
