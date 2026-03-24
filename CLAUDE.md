# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Proteus is an Ancient Greek phonological search engine that finds lexically related word forms across dialects (Attic, Ionic, Doric, Koine) using BLAST-like phonological alignment. It implements a three-stage search pipeline (Seed → Extend → Filter) operating over IPA phonological space rather than text.

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
uv run uvicorn proteus.api.main:app --reload

# Regenerate the Attic-Doric distance matrix
uv run python -m proteus.phonology.matrix_generator

# Build wheel
uv build
```

## Architecture

### Source Layout (`src/proteus/`)

The package uses `src/` layout with hatchling as the build backend.

**`phonology/`** — Core computation layer:
- **`ipa_converter.py`** — Greek script → IPA conversion. Greedy left-to-right tokenizer handles diphthongs, rough breathing, diaeresis, iota subscript. `greek_to_ipa()` returns phone list; `tokenize_ipa()` parses compact/space-separated IPA strings back into phone tokens using a priority-ordered inventory.
- **`distance.py`** — Weighted edit distance (Needleman-Wunsch) over IPA sequences. Two scoring modes: raw (using `phone_distance()` with DEFAULT_COST=5.0) and normalized 0.0-1.0 (capping substitutions at 1.0). Matrix loading includes TOCTOU-safe symlink/path traversal checks.
- **`search.py`** — Three-stage search pipeline (`seed_stage`, `extend_stage`, `filter_stage`) with `SearchResult` and `SearchConfig` dataclasses.
- **`explainer.py`** — Loads YAML phonological rules from `data/rules/`, validates structure and unique IDs. `explain_alignment()` is stubbed.
- **`matrix_generator.py`** — Reads/validates/regenerates `data/matrices/attic_doric.json`. Enforces symmetry, completeness, and [0.0, 1.0] bounds on sound-class sub-matrices (vowels, stops, dialect_pairs).
- **`_paths.py`** — Shared utility to locate `data/` directories by walking up from the module to find `pyproject.toml`.

**`api/`** — FastAPI REST layer:
- **`main.py`** — Endpoints: `GET /` (frontend HTML), `POST /search`, `GET /health`. Serves static assets from `web/static/`. Pydantic models define the full request/response schema including `RuleStep` and `SearchHit`.

### Data Files (`data/`)

- **`matrices/attic_doric.json`** — Phonological distance matrix with `sound_classes.vowels`, `sound_classes.stops`, and `sound_classes.dialect_pairs` sections.
- **`rules/ancient_greek/`** — YAML rule files (`vowel_shifts.yaml`, `consonant_changes.yaml`) with structured rule entries (id, name, from/to phones, dialect, weight).
- **`lexicon/greek_lemmas.json`** — LSJ headword list with IPA transcriptions; validated against `greek_lemmas.schema.json`.

Data files are bundled into the wheel via `[tool.hatch.build.targets.wheel.force-include]` and resolved at runtime through `_paths.py` (repo layout) or `importlib.resources` (installed package).

### Key Design Decisions

- **Dual data resolution**: `_paths.py` walks up to find repo-root `data/`, while `distance.py` also tries `importlib.resources` for installed-package paths. `PROTEUS_TRUSTED_MATRICES_DIR` env var overrides matrix location.
- **Normalized vs raw distance**: API consumers use normalized 0.0-1.0 distances; internal computation uses raw costs. These are separate code paths, not a simple division.
- **Matrix flattening**: `load_matrix()` recursively flattens nested JSON structures, skipping `_`-prefixed keys and `dialect_pairs` metadata to extract only phone-distance rows.

## Testing

Tests use `pytest` with a shared `conftest.py` providing a FastAPI `TestClient` fixture. Test files mirror source modules (e.g., `test_distance.py`, `test_ipa_converter.py`). CI runs on Python 3.11 and 3.12.

## Language

Project documentation (`project.md`, CodeRabbit config) is in Japanese. Code, docstrings, and commit messages are in English.
