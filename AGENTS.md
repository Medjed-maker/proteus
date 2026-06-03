# Repository Guidelines

## Project Structure & Module Organization

Proteus uses a `src/` layout. Core code lives in `src/`: `phonology/` contains IPA conversion, distance logic, rule explanation, and matrix generation; `api/` exposes the FastAPI app; `web/` holds the packaged frontend HTML. Repository data files live under `data/` (`matrices/`, `rules/ancient_greek/`, `lexicon/`). Tests are in `tests/` and generally mirror the source modules, for example `src/phonology/distance.py` maps to `tests/test_distance.py`.

## Build, Test, and Development Commands

Install dependencies with `uv sync --all-extras --dev`. Run the full test suite with `uv run pytest`. For focused work, use `uv run pytest tests/test_distance.py` or `uv run pytest tests/test_api_main.py::TestHealthEndpoint::test_api_health -v`. Start the local API with `uv run uvicorn api.main:app --reload`. Regenerate the bundled distance matrix with `uv run python -m phonology.languages.ancient_greek.matrix_generator`. Build a wheel with `uv build`.

## Coding Style & Naming Conventions

Target Python 3.11+ and follow PEP 8 with 4-space indentation. Prefer explicit type hints, small pure functions, and concise Google-style docstrings on non-trivial public functions and classes. Use `snake_case` for modules, functions, and variables; use `PascalCase` for classes and Pydantic models. Keep code, docstrings, and commit messages in English; repository-level discussion docs may be Japanese.

## Design Principles for AI-Assisted Changes

When using AI agents to implement or refactor code, explicitly preserve maintainability rather than only making the feature work. Follow these principles, especially in core phonology and search logic:

- **Encapsulation**: keep state and behavior together behind clear public APIs. Avoid exposing mutable internals or spreading invariants across callers.
- **Separation of concerns**: keep validation, domain logic, persistence, I/O, API schema handling, and presentation concerns in distinct functions or modules. Each class or function should have one clear responsibility.
- **Design by contract**: make preconditions, postconditions, and invariants explicit through type hints, validation, assertions where appropriate, docstrings, and focused tests.
- **Side-effect isolation**: prefer pure functions for business and phonological logic. Keep filesystem, network, environment-variable, and global-state interactions in narrow adapter or service boundaries.
- **Domain language**: use names that reflect the historical phonology domain accurately. Invest more modeling effort in core domain concepts, and keep peripheral glue code simple.

These guidelines are inspired by the practical design lessons in 『良いコード／悪いコードで学ぶ設計入門』 by MinoDriven. For refactoring work, it is acceptable to ask agents to improve code against these principles before adding new behavior.

## Testing Guidelines

Tests use `pytest` with shared fixtures in `tests/conftest.py`. Name files `test_<module>.py` and keep test classes and function names descriptive. Cover boundary cases, invalid input, and path-security behavior where relevant. No coverage threshold is enforced in `pyproject.toml`, but `pytest-cov` is available; run `uv run pytest --cov=src` for non-trivial changes. CI runs the suite on Python 3.11 and 3.12.

## Commit & Pull Request Guidelines

Recent history follows Conventional Commit style such as `feat: implement secure phonological distance matrix loading`. Continue using prefixes like `feat:`, `fix:`, and `test:` with concise summaries. Pull requests should describe the behavior change, list affected modules or data files, and note test coverage added or updated. Include screenshots only when changing `src/web/index.html` or other user-visible output.

## Data & Configuration Notes

Do not hardcode external paths. Matrix and rule loading rely on repository-relative resolution and packaged data inclusion defined in `pyproject.toml`. When editing files under `data/`, add or update tests that validate schema, packaging, or runtime loading.
