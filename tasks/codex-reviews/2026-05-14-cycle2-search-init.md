# Codex Review — Cycle 2: search/__init__.py split

**Date:** 2026-05-14
**Target:** working-tree diff (1844-line module split across new private siblings)
**Reviewer:** `/codex:review --wait` (codex-companion 1.0.4)
**Thread:** `019e26f7-371a-74e3-95e3-16f73ee28683`

## Scope

- `src/phonology/search/__init__.py`: 1844 → 775 lines
- New `src/phonology/search/_dependencies.py` (145): `PreparedQueryIpa`, `_FallbackLimits`, `_FinalizationResult`, `SearchExecutionResult`, `IpaConverter` Protocol, `_LazySearchDependencies` (with late-import of `_build_lexicon_map_for_inventory` to preserve monkeypatch seam)
- New `src/phonology/search/_registry.py` (134): `_load_rules_cached`, `_get_tokenized_rules`, `get_rules_registry`, `_TOKENIZED_RULES_CACHE_MAXSIZE` (with `sys.modules['phonology.search']` resolution of `get_rules_registry`, `tokenize_rules_for_matching`, `load_rules`, `get_language_profile`, `get_default_language_profile` so test monkeypatches still take effect)
- New `src/phonology/search/_orchestration.py` (564): three mode-aware finalizers, token-count proximity ranker, `_execute_search` orchestrator (with late `from . import` of every internal helper that crosses the package boundary)
- New `src/phonology/search/_selection.py` (503): four candidate-selection paths plus the length-proximate injector (with `sys.modules` fallback for `_select_partial_seed_candidates`, `_select_partial_fallback_candidates`, and `_select_partial_token_fallback_candidates`)
- `src/phonology/search/compat.py`: `TYPE_CHECKING` import retargeted to `._dependencies` to keep mypy-strict happy after `IpaConverter`/`PreparedQueryIpa` moved out of `__init__`.

## Outcome

**No issues found.** Codex confirmed the structure preserves behaviour by inspecting the diff against `HEAD`, importing `phonology.search`, listing the surviving symbol set, and verifying that every test seam (`to_ipa`, `load_rules`, `_score_stage`, `_seed_stage_core`, `_build_lexicon_map_for_inventory`, `_select_partial_seed_candidates`, `_select_partial_fallback_candidates`, `_select_partial_token_fallback_candidates`, `_summarize_query_ipa_for_logs`, `tokenize_rules_for_matching`, `_load_rules_cached.cache_clear`, `_get_tokenized_rules.cache_clear`) still resolves on the package.

> Codex (verbatim): 確認した範囲では、リファクタリング後の検索パイプラインに既存挙動を壊す明確な不具合は見つかりませんでした。サンドボックス制約により pytest は実行できませんでしたが、主要モジュールの import と基本的な検索スモークは通っています。

## Local Verification

- `uv run pytest -q` — **1621 passed** (all suites, ~13 min)
- `uv run mypy` — **0 issues across 52 source files**
- API smoke (`POST /search` for `λόγος`) returns expected `lóɡos` hit with distance 0.00 plus near matches `ἄλογος` (0.17) and `ἄλγος` (0.30).
- File sizes (post-refactor):
  - `__init__.py` 775 (was 1844)
  - `_orchestration.py` 564
  - `_selection.py` 503
  - `compat.py` 382 (unchanged in size, one import retargeted)
  - all other siblings ≤ 323

## Test-Seam Resolution Pattern

To keep `monkeypatch.setattr(search_module, "X", ...)` working after the split, every cross-module call in `_dependencies.py`, `_registry.py`, `_orchestration.py`, and `_selection.py` resolves `X` via the package namespace at call time:

- For sibling-module symbols re-exported from `__init__.py`: `from . import X` inside the function body.
- For dynamically-monkeypatched symbols when a static fallback is preferable: `getattr(sys.modules["phonology.search"], "X", default)`.

This avoids both circular-import issues at package init time and stale bindings during tests.

## Follow-ups

None.
