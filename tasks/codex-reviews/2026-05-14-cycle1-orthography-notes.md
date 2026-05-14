# Codex Review — Cycle 1: orthography_notes split

**Date:** 2026-05-14
**Target:** working-tree diff (file replaced with sub-package)
**Reviewer:** `/codex:review --wait` (codex-companion 1.0.4)
**Thread:** `019e2550-7fac-7e81-9bfb-d0480107e2ed`

## Scope

- Deleted `src/phonology/languages/ancient_greek/orthography_notes.py` (631 lines)
- Created `src/phonology/languages/ancient_greek/orthography_notes/` sub-package:
  - `__init__.py` (132) — facade keeping `_load_correspondence_entries`, `build_orthographic_notes`, `prepare_orthographic_data` in package globals so `monkeypatch.setattr(module, "_orthography_data_path", ...)` and `setattr(module, "_load_correspondence_entries", ...)` continue to take effect
  - `schema.py` (85) — data classes, constants, `_nfc`
  - `_paths.py` (17) — `_orthography_data_path`
  - `_validators.py` (263) — field-level validators
  - `_parser.py` (106) — `_parse_entry`, `_load_yaml_mapping`
  - `_messages.py` (133) — language-aware message builders

## Outcome

**No issues found.** Codex independently verified the split is a pure structural refactor and ran `pytest tests/test_orthography_notes.py` — green.

> Codex (verbatim): 変更は既存モジュールをパッケージ分割したリファクタリングで、確認した範囲では動作差分や破壊的な問題は見つかりませんでした。関連する orthography notes のテストも通過しています。

## Local Verification

- `uv run pytest tests/test_orthography_notes.py tests/test_language_profiles.py tests/test_api_main.py -x` — **274 passed**
- `uv run mypy` — **0 issues across 48 source files**
- Max file in sub-package: `_validators.py` at 263 lines (well under 800)

## Follow-ups

None.
