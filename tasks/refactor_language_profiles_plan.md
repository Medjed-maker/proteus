# LanguageProfile Refactoring Implementation Plan

## Summary

Goal: refactor Proteus from an Ancient Greek-specific pilot into a
language-independent phonological search framework with an `ancient_greek`
language plugin, while preserving existing Ancient Greek search behavior and API
compatibility.

## Implementation Checklist

- [x] Create this Markdown implementation plan
- [x] Add `LanguageProfile` dataclass and profile registry
- [x] Register the built-in `ancient_greek` profile
- [x] Move Ancient Greek Unicode-to-IPA conversion into a language plugin
- [x] Move generic IPA tokenization into `phonology.core`
- [x] Keep `phonology.ipa_converter` as a backward-compatible wrapper
- [x] Move data to `data/languages/ancient_greek/{rules,matrices,lexicon}`
- [x] Update runtime path resolution for the new language data layout
- [x] Keep legacy matrix/rule path inputs working where existing APIs accept them
- [x] Update packaging configuration for the new data layout
- [x] Add `SearchRequest.language` with default `"ancient_greek"`
- [x] Preserve legacy `language="en"|"ja"` as an alias for response prose `lang`
- [x] Add `SearchRequest.response_language` as the preferred i18n field
- [x] Load `/search` lexicon, matrix, rules, and converter via `LanguageProfile`
- [x] Cache API search dependencies per language profile
- [x] Allow `search_execution()` to receive a profile-provided converter
- [x] Return `query_ipa` and `query_mode` from `search_execution()`
- [x] Load search rules through registered profile `rules_dir`
- [x] Add a `toy_language` integration test using only toy profile/data/converter
- [x] Update README API and data layout documentation

## Public API / Interface Changes

- [x] New: `phonology.profiles.LanguageProfile`
- [x] New: `phonology.profiles.get_language_profile(language_id: str)`
- [x] New: `phonology.profiles.register_language_profile(profile: LanguageProfile)`
- [x] Changed: `/search` accepts `language`, defaulting to `"ancient_greek"`
- [x] New: `/search` accepts `response_language` for response prose language
- [x] Preserved: `lang` remains a response prose language alias
- [x] Deprecated: legacy `language="en"|"ja"` emits migration headers
- [x] Preserved: `phonology.ipa_converter.to_ipa`, `greek_to_ipa`, `tokenize_ipa`
- [x] Preserved: legacy `/search` aliases `query`, `dialect`, and `max_results`

## Test Plan

- [x] `uv run pytest tests/test_ipa_converter.py tests/test_paths.py -q`
- [x] `uv run pytest tests/test_language_profiles.py -q`
- [x] `uv run pytest tests/test_api_main.py tests/test_i18n.py -q`
- [x] `uv run pytest tests/test_search.py tests/test_search_short_query.py ... -q`
- [x] `uv run pytest tests/test_build_lexicon.py tests/test_lsj_extractor.py tests/test_matrix_generator.py -q`
- [x] `uv run pytest tests/test_ipa_converter.py tests/test_paths.py tests/test_language_profiles.py tests/test_distance.py tests/test_explainer.py tests/test_data_files.py tests/test_validate_matrix.py tests/test_buck_data_files.py -q`
- [x] Targeted packaging checks for force-include/runtime layout and CI cache paths
- [x] `uv run pytest`

## Completion Criteria

- [x] Ancient Greek remains the default language profile.
- [x] `/search` accepts `language="ancient_greek"`.
- [x] Existing clients that omit `language` continue to work.
- [x] Existing clients that send `language="en"` or `"ja"` for i18n continue to work.
- [x] Existing clients that send `language="en"` or `"ja"` receive deprecation headers.
- [x] A registered `toy_language` can run `search_execution()` without core code changes.
- [x] Core search can receive language-specific converter/rule configuration externally.

## Deferred Technical Debt

The following issues were identified in post-implementation code review and are
now complete.

- [x] MEDIUM #4 — `_get_profile_converter` test monkeypatch seam (`api/main.py`)
- [x] MEDIUM #5 — `_call_with_language` `lru_cache` key hack (`api/main.py`)
- [x] MEDIUM #6 — `_models.SearchRequest` backward-compat shim for `language="en"|"ja"`

### MEDIUM #4 — `_get_profile_converter` test monkeypatch seam (`api/main.py`)

`api.main._get_profile_converter` returns the module-level `to_ipa` reference
for the default profile so that `tests/test_api_main.py` patches of
`api.main.to_ipa` propagate correctly. For custom-language profiles the
function returns `profile.converter` directly.

**Problem**: The logic couples API tests to implementation internals. If a
caller registers a custom converter under the default language id, the guard
raises a `RuntimeError` rather than routing cleanly.

**Recommended fix**: Decouple by having tests patch `phonology_search.search_execution`
instead of `api.main.to_ipa`. Move IPA conversion fully inside the search layer
so the API no longer needs a separate converter reference.

**Status**: Complete. The API no longer imports `to_ipa` or exposes
`_get_profile_converter`; `/search` forwards `profile.converter` to
`phonology_search.search_execution()`.

### MEDIUM #5 — `_call_with_language` `lru_cache` key hack (`api/main.py`)

Default-language calls invoke cached loaders with zero arguments
(`func()`) to reuse the same cache slot that tests rely on when they
monkeypatch the zero-arg default path.

**Problem**: The zero-arg slot is fragile. If the default language id ever
changes, the monkeypatch target changes silently. The hack also means
custom-language callers get a different code path than default-language callers.

**Recommended fix**: Remove `_call_with_language` and make all cached loaders
accept an explicit `language: str` argument. Update tests to patch at the
`phonology_search.search_execution` boundary rather than individual loaders.

**Status**: Complete. `_call_with_language` was removed and API dependency
loading now passes an explicit language id through the cached loader stack.

### MEDIUM #6 — `_models.SearchRequest` backward-compat shim for `language="en"|"ja"`

`SearchRequest` validates `language` against the registered profile registry
but special-cases `"en"` and `"ja"` as i18n pass-through values (not phonology
language IDs). This creates an invisible two-tier language field.

**Problem**: Existing clients that send `language="en"` for response locale
would receive a validation error if the shim is ever removed without a
migration notice. The field's dual semantics (phonology language vs. response
locale) is undocumented at the API boundary.

**Recommended fix**: Introduce a separate `response_language` field for i18n,
deprecate `language="en"|"ja"` via a migration header, and drop the shim in a
future minor version after a grace period.

**Status**: Complete. `response_language` is the preferred field, `lang`
remains an alias, and legacy `language="en"|"ja"` requests receive deprecation
and migration headers.
