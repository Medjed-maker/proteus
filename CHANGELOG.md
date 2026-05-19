# Changelog

## Unreleased

### Changed

- Clarified Ancient Greek orthographic-note wording and UI grouping so
  `παιδίο` can rank the current candidate `παιδίον` while separately presenting
  `παιδίου (paidiou)` as an alternative orthographic reading.

## [0.3.0] - 2026-05-18

Phase 2 REST API and MCP prototype surface is now stable. See `docs/API.md`
and `docs/MCP.md` for the full Phase 2 contract. This release also finalizes
the Phase 0 core/plugin separation and Phase 1 rule schema validation work.

### Added

- `GET /languages` and `GET /version` endpoints.
- Search response `meta` envelope (`api_version`, `schema_version`,
  `engine_version`, `request_id`, `timestamp`, `verification_url`,
  `request_echo`, `data_versions`, `ruleset_versions`).
- `X-Request-ID` middleware and response header.
- Deterministic `meta.verification_url` derived from the validated request.
- `proteus-mcp` console script (stdio MCP server) exposing the
  `ancient_phonology.search` tool.
- Entry-level review metadata validation (`review_status`, `citation_ready`,
  `source_type`, `source_ids`, `references`, `reference_urls`, `review_notes`,
  `reviewed_by`, `reviewed_at`) for Ancient Greek runtime
  orthographic-note YAML.
- `ReviewStatus` and `SourceType` literal type aliases in the Ancient Greek
  orthography-note module.

### Changed

- Preserved the top-level `data_versions` field in `SearchResponse`; it
  mirrors `meta.data_versions`.
- Runtime validation now keeps orthographic-note `references` and `source_ids`
  URL-free, restricts `reference_urls` to `http` / `https`, and rejects
  `evidence_excerpt` in packaged YAML.

### Deprecated

- `SearchRequest.orthography_hint` is marked `deprecated: true` in the OpenAPI
  schema. The field is still accepted for backward compatibility but is
  ignored during note generation. It will be removed in a future release.
- `build_orthographic_notes(orthography_hint=...)` now emits a
  `DeprecationWarning` and ignores the argument.

### Breaking Changes

- API `POST /search` no longer returns a `pre_403_2_attic` orthographic note
  when only the `orthography_hint="pre_403_2_attic"` query field is supplied
  without a curated runtime YAML entry. Consumers that relied on hint-only
  historical notes will see an empty `orthographic_notes` list for those
  candidates.
- Removed the unverified `pre_403_2_attic` tag from the provisional
  `παιδίο -> παιδίου` seed entry, so that seed no longer emits a historical
  Attic spelling note until direct source evidence is recorded.
- `phonology.search.tokenize_ipa` no longer applies Ancient-Greek-aware
  greedy tokenization. The search module now exposes a language-agnostic
  one-argument shim that splits IPA text into literal characters
  (e.g. `"kʰ"` → `["k", "ʰ"]`). Callers that need the previous
  Ancient-Greek-aware behavior should import from `phonology.ipa_converter`:
  `from phonology.ipa_converter import tokenize_ipa`. This change reflects
  the Phase 0 core/plugin separation; the search-side shim is intentionally
  language-agnostic.

### Documentation

- Added `docs/API.md` describing the full REST contract, error responses,
  reproducibility, deprecation policy, and environment variables.
- Added `docs/MCP.md` describing the MCP prototype, the
  `ancient_phonology.search` tool, and Claude Desktop config example.
- Committed OpenAPI artifact at `docs/api/openapi.json` and MCP tool schema
  at `docs/mcp/tools.json`, with drift-check scripts under `scripts/`.
- Added `docs/CODEMAPS/mcp.md`; updated `docs/CODEMAPS/api.md` and the
  README REST API / MCP Server sections.

### Infrastructure

- **Phase 0 Complete: Core/Plugin Separation**
  - Created `src/phonology/languages/ancient_greek/profile.py` with `build_profile()` factory
  - Refactored `src/phonology/profiles.py` to use lazy imports, removing eager dependency on Ancient Greek modules
  - Removed generic search/index/tokenization imports of Ancient Greek conversion helpers
  - Ancient Greek behavior is supplied through `LanguageProfile` configuration, with public backward compatibility maintained at the search/profile boundary

- **Phase 1 Complete: Rule Schema Validation**
  - Created standalone JSON Schema at `data/schemas/phonology_rule_file.schema.json`
  - Created validation tool at `tools/validate_rule_files.py` with CLI
  - Added packaged rule validation and focused negative schema tests for CI integration
  - Packaged the shared rule schema in wheel and sdist artifacts
  - All Ancient Greek rule files (consonant, vowel, morphophonemic) validate against the schema

- **Phase 0/1 Follow-up: Internal Cleanup**
  - Refactored `build_lexicon_map` into a public wrapper plus internal
    `_build_lexicon_map_core`, eliminating the `language=""` sentinel from
    internal callers.
  - Tightened JSON Schema `additionalProperties: false` for `meta`, `rule`,
    and `example` blocks; the document root remains permissive for forward
    compatibility, and `lemma_constraints` keeps `additionalProperties: true`.
  - Documented the `to_ipa` module-level monkeypatch seam (purpose, call
    conditions, and `isolated_language_registry` interaction).
  - Added a shared `_score_stage` mock kwargs assertion helper at
    `tests/_helpers/score_stage_mock.py` so a new keyword forwarded to the
    real `_score_stage` is detected by every existing fake.
  - Aligned the public stage APIs (`seed_stage`, `extend_stage`,
    `prepare_query_ipa`) with `build_kmer_index` / `build_lexicon_map`:
    passing `language="ancient_greek"` (default) now fills in the Ancient
    Greek phone inventory consistently across the seed/extend/prepare path,
    so direct callers no longer lose multi-character IPA phones (e.g.
    `pʰ`) at seed time.
