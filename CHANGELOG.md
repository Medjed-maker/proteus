# Changelog

## Unreleased

### Breaking Changes

- API `POST /search` no longer returns a `pre_403_2_attic` orthographic note
  when only the `orthography_hint="pre_403_2_attic"` query field is supplied
  without a curated runtime YAML entry. Consumers that relied on hint-only
  historical notes will see an empty `orthographic_notes` list for those
  candidates.
- Removed the unverified `pre_403_2_attic` tag from the provisional
  `παιδίο -> παιδίου` seed entry, so that seed no longer emits a historical
  Attic spelling note until direct source evidence is recorded.

### Deprecated

- `SearchRequest.orthography_hint` is marked `deprecated: true` in the OpenAPI
  schema. The field is still accepted for backward compatibility but is
  ignored during note generation. It will be removed in a future release.
- `build_orthographic_notes(orthography_hint=...)` now emits a
  `DeprecationWarning` and ignores the argument.

### Added

- Entry-level review metadata validation (`review_status`, `citation_ready`,
  `source_type`, `source_ids`, `references`, `reference_urls`, `review_notes`,
  `reviewed_by`, `reviewed_at`) for Ancient Greek runtime
  orthographic-note YAML.
- `ReviewStatus` and `SourceType` literal type aliases in the Ancient Greek
  orthography-note module.
- Runtime validation now keeps orthographic-note `references` and `source_ids`
  URL-free, restricts `reference_urls` to `http` / `https`, and rejects
  `evidence_excerpt` in packaged YAML.

