# Lexicon Schema Migration

Proteus lexicon files now use a top-level `schema_version` field to version the
data contract independently from the JSON Schema document path.

## Current format

- Set root `schema_version` to `2.0.0`.
- Rename `_meta.schema` to `_meta.data_schema_ref`.
- `_meta.license`, `_meta.contributors`, `_meta.data_schema_ref`, and
  `_meta.description` are optional again.

## Migration example

Before:

```json
{
  "_meta": {
    "schema": "data/lexicon/greek_lemmas.schema.json"
  },
  "lemmas": []
}
```

After:

```json
{
  "schema_version": "2.0.0",
  "_meta": {
    "data_schema_ref": "data/lexicon/greek_lemmas.schema.json"
  },
  "lemmas": []
}
```

Existing files that still use `_meta.schema` should be updated to
`_meta.data_schema_ref` before validating against the current schema.
