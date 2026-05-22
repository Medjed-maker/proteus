# Hard Query Evaluation Data

This directory contains public, repo-safe hard query cases for Phase 3
scholarly validation. These files are quality-evaluation fixtures, not the
search-latency benchmark used by `tools/benchmark_search_latency.py`.

## Public Boundary

Public files in this directory may contain:

- search input forms that are safe to publish
- expected candidate headwords
- short source identifiers and short citations
- summarized reasoning and failure notes
- public reviewer handles or initials when consent is clear

Public files must not contain:

- unpublished collaborator data
- embargoed research notes
- long source-text quotations
- private reviewer comments or personal information
- URLs in any string field of a hard query case (defense in depth: the
  schema enforces this for `source_id` and `short_citation`, and the
  validator additionally rejects URLs anywhere in the case body)

Cases with `visibility: private_collaborator` or `visibility: embargoed` belong
in a non-public collection log until they can be anonymized or cleared for
publication. The validator rejects those visibility states in `--public-only`
mode.

## Files

- `public_seed_cases.yaml` contains a small public seed dataset that exercises
  the schema, validator, and evaluator.
- `../../schemas/hard_query_case.schema.json` defines the case-level schema.

Phase 3 acceptance requires at least 20 real or semi-real cases across public
or private collection logs. This seed file is intentionally smaller; it is a
repo-safe starting point, not the full collaborator collection.

## Validation

```bash
uv run python tools/validate_hard_queries.py --public-only data/evaluation/hard_queries/public_seed_cases.yaml
```

## Evaluation

```bash
uv run python tools/evaluate_hard_queries.py \
  --cases data/evaluation/hard_queries/public_seed_cases.yaml \
  --output-json /tmp/proteus-hard-query-eval.json
```
