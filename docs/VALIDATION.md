# Scholarly Validation Workflow

Phase 3 validates whether Proteus solves real research pain, not just whether
the API returns well-formed JSON. The validation workflow collects hard query
cases, compares manual search with tool-assisted search, tracks false
positives and false negatives, and feeds reviewed findings back into rules and
orthographic-note data.

## Data Boundary

Public repository data may include:

- schema and templates for case collection
- public or anonymized seed cases
- expected candidate headwords
- short source identifiers and citations
- concise reasoning and failure summaries

Public repository data must not include:

- unpublished collaborator datasets
- embargoed research notes
- long source-text quotations
- private reviewer comments
- personal information or real names without consent
- URLs in any string field of a hard query case (defense in depth: the
  schema rejects URL-shaped strings in `source_id` and `short_citation`,
  and the validator additionally rejects URLs anywhere else in the case
  body, including notes and summaries)

Use `visibility: private_collaborator` or `visibility: embargoed` in a
non-public collection log until a case is cleared for publication. The public
seed dataset is validated with `--public-only`, which rejects those visibility
states.

## Collection Workflow

1. Record one case using `docs/hard_query_collection_template.md`.
2. Capture the input form, language, dialect hint, expected candidate, and
   source notes.
3. Record what manual search found and where exact or existing search failed.
4. Run Proteus with `tools/evaluate_hard_queries.py`.
5. Review the rank, confidence, applied rules, orthographic notes, false
   positives, and false negatives.
6. Keep private material outside the public repository until publication is
   approved.

## Manual vs Tool-Assisted Comparison

Manual search records what a scholar or collaborator could find without
Proteus. Tool-assisted search records what Proteus returns for the same input.
The important comparison points are:

- whether the expected candidate appears in the requested top-N
- whether a non-expected candidate outranks the expected one
- whether applied rules and explanations are useful
- whether orthographic notes are helpful but still correctly marked provisional

## Orthographic Note Review

The `παιδίο -> παιδίου` seed remains the first Phase 3 review pilot. Promotion
to `citation_ready: true` requires source identifiers, short references,
reviewer metadata, an expert-reviewed decision, and an ISO review date in the
runtime orthographic-note data.

If source evidence is still insufficient, keep the entry at
`needs_expert_review` and do not describe it as citation-ready. Pre-403/2 BCE
Attic orthographic notes require inscriptional source evidence or explicit
expert judgment before they can be promoted.

## Commands

Validate the public seed dataset:

```bash
uv run python tools/validate_hard_queries.py --public-only data/evaluation/hard_queries/public_seed_cases.yaml
```

Evaluate the public seed dataset:

```bash
uv run python tools/evaluate_hard_queries.py \
  --cases data/evaluation/hard_queries/public_seed_cases.yaml \
  --output-json /tmp/proteus-hard-query-eval.json
```

Check the Phase 3 collection target when private logs are available:

```bash
uv run python tools/validate_hard_queries.py --min-cases 20 path/to/non-public-cases.yaml
```

Phase 3 evaluator tests are marked `integration` because they exercise the
real search runner. CI or local quick passes can skip them with:

```bash
uv run pytest -m "not integration"
```
