# Domain docs

Engineering skills consume this repository's domain documentation as a single
context.

## Before exploring

- Read `CONTEXT.md` at the repository root when it exists.
- Read relevant ADRs under `docs/adr/` when that directory exists.
- If either location is absent, proceed silently. Domain documents are created
  lazily when terminology or architectural decisions need to be recorded.

## Vocabulary

Use domain terms as defined in `CONTEXT.md` in issue titles, tests, plans, and
implementation names. Avoid introducing synonyms for established terms.

If a required concept is not present, first check whether existing repository
language already expresses it. Record a genuine terminology gap for later
domain-document work rather than inventing an inconsistent term.

## ADR conflicts

Surface any conflict with an existing ADR explicitly. Do not silently override
an architectural decision.
