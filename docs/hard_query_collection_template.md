# Hard Query Collection Template

Use this template to collect one Phase 3 hard query case at a time. Public
repository copies must contain only source identifiers, short citations, and
safe summaries. Keep unpublished collaborator notes, long source quotations,
and private reviewer comments out of the public repo.

## Case Metadata

```yaml
case_id: "hq-ag-0000"
visibility: "private_collaborator"  # public_seed | public_anonymized | private_collaborator | embargoed
input_form: ""
language: "ancient_greek"            # extend the schema enum when adding a new language profile
dialect_hint: "attic"                # extend the schema enum when a profile adds a dialect
max_candidates: 20
review_status: "not_reviewed"  # not_reviewed | collected | source_checked | expert_reviewed | rejected
reviewer: ""  # ASCII handle or initials; full name only with consent
```

## Expected Candidate

```yaml
expected_candidates:
  - headword: ""
    acceptable_forms: []
    note: ""
```

## Source Notes

Do not copy long Greek source text here. Record only identifiers and short,
publicly safe notes.

```yaml
source_notes:
  - source_id: ""
    short_citation: ""
    note: ""
```

## Manual Search

```yaml
manual_search:
  checked_by: ""
  checked_at: "YYYY-MM-DD"
  summary: ""
```

## Existing Search Failure

```yaml
existing_search_failure:
  tool: ""
  query: ""
  failure_type: "not_yet_checked"  # exact_search_miss | rank_too_low | wrong_candidate | manual_only | not_yet_checked
  notes: ""
```

## Proteus Tool-Assisted Search

```yaml
tool_assisted_search:
  checked_at: "YYYY-MM-DD"
  summary: ""
```

## Reasoning

```yaml
reasoning: ""
```

## Error Analysis

```yaml
false_positive_notes: []
false_negative_notes: []
```

## Public-Repo Safety Checklist

- [ ] No unpublished research data is copied into the public case.
- [ ] No long source-text quotation is copied into the case.
- [ ] No URLs appear in any string field of the case (schema enforces this
      for `source_id` and `short_citation`; the validator additionally
      rejects URLs anywhere in the case body).
- [ ] Reviewer identity is a consented public handle or initials.
- [ ] External reviewer comments are summarized, not copied verbatim.
