# Orthographic Notes Review Packet Template

このテンプレートは、Ancient Greek orthographic note の runtime YAML entry を
reviewer に渡し、source evidence、reviewer decision、実装反映内容を entry 単位で
記録するために使う。

`romanization` は `original` ではなく `normalized` 形に対応する値を書く。`source_ids`
には canonical identifier のみを入れ、URL は `reference_urls` に分離する。`rejected`
decision は runtime YAML に残さず、review log、docs、issue、または PR discussion に
記録する。

## Entry Metadata

```yaml
original: ""
normalized: ""
candidate_headwords: []
romanization: ""  # romanization of normalized, not original
kind: "orthographic_correspondence"
tags: []
confidence: "medium"
```

## Review Metadata

```yaml
review_status: "not_expert_reviewed"
citation_ready: false
reviewed_by: ""  # ASCII handle or initials; full name only with consent
reviewed_at: ""  # ISO date, required for expert_reviewed / citation_ready
```

`citation_ready: true` requires `review_status: expert_reviewed`, non-empty
`source_type`, `source_ids`, `references`, `reviewed_by`, and ISO-date
`reviewed_at`.

## Evidence

```yaml
source_type: []      # list, for example ["aio"] or ["aio", "expert_note"]
source_ids: []       # canonical identifiers only; do not include URLs
references: []       # short human-readable citations
reference_urls: []   # optional
dates: ""
place: ""
dialect_or_region: ""
publication_reference: ""
```

Evidence notes:

- Do not store long Greek source text in runtime YAML.
- Use `references` for short citation strings that may be shown through API/UI.
- Keep `references` and `source_ids` URL-free. Runtime validation rejects
  URL-like values in those fields.
- Use `reference_urls` only for `http` / `https` verification links.
- Use `review_notes` for editor-only notes, not public display text.
- Do not add `evidence_excerpt` to runtime YAML; excerpt handling requires a
  separate policy for length, copyright, and source terms.
- If `pre_403_2_attic` is proposed, the evidence must identify a pre-403/2 BCE
  Attic inscription source or an explicit expert judgment.

## Reviewer Decision

Choose exactly one primary decision:

- [ ] Keep entry
- [ ] Change normalized form
- [ ] Change candidate headword
- [ ] Change tags
- [ ] Change confidence
- [ ] Reject entry
- [ ] Needs another source

Decision details:

```text
Summary:

Required changes:

Reviewer notes:
```

If the decision is `Reject entry`, remove the entry from runtime YAML or keep the
rejection only in a non-runtime review log. Committed runtime YAML rejects
`review_status: rejected`.

## Implementation Action

Check all required follow-up actions:

- [ ] YAML metadata update required
- [ ] Tests update required
- [ ] Docs update required
- [ ] No runtime change required

Implementation notes:

```text
YAML changes:

Test changes:

Docs / issue / PR notes:
```

## Final Review State

```yaml
review_status: ""
citation_ready: false
source_type: []
source_ids: []
references: []
reference_urls: []
review_notes: ""
reviewed_by: ""
reviewed_at: ""
```
