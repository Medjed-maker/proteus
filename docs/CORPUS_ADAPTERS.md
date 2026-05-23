# Corpus Adapters

Corpus adapters attach external source metadata to search results after the
core phonological search has ranked and explained candidates. They do not affect
ranking, distance calculation, rule application, or orthographic-note logic.

## Runtime Contract

Adapters implement `CorpusAdapter.lookup(entry_id, headword, language)` and
return `SourceReference` records. A source reference contains:

- `source_id`
- `corpus`
- `short_citation`
- `external_url`
- `license_note`
- `access_policy`
- `citation_ready`

The runtime catches adapter lookup failures, logs a warning, and returns an
empty `source_references` list for that candidate. This keeps source enrichment
separate from the core search path.

## Static Metadata Format

The Phase 4 proof of concept uses
`data/languages/ancient_greek/corpus_sources/perseus_scaife_sources.yaml`.
The file maps lexicon `entry_id` values to one or more public source references.
It is validated by `data/schemas/corpus_source_reference.schema.json`.

This file is metadata-only. It may contain stable identifiers, short citations,
provider landing-page links, and license notes. It must not contain source text,
passage text, evidence excerpts, or long quotations.

## URL and Identifier Encoding

Both `source_id` and `external_url` use **percent-encoded** representations of
special characters. This matches the link form that Perseus and Scaife-aware
tools emit (for example `entry=lo%2Fgos`, not `entry=lo/gos`) and keeps
identifiers and link targets consistent within one record. When ingesting new
sources, normalise URN-like identifiers to the same encoding as the URL so
clients can reconstruct one from the other without ambiguity.

## Licensing Boundary

Restricted corpora are linked rather than redistributed. Runtime search results
may point to a provider page, but Proteus does not copy the provider's source
text into API, MCP, YAML, fixture, or test outputs.

Perseus / Scaife references in the PoC are not a claim that every candidate is
citation-ready. `citation_ready: false` remains the default until a source
reference has been reviewed for the relevant scholarly use.

## Candidate Generation Boundary

papyri.info, PHI, AIO, and similar sources may be used later for candidate
generation or reviewer workflow support. Automatically ingested metadata from
those sources must not become runtime citation-ready note data without human
review. Pre-403/2 BCE Attic orthographic notes still require expert-reviewed
inscriptional evidence before promotion.
