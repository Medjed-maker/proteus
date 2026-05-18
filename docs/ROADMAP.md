# Roadmap

## 1. Project Direction

Proteus / HPSI is being developed as a language-independent framework for explainable reverse phonological search.

The current Ancient Greek implementation is a pilot, not the final product boundary.

The roadmap prioritizes:

1. separating the language-independent core from the Ancient Greek plugin
2. validating the search/explanation model with Ancient Greek
3. exposing the engine through REST API and MCP
4. preparing for additional languages
5. building toward scholarly review, citation, and institutional use

---

## 2. Phase 0: Core Refactor and Ancient Greek Pilot Stabilization ✅ COMPLETE

### Goal

Convert the current Ancient Greek prototype into:

language-independent core
+ ancient_greek plugin

### Completed Work
- Introduced LanguageProfile with lazy loading
- Introduced language registry in `profiles.py`
- Moved Ancient Greek-specific logic into `languages/ancient_greek/profile.py`
- Added language parameter to /search
- Maintained backward compatibility with existing Ancient Greek search at the public search/profile boundary
- Separated generic search/indexing/tokenization from Ancient Greek-specific conversion; Koine skeletons are supplied only by `LanguageProfile`
- toy_language fixture tests true language independence
- Core modules no longer eagerly import Ancient Greek-specific modules

### Acceptance Criteria Status
- Existing Ancient Greek tests pass (146 tests) — `uv run pytest -q` passes all search/api tests
- /search works with language="ancient_greek" — `tests/test_api_main.py::TestSearchEndpoint`
- language can be omitted and defaults to ancient_greek — `tests/test_api_main.py::TestSearchEndpoint`
- toy_language works without modifying core search logic — `tests/conftest.py` `isolated_language_registry` fixture
- Core search modules do not import Ancient Greek-specific conversion helpers; compatibility defaults are resolved through the default profile — `tests/test_search_pipeline.py`
- README states that the project is a framework with an Ancient Greek pilot plugin — `README.md`

## 3. Phase 1: Rule Set v0.1 and Search Quality ✅ COMPLETE

### Goal
Create a usable Ancient Greek rule set and improve candidate ranking.

### Completed Work
- Defined Ancient Greek phonological rules in YAML (consonant_changes.yaml, vowel_shifts.yaml, morphophonemic_alternations.yaml)
- Created standalone JSON Schema: `data/schemas/phonology_rule_file.schema.json`
- Created validation tool: `tools/validate_rule_files.py`
- Added schema validation tests in `tests/test_data_files.py` and focused negative tests for the validation helper
- All rule examples and references follow consistent format
- Phonological distance matrices improved and validated
- Alignment output and candidate scoring operational
- Benchmark test cases added
- Provisional status documented in rule metadata
- Citation metadata in place (`citation_ready: false` pending expert review)

### Deliverables
- **Schema**: `data/schemas/phonology_rule_file.schema.json` - Machine-readable rule schema for all languages, packaged with wheel/sdist artifacts
- **Validation Tool**: `tools/validate_rule_files.py` - Standalone CLI for rule validation
- **CI Integration**: Rule files validated via `test_rule_file_validates_against_schema` in test suite

### Acceptance Criteria Status
- 50+ Ancient Greek phonological rules represented (16 consonant, 21 vowel, 17 morphophonemic) — `data/languages/ancient_greek/rules/`
- Each rule has ID, name_en, name_ja, input, output, context, dialects, period, references, and examples — `data/schemas/phonology_rule_file.schema.json`
- Search results include applied rule IDs (`rules_applied`) — `tests/test_search.py`
- Search results include confidence levels (0.0-1.0 normalized) — `tests/test_search.py`
- Rule files pass schema validation in CI (`test_rule_file_validates_against_schema`) — `tests/test_data_files.py::test_rule_file_validates_against_schema`
- 5+ representative Ancient Greek test cases pass — `tests/test_api_main.py`

## 4. Phase 2: REST API and MCP Prototype ✅ COMPLETE

### Goal
Expose the search engine for external use and LLM grounding.

### Completed Work
- Stabilized REST API response schema with a `meta` envelope
- Added `/languages` endpoint
- Added `/version` endpoint
- Added MCP server prototype over stdio transport
- Implemented MCP `ancient_phonology.search` tool
- Included engine version and ruleset version in REST and MCP responses
- Added deterministic verification URL and reproducibility metadata
- Documented REST API and MCP usage, including generated schema artifacts

### Acceptance Criteria Status
- API returns structured candidates — `tests/test_api_main.py::TestSearchEndpoint`
- API returns applied rules and explanations — `tests/test_search.py`, `tests/test_search_annotation.py`
- MCP server can answer a query through the search engine — `tests/test_mcp_search_tool.py`
- MCP response includes candidates, confidence, applied rules, and metadata — `tests/test_mcp_search_tool.py`
- API and MCP output schemas are documented — `docs/API.md`, `docs/MCP.md`, `docs/api/openapi.json`, `docs/mcp/tools.json`

## 5. Phase 3: Scholarly Validation and Hard Query Collection

### Goal
Validate whether the tool solves real research pain.

### Required Work
- Collect hard queries from researchers
- Record cases where existing search fails
- Compare manual search vs tool-assisted search
- Track false positives and false negatives
- Improve rule set based on feedback
- Add student-facing inscriptional orthography aid for documented spelling systems
- Run an expert-review workflow for orthographic-note seeds using review packets
- Promote the first reviewed seed only after source identifiers, short
  references, reviewer metadata, and `citation_ready: true` are recorded
- Prepare benchmark dataset
- Prepare documentation for scholarly collaborators

### Acceptance Criteria
- At least 20 real or semi-real hard query cases documented
- Each case includes input form, expected candidate, reasoning, and source notes
- Student-facing inscriptional orthography aid is clearly marked provisional
- The first reviewed orthographic-note seed has a recorded reviewer decision
  before it is described as citation-ready
- Expert feedback is recorded separately from public code if needed
- Sensitive or unpublished research data is not committed to the public repository

## 6. Phase 4: Corpus Adapter Proof of Concept

### Goal
Connect search results to real source metadata without violating data licenses.

### Candidate Sources
- Perseus / Scaife
- papyri.info
- PHI Greek Inscriptions
- EAGLE / EpiDoc
- Morpheus / LSJ-derived lexicon data

### Required Work
- Define corpus adapter interface
- Add source metadata model
- Add external link support
- Avoid storing restricted source texts
- Add license notes per corpus
- Add example adapter for an open-source corpus
- Keep papyri.info / PHI / AIO ingestion automation as candidate generation
  until human review promotes entries into runtime orthographic-note YAML

### Acceptance Criteria
- Search result can include source references
- Restricted corpora are linked rather than redistributed
- Data source attribution is documented
- Corpus adapter logic is separate from core search logic
- pre-403/2 BCE Attic orthographic notes use expert-reviewed inscriptional
  data sources and are not inferred from papyri.info metadata alone
- Automatically ingested source metadata is not treated as citation-ready
  runtime note data without review.

## 7. Phase 5: Multi-Language Expansion

### Goal
Demonstrate that the framework can support another historical language.

### Candidate Languages
- Latin
- Coptic
- Ancient Semitic languages
- Akkadian
- Old English / Middle English

### Required Work
- Add second real language plugin
- Define converter, rules, matrix, and lexicon
- Add language-specific tests
- Compare implementation effort with Ancient Greek plugin
- Identify missing abstractions in the core

### Acceptance Criteria
- Second language works without major core changes
- Language plugin interface is sufficient
- Documentation explains how to add a new language
- Core architecture is validated as language-independent

## 8. Phase 6: Open-Core and Hosted Layer

### Goal
Prepare for sustainable operation.

### Possible Paid Components
- hosted API
- high-throughput search
- MCP deployment
- institution-specific integrations
- custom language plugin development
- expert-reviewed datasets
- SLA-backed institutional access

### Required Work
- Define public vs private components
- Define data licensing boundaries
- Add contribution guidelines
- Add security and privacy policy
- Prepare institutional documentation

## 9. Non-Goals
The following are not immediate roadmap goals:

- full automatic translation
- OCR/HTR
- automatic restoration of damaged inscriptions
- complete PHI/TLG corpus redistribution
- replacing expert judgment
- claiming scholarly authority before expert review
- supporting all ancient languages at once

## 10. Current Priority
The current priority is:

Refactor the current Ancient Greek prototype into a language-independent framework with an Ancient Greek pilot plugin.
This must be completed before adding major new features.
