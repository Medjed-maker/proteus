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

## 2. Phase 0: Core Refactor and Ancient Greek Pilot Stabilization

### Goal

Convert the current Ancient Greek prototype into:

language-independent core
+ ancient_greek plugin

### Required Work
- Introduce LanguageProfile
- Introduce language registry
- Move Ancient Greek-specific logic into an ancient_greek plugin
- Add language parameter to /search
- Keep backward compatibility with existing Ancient Greek search
- Separate generic IPA tokenization from Ancient Greek conversion
- Add toy_language fixture to test true language independence
- Add tests proving that core logic does not depend on Ancient Greek

### Acceptance Criteria
- Existing Ancient Greek tests pass
- /search works with language="ancient_greek"
- language can be omitted and defaults to ancient_greek
- toy_language works without modifying core search logic
- Core modules do not directly reference Ancient Greek-specific concepts
- README states that the project is a framework with an Ancient Greek pilot plugin

## 3. Phase 1: Rule Set v0.1 and Search Quality

### Goal
Create a usable Ancient Greek rule set and improve candidate ranking.

### Required Work
- Define Ancient Greek phonological rules in YAML/JSON
- Add schema validation for rules
- Add rule examples and references
- Improve phonological distance matrices
- Improve alignment output
- Improve candidate scoring
- Add benchmark test cases
- Document provisional status of rules
- Add basic citation metadata

### Acceptance Criteria
- At least 20 Ancient Greek phonological rules are represented
- Each rule has an ID, name, input, output, context, dialect, references, and examples
- Search results include applied rule IDs
- Search results include confidence levels
- Rule files pass schema validation in CI
- At least 5 representative Ancient Greek test cases pass

## 4. Phase 2: REST API and MCP Prototype

### Goal
Expose the search engine for external use and LLM grounding.

### Required Work
- Stabilize REST API response schema
- Add /languages endpoint
- Add /version endpoint
- Add MCP server prototype
- Implement MCP search tool
- Include engine version and ruleset version in responses
- Add verification URL or reproducibility metadata
- Document API usage

### Acceptance Criteria
- API returns structured candidates
- API returns applied rules and explanations
- MCP server can answer a query through the search engine
- MCP response includes candidates, confidence, applied rules, and metadata
- API and MCP output schemas are documented

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
- Prepare benchmark dataset
- Prepare documentation for scholarly collaborators

### Acceptance Criteria
- At least 20 real or semi-real hard query cases documented
- Each case includes input form, expected candidate, reasoning, and source notes
- Student-facing inscriptional orthography aid is clearly marked provisional
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

### Acceptance Criteria
- Search result can include source references
- Restricted corpora are linked rather than redistributed
- Data source attribution is documented
- Corpus adapter logic is separate from core search logic
- pre-403/2 BCE Attic orthographic notes use expert-reviewed inscriptional
  data sources and are not inferred from papyri.info metadata alone

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
