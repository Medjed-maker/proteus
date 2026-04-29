# Data License and Data Use Policy

## 1. Purpose

This document explains what data is included in this repository, how it may be used, and what data is intentionally excluded.

This project contains code, rule specifications, provisional linguistic data, examples, tests, and documentation. These categories may have different licensing and attribution requirements.

This document is not legal advice. Licenses and data permissions should be reviewed before any stable public release, institutional deployment, or commercial use.

This file lives at the repository root so public viewers can distinguish the code license from data, rules, examples, and generated artifacts.

---

## 2. Repository Data Categories

This repository may contain the following data categories.

### 2.1 Project Code

Examples:

- Python source code
- API code
- MCP server code
- tests
- utility scripts

License:

MIT. See `LICENSE`.

### 2.2 Rule Specifications

Examples:

- YAML/JSON schema for phonological rules
- example rule format
- validation logic
- documentation of required fields

Intended status:

Public.

Possible license:

CC BY 4.0, CC BY-SA 4.0, or the repository code license

Final license to be confirmed before stable release.

### 2.3 Provisional Rule Data
Examples:

- Ancient Greek sound change rules
- dialectal variation rules
- example transformations
- provisional references

Current status:

Provisional research data.
Not yet expert-reviewed.
Not citation-ready unless explicitly marked as a release.

Users should not treat provisional rules as authoritative scholarly claims.

### 2.4 Lexicon Data
Examples:

- lemma lists
- glosses
- part-of-speech labels
- source IDs
- generated test lexica
- toy language fixtures
- generated LSJ-derived lexicon artifacts included in wheel or sdist builds

Policy:

- Toy language data may be freely redistributed.
- Project-created minimal test data may be redistributed under this policy.
- Third-party lexicon data must follow the license of the original source.
- Restricted or copyrighted lexicon data must not be committed unless redistribution is permitted.
- LSJ-derived artifacts generated from PerseusDL/lexica are governed by CC BY-SA 4.0, including attribution and share-alike obligations.

### 2.5 Corpus Data
Examples:

- inscription texts
- papyrus texts
- literary passages
- corpus-derived examples
- source metadata

Policy:

Corpus data must follow the license and terms of the original corpus provider.

The repository should prefer:

- metadata
- source IDs
- external links
- small permitted examples
- generated fixtures
- citation references

The repository should avoid storing restricted corpus texts.

### 2.6 Hard Query Data
Hard queries are examples of difficult forms that researchers submit or that are collected during validation.

Policy:

Hard query data must not be committed publicly unless:

- the submitter has given permission
- the data does not reveal unpublished research
- personal information has been removed
- the source license allows publication
- the example has been reviewed for sensitivity

Private hard query collections should be stored outside the public repository.

### 2.7 Benchmark Data
Benchmark data may be public or restricted.

Public benchmark data should include:

- source
- license
- expected answer
- explanation
- version
- citation guidance

Restricted benchmark data should not be committed to the public repository.

## 3. Current Included Data
At the current pre-alpha stage, the repository may include:

- provisional Ancient Greek rule examples
- small test lexica
- generated fixtures
- toy language data
- example matrices
- generated LSJ-derived lexicon artifacts in distribution packages
- documentation
- tests

These are intended for development and demonstration.

Unless explicitly stated otherwise, these data should be considered:

provisional
not expert-reviewed
not citation-ready
not suitable as final scholarly evidence

## 4. Excluded Data
The following must not be included in the public repository without explicit permission and license review:

- TLG corpus data
- restricted PHI corpus data
- full copyrighted dictionaries
- non-redistributable lexicon data
- private researcher-submitted hard queries
- unpublished examples from collaborators
- personal information
- private correspondence
- API keys
- institution-specific data
- copyrighted text beyond permitted quotation or license terms

## 5. Third-Party Data
When using third-party data, contributors must record:

- source name
- source URL
- license
- attribution requirement
- redistribution permission
- transformation performed
- date accessed or generated
- script used to generate derived data

Recommended metadata format:

source:
  name: "Example Source"
  url: "https://example.org"
  license: "CC BY-SA 4.0"
  accessed: "YYYY-MM-DD"
  derived_by: "scripts/example_extractor.py"
  redistribution_allowed: true
  notes: "..."

## 6. Derived Data
Some files may be derived from public or licensed sources.

Examples:

- normalized lemma lists
- extracted metadata
- phoneme sequences
- distance matrices
- test cases

Derived data must preserve the license obligations of the original source.

If the original source does not allow redistribution, derived data should not be committed unless legally permitted.

## 7. Suggested Licensing Structure
Recommended structure:

LICENSE
  License for source code.

DATA_LICENSE.md
  Repository-root license and policy for data, rules, examples, and benchmarks.

NOTICE
  Third-party attribution and source notices.

docs/licensing.md
  More detailed licensing explanation if needed.

Possible division:

| Category | Suggested Treatment |
| --- | --- |
| Source code | MIT |
| Rule schema | Same as code or CC BY 4.0 |
| Provisional rules | CC BY 4.0 or CC BY-SA 4.0, pending decision |
| Expert-reviewed rules | Versioned release, possibly CC BY 4.0 |
| Toy fixtures | Public / permissive |
| Third-party lexica | Original license applies |
| Restricted corpora | Do not redistribute |
| Hard queries | Private unless permission is granted |
| Benchmarks | Case-by-case |

## 8. Attribution
Users of public rule sets or datasets should cite:

- project name
- version
- repository URL
- rule set version
- DOI, if available
- original data sources, if applicable

Suggested citation placeholder:

HPSI Project Contributors. Historical Phonological Search Infrastructure: Ancient Greek Ruleset, version 0.1.0. Repository: https://github.com/Medjed-maker/proteus

This citation format is provisional until a formal DOI release exists.

## 9. Contributor Requirements
Contributors who add data must confirm that:

1. they have the right to contribute the data;
2. the data does not violate third-party licenses;
3. the data does not contain private or unpublished research without permission;
4. the source is documented;
5. any required attribution is included;

Rule contributions should include references whenever possible.

## 10. Provisional Status Notice
All linguistic data in this repository is provisional unless explicitly marked otherwise.

This includes:

- phonological rules
- distance matrices
- example transformations
- dialectal labels
- confidence scores
- candidate explanations

Do not use provisional data as final scholarly authority without independent verification.

## 11. Future Releases
Before the first stable data release, the project should:

- choose a final data license
- audit third-party sources
- remove restricted data
- verify attribution requirements
- mark provisional vs reviewed rules
- prepare a DOI release if appropriate
- add a formal citation file, such as CITATION.cff

## 12. Contact
For licensing questions, open a GitHub issue or contact the project maintainer.
