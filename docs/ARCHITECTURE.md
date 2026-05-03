# Architecture

## 1. Purpose

Proteus / HPSI is a language-independent framework for explainable reverse phonological search across historical languages.

The current implementation includes an Ancient Greek pilot plugin, but the long-term architecture is not limited to Ancient Greek. Ancient Greek is treated as the first language plugin used to validate the framework.

## 2. Architectural Principle

The system is divided into three layers:

1. Language-independent core
2. Language-specific plugins
3. Corpus adapters

The core must not contain Ancient Greek-specific logic. All language-specific behavior should be injected through language profiles and plugins.


HPSI / Proteus
├── Core framework
│   ├── search
│   ├── distance
│   ├── alignment
│   ├── scoring
│   ├── explanation
│   └── result formatting
│
├── Language plugins
│   ├── ancient_greek
│   │   ├── converter
│   │   ├── phoneme inventory
│   │   ├── rules
│   │   ├── matrices
│   │   ├── lexicon
│   │   └── orthographic note builder
│   └── future languages
│       ├── latin
│       ├── coptic
│       ├── akkadian
│       └── others
│
└── Corpus adapters
    ├── PHI
    ├── Perseus / Scaife
    ├── papyri.info
    ├── TLG
    ├── CIL / EDCS
    └── other corpora

## 3. Core Layer
The core layer contains reusable logic that should work across historical languages.

Core responsibilities:

- phonological distance calculation
- sequence alignment
- candidate generation
- candidate ranking
- rule matching
- explanation generation
- query normalization interface
- search result formatting
- API/MCP-compatible response structure

The core layer must not directly reference:

- Greek characters
- Ancient Greek dialect names
- PHI-specific data structures
- LSJ-specific assumptions
- language-specific phoneme inventories
- language-specific sound changes

## 4. Language Plugin Layer
Each language is implemented as a plugin.

A language plugin provides:

- language ID
- display name
- supported dialects
- default dialect
- orthography-to-phoneme converter
- phoneme inventory
- vowel/consonant definitions
- phonological rules
- phonological distance matrices
- lexicon data
- optional orthographic note builder
- examples and test cases

Example:

data/languages/ancient_greek/
├── profile.yaml
├── rules/
├── matrices/
├── lexicon/
├── orthography/
└── examples/

## 5. LanguageProfile
Each language must be registered through a LanguageProfile.

Required fields:

```python
@dataclass(frozen=True)
class LanguageProfile:
    language_id: str
    display_name: str
    default_dialect: str | None
    supported_dialects: tuple[str, ...]
    converter: Callable
    phone_inventory: frozenset[str]
    vowel_phones: frozenset[str]
    lexicon_path: Path
    matrix_path: Path
    rules_dir: Path
    orthographic_note_builder: OrthographicNoteBuilder | None = None
```

The search engine must receive all language-dependent behavior through this profile.

`orthographic_note_builder` is an optional language-specific hook. It may add
candidate-level comments about writing systems, spelling conventions,
normalized forms, or beginner reading aids. The core search engine treats the
hook as optional and receives an empty note list for languages that do not
provide one.

## 6. Corpus Adapter Layer
Corpus adapters connect the framework to external corpora or corpus-derived metadata.

Adapters should handle:

- source-specific IDs
- citation URLs
- text references
- license constraints
- metadata mapping
- source-specific normalization
- external links

The corpus adapter layer must be separate from the search core.

## 7. API Layer
The REST API exposes the core search engine.

Primary endpoints:

```text
POST /search
GET /languages
GET /languages/{language_id}
GET /health
GET /version
```

Search requests should include:

```json
{
  "query_form": "ΔΑΜΟΣΘΕΝΑΣ",
  "language": "ancient_greek",
  "dialect_hint": "attic",
  "max_candidates": 10
}
```
If language is omitted, the system may default to ancient_greek for backward compatibility during the pilot phase.

## 8. MCP Layer
The MCP server exposes the search engine as a tool for LLM clients.

Initial tool:
ancient_phonology.search

Future tool name:
historical_phonology.search
MCP responses should include:

- candidates
- scores
- applied rules
- dialect hypotheses
- citations
- rule IDs
- engine version
- ruleset version
- verification URL

## 9. Explanation Model
Every candidate should be explainable.

A search result should include:

- input form
- normalized form
- phoneme sequence
- candidate lemma
- alignment
- applied rules
- orthographic notes
- rule references
- score
- confidence level
- data version
- engine version

## 10. Design Constraints
The following are prohibited:

- hardcoding Ancient Greek logic in the core layer
- adding new languages by modifying core search logic
- storing third-party restricted corpus data without permission
- treating provisional rules as expert-reviewed data
- using LLM output as the primary source of scholarly authority

## 11. Current Status
This repository is a pre-alpha research prototype.

Current scope:

- Ancient Greek pilot plugin
- reverse phonological search
- rule-based explanation
- REST API prototype
- early MCP design

Out of scope for now:

- full corpus ingestion
- production authentication
- paid API access
- OCR/HTR
- automatic scholarly judgment
- complete multi-language support

## 12. Target Architecture
The target architecture is:

language-independent core
+ language plugins
+ corpus adapters
+ REST API
+ MCP server
+ future hosted execution layer

The long-term goal is to support historical phonological search across multiple ancient and historical languages.
