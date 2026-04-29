# Open-Core Strategy

## 1. Purpose

This document defines what parts of Proteus / HPSI are intended to be public, what parts may remain private or paid, and why.

Proteus / HPSI is not intended to be a closed black-box product. Scholarly trust requires transparency, reproducibility, citation, and community review.

At the same time, long-term sustainability may require paid hosted services, integrations, support, and expert-reviewed datasets.

The intended model is therefore:


open scholarly core
+ public rule specifications
+ selected public datasets
+ paid hosted execution and integration services

## 2. Why Open-Core
The project operates in a scholarly context where trust matters more than pure feature ownership.

Public development supports:

- scholarly review
- reproducibility
- transparent rule inspection
- citation
- collaboration
- grant applications
- community contributions
- long-term preservation

However, maintaining infrastructure, hosted APIs, MCP servers, institutional integrations, and expert-reviewed datasets requires sustainable funding.

## 3. Public Components
The following components are intended to be public, unless third-party licenses or collaborator agreements prevent publication.

### 3.1 Framework Code
Public:

- search core
- distance calculation
- alignment logic
- scoring framework
- explanation framework
- language plugin interface
- API schema
- MCP schema or prototype code
- tests
- documentation

License:

MIT. See the repository-root `LICENSE`.

### 3.2 Language Plugin Specifications
Public:

- plugin interface
- LanguageProfile schema
- example language plugin
- toy language fixture
- provisional Ancient Greek plugin structure

### 3.3 Rule Format
Public:

- YAML/JSON rule schema
- example rules
- validation logic
- rule documentation format

### 3.4 Selected Rule Sets
Public, with caution:

- provisional rule sets
- reviewed rule sets
- examples with proper attribution
- versioned releases

Rule sets should clearly state whether they are:

- provisional
- expert-reviewed
- citation-ready
- deprecated

### 3.5 Documentation
Public:

- architecture
- roadmap
- requirements
- contribution guide
- data license notes
- citation guidance
- API documentation

## 4. Private or Paid Components
The following components may remain private, restricted, or paid.

### 4.1 Hosted Execution Layer
Examples:

- hosted API
- high-throughput search
- managed database
- caching
- monitoring
- scaling
- uptime guarantees

Reason:

Running infrastructure creates ongoing cost.

### 4.2 MCP Server Deployment
The MCP protocol integration can be public, but managed deployment may be paid.

Examples:

- hosted MCP endpoint
- institution-specific MCP server
- authenticated access
- usage limits
- audit logs
- support

### 4.3 Institution-Specific Integrations
Examples:

- Scaife / Perseus integration
- papyri.info integration
- private corpus integration
- library system integration
- custom UI widgets

Reason:

Integrations require implementation, maintenance, support, and sometimes legal review.

### 4.4 Expert-Reviewed Datasets
Some datasets may be public, while others may be restricted.

Potential restricted data:

- expert-reviewed high-confidence rules
- collaborator-provided hard queries
- unpublished research examples
- private annotations
- evaluation data from research partners

### 4.5 Custom Language Development
Paid service examples:

- Latin plugin development
- Coptic plugin development
- Akkadian plugin development
- institution-specific rule set
- project-specific annotation pipeline

## 5. Data Boundary
The project must distinguish between:

1. code
2. rule specifications
3. rule data
4. lexicon data
5. corpus data
6. user-submitted queries
7. expert annotations
8. benchmark data

Each category may have a different license or access policy.

## 6. What Should Never Be Public Without Review
The following must not be committed to the public repository without explicit permission and license review:

- restricted corpus texts
- TLG data
- PHI data if redistribution is not permitted
- unpublished hard query examples from researchers
- personally identifying information
- private emails or correspondence
- collaborator notes not intended for publication
- API keys
- private benchmark results under embargo
- copyrighted dictionary data beyond permitted use
- LLM-generated scholarly claims presented as verified facts

## 7. Contribution Model
The project may accept contributions in the following areas:

- code improvements
- tests
- documentation
- rule corrections
- references
- examples
- language plugin proposals
- bug reports
- API feedback

Rule contributions should include:

- rule ID
- description
- input form
- output form
- context
- dialect
- period
- reference
- example
- confidence level
- contributor note

## 8. Citation and Versioning
Public rule sets should eventually be versioned and citable.

Possible release pattern:

hpsi-rules-ancient-greek-v0.1.0
hpsi-rules-ancient-greek-v0.2.0
hpsi-core-v0.1.0

Stable rule releases may be archived on Zenodo or a similar repository to obtain DOIs.

Pre-alpha rules should not be cited as authoritative.

## 9. Commercialization Boundary
The project may monetize:

- hosted API
- MCP hosting
- enterprise/institutional access
- integration support
- custom corpus adapters
- custom language plugins
- consulting
- expert-reviewed premium datasets
- training workshops
- maintenance contracts

The project should not monetize by hiding all scholarly logic. The core value proposition depends on transparency and trust.

## 10. Current Policy
Current repository status:

Pre-alpha public research prototype.

Current intended policy:

Public framework development with a provisional Ancient Greek pilot.
Restricted corpus redistribution and private hard query data are excluded from the repository.
No claim of expert-reviewed scholarly authority yet.

## 11. Future Review
This strategy should be reviewed before:

- first public release
- first DOI publication
- first external collaborator contribution
- first hosted API launch
- first institutional pilot
- first paid contract
