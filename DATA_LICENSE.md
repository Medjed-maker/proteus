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

Phase 4 corpus adapter proof-of-concept data follows this policy by storing
only the following under `data/languages/*/corpus_sources/`:

- source identifiers
- external links
- license notes
- review status
- short citations (limited to 200 characters or 40 words, whichever is shorter)

Short citations may include:
- titles and authors
- publication years
- DOIs
- brief bibliographic metadata (under 50 characters)
- a single sentence, only if the entire citation including all metadata remains under the 200 character or 40 word limit

It must NOT store:
- source text, passage text, or evidence excerpts
- multi-sentence passages or full paragraphs
- long quotations
- any excerpt exceeding the short-citation limit

Citations over that limit must be converted to metadata-only entries (source identifiers, links, and license notes only).

Metadata ingested from papyri.info, PHI, AIO, or similar services may support candidate
generation, but it is not citation-ready runtime note data until reviewed.

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

### 5.1 Benchmark Sources

Used by `tools/benchmarks/dilemma/` and documented in
`docs/benchmarks/dilemma_papygreek_baseline.md` and
`docs/benchmarks/reverse_lookup_baseline_ladder.md`.

    source:
      name: "PapyGreek Treebanks v1.01"
      url: "https://doi.org/10.5281/zenodo.5074307"
      license: "CC BY-SA 4.0"
      attribution: "Vierros, Marja & Erik Henriksson, University of Helsinki"
      citation: "https://doi.org/10.5334/johd.55"
      accessed: "2026-08-10"
      derived_by: "tools/benchmarks/dilemma/build_dataset.py"
      redistribution_allowed: true
      notes: >
        Evaluation only. The bulk derived dataset (dataset.jsonl, 33,615 tokens
        stratified by whether the spelling needed editorial regularisation) and
        the gold POS dump (postags.jsonl) are NOT committed to this repository;
        regenerate both from the Zenodo archive.

        The rule for everything else: any committed file under
        tools/benchmarks/dilemma/ or docs/benchmarks/ that carries PapyGreek
        surface forms, lemmas or document filenames is a CC BY-SA 4.0
        derivative, redistributed here under that licence with the attribution
        above. Share-alike therefore binds those files and anything derived
        from them. It does not reach the repository's own code, which is
        separately licensed (see LICENSE), nor the purely numeric result
        summaries.

        "Document filenames" means PapyGreek's own EpiDoc filenames --
        bgu.16.2604.xml and the 394 others enumerated in papygreek_ids.json.
        A bare Trismegistos number is deliberately not a marker of file
        content: TM ids are 4-6 digit integers, they collide with the counts
        and percentages that fill the numeric result summaries, and treating
        them as content would sweep in files carrying nothing from PapyGreek.
        (This is separate from the DDbDP entry below, where TM numbers are
        used as EpiDoc *filenames* to exclude PapyGreek documents from a
        corpus. A filename is an identifier; an integer in a results table is
        not.)

        As of 2026-08-14 the CC BY-SA 4.0 derivatives committed here are:

          `docs/benchmarks/dilemma_papygreek_baseline.md`
              Greek forms quoted in the write-up
          `docs/benchmarks/reranking_and_statistics.md`
              same
          `docs/benchmarks/reverse_lookup_baseline_ladder.md`
              same
          `tools/benchmarks/dilemma/README.md`
              PapyGreek forms quoted while documenting the cleaning layer
          `tools/benchmarks/dilemma/preregistration_3rd.md`
              PapyGreek and DDbDP forms quoted throughout the hypotheses and
              their worked examples
          `tools/benchmarks/dilemma/preregistration_4th.md`
              same
          `tools/benchmarks/dilemma/categories.json`
              per-alternation counts plus up to 8 worked examples each (~1,800
              orig -> reg pairs with gold lemma); the evidence behind section 5
              of dilemma_papygreek_baseline.md
          `tools/benchmarks/dilemma/alternations_undescribed.json`
              same shape for the alternations no rule covers, with document
              filenames
          `tools/benchmarks/dilemma/papygreek_ids.json`
              the 395 evaluated documents: PapyGreek filename, Trismegistos id,
              series and date range. Carries no PapyGreek text -- it is listed
              because the filenames are themselves PapyGreek's. (The one Greek
              string in it is a GLAUx work title; see the GLAUx entry.)
          `tools/benchmarks/dilemma/splits.json`
              the frozen dev/test split: 88 dev documents of 310, named by
              PapyGreek filename
          `tools/benchmarks/dilemma/results_b0.json`
              B0 miss list (~239 forms)
          `tools/benchmarks/dilemma/results_decision_B3u.json`
              40+40 worked examples, form/reg/gold
          `tools/benchmarks/dilemma/results_propnoun_test.json`
              capitalisation buckets (~154 forms)

        The remaining committed results_*.json and ddb_*.json files are numeric
        and carry no PapyGreek content. The two per-token dumps,
        results_b3u.json and results_strict_fixed.json, are deliberately not
        committed (see .gitignore).

        The list covers data and prose, not the benchmark scripts. Several
        rlb_*.py files do contain Greek, but illustratively: phone and vowel
        inventories, generic example words, standard lexical pairs, and a
        handful of forms cited in a comment to justify a parsing decision --
        some of them from DDbDP (CC BY 3.0) rather than PapyGreek. The scripts
        are the repository's own code, separately licensed (see LICENSE), and
        declaring them would encumber that code rather than protect anyone's
        attribution. The exemption is bounded, not assumed: the test below also
        asserts that no committed script names an evaluated PapyGreek document
        or accumulates Greek at dataset scale.

        This list is not maintained by hand. `tests/test_benchmark_licence_manifest.py`
        re-derives it from the two markers above over everything git would
        commit, and fails if a file is missing from the list or listed but no
        longer committed. Run `uv run pytest tests/test_benchmark_licence_manifest.py`
        after adding any artefact.

    source:
      name: "Dilemma data layer (lookup.db, spell_index.db, corpus_freq.json,
              lemma_attestation.json)"
      url: "https://github.com/open-greek/dilemma"
      license: "MIT"
      accessed: "2026-08-11"
      commit: "f82f15a62ddce5d55c19b299c34a6c89476af5ce"
      derived_by: "tools/benchmarks/dilemma/rlb_build.py"
      redistribution_allowed: true
      notes: >
        Read-only, from the user's own `python -m dilemma download` cache.
        Nothing from it is committed here. The frequency tables derive from
        GLAUx, Diorisis and PatristicTextArchive, none of which overlaps
        PapyGreek -- which is what keeps the benchmark's ranking signal free of
        evaluation-set leakage.

    source:
      name: "GLAUx (Greek Language Automated Text Corpus)"
      url: "https://github.com/alekkeersmaekers/glaux"
      license: "CC BY-SA 4.0"
      attribution: "Alek Keersmaekers, KU Leuven"
      accessed: "2026-08-11"
      derived_by: "tools/benchmarks/dilemma/rlb_lm.py (fetch / local)"
      redistribution_allowed: true
      notes: >
        Fetched directly as XML, not only transitively through Dilemma's
        frequency tables, and used to train the context trigram LM. Only lemma
        sequences are extracted; the derived corpus is not committed. Verified
        in `docs/benchmarks/reranking_and_statistics.md` §3.1 to contain no
        papyrus or documentary genre and therefore no PapyGreek overlap -- the
        37 apparent Trismegistos-id matches are a namespace collision (TM 705
        is Demosthenes in GLAUx and a Zeno-archive papyrus in PapyGreek).
        GLAUx ships NFD where everything else here is NFC.

    source:
      name: "Duke Databank of Documentary Papyri (DDbDP), via papyri/idp.data"
      url: "https://github.com/papyri/idp.data"
      license: "CC BY 3.0"
      attribution: "Duke Databank of Documentary Papyri, papyri.info"
      accessed: "2026-08-12"
      commit: "2249c22c92f634f74d6bea58ff828c68cf0bffa0"
      derived_by: "tools/benchmarks/dilemma/rlb_ddb.py"
      redistribution_allowed: true
      notes: >
        67,980 EpiDoc documents, sparse shallow checkout of the DDbDP subtree
        only. Two derived artefacts, neither committed: an in-domain lemma
        corpus for the context LM, and an orig->reg normalisation benchmark
        built from 160,815 <choice><reg>/<orig> pairs. Filenames are
        Trismegistos numbers, so PapyGreek exclusion is a filename test;
        both derivations additionally drop every document sharing a
        publication volume with a PapyGreek text (13.7% of documents), as a
        proxy for archive-level separation. Provenance is recorded in
        `ddb_manifest.json`. Being CC BY rather than CC BY-SA, this benchmark
        carries no share-alike obligation onto derived rulesets.

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
