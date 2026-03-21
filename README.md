# Proteus

Ancient Greek phonological search engine — find lexically related forms across dialects and time periods using BLAST-like phonological alignment.

## Overview

Proteus implements a three-stage search pipeline inspired by NCBI BLAST, operating over phonological space rather than nucleotide sequences:

1. **Seed** — candidate words sharing a phoneme k-mer with the query
2. **Extend** — full weighted edit distance over IPA segments
3. **Filter** — rank by phonological distance, apply threshold

Phonological distances are computed using dialect-aware rules drawn from comparative Greek linguistics (Attic, Ionic, Doric, Koine).

## Project Structure

```
proteus/
├── data/
│   ├── rules/ancient_greek/     # YAML phonological change rules
│   │   ├── vowel_shifts.yaml
│   │   └── consonant_changes.yaml
│   ├── lexicon/
│   │   └── greek_lemmas.json    # LSJ headword list with IPA
│   └── matrices/
│       └── attic_doric.json     # Phonological distance matrix
├── docs/
│   └── phonology_rules.md       # Rule context notation and examples
├── src/
│   └── proteus/
│       ├── phonology/
│       │   ├── ipa_converter.py # Greek script → IPA
│       │   ├── distance.py      # Weighted edit distance
│       │   ├── search.py        # Three-stage search
│       │   └── explainer.py     # Human-readable rule explanations
│       ├── api/
│       │   └── main.py          # FastAPI endpoints
│       └── web/
│           └── index.html       # Frontend
├── tests/
├── pyproject.toml
└── README.md
```

## Requirements

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) (recommended)

## Setup

```bash
# Install dependencies
uv sync --all-extras

# Run development server
uv run uvicorn proteus.api.main:app --reload

# Run tests
uv run pytest
```

## API

### `POST /search`

```json
{
  "query": "ἄνθρωπος",
  "dialect": "attic",
  "max_results": 20
}
```

`/search` is not implemented yet and currently returns HTTP 501. The example below documents the response model shape that the API advertises once the search backend is wired in.

**Response Model Example**

```json
{
  "query": "ἄνθρωπος",
  "query_ipa": "ántʰrɔːpos",
  "hits": [
    {
      "headword": "ἀνήρ",
      "ipa": "anɛːr",
      "distance": 0.18,
      "rules_applied": [
        {
          "rule_id": "VSH-001",
          "rule_name": "Attic-Ionic long alpha to eta shift",
          "from_phone": "aː",
          "to_phone": "ɛː",
          "position": 1
        }
      ],
      "explanation": "The match reflects the same lexical root with an Attic-Ionic vowel correspondence. The rules_applied entry records the segment-level eta shift behind that summary."
    }
  ]
}
```

`distance` is a normalized phonological distance on a 0.0-1.0 scale: `0.0` means an exact phonological match, and smaller values indicate closer similarity. As a rule of thumb, values below `0.2` are high-similarity matches, around `0.2-0.5` are plausible dialectal or historical matches, and values above `0.5` are relatively distant.

Each object in `rules_applied` records one explanatory rule step: `rule_id` is the stable identifier, `rule_name` is the display label, `from_phone` and `to_phone` capture the segment change, and `position` is the zero-based aligned phone index where that step applies.

`hits[].explanation` is the human-readable companion to `hits[].rules_applied` and the API field `SearchHit.explanation`. It is a readable plain-text string, not HTML or Markdown, because the packaged frontend renders it via `textContent` in `src/proteus/web/index.html`.

Typical `explanation` content is a short 1-2 sentence summary of why the hit is plausible: a compact etymological note, a summary of the rule sequence already listed in `rules_applied`, or a brief statement about the dialectal correspondence. It should stay concise, usually one short sentence and at most two, and it should not embed reference links, raw markup, or confidence-score fields. If richer provenance is needed later, add separate structured fields instead of overloading `explanation`.

### `GET /health`

Liveness probe — returns `{"status": "ok"}`.

## Data Sources

- **Lexicon**: [LSJ — Liddell-Scott-Jones Greek-English Lexicon](http://stephanus.tlg.uci.edu/lsj/)
- **Phonology**: Allen, W.S. *Vox Graeca* (3rd ed., 1987); Horrocks, G. *Greek: A History of the Language and its Speakers* (2010)
- **IPA system**: Scholarly Ancient Greek pronunciation (Attic, c. 400 BCE default)
- **Rule notation**: See `docs/phonology_rules.md` for context notation used in the committed YAML rule files.

## License

MIT
