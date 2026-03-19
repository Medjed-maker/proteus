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
├── src/
│   ├── phonology/
│   │   ├── ipa_converter.py     # Greek script → IPA
│   │   ├── distance.py          # Weighted edit distance
│   │   ├── search.py            # Three-stage search
│   │   └── explainer.py         # Human-readable rule explanations
│   ├── api/
│   │   └── main.py              # FastAPI endpoints
│   └── web/
│       └── index.html           # Frontend
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
uv run uvicorn src.api.main:app --reload

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

**Response**

```json
{
  "query": "ἄνθρωπος",
  "query_ipa": "ántʰrɔːpos",
  "hits": [
    {
      "headword": "ἀνήρ",
      "ipa": "anɛːr",
      "distance": 0.42,
      "rules_applied": [...],
      "explanation": "Shares the root *aner- ..."
    }
  ]
}
```

### `GET /health`

Liveness probe — returns `{"status": "ok"}`.

## Data Sources

- **Lexicon**: [LSJ — Liddell-Scott-Jones Greek-English Lexicon](http://stephanus.tlg.uci.edu/lsj/)
- **Phonology**: Allen, W.S. *Vox Graeca* (3rd ed., 1987); Horrocks, G. *Greek: A History of the Language and its Speakers* (2010)
- **IPA system**: Scholarly Ancient Greek pronunciation (Attic, c. 400 BCE default)

## License

MIT
