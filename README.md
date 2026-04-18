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
# Install dependencies for editable development
uv sync --all-extras

# Generate the lexicon before using /ready or /search
uv run --extra extract python -m phonology.build_lexicon --if-missing

# Run development server
uv run uvicorn api.main:app --reload

# Run tests
uv run pytest
```

If you skip the extraction step, the editable install still succeeds, but `/ready`
and `/search` return HTTP 503 until the lexicon is generated.

`bash scripts/extract-lsj.sh --if-missing` remains available as a convenience
wrapper on shell-friendly platforms, but the Python module command above is the
canonical cross-platform workflow, including native Windows environments.

Frontend styles are generated from `src/web/input.css` into
`src/web/static/styles.css` by `scripts/build-css.sh`. The generated
`styles.css` stays tracked because the packaged wheel serves it as a runtime
asset under `web/static`, so builds and installs do not depend on fetching
Tailwind at runtime.

```bash
# Regenerate the packaged frontend CSS after UI or utility class changes
bash scripts/build-css.sh
```

## Data Setup

The lexicon is generated from Perseus LSJ XML and not stored in the repository.
`uv build` reuses a fresh local `data/lexicon/greek_lemmas.json` when a matching
`data/lexicon/greek_lemmas.meta.json` exists. It regenerates the lexicon when a local
LSJ checkout is available. The build hook does not clone LSJ data during the build;
if the lexicon is missing or stale, provide a local checkout via the `PROTEUS_LSJ_REPO_DIR`
environment variable, or pre-generate the lexicon before building.
Generated `sdist` artifacts bundle the current `greek_lemmas.json` and
`greek_lemmas.meta.json` so `sdist -> wheel` rebuilds can stay offline. `wheel`
artifacts still include the lexicon JSON but omit the freshness metadata.

```bash
# Generate or refresh lexicon explicitly
# (--if-missing: skip generation if fresh output exists)
uv run --extra extract python -m phonology.build_lexicon [--xml-dir DIR] [--lsj-repo-dir DIR] [--if-missing]
```

The `--xml-dir` and `--lsj-repo-dir` options apply directly to
`python -m phonology.build_lexicon`. `scripts/extract-lsj.sh` forwards the same
arguments to that module, while `PROTEUS_LSJ_REPO_DIR` is read by both
`uv build` and the extraction workflow.

## API

### `POST /search`

```json
{
  "query": "ἄνθρωπος",
  "dialect": "attic",
  "max_results": 20
}
```


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
          "rule_name": "Ionic long alpha to eta shift",
          "from_phone": "aː",
          "to_phone": "ɛː",
          "position": 1
        }
      ],
      "explanation": "The match reflects the same lexical root with an Ionic vowel correspondence. The rules_applied entry records the segment-level eta shift behind that summary."
    }
  ]
}
```

`distance` is a normalized phonological distance on a 0.0-1.0 scale: `0.0` means an exact phonological match, and smaller values indicate closer similarity. As a rule of thumb, values below `0.2` are high-similarity matches, around `0.2-0.5` are plausible dialectal or historical matches, and values above `0.5` are relatively distant.

Each object in `rules_applied` records one explanatory rule step: `rule_id` is the stable identifier, `rule_name` is the display label, `from_phone` and `to_phone` capture the segment change, and `position` is the zero-based aligned phone index where that step applies.

`hits[].explanation` is the human-readable companion to `hits[].rules_applied` and the API field `SearchHit.explanation`. It is a readable plain-text string, not HTML or Markdown, because the packaged frontend renders it via `textContent` in `src/web/index.html`.

Typical `explanation` content is a short 1-2 sentence summary of why the hit is plausible: a compact etymological note, a summary of the rule sequence already listed in `rules_applied`, or a brief statement about the dialectal correspondence. It should stay concise, usually one short sentence and at most two, and it should not embed reference links, raw markup, or confidence-score fields. If richer provenance is needed later, add separate structured fields instead of overloading `explanation`.

### `GET /health`

Liveness probe — returns `{"status": "ok"}`.

## Deployment & Operations

- **Rate limiting**: Proteus intentionally does not ship an in-process rate
  limiter. When exposing the API publicly, enforce rate limits at the fronting
  proxy (nginx, Cloudflare, Vercel, etc.). A conservative starting point is
  `10 req/min/IP` on `/search`. `/health` and `/ready` should remain uncapped
  for orchestration probes. The `query` field is already length-bounded
  (≤64 chars) so the edit-distance cost is worst-case constant, but request
  volume still needs to be controlled upstream.
- **Interactive docs (`/docs`)**: Disabled by default. Set
  `PROTEUS_ENABLE_API_DOCS=1` in development or behind authenticated staging
  to enable the Swagger UI and the OpenAPI schema at `/openapi.json`.
  The env var is read at import time, so export it before starting the server.
  For production deployments, ensure these endpoints remain disabled or are
  blocked at the fronting proxy.
- **Security headers**: The app emits `X-Content-Type-Options`,
  `X-Frame-Options`, `Referrer-Policy`, and a restrictive `Permissions-Policy`
  on every response. HSTS and CSP are intentionally delegated to the fronting
  proxy that terminates TLS and knows the deployment's asset origins.

## Data Sources

- **Lexicon**: [LSJ — Liddell-Scott-Jones Greek-English Lexicon](http://stephanus.tlg.uci.edu/lsj/)
  — XML source provided by [Perseus Digital Library](http://www.perseus.tufts.edu/)
  ([PerseusDL/lexica](https://github.com/PerseusDL/lexica)); licensed
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Morpheus
  ([PerseusDL/morpheus](https://github.com/PerseusDL/morpheus)) is a separate
  morphological analysis system.
- **Phonology**: Allen, W.S. *Vox Graeca* (3rd ed., 1987); Horrocks, G. *Greek: A History of the Language and its Speakers* (2010)
- **IPA system**: Scholarly Ancient Greek pronunciation (Attic, c. 400 BCE default)
- **Rule notation**: See `docs/phonology_rules.md` for context notation used in the committed YAML rule files.

## License

MIT
