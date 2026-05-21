# Proteus

Historical Phonological Search Infrastructure (HPSI)

Proteus is a language-independent framework for explainable reverse phonological search across historical languages. The project is also developed under the working name HPSI.

The current implementation includes PhonoTrace Engine and an Ancient Greek
pilot plugin. The pilot is intended to support both researchers working with
non-standard historical forms and students learning to read inscriptional
spelling conventions.

## Status

Pre-alpha research prototype.

This repository is not yet a production scholarly tool. The Ancient Greek
rules, matrices, examples, and orthographic note data are provisional and
require expert review before citation or research use.

## Open-core strategy

The framework code and selected rule/data specifications are developed in public to support scholarly review, reproducibility, and collaboration.

Hosted APIs, high-throughput execution, MCP deployment, institution-specific integrations, and custom language rule-set development may be offered as paid services in the future.

## Overview

Proteus implements a three-stage search pipeline inspired by NCBI BLAST, operating over phonological space rather than nucleotide sequences:

1. **Seed** — candidate words sharing a phoneme k-mer with the query
2. **Extend** — full weighted edit distance over IPA segments
3. **Filter** — rank by phonological distance, apply threshold

Phonological distances are computed through registered language profiles. The bundled Ancient Greek profile currently provides the Attic/Koine pilot data.

Candidate cards may also include `Orthographic note` entries. These are
student-facing and researcher-facing comments about writing systems, spelling
conventions, or normalized-form correspondences; they are kept separate from
phonological rule explanations.

Runtime orthographic notes carry internal review metadata in the packaged YAML,
but the public API currently exposes only display-oriented note fields such as
`references`. A note should not be treated as citation-ready unless the source
data explicitly records expert review and citation readiness. Non-empty
`references` are short source labels, not a guarantee that the full note has
passed expert review.

Citation-ready orthographic notes are a data-review roadmap item. The current
packaged Ancient Greek orthographic-note seeds remain provisional until their
source evidence, reviewer decision, and citation-ready metadata are recorded in
runtime data.

## Project Structure

```text
proteus/
├── data/
│   ├── languages/
│   │   └── ancient_greek/
│   │       ├── rules/           # YAML phonological change rules
│   │       ├── lexicon/         # LSJ headword list with IPA
│   │       ├── matrices/        # Phonological distance matrix
│   │       └── orthography/     # Provisional orthographic-note seeds
│   └── schemas/
│       └── phonology_rule_file.schema.json  # Machine-readable rule schema
├── docs/
│   ├── phonology_rules.md       # Rule context notation and examples
│   ├── OPEN_CORE_STRATEGY.md    # Public/private boundary notes
│   └── ROADMAP.md               # Research and product roadmap
├── src/
│   ├── phonology/
│   │   ├── profiles.py          # LanguageProfile registry (lazy loading)
│   │   ├── languages/
│   │   │   └── ancient_greek/   # Ancient Greek pilot plugin
│   │   │       └── profile.py   # LanguageProfile factory
│   │   ├── distance.py          # Weighted edit distance
│   │   ├── search.py            # Three-stage search (language-agnostic)
│   │   └── explainer.py         # Human-readable rule explanations
│   ├── api/
│   │   └── main.py              # FastAPI endpoints
│   └── web/
│       └── index.html           # Frontend
├── tests/
├── tools/
│   └── validate_rule_files.py   # Standalone rule validation CLI
├── DATA_LICENSE.md
├── LICENSE
├── NOTICE
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

# Validate rule files against JSON Schema
uv run python tools/validate_rule_files.py
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
`uv build` reuses a fresh local `data/languages/ancient_greek/lexicon/greek_lemmas.json` when a matching
`data/languages/ancient_greek/lexicon/greek_lemmas.meta.json` exists. It regenerates the lexicon when a local
LSJ checkout is available. The build hook does not clone LSJ data during the build;
if the lexicon is missing or stale, provide a local checkout via the `PROTEUS_LSJ_REPO_DIR`
environment variable, or pre-generate the lexicon before building.
Generated `sdist` artifacts bundle the current `greek_lemmas.json` and
`greek_lemmas.meta.json` so `sdist -> wheel` rebuilds can stay offline. `wheel`
artifacts still include the lexicon JSON but omit the freshness metadata. These
generated LSJ-derived artifacts remain subject to the PerseusDL/lexica
CC BY-SA 4.0 license and attribution requirements; see `DATA_LICENSE.md` and
`NOTICE`.

```bash
# Generate or refresh lexicon explicitly
# (--if-missing: skip generation if fresh output exists)
uv run --extra extract python -m phonology.build_lexicon [--xml-dir DIR] [--lsj-repo-dir DIR] [--if-missing]
```

The `--xml-dir` and `--lsj-repo-dir` options apply directly to
`python -m phonology.build_lexicon`. `scripts/extract-lsj.sh` forwards the same
arguments to that module, while `PROTEUS_LSJ_REPO_DIR` is read by both
`uv build` and the extraction workflow.

Runtime trusted-directory overrides for Buck rules and matrix assets are
disabled by default. To use `PROTEUS_TRUSTED_BUCK_DIR` or
`PROTEUS_TRUSTED_MATRICES_DIR`, you must also set
`PROTEUS_ALLOW_TRUSTED_DIR_OVERRIDES=1`. When the opt-in flag is absent,
Proteus rejects those override env vars instead of silently honoring them.
The override directory must be a real directory and must not itself be a
symlink, nor may any path components resolve through symlinks.

Example:

```bash
export PROTEUS_ALLOW_TRUSTED_DIR_OVERRIDES=1
export PROTEUS_TRUSTED_BUCK_DIR=/absolute/path/to/buck
export PROTEUS_TRUSTED_MATRICES_DIR=/absolute/path/to/matrices
uv run uvicorn api.main:app --host 127.0.0.1 --port 8000
```

If the opt-in flag is missing, Proteus rejects `PROTEUS_TRUSTED_BUCK_DIR` and
`PROTEUS_TRUSTED_MATRICES_DIR` during dependency loading with a clear
`...requires PROTEUS_ALLOW_TRUSTED_DIR_OVERRIDES=1` error instead of accepting
them silently. In that state, readiness and search dependency loading fail, so
`/ready` returns not-ready and `/search` cannot serve requests until the
configuration is fixed.

Validation also fails with clear errors when an override path does not exist, is
not a directory, or resolves through a symlinked path component. The symlink
restriction prevents path traversal and privilege-escalation risks via
symlink-swapped directories.

## REST API

Proteus exposes a Phase 2 REST API for search, language-profile discovery, and
runtime version metadata. Full endpoint and schema documentation lives in
[`docs/API.md`](docs/API.md), with the committed OpenAPI artifact at
[`docs/api/openapi.json`](docs/api/openapi.json).

日本語要約: REST API の詳細な schema とエラー仕様は `docs/API.md` に移動しました。
README では最小の利用例だけを示します。

### `POST /search`

```bash
curl -sS http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query_form":"λόγος","language":"ancient_greek","dialect_hint":"attic","max_candidates":5,"response_language":"en"}'
```

`language` selects the phonological profile and defaults to `"ancient_greek"`.
`response_language` selects generated prose language and accepts `"en"` or
`"ja"`. The older `lang` field remains accepted as an alias. For backward
compatibility, legacy `language: "en"` or `"ja"` is still treated as the
response prose language, but responses include deprecation headers; new clients
should use `response_language`.

Search responses include structured candidates in `hits`, explanatory
`rules_applied`, candidate-level `orthographic_notes`, and a Phase 2 `meta`
envelope with `api_version`, `schema_version`, `engine_version`, `request_id`,
`verification_url`, and `request_echo`. The top-level `data_versions` field is
kept for compatibility and mirrors `meta.data_versions`.

`distance` is a normalized phonological distance on a 0.0-1.0 scale: `0.0`
means an exact phonological match, and smaller values indicate closer
similarity. Values below `0.2` are high-similarity matches, around `0.2-0.5`
are plausible dialectal or historical matches, and values above `0.5` are
relatively distant.

`rules_applied` records phonological rule steps. `orthographic_notes` records
writing-system, spelling, or beginner-facing comments and should not be merged
into `rules_applied`.
For example, `παιδίο` can rank the current candidate `παιδίον` while the
orthographic note separately presents the alternative reading `παιδίου (paidiou)`.
In that display, `παιδίο -> παιδίον` is the search candidate chosen by ranking,
while `παιδίο -> παιδίου` is a writing-system reading aid. The normalized form
`παιδίου (paidiou)` is the later or standard-form reading; `παιδίο (paidiō)` is
shown as the pre-403/2 BCE Attic inscriptional spelling and reading.

### Discovery and probes

```bash
curl -sS http://127.0.0.1:8000/languages
curl -sS http://127.0.0.1:8000/version
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/ready
```

- `GET /languages` lists registered language profiles. Ancient Greek is the
  bundled pilot profile.
- `GET /version` returns engine, API, schema, build, Git, Python, and MCP
  server version metadata.
- `GET /health` is a liveness probe.
- `GET /ready` verifies that search dependencies are available.

**Response compatibility**

`SearchResponse` may gain new fields during Phase 2 without treating the
addition as a breaking API change. Field additions are considered compatible
because typical HTTP/JSON clients ignore unknown response keys by default.
Removing existing top-level fields or changing their types remains a breaking
change.

## MCP Server

Proteus also ships a Phase 2 MCP prototype over stdio transport. Full MCP
documentation lives in [`docs/MCP.md`](docs/MCP.md), and the committed tool
schema artifact lives at [`docs/mcp/tools.json`](docs/mcp/tools.json).

```bash
proteus-mcp
```

The current tool is `ancient_phonology.search`. It accepts `query_form`,
`source_language`, `dialect_hint`, `max_candidates`, and `response_language`,
then returns ranked `candidates` plus the shared `ResponseMeta` envelope.

Claude Desktop configuration example:
[`docs/mcp/example-claude-desktop.json`](docs/mcp/example-claude-desktop.json).

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
- **`PROTEUS_PUBLIC_BASE_URL`**: Optional. Sets the absolute base URL used to
  build `meta.verification_url` in `/search` responses (a deterministic URL
  that reproduces the request). Must be an absolute URL with scheme and host
  and no query/fragment, e.g. `https://proteus.example`. Invalid values
  (relative URLs, embedded query strings) abort startup with a clear
  `RuntimeError`. When unset, Proteus falls back to the request's own
  `base_url`, which is sufficient for development.
- **Response metadata vs. server-side query redaction**: `meta.request_echo`
  and `meta.verification_url` always carry the client-supplied
  `query_form`/`q` so the request is reproducible. They are independent of
  `PROTEUS_LOG_RAW_SEARCH_QUERY`, which only redacts the raw query from
  *server logs*. A reverse proxy or browser extension that logs response
  bodies will still observe the raw query. If response-side redaction is
  required, strip these fields at the fronting proxy.

## Data Sources

- **Lexicon**: [LSJ — Liddell-Scott-Jones Greek-English Lexicon](http://stephanus.tlg.uci.edu/lsj/)
  — XML source provided by [Perseus Digital Library](http://www.perseus.tufts.edu/)
  ([PerseusDL/lexica](https://github.com/PerseusDL/lexica)); licensed
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Morpheus
  ([PerseusDL/morpheus](https://github.com/PerseusDL/morpheus)) is a separate
  morphological analysis system.
- **Phonology**: Allen, W.S. _Vox Graeca_ (3rd ed., 1987); Horrocks, G. _Greek: A History of the Language and its Speakers_ (2010)
- **IPA system**: Scholarly Ancient Greek pronunciation (Attic, c. 400 BCE default)
- **Rule notation**: See `docs/phonology_rules.md` for context notation used in the committed YAML rule files.

## License

- **Source code**: MIT. See `LICENSE`.
- **Rule specifications and provisional rule data**: see `DATA_LICENSE.md`.
- **LSJ-derived lexicon artifacts**: generated from PerseusDL/lexica and
  distributed under CC BY-SA 4.0 when included in wheel or sdist artifacts.
- **Restricted corpora, private hard queries, unpublished collaborator data**:
  not included in this public repository.

![CodeRabbit Pull Request Reviews](https://img.shields.io/coderabbit/prs/github/Medjed-maker/proteus?utm_source=oss&utm_medium=github&utm_campaign=Medjed-maker%2Fproteus&labelColor=171717&color=FF570A&link=https%3A%2F%2Fcoderabbit.ai&label=CodeRabbit+Reviews)
