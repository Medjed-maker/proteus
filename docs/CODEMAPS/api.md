# API Codemap

**Last Updated:** 2026-05-17
**Entry Points:** `src/api/main.py` (FastAPI app instance)

## Architecture

```
HTTP REQUEST
    ↓
┌─────────────────────────────────────────────────────────────┐
│  Request ID Middleware (_add_request_id)                      │
│  - Accepts valid X-Request-ID or generates UUID4 hex          │
│  - Adds X-Request-ID to every response                        │
├─────────────────────────────────────────────────────────────┤
│  Security Headers Middleware                                  │
│  - X-Content-Type-Options, X-Frame-Options, Referrer-Policy  │
├─────────────────────────────────────────────────────────────┤
│  CORS Middleware (CORSMiddleware)                            │
│  - Allow-Origins from PROTEUS_ALLOWED_ORIGINS env var       │
│  - Methods: GET, POST, OPTIONS                              │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│  FASTAPI ROUTES                                             │
├─────────────────────────────────────────────────────────────┤
│ GET  /              → root()         → HTMLResponse (frontend)
│ GET  /changelog     → changelog()    → HTMLResponse (changelog)
│ POST /search        → search()       → SearchResponse JSON
│ GET  /languages     → languages()    → LanguagesResponse JSON
│ GET  /version       → version()      → VersionInfo JSON
│ GET  /health        → health()       → {"status": "ok"}
│ GET  /ready         → ready()        → {"status": "ok"}
│ GET  /static/*      → StaticFiles    → CSS, JS, images
│ GET  /docs          → Swagger UI     → gated by PROTEUS_ENABLE_API_DOCS
└─────────────────────────────────────────────────────────────┘
    ↓
HTTP RESPONSE (JSON or HTML)
```

## Endpoints

| Endpoint     | Method   | Handler                           | Input                      | Output                         | Behavior                                                                                     |
| ------------ | -------- | --------------------------------- | -------------------------- | ------------------------------ | -------------------------------------------------------------------------------------------- |
| `/`          | GET/HEAD | `root()` / `root_head()`          | None                       | `HTMLResponse` / empty response | Serves `src/web/index.html` (frontend)                                                       |
| `/changelog` | GET/HEAD | `changelog()` / `changelog_head()` | None                       | `HTMLResponse` / empty response | Serves packaged changelog HTML                                                               |
| `/search`    | POST     | `search()`                        | `SearchRequest` (Pydantic) | `SearchResponse` (Pydantic)     | Three-stage pipeline with full dependency loading, query normalization, dialect-aware search |
| `/languages` | GET/HEAD | `languages()` / `languages_head()` | None                       | `LanguagesResponse`             | Lists registered language profiles and runtime metadata                                      |
| `/version`   | GET/HEAD | `version()` / `version_head()`    | None                       | `VersionInfo`                   | Returns engine/API/schema/build/runtime version metadata                                     |
| `/health`    | GET/HEAD | `health()` / `health_head()`      | None                       | `{"status": "ok"}`              | Liveness probe                                                                               |
| `/ready`     | GET      | `ready()`                         | None                       | `{"status": "ok"}`              | Readiness probe (checks all search dependencies loaded; returns 503 if not)                  |
| `/static/*`  | GET      | StaticFiles                       | Any                        | Static file                     | Maps `src/web/static/` directory                                                             |

## Pydantic Models

### Inbound (Request)

**`SearchRequest`**

```
- query_form: str (non-empty after strip_whitespace)
- language: str = "ancient_greek"
- dialect_hint: Literal["attic", "koine"] = "attic"
- max_candidates: int (default 20, range 1-100)
- response_language: Literal["en", "ja"] = "en"
- orthography_hint: str | None (deprecated; accepted but ignored)
```

### Outbound (Response)

**`SearchResponse`**

```
- query: str (the original query)
- query_ipa: str (normalized IPA transcription)
- query_mode: Literal["Full-form", "Short-query", "Partial-form"]
- hits: list[SearchHit]
- truncated: bool
- data_versions: DataVersions
- meta: ResponseMeta
```

**`ResponseMeta`**

```text
- api_version: str
- schema_version: str
- engine_version: str
- data_versions: DataVersions
- ruleset_versions: dict[str, str]
- request_id: str
- timestamp: str
- verification_url: str
- request_echo: RequestEcho | None
```

Note: `SearchResponse` includes a top-level `data_versions` field that mirrors `meta.data_versions` for backward compatibility. This duplicate is intentional.

**`RequestEcho`**

```text
- query_form: str
- language: str
- dialect_hint: str
- max_candidates: int
- response_language: Literal["en", "ja"]
```

**`LanguageInfo`**

```text
- language_id: str
- display_name: str
- default_dialect: str
- supported_dialects: list[str]
- status: Literal["pilot", "experimental", "stable"]
- ruleset_version: str
- lexicon_schema_version: str
- matrix_version: str
- description: str
```

**`LanguagesResponse`**

```text
- languages: list[LanguageInfo]
- meta: VersionInfo
```

**`VersionInfo`**

```text
- engine_version: str
- api_version: str
- schema_version: str
- rule_schema_version: str
- build_timestamp: str
- git_sha: str
- python_version: str
- mcp_server_version: str
```

**`SearchHit`**

```
- headword: str (display headword)
- ipa: str (IPA for lexicon entry)
- distance: float (normalized phonological distance, 0.0-1.0)
- confidence: float (0.0-1.0 normalized distance as confidence)
- dialect_attribution: str (e.g., "Attic", "Doric")
- alignment_visualization: str (aligned query/headword pair visualization)
- match_type: Literal["Exact", "Rule-based", "Distance-only", "Low-confidence"]
- rule_support: bool (at least one explicit catalogued rule supports the candidate)
- applied_rule_count: int (count of explicit catalogued rules applied)
- observed_change_count: int (count of uncatalogued observed changes)
- alignment_summary: str (human-readable diff summary)
- why_candidate: list[str] (short bullet points explaining why the candidate ranked highly)
- uncertainty: Literal["Low", "Medium", "High"]
- candidate_bucket: Literal["Supported", "Exploratory"]
- rules_applied: list[RuleStep]
- orthographic_notes: list[OrthographicNote]
- explanation: str (human-readable prose summary of the derivation)
```

**`candidate_bucket` determination**

The bucket is computed in two steps:

1. *Default bucket* (`_build_candidate_bucket` in `api/_hit_formatting.py`):
   - Short-query / Partial-form mode: "Supported" for Exact or Rule-based matches only; otherwise "Exploratory"
   - Full-form mode: "Supported" for Exact, Rule-based, or Low-uncertainty matches; otherwise "Exploratory"

2. *Orthographic promotion* (`_promote_bucket_for_orthographic_notes`):
   - If `orthographic_notes` contains any note with `kind == "orthographic_correspondence"`, the candidate is unconditionally promoted to "Supported" regardless of step 1.
   - `beginner_aid` and `pre_403_2_attic` notes do **not** affect the bucket.

**`RuleStep`** (element of `rules_applied`)

```
- rule_id: str (unique rule identifier)
- rule_name: str (human-readable rule name)
- rule_name_en: str (English display name for the rule)
- from_phone: str (IPA phone that changed)
- to_phone: str (IPA phone it became)
- position: int (alignment position, or -1 if unknown)
```

**`OrthographicNote`** (element of `orthographic_notes`)

```text
- kind: Literal["orthographic_correspondence", "beginner_aid", "pre_403_2_attic"]
- label: str (short display label)
- messages: list[str] (candidate-level writing-system or spelling comments)
- normalized_form: str | None
- romanization: str | None
- period_label: str | None
- references: list[str]
- confidence: Literal["low", "medium", "high"]
- pre_reform_spelling: str | None
- pre_reform_romanization: str | None
```

`orthographic_notes` is separate from `rules_applied`: it explains writing
systems, spelling conventions, normalized-form correspondences, and
beginner-facing reading aids rather than phonological rule steps. However,
`orthographic_correspondence` notes are an exception: they affect
`candidate_bucket` (see **`candidate_bucket` determination** above).
For `παιδίο`, the current candidate can remain `παιδίον` while the note
separately presents the alternative orthographic reading `παιδίου (paidiou)`,
and the optional `pre_reform_spelling` / `pre_reform_romanization` fields can
expose the pre-403/2 BCE Attic inscriptional form `παιδίο (paidiō)` alongside
it.

`references` contains short citation strings loaded from runtime orthography
data when available. These strings are intentionally URL-free; runtime data
stores verification links separately in `reference_urls`, which is not exposed
in the current API shape. The API keeps those evidence-management fields
private until the publication policy for provisional versus citation-ready
notes is decided. Entry-level review metadata such as `review_status` and
`citation_ready` is validated at load time but is not exposed in this API shape.
An `orthographic_notes` entry, even one with non-empty `references`, is
therefore not automatically citation-ready.

## Data Flow: POST /search

```
HTTP POST { "query_form": "λόγος", "dialect_hint": "attic", "max_candidates": 5 }
    ↓ Pydantic validation
SearchRequest(query_form="λόγος", dialect_hint="attic", max_candidates=5)
    ↓
search(request: SearchRequest):
    1. api._dependencies._load_search_dependencies()
       → orchestrates cached sub-loaders
       → Lexicon (tuple of dicts)
       → Distance matrix (nested dict)
       → Rules registry (YAML-parsed)
       → Search index (k-mer → headwords)
       → Unigram index (k=1, fallback)
       → Lexicon map (entry_id → LexiconRecord)
    2. normalize_query_for_search(request.query_form)
       → "logos"
    3. to_ipa("logos", dialect="attic")
       → "lɔːɡos" (for distance calculation)
    4. classify_query_mode("logos")
       → "Full-form"
    5. phonology_search.search(
         query_form, lexicon, matrix, index, ...,
         query_ipa,
         similarity_fallback_limit=2000,
         unigram_fallback_limit=2000
       )
       → [SearchResult(...), SearchResult(...), ...]
    6. For each result:
       - _build_search_hit(result, query_ipa, rules_registry, query_mode)
         • Extract IPA, apply rule-matching via explain_alignment()
         • Classify match_type, uncertainty, base candidate_bucket
         • Generate alignment summary & visualization
         • Call the language profile's orthographic_note_builder when present
         • Use [] for orthographic_notes when the profile has no builder or no note matches
         • Promote candidate_bucket to "Supported" if any orthographic_correspondence note present
         • Return SearchHit (Pydantic model)
    7. Build ResponseMeta (request id, versions, request echo, verification URL)
    8. Return SearchResponse([SearchHit, SearchHit, ...], meta=ResponseMeta)
    ↓
HTTP 200 JSON
{
  "query": "λόγος",
  "query_ipa": "lɔːɡos",
  "query_mode": "Full-form",
  "hits": [
    {
      "headword": "λόγος",
      "ipa": "lɔːɡos",
      "distance": 0.25,
      "confidence": 0.75,
      "dialect_attribution": "lemma dialect: attic",
      "alignment_visualization": "query: l ɔː ɡ o s\n       : :  . : :\nlemma: l ɔː k o s",
      "match_type": "Rule-based",
      "rule_support": true,
      "applied_rule_count": 1,
      "observed_change_count": 0,
      "alignment_summary": "1 matched rule across 1 position.",
      "why_candidate": [
        "1 explicit rule supports the match.",
        "Moderate phonological similarity.",
        "No fallback edits required."
      ],
      "uncertainty": "Medium",
      "candidate_bucket": "Supported",
      "rules_applied": [
        {
          "rule_id": "CCH-001",
          "rule_name": "Voicing assimilation",
          "rule_name_en": "Voicing assimilation",
          "from_phone": "k",
          "to_phone": "ɡ",
          "position": 2
        }
      ],
      "orthographic_notes": [],
      "explanation": "Applied CCH-001 to explain the consonant correspondence; distance 0.250."
    }
  ],
  "meta": {
    "api_version": "1.0",
    "schema_version": "1.0.0",
    "engine_version": "<engine-version>",
    "data_versions": {
      "lexicon": "2.0.0",
      "matrix": "1.0.0",
      "rules": "0.1.0"
    },
    "ruleset_versions": {},
    "request_id": "8e52f0dc35b2427c944f79d5c57ef3c8",
    "timestamp": "2024-01-01T00:00:00Z",
    "verification_url": "http://127.0.0.1:8000/?q=%CE%BB%CF%8C%CE%B3%CE%BF%CF%82&language=ancient_greek&dialect=attic&max_candidates=5&response_language=en",
    "request_echo": {
      "query_form": "λόγος",
      "language": "ancient_greek",
      "dialect_hint": "attic",
      "max_candidates": 5,
      "response_language": "en"
    }
  }
}
```

## Request Metadata

`_add_request_id` runs as HTTP middleware and stores the accepted/generated
request id on `request.state.request_id`. `search()` passes that value into the
shared search runner so `SearchResponse.meta.request_id` and the
`X-Request-ID` response header stay aligned.

`api._request_context` owns public-base-url resolution, verification URL
construction, request-id generation, and request echo construction. REST uses
the incoming FastAPI request base URL when `PROTEUS_PUBLIC_BASE_URL` is unset.

Unhandled exceptions are caught by `_unhandled_exception_x_request_id`
(`src/api/main.py`), which falls back to `request.state.request_id` or a
freshly generated id so error responses still emit `X-Request-ID`.

## Dependency Loading

Orchestrated by `api._dependencies._load_search_dependencies()`. The aggregate
loader is not itself cached; per-process LRU caching lives on selected
sub-loaders in `src/api/_dependencies.py`. `api.main` re-exports the canonical
dependency functions as thin aliases for backward-compatible names. Tests that
stub sub-loader behavior must patch `api._dependencies`; whole-function swaps
consumed by this module's endpoints/warmup may still patch `api.main` because
those names are resolved here at call time.

```python
@lru_cache(maxsize=8)
def _load_lexicon_document(language_id: str) → dict[str, Any]
  → from data/languages/ancient_greek/lexicon/greek_lemmas.json

@lru_cache(maxsize=8)
def _load_lexicon_entries(language_id: str) → tuple[dict[str, Any], ...]
  → parsed lemmas from _load_lexicon_document(language_id)

@lru_cache(maxsize=8)
def _load_distance_matrix_with_meta(language_id: str) → tuple[MatrixData, dict[str, Any]]
  → from data/languages/ancient_greek/matrices/attic_doric.json via distance.load_matrix()

def _load_distance_matrix(language: str | None = None) → MatrixData
  → wrapper around _load_distance_matrix_with_meta()

def _load_rules_registry(language: str | None = None) → dict[str, dict[str, Any]]
  → from data/languages/ancient_greek/rules/*.yaml via explainer.load_rules()
  → delegates to phonology_search.get_rules_registry()

@lru_cache(maxsize=8)
def _load_search_index(language_id: str) → KmerIndex
  → build_kmer_index(lexicon, k=2)

@lru_cache(maxsize=8)
def _load_unigram_index(language_id: str) → KmerIndex
  → build_kmer_index(lexicon, k=1)

@lru_cache(maxsize=8)
def _load_lexicon_map(language_id: str) → LexiconMap
  → build_lexicon_map(lexicon)

@lru_cache(maxsize=8)
def _load_ipa_index(language_id: str) → IpaIndex
  → build_ipa_index(_load_lexicon_map(language_id))

@lru_cache(maxsize=8)
def _get_rules_version_cached(rules_dir: Path) → dict[str, str]
  → get_rules_version(rules_dir)
```

On `/ready` or first `/search`, if any loading fails:

- Raise `SearchDependenciesNotReadyError` with detail message
- Return HTTP 503 Service Unavailable with actionable guidance

**Startup Warmup:** Runs in background thread (non-blocking). If dependencies fail, subsequent requests report 503.

## Static File Serving

```python
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
```

Mounts `src/web/static/` at `/static/*`. Conditional on directory existence.

## Environment Variables

| Variable                              | Purpose                                                                             | Example                                       |
| ------------------------------------- | ----------------------------------------------------------------------------------- | --------------------------------------------- |
| `PROTEUS_ALLOWED_ORIGINS`             | CORS allowlist (comma-separated)                                                    | `"http://localhost:3000,https://example.com"` |
| `PROTEUS_LOG_RAW_SEARCH_QUERY`        | Enable raw query logging (debug only)                                               | `"true"`                                      |
| `PROTEUS_LSJ_REPO_DIR`                | Override LSJ checkout location (used by build_lexicon)                              | `"/path/to/lexica"`                           |
| `PROTEUS_ALLOW_TRUSTED_DIR_OVERRIDES` | Must be enabled before `PROTEUS_TRUSTED_*_DIR` runtime overrides are honored        | `"true"`                                      |
| `PROTEUS_TRUSTED_BUCK_DIR`            | Override Buck rules directory when `PROTEUS_ALLOW_TRUSTED_DIR_OVERRIDES` is enabled | `"/path/to/buck"`                             |
| `PROTEUS_TRUSTED_MATRICES_DIR`        | Override matrix directory when `PROTEUS_ALLOW_TRUSTED_DIR_OVERRIDES` is enabled     | `"/path/to/matrices"`                         |

## Related Areas

- **[phonology.md](./phonology.md)** — Core search pipeline, distance calculation, rule loading
- **Frontend:** `src/web/index.html`, `src/web/static/`
- **Configuration:** Environment variables (see above)
