# API Codemap

**Last Updated:** 2026-04-11  
**Entry Points:** `src/api/main.py` (FastAPI app instance)

## Architecture

```
HTTP REQUEST
    ↓
┌─────────────────────────────────────────────────────────────┐
│  CORS Middleware (CORSMiddleware)                            │
│  - Allow-Origins from PROTEUS_ALLOWED_ORIGINS env var       │
│  - Methods: GET, POST, OPTIONS                              │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│  FASTAPI ROUTES                                             │
├─────────────────────────────────────────────────────────────┤
│ GET  /              → root()         → HTMLResponse (frontend)
│ POST /search        → search()       → SearchResponse JSON
│ GET  /health        → health()       → {"status": "ok"}
│ GET  /ready         → ready()        → {"status": "ok"}
│ GET  /static/*      → StaticFiles    → CSS, JS, images
│ GET  /docs          → Swagger UI     → OpenAPI documentation
└─────────────────────────────────────────────────────────────┘
    ↓
HTTP RESPONSE (JSON or HTML)
```

## Endpoints

| Endpoint | Method | Handler | Input | Output | Behavior |
|----------|--------|---------|-------|--------|----------|
| `/` | GET | `root()` | None | `HTMLResponse` | Serves `src/web/index.html` (frontend) |
| `/search` | POST | `search()` | `SearchRequest` (Pydantic) | `SearchResponse` (Pydantic) | Three-stage pipeline with full dependency loading, query normalization, dialect-aware search |
| `/health` | GET | `health()` | None | `{"status": "ok"}` | Liveness probe (always succeeds if server is running) |
| `/ready` | GET | `ready()` | None | `{"status": "ok"}` | Readiness probe (checks all search dependencies loaded; returns 503 if not) |
| `/static/*` | GET | StaticFiles | Any | Static file | Maps `src/web/static/` directory |

## Pydantic Models

### Inbound (Request)

**`SearchRequest`**
```
- query_form: str (non-empty after strip_whitespace)
- dialect_hint: Literal["attic", "koine"] = "attic"
- max_candidates: int (default 20, range 1-100)
```

### Outbound (Response)

**`SearchResponse`**
```
- query: str (the original query)
- query_ipa: str (normalized IPA transcription)
- query_mode: Literal["Full-form", "Short-query", "Partial-form"]
- hits: list[SearchHit]
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
- explanation: str (human-readable prose summary of the derivation)
```

**`RuleStep`** (element of `rules_applied`)
```
- rule_id: str (unique rule identifier)
- rule_name: str (human-readable rule name)
- rule_name_en: str (English display name for the rule)
- from_phone: str (IPA phone that changed)
- to_phone: str (IPA phone it became)
- position: int (alignment position, or -1 if unknown)
```

## Data Flow: POST /search

```
HTTP POST { "query_form": "λόγος", "dialect_hint": "attic", "max_candidates": 100 }
    ↓ Pydantic validation
SearchRequest(query_form="λόγος", dialect_hint="attic", max_candidates=100)
    ↓
search(request: SearchRequest):
    1. _load_search_dependencies()  [cached]
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
         • Classify match_type, uncertainty, candidate_bucket
         • Generate alignment summary & visualization
         • Return SearchHit (Pydantic model)
    7. Return SearchResponse([SearchHit, SearchHit, ...])
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
      "explanation": "Applied CCH-001 to explain the consonant correspondence; distance 0.250."
    }
  ]
}
```

## Dependency Loading

Managed by `_load_search_dependencies()` with per-process LRU caching:

```python
@lru_cache(maxsize=1)
def _load_lexicon_entries() → tuple[dict[str, Any], ...]
  → from data/languages/ancient_greek/lexicon/greek_lemmas.json

@lru_cache(maxsize=1)
def _load_distance_matrix() → MatrixData
  → from data/languages/ancient_greek/matrices/attic_doric.json via distance.load_matrix()

@lru_cache(maxsize=1)
def _load_rules_registry() → dict[str, dict[str, Any]]
  → from data/languages/ancient_greek/rules/*.yaml via explainer.load_rules()

@lru_cache(maxsize=1)
def _load_search_index() → KmerIndex
  → build_kmer_index(lexicon, k=2)

@lru_cache(maxsize=1)
def _load_unigram_index() → KmerIndex
  → build_kmer_index(lexicon, k=1)

@lru_cache(maxsize=1)
def _load_lexicon_map() → LexiconMap
  → build_lexicon_map(lexicon)
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

| Variable | Purpose | Example |
|----------|---------|---------|
| `PROTEUS_ALLOWED_ORIGINS` | CORS allowlist (comma-separated) | `"http://localhost:3000,https://example.com"` |
| `PROTEUS_LOG_RAW_SEARCH_QUERY` | Enable raw query logging (debug only) | `"true"` |
| `PROTEUS_LSJ_REPO_DIR` | Override LSJ checkout location (used by build_lexicon) | `"/path/to/lexica"` |
| `PROTEUS_ALLOW_TRUSTED_DIR_OVERRIDES` | Must be enabled before `PROTEUS_TRUSTED_*_DIR` runtime overrides are honored | `"true"` |
| `PROTEUS_TRUSTED_BUCK_DIR` | Override Buck rules directory when `PROTEUS_ALLOW_TRUSTED_DIR_OVERRIDES` is enabled | `"/path/to/buck"` |
| `PROTEUS_TRUSTED_MATRICES_DIR` | Override matrix directory when `PROTEUS_ALLOW_TRUSTED_DIR_OVERRIDES` is enabled | `"/path/to/matrices"` |

## Related Areas

- **[phonology.md](./phonology.md)** — Core search pipeline, distance calculation, rule loading
- **Frontend:** `src/web/index.html`, `src/web/static/`
- **Configuration:** Environment variables (see above)
