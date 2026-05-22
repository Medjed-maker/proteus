# Proteus REST API

Proteus exposes the Phase 2 REST API as a field-versioned JSON contract. The
current public API version is `1.0`; response schema compatibility is tracked by
`schema_version` in response metadata and by the committed OpenAPI artifact at
[`docs/api/openapi.json`](api/openapi.json).

日本語要約: Phase 2 の REST API は URL ではなく `meta.api_version` と
`meta.schema_version` で契約を管理します。詳細な機械可読 schema は
`docs/api/openapi.json` を参照してください。

## Versioning

Proteus keeps the Phase 2 REST path surface stable. The search endpoint remains
`POST /search`; version information is surfaced in response bodies instead of
path prefixes.

| Field | Meaning |
| --- | --- |
| `meta.api_version` | REST API contract version, currently `1.0`. |
| `meta.schema_version` | Public response schema version, currently `1.0.0`. |
| `meta.engine_version` | Installed `proteus` package version. |
| `data_versions` | Data source versions used by the search response. |

Adding JSON fields is considered backward-compatible. Removing existing fields
or changing their types is breaking. Clients should ignore unknown response
keys.

Phase 3 hard-query validation data does not change this REST contract. The
dataset and evaluator live under `data/evaluation/hard_queries/` and `tools/`;
they call the same search runner to measure quality, false positives, and false
negatives without adding response fields.

## Endpoints

### `POST /search`

Runs reverse phonological search for a historical form.

```bash
curl -sS http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query_form": "λόγος",
    "language": "ancient_greek",
    "dialect_hint": "attic",
    "max_candidates": 5,
    "response_language": "en"
  }'
```

Request fields:

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `query_form` | string | required | Search query, trimmed and length-bounded. |
| `language` | string | `ancient_greek` | Registered language profile id. Legacy values `en` and `ja` are accepted only as deprecated prose-language aliases. |
| `dialect_hint` | string | profile default | Dialect used by the language profile. `dialect` is also accepted as an alias. |
| `max_candidates` | integer | `20` | Range `1` to `100`. |
| `response_language` | `en` or `ja` | `en` | Language for generated prose fields. |
| `lang` | `en` or `ja` | none | Deprecated alias for `response_language`. |
| `orthography_hint` | `standard`, `inscriptional`, or `pre_403_2_attic` | none | Deprecated; these three legacy values are accepted for compatibility but ignored during note generation. Other values fail validation with HTTP 422. |

Response fields:

| Field | Type | Notes |
| --- | --- | --- |
| `query` | string | Original query string. |
| `query_ipa` | string | IPA transcription computed for the query. |
| `query_mode` | string | `Full-form`, `Short-query`, or `Partial-form`. |
| `hits` | array of `SearchHit` | Ranked candidates. |
| `truncated` | boolean | True when annotation was truncated by candidate-window or batch limits. |
| `data_versions` | `DataVersions` | Backward-compatible top-level copy of `meta.data_versions`. |
| `meta` | `ResponseMeta` | Version, request, and reproducibility metadata. |

Example response excerpt:

```json
{
  "query": "λόγος",
  "query_ipa": "lɔːɡos",
  "query_mode": "Full-form",
  "hits": [
    {
      "headword": "λόγος",
      "ipa": "lɔːɡos",
      "distance": 0.0,
      "confidence": 1.0,
      "rules_applied": [],
      "orthographic_notes": [],
      "explanation": "Exact phonological match."
    }
  ],
  "truncated": false,
  "data_versions": {
    "lexicon": "2.0.0",
    "matrix": "1.0.0",
    "rules": "0.1.0"
  },
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

### `GET /languages`

Lists registered language profiles and runtime version metadata.

```bash
curl -sS http://127.0.0.1:8000/languages
```

The response contains `languages: LanguageInfo[]` and `meta: VersionInfo`.
Ancient Greek is currently the bundled pilot profile.

### `GET /version`

Returns runtime metadata:

| Field | Meaning |
| --- | --- |
| `engine_version` | Installed `proteus` package version. |
| `api_version` | REST API version. |
| `schema_version` | Public response schema version. |
| `rule_schema_version` | Rule-file JSON schema id or version. |
| `build_timestamp` | Optional deployment timestamp from `PROTEUS_BUILD_TIMESTAMP`. |
| `git_sha` | Optional deployment commit from `PROTEUS_GIT_SHA`. |
| `python_version` | Python runtime version. |
| `mcp_server_version` | MCP server version exposed by this deployment. |

### `GET /health`

Liveness probe. Returns `{"status": "ok"}` when the app process can respond.

### `GET /ready`

Readiness probe. Returns `{"status": "ok"}` when search dependencies are
loaded. Returns HTTP 503 when lexicon, matrix, rules, or profile dependencies
are unavailable.

### `GET /`

Serves the packaged HTML frontend.

### `GET /changelog`

Serves the packaged changelog HTML page.

## Request / Response Models

| Model | Purpose | Important fields |
| --- | --- | --- |
| `SearchRequest` | Search input. | `query_form`, `language`, `dialect_hint`, `max_candidates`, `response_language`; deprecated `orthography_hint`, `lang`. |
| `SearchResponse` | Top-level search output. | `query`, `query_ipa`, `query_mode`, `hits`, `truncated`, `data_versions`, `meta`. |
| `SearchHit` | Ranked candidate. | `headword`, `ipa`, `distance`, `confidence`, `dialect_attribution`, `match_type`, `rules_applied`, `orthographic_notes`, `explanation`. |
| `RuleStep` | One phonological rule application. | `rule_id`, `rule_name`, `rule_name_en`, `from_phone`, `to_phone`, `position`. |
| `OrthographicNote` | Candidate-level spelling or writing-system note. | `kind`, `label`, `messages`, `normalized_form`, `romanization`, `period_label`, `references`, `confidence`, `pre_reform_spelling`, `pre_reform_romanization`. |
| `DataVersions` | Data source metadata. | `lexicon`, `lexicon_updated_at`, `matrix`, `matrix_generated_at`, `rules`. |
| `ResponseMeta` | Search response metadata. | `api_version`, `schema_version`, `engine_version`, `data_versions`, `ruleset_versions`, `request_id`, `timestamp`, `verification_url`, `request_echo`. |
| `RequestEcho` | Sanitized validated request echo. | `query_form`, `language`, `dialect_hint`, `max_candidates`, `response_language`. |
| `LanguageInfo` | Registered language profile metadata. | `language_id`, `display_name`, `default_dialect`, `supported_dialects`, `status`, `ruleset_version`, `lexicon_schema_version`, `matrix_version`, `description`. |
| `LanguagesResponse` | Language profile list. | `languages`, `meta`. |
| `VersionInfo` | Runtime version metadata. | `engine_version`, `api_version`, `schema_version`, `rule_schema_version`, `build_timestamp`, `git_sha`, `python_version`, `mcp_server_version`. |

日本語要約: `rules_applied` は音韻変化、`orthographic_notes` は表記・綴字上の注記です。この 2 つは用途が異なるため統合しません。たとえば `παιδίο` の検索で現在候補が `παιδίον` の場合でも、表記体系コメントは別読解として `παιδίου (paidiou)` を提示し、必要に応じて改革前アッティカ碑文表記 `παιδίο (paidiō)` を `pre_reform_spelling` / `pre_reform_romanization` フィールドで併記します。

### `SearchHit.explanation` content guidelines

`hits[].explanation` is the human-readable companion to `rules_applied`. It is
a plain-text string, not HTML or Markdown, because the packaged frontend
renders it via `textContent` in `src/web/index.html`.

Recommended content shape:

- One short sentence, at most two.
- Compact etymological note, a summary of the rules already listed in
  `rules_applied`, or a brief statement of the dialectal correspondence.
- No embedded reference links, raw markup, or numeric confidence values.

If richer provenance is needed in the future, add separate structured fields
on `SearchHit` instead of overloading `explanation`.

## Error Responses

| Status | Endpoint | Shape | Typical cause |
| --- | --- | --- | --- |
| 400 | `POST /search` | `{"detail": "Invalid search query"}` | Query rejected after validation. |
| 422 | `POST /search` | FastAPI validation detail | Missing or malformed request fields, including unknown language profiles and unsupported `dialect_hint` values. |
| 503 | `POST /search`, `GET /ready` | `{"detail": "...not ready..."}` | Search dependencies are unavailable. |
| 503 | `GET /languages` | `{"detail": "No language profiles registered"}` | No profiles are registered. |

All HTTP responses include `X-Request-ID`.

## Reproducibility

`meta.request_id` is a request correlation id. Clients may send `X-Request-ID`
with an 8-64 character hex value; otherwise Proteus generates a UUID4 hex id.

`meta.verification_url` is a deterministic URL derived from the validated
search request. If `PROTEUS_PUBLIC_BASE_URL` is set, that base URL is used.
Otherwise REST responses use the incoming FastAPI request base URL.

`meta.request_echo` contains validated request parameters. It intentionally
includes the raw `query_form` so that the response is reproducible; it is not
affected by `PROTEUS_LOG_RAW_SEARCH_QUERY`, which only controls server logs.

## Deprecation Policy

Deprecated request paths remain accepted during Phase 2 when they can be mapped
without ambiguity.

| Deprecated input | Replacement | Signal |
| --- | --- | --- |
| `language: "en"` or `"ja"` as a prose-language selector | `response_language` | `Deprecation: true`, `Link` (only when interactive docs enabled), `X-Proteus-Migration`. |
| `lang` | `response_language` | OpenAPI deprecation metadata. |
| `orthography_hint` | runtime orthographic-note data | OpenAPI deprecation metadata; only `standard`, `inscriptional`, and `pre_403_2_attic` are accepted, and accepted values are ignored. |

`Deprecation: true` and `X-Proteus-Migration` are emitted only on responses to
requests that supplied `language: "en"` or `"ja"` as a prose-language
selector. The accompanying `Link: <docs-url>; rel="deprecation"` header is
added only when interactive API docs are enabled
(`PROTEUS_ENABLE_API_DOCS=1`); otherwise it is omitted. The `lang` and
`orthography_hint` deprecations are surfaced through OpenAPI metadata only and
do not add per-response headers.

## Environment Variables

| Variable | Purpose |
| --- | --- |
| `PROTEUS_ALLOWED_ORIGINS` | Comma-separated CORS allowlist. |
| `PROTEUS_PUBLIC_BASE_URL` | Absolute public base URL for `meta.verification_url`. |
| `PROTEUS_APP_VERSION` | Override for `engine_version`. |
| `PROTEUS_ENABLE_API_DOCS` | Enables Swagger UI and `/openapi.json` when set to `1`. |
| `PROTEUS_LOG_RAW_SEARCH_QUERY` | Allows raw query logging when enabled; response metadata still includes the query. |
| `PROTEUS_DISABLE_STARTUP_WARMUP` | Skips startup warmup when set. |
| `PROTEUS_GIT_SHA` | Optional deployment commit surfaced by `/version`. |
| `PROTEUS_BUILD_TIMESTAMP` | Optional deployment timestamp surfaced by `/version`. |

## OpenAPI Schema

The committed schema artifact is [`docs/api/openapi.json`](api/openapi.json).
Regenerate it after API model or endpoint changes:

```bash
uv run python scripts/export_openapi.py --output docs/api/openapi.json
```

Check for drift without rewriting the file:

```bash
uv run python scripts/export_openapi.py --check
```

## Examples

### curl

```bash
curl -sS http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query_form":"λόγος","max_candidates":3}'
```

### httpie

```bash
http POST :8000/search query_form=λόγος max_candidates:=3
```

### Python

```python
import httpx

payload = {"query_form": "λόγος", "max_candidates": 3}
response = httpx.post("http://127.0.0.1:8000/search", json=payload, timeout=10)
response.raise_for_status()
print(response.json()["hits"])
```
