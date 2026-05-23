# Proteus MCP Server

The Proteus MCP server is a Phase 2 prototype that exposes the phonological
search engine to MCP clients over stdio transport. It currently registers one
tool: `ancient_phonology.search`.

日本語要約: Phase 2 の MCP server は stdio 専用の prototype です。Hosted
HTTP/SSE transport、認証、複数 tool の本格展開は将来フェーズの対象です。

## Overview

`ancient_phonology.search` accepts an Ancient Greek query, runs the shared
Proteus search implementation, and returns structured candidates plus the same
metadata envelope used by REST search responses.

The runtime tool schema is committed at [`mcp/tools.json`](mcp/tools.json).

Phase 3 hard-query validation does not change the MCP tool schema. The
evaluation dataset and runner are repo-side quality tooling that exercise the
same search engine and record expected-candidate hits, false positives, and
false negatives outside the MCP wire contract.

## Installation

For local development:

```bash
uv sync --all-extras --dev
```

For installed use after packaging:

```bash
uv pip install proteus
# or
pipx install proteus
```

The console script is `proteus-mcp`.

## Running

Start the server on stdio:

```bash
proteus-mcp
```

For reproducible verification links in MCP responses, set a public base URL:

```bash
PROTEUS_PUBLIC_BASE_URL=https://proteus.example proteus-mcp
```

When `PROTEUS_PUBLIC_BASE_URL` is unset, MCP responses use an empty
`meta.verification_url` because there is no HTTP request base URL in stdio
context.

## Claude Desktop Config

Example config:

```json
{
  "mcpServers": {
    "proteus": {
      "command": "proteus-mcp",
      "env": {
        "PROTEUS_PUBLIC_BASE_URL": "https://proteus.example/"
      }
    }
  }
}
```

The same example is committed at
[`mcp/example-claude-desktop.json`](mcp/example-claude-desktop.json).

## Tool Reference: `ancient_phonology.search`

The tool argument object contains one field, `request`, whose value follows
`McpSearchInput`.

Input fields:

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `query_form` | string | required | Greek word or form to search for. |
| `source_language` | string | `ancient_greek` | Language profile id. |
| `dialect_hint` | string or null | `null` | `null` defers to the selected profile default. |
| `max_candidates` | integer | `20` | Range `1` to `100`. |
| `response_language` | `en` or `ja` | `en` | Language for generated prose fields. |

Example logical call:

```json
{
  "request": {
    "query_form": "λόγος",
    "source_language": "ancient_greek",
    "dialect_hint": "attic",
    "max_candidates": 5,
    "response_language": "en"
  }
}
```

## Output Schema

`ancient_phonology.search` returns `McpSearchOutput`.

| Field | Type | Notes |
| --- | --- | --- |
| `candidates` | array of `SearchHit` | Ranked search candidates. Same candidate model as REST `hits`. |
| `query` | string | Original query string. |
| `query_ipa` | string | IPA transcription computed for the query. |
| `query_mode` | string | `Full-form`, `Short-query`, or `Partial-form`. |
| `truncated` | boolean | True when candidate annotation was truncated. |
| `meta` | `ResponseMeta` | Version, request, and reproducibility metadata. |

`meta.request_echo.language` is populated from the REST-compatible
`SearchRequest.language` field after the MCP `source_language` value is
validated and normalized.

Each candidate uses the same `SearchHit` model as REST. If
`source_references` is present, MCP clients should treat it as metadata-only
attribution and link data. The field must not be expanded into quoted source
text, and `citation_ready=false` references require human review before
scholarly citation.

## Verification & Reproducibility

Each tool response includes a generated `meta.request_id` and a
`meta.request_echo` object. `meta.verification_url` matches the REST
verification URL format when `PROTEUS_PUBLIC_BASE_URL` is configured. Without
that environment variable, the value is `""`.

Clients should treat `meta.verification_url == ""` as "no public URL is
available" rather than as an error. When a deployment exposes the REST surface
alongside the MCP server, clients can reconstruct an equivalent URL from
`meta.request_echo` against their own known base URL.

`PROTEUS_LOG_RAW_SEARCH_QUERY` only controls server-side logging. It does not
remove `query_form` from MCP response bodies.

## Limitations

- Phase 2 supports stdio transport only.
- The prototype has no built-in authentication or authorization layer.
- Stdio transport runs with the same OS-level permissions as the launching
  process; the surrounding MCP client (for example Claude Desktop) is
  responsible for any authorization boundary.
- The only registered tool is `ancient_phonology.search`.
- The bundled Ancient Greek data remains provisional and requires expert review
  before citation or research use.

## Future Tools

Future MCP tools tracked by the requirements include:

- `ancient_phonology.explain_rule`
- `ancient_phonology.list_languages`
- `ancient_phonology.list_rules`
- `ancient_phonology.compare_candidates`

## Schema Artifact

Regenerate the committed MCP schema artifact:

```bash
uv run python scripts/export_mcp_schema.py --output docs/mcp/tools.json
```

Check for drift without rewriting:

```bash
uv run python scripts/export_mcp_schema.py --check
```

## Smoke Testing

Phase 2 acceptance is verified by the automated test suite
(`tests/test_mcp_search_tool.py`, `tests/test_mcp_schema.py`,
`tests/test_packaging.py`) and by `scripts/export_mcp_schema.py --check` drift
detection in CI. End-to-end smoke testing against a real MCP client
(for example Claude Desktop) is performed by an operator after release using
the configuration in [`mcp/example-claude-desktop.json`](mcp/example-claude-desktop.json).
