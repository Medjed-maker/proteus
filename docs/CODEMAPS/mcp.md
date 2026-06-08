# MCP Codemap

**Last Updated:** 2026-06-08
**Entry Points:** `src/mcp_server/server.py`, `proteus-mcp`

## Architecture

```text
MCP CLIENT
    |
    | stdio
    v
src/mcp_server/server.py
    |
    | registers tools on FastMCP
    v
src/mcp_server/tools/search.py
    |
    | validates McpSearchInput and delegates
    v
src/mcp_server/_search_adapter.py
    |
    | converts to api._models.SearchRequest
    | loads dependencies through api._dependencies._load_search_dependencies
    | reads APP_VERSION/ruleset metadata from api._runtime_metadata/api._dependencies
    | executes api._search_runner.run_search
    v
src/phonology/search/ and language profiles
    |
    v
McpSearchOutput JSON
```

## Runtime Components

| Component                                     | Responsibility                                                                                             |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `src/mcp_server/server.py`                    | Creates the `FastMCP` app, registers MCP tools, and exposes `main()` for the `proteus-mcp` console script. |
| `src/mcp_server/tools/search.py`              | Defines `McpSearchInput`, `McpSearchOutput`, and registers `ancient_phonology.search`.                     |
| `src/mcp_server/_search_adapter.py`           | Bridges MCP input to the shared REST/search execution path, builds request ids, metadata, and MCP output.  |
| `api._dependencies._load_search_dependencies` | Canonical dependency loader reused by REST and MCP so both surfaces resolve profile data consistently.     |
| `api._runtime_metadata.APP_VERSION`           | Shared engine version used by REST and MCP without importing the REST app.                                 |
| `api._search_runner.run_search`               | Shared search orchestration used by both `POST /search` and MCP search.                                    |

## Tool Surface

| Tool                       | Input model      | Output model      | Behavior                                                                                  |
| -------------------------- | ---------------- | ----------------- | ----------------------------------------------------------------------------------------- |
| `ancient_phonology.search` | `McpSearchInput` | `McpSearchOutput` | Runs Ancient Greek phonological search and returns ranked candidates plus `ResponseMeta`. |

`McpSearchInput.source_language` maps to REST `SearchRequest.language`.
`McpSearchOutput.candidates` uses the same `SearchHit` model as REST
`SearchResponse.hits`.

## Metadata Flow

1. `_search_adapter._run_search_for_mcp()` generates a UUID4 hex request id.
2. `_build_search_request()` converts MCP input into `SearchRequest`.
3. `execute_search()` calls the shared search runner with `APP_VERSION` from
   `api._runtime_metadata` and ruleset versions from
   `api._dependencies._build_ruleset_versions`.
4. `build_mcp_response()` copies REST hits into `candidates` and preserves the
   shared `ResponseMeta` envelope.

When `PROTEUS_PUBLIC_BASE_URL` is unset, MCP uses an empty
`meta.verification_url` because stdio requests do not provide an HTTP base URL.

## Tests and Artifacts

| File                                   | Purpose                                                                                                                           |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `tests/test_mcp_search_tool.py`        | In-process FastMCP tool tests for listing, validation, candidate output, metadata, Japanese prose, and verification URL behavior. |
| `tests/test_mcp_server_init.py`        | Import, entry point, and version consistency checks.                                                                              |
| `docs/mcp/tools.json`                  | Generated tool schema artifact from `scripts/export_mcp_schema.py`.                                                               |
| `docs/mcp/example-claude-desktop.json` | Example MCP client configuration.                                                                                                 |

Use `uv run python scripts/export_mcp_schema.py --check` to detect schema drift.
