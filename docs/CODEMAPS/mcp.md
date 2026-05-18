# MCP Codemap

**Last Updated:** 2026-05-17
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
    | loads dependencies through api.main.load_search_dependencies
    | executes api._search_runner.run_search
    v
src/phonology/search.py and language profiles
    |
    v
McpSearchOutput JSON
```

## Runtime Components

| Component | Responsibility |
| --- | --- |
| `src/mcp_server/server.py` | Creates the `FastMCP` app, registers MCP tools, and exposes `main()` for the `proteus-mcp` console script. |
| `src/mcp_server/tools/search.py` | Defines `McpSearchInput`, `McpSearchOutput`, and registers `ancient_phonology.search`. |
| `src/mcp_server/_search_adapter.py` | Bridges MCP input to the shared REST/search execution path, builds request ids, metadata, and MCP output. |
| `api.main.load_search_dependencies` | Public dependency loader reused by MCP so REST and MCP resolve profile data consistently. |
| `api._search_runner.run_search` | Shared search orchestration used by both `POST /search` and MCP search. |

## Tool Surface

| Tool | Input model | Output model | Behavior |
| --- | --- | --- | --- |
| `ancient_phonology.search` | `McpSearchInput` | `McpSearchOutput` | Runs Ancient Greek phonological search and returns ranked candidates plus `ResponseMeta`. |

`McpSearchInput.source_language` maps to REST `SearchRequest.language`.
`McpSearchOutput.candidates` uses the same `SearchHit` model as REST
`SearchResponse.hits`.

## Metadata Flow

1. `_search_adapter._run_search_for_mcp()` generates a UUID4 hex request id.
2. `_build_search_request()` converts MCP input into `SearchRequest`.
3. `execute_search()` calls the shared search runner with `APP_VERSION` and
   ruleset versions from `api.main`.
4. `build_mcp_response()` copies REST hits into `candidates` and preserves the
   shared `ResponseMeta` envelope.

When `PROTEUS_PUBLIC_BASE_URL` is unset, MCP uses an empty
`meta.verification_url` because stdio requests do not provide an HTTP base URL.

## Tests and Artifacts

| File | Purpose |
| --- | --- |
| `tests/test_mcp_search_tool.py` | In-process FastMCP tool tests for listing, validation, candidate output, metadata, Japanese prose, and verification URL behavior. |
| `tests/test_mcp_server_init.py` | Import, entry point, and version consistency checks. |
| `docs/mcp/tools.json` | Generated tool schema artifact from `scripts/export_mcp_schema.py`. |
| `docs/mcp/example-claude-desktop.json` | Example MCP client configuration. |

Use `uv run python scripts/export_mcp_schema.py --check` to detect schema drift.
