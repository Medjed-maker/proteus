# Phase 2 REST API and MCP Prototype Implementation Plan

作成日: 2026-05-15
対象: `docs/ROADMAP.md` Phase 2 — REST API stabilization, `/languages`, `/version`, MCP server prototype, MCP `ancient_phonology.search` tool, engine/ruleset version surfacing, reproducibility/verification metadata, API and MCP documentation
目的: Phase 1 で固まった検索エンジンを、(a) 安定した契約を持つ REST API、(b) LLM クライアントから呼び出せる MCP server、の二つの面から外部公開する。Public 出力に engine/ruleset version と verification metadata を含め、API/MCP の出力スキーマを文書化する。Ancient Greek の検索挙動・ランキング・説明出力は変更しない。

## Context

Phase 0 (core/plugin 分離) と Phase 1 (rule schema v0.1) が完了し、`POST /search` は構造化候補・適用ルール・説明文を返す状態にある (`src/api/main.py:551-660`, `src/api/_models.py:294-405`)。一方で:

- `/languages` `/version` が未実装。
- レスポンスに engine version (アプリ version) が surface されていない。
- 再現性メタデータ (request_id, verification URL, request echo) が無い。
- MCP scaffolding (パッケージ、依存、tool) が無い。
- API 出力スキーマ・MCP 出力スキーマのドキュメントが無い (`docs/API.md` `docs/MCP.md` `docs/api/openapi.json` いずれも未作成)。

本 plan は ROADMAP §4 (Phase 2) の Required Work と Acceptance Criteria を atomic checklist に分解し、Ancient Greek pilot の挙動を壊さずに上記ギャップを埋める道筋を示す。

## 確定済み設計判断 (本 plan で fix)

- **API versioning**: URL path はそのまま (`/search` のまま)。`meta.api_version` / `meta.schema_version` の field-based 管理。`API_VERSION = "1.0"` / `SCHEMA_VERSION = "1.0.0"` を Phase 2 で確立。
- **verification_url**: deterministic URL。`{base}/?q=<urlencoded>&language=<id>&dialect=<d>&max_candidates=<n>&response_language=<en|ja>`。永続化なし。同じクエリで同じ URL。`PROTEUS_PUBLIC_BASE_URL` env override 可。
- **MCP transport**: stdio のみ。Streamable HTTP/SSE は Phase 6 (Hosted Layer) に持ち越し。
- **/version フィールド**: ミニマル (`engine_version`, `api_version`, `schema_version`, `rule_schema_version`, `python_version`, `build_timestamp`, `git_sha`, `mcp_server_version`)。per-language ruleset version は `/languages` 経由。
- **`data_versions` トップレベル**: 既存 client 互換のため `SearchResponse` トップレベルに残し、`meta.data_versions` にも同一値を出す (両出し)。Phase 3 以降に deprecation を検討。
- **MCP SDK**: 高位 `FastMCP` (`mcp.server.fastmcp.FastMCP`) を採用。
- **MCP tool name**: `ancient_phonology.search` (REQUIREMENTS §4.4 準拠)。
- **MCP entry point**: `proteus-mcp = "mcp_server.server:main"` (`[project.scripts]`)。
- **Wheel packages**: `src/api`, `src/phonology`, `src/mcp_server` を `[tool.hatch.build.targets.wheel].packages` に追加。
- **OpenAPI artifact**: `docs/api/openapi.json` を repo にコミットし、CI で drift 検知 (`scripts/export_openapi.py --check`)。
- **`/languages` staged migration**: 章 2 の `LanguagesResponse.meta` は一時的に `VersionInfo` を返す。章 1 完了時に `VersionInfo` → `ResponseMeta` へ置換し、その PR で `scripts/export_openapi.py`, `docs/api/openapi.json`, 関連 tests / CI expectation を同時更新する。各 migration step は独立 PR とし、章またぎの atomic change は行わない。

## 進捗凡例

- [ ] 未着手
- [x] 完了
- [~] 実装中または要再検討

## この plan の使い方

- 章番号は topic の章立て。実装順は最末尾の `Suggested Implementation Order` を正とする。
- `(decide:)` で始まる行は、本 plan で完全には固定していない補助的判断。実装着手時に PR 上で確定してから checklist を更新する (主要な判断は上記「確定済み設計判断」で fix 済み)。
- 章 0–8 は個別 PR に分けて良い。実装順は最末尾参照。

---

## Phase 2 受け入れマッピング (overview)

ROADMAP `docs/ROADMAP.md` lines 91-96 の受け入れ基準と本 plan のマッピング。章 8 でこの表が緑になることが Phase 2 完了条件。

- [ ] API returns structured candidates → 既存 `POST /search` (章 0 baseline で確認、章 1 で envelope を追加しても shape は backwards-compatible)。
- [ ] API returns applied rules and explanations → 既存 `SearchHit.rules_applied` / `explanation` (章 0 baseline で確認)。
- [ ] MCP server can answer a query through the search engine → 章 5 で `ancient_phonology.search` tool 実装、章 7 で test。
- [ ] MCP response includes candidates, confidence, applied rules, and metadata → 章 5 でツール出力スキーマを REST `SearchResponse` 共有に揃える。
- [ ] API and MCP output schemas are documented → 章 6 で `docs/API.md`, `docs/MCP.md`, `docs/api/openapi.json` を整備。

---

## 0. Baseline & Guardrails

### Goal
Phase 2 着手前に、現行の安定挙動と test baseline をスナップショットし、Phase 2 で壊してはならない不変量を確定する。

### Current gap
Phase 2 は API レスポンスのトップレベル形状に新規フィールド (`meta`, `api_version` 等) を追加するため、既存 client が壊れないことを stage ごとに確認する基準点が必要。

### Atomic checklist

- [x] 現行 test baseline を記録する。
  - [x] `uv run pytest tests/test_api_main.py -q` の pass 数を記録。
  - [x] `uv run pytest tests/test_language_profiles.py -q` の pass 数を記録。
  - [x] `uv run pytest tests/test_packaging.py -q` の pass 数を記録。
  - [x] `uv run pytest -q` (全体) の pass 数を Phase 2 開始時点として記録。
- [x] 維持すべき不変量を列挙する。
  - [x] `POST /search` は既存 client のために `query`, `query_ipa`, `query_mode`, `hits[]`, `truncated`, `data_versions` を引き続き返す (`src/api/_models.py:379-405`)。
  - [x] `SearchHit` の既存フィールドは追加可・削除/型変更不可 (`src/api/_models.py:294-377`)。
  - [x] `GET /health` は `{"status": "ok"}` を返す (`src/api/main.py:663-666`)。
  - [x] `GET /ready` は 200 (deps ready) または 503 (`src/api/main.py:675-687`)。
  - [x] `HEAD /` / `GET /` / `GET /changelog` の挙動は変更しない (`src/api/main.py:523-548`)。
  - [x] CORS / security headers middleware の挙動を維持する (`src/api/main.py:203-228`)。
  - [x] `_load_app_version()` の優先順 (env → metadata → pyproject) を維持する (`src/api/main.py:82-108`)。
  - [x] `legacy_language_alias_used` 経由の `Deprecation: true` ヘッダ動作を維持する (`src/api/main.py:651-659`)。
- [x] `SearchResponse` への新フィールド追加は破壊的変更とみなさない、を本 plan の規約として明文化する。README に次の 3 点を明記済み。`docs/API.md` は章 6 で作成時に同内容を移植する: (a) `SearchResponse` の `model_config = ConfigDict(extra="ignore")` は **サーバー側** Pydantic モデルが未知フィールドを無視する設定であり、client の JSON parser の挙動を規定するものではない、(b) フィールド追加が非破壊である根拠はサーバーが新フィールドを安全に追加できる点と、HTTP/JSON client が JSON レスポンスの未知キーを通常無視できる点にある、(c) したがって新フィールド追加後も既存 client は互換のまま動作する。
- [x] `CHANGELOG.md` にこの Phase 2 全体の section heading を予約する (既存の `## Unreleased` 配下に `### Phase 2: REST API and MCP Prototype`)。

### Baseline record

Recorded on 2026-05-15 with Python 3.12.9 and pytest 9.0.2:

- `uv run pytest tests/test_api_main.py -q`: 205 passed.
- `uv run pytest tests/test_language_profiles.py -q`: 17 passed.
- `uv run pytest tests/test_packaging.py -q`: 18 passed.
- `uv run pytest -q`: 1639 passed.

### Files to inspect (no changes)
- `src/api/main.py`
- `src/api/_models.py`
- `src/phonology/profiles.py`
- `docs/ROADMAP.md` lines 76-96
- `docs/REQUIREMENTS.md` §3.3.3, §4.3, §4.4, §5.5, §5.7, §6.7, §7.5, §7.6

### Test commands
- `uv run pytest tests/test_api_main.py -q`
- `uv run pytest tests/test_language_profiles.py -q`
- `uv run pytest tests/test_packaging.py -q`
- `uv run pytest -q`

### Acceptance
- Baseline 数値が本 plan または PR description に記録されている。

---

## 1. REST API Response Schema Stabilization

### Goal
`SearchResponse` に engine/ruleset version、`api_version`、`request_id`、verification metadata、リクエスト echo を後方互換な形で追加する。`data_versions` は維持しつつ、新規 `meta` envelope に同等以上の情報を提供する。

Staged migration note: 章 1 で `ResponseMeta` を導入したら、章 2 で一時的に使っている `LanguagesResponse.meta: VersionInfo` を `ResponseMeta` へ置換し、OpenAPI artifact と tests / CI expectation を同じ PR で更新する。

### Current gap
- `SearchResponse.data_versions` (`src/api/_models.py:22-58`) は lexicon / matrix / rules の version 文字列のみで、engine version (アプリ version) を返していない。
- `api_version` / `schema_version` / `request_id` / `verification_url` が存在しない。
- リクエストパラメータの echo が response に含まれない。
- OpenAPI 出力は `PROTEUS_ENABLE_API_DOCS=1` の時しか公開されず、固定された artifact がない (`src/api/main.py:196-199`)。

### Atomic checklist

#### 1.1 新規 Pydantic モデル: `ResponseMeta`
- [ ] `src/api/_models.py` に `ResponseMeta(BaseModel)` を追加。フィールド:
  - [ ] `api_version: str` — REST API 契約バージョン (例: `"1.0"`)。
  - [ ] `schema_version: str` — `SearchResponse` JSON Schema の semantic version (例: `"1.0.0"`)。
  - [ ] `engine_version: str` — `proteus` package version (`_load_app_version()` を再利用)。
  - [ ] `data_versions: DataVersions` — 既存 `DataVersions` をネスト。
  - [ ] `ruleset_versions: dict[str, str]` — 言語別 rules version (例: `{"ancient_greek": "0.1.0"}`)。`get_rules_version()` を呼ぶ。
  - [ ] `request_id: str` — UUID4 hex string (length 32)。
  - [ ] `timestamp: str` — ISO 8601 UTC string (`datetime.now(timezone.utc).isoformat()`)。
  - [ ] `verification_url: str` — deterministic URL (確定済み設計判断参照)。
  - [ ] `request_echo: RequestEcho | None` — sanitized request echo (1.2 参照)。`/languages` `/version` では `None` 可。
- [ ] `ResponseMeta` は `VersionInfo` と別モデルとして設計する。`ResponseMeta` は `VersionInfo` と共通の version fields を共有し、request metadata / data metadata を追加で提供する。`VersionInfo` の既存 field 名・型・意味は変更せず、`LanguagesResponse.meta` の移行後も client が未知 field を無視すれば継続動作できるようにする。
- [ ] `ResponseMeta` の `timestamp` field validator は `DataVersions._validate_timestamp` と同じパターンで検証する。これは `VersionInfo` の validator ではなく、関連する別モデルである `ResponseMeta` 側の追加 metadata 検証として実装する。
- [ ] `ResponseMeta` を `__all__` に追加。

#### 1.2 新規 Pydantic モデル: `RequestEcho`
- [ ] `src/api/_models.py` に `RequestEcho(BaseModel)` を追加。フィールド:
  - [ ] `query_form: str`
  - [ ] `language: str`
  - [ ] `dialect_hint: str`
  - [ ] `max_candidates: int`
  - [ ] `response_language: Literal["en", "ja"]`
- [ ] `RequestEcho` は `model_config = ConfigDict(frozen=True)`。
- [ ] `orthography_hint` (deprecated) は echo に含めない。
- [ ] `RequestEcho` を `__all__` に追加。

#### 1.3 `SearchResponse` への組み込み
- [ ] `SearchResponse` (`src/api/_models.py:379-405`) に `meta: ResponseMeta` を追加。default factory なし — `main.py` が必ず構築。
- [ ] `data_versions` トップレベルは互換のため残す。docstring に `meta.data_versions` も同値であることを明記。
- [ ] `model_config = ConfigDict(extra="ignore")` を `SearchResponse` に明示し、unknown fields の追加を非破壊扱いとする規約を schema に固定。

#### 1.4 Request ID & verification URL 生成 helper
- [ ] `src/api/_request_context.py` (新規) を作成。
  - [ ] `generate_request_id() -> str` — `uuid.uuid4().hex` を返す。
  - [ ] `resolve_public_base_url(fastapi_request: Request) -> str` — env `PROTEUS_PUBLIC_BASE_URL` 優先、未設定なら `str(fastapi_request.base_url)`。
  - [ ] `build_verification_url(base: str, request: SearchRequest) -> str` — `urllib.parse.urlencode` で組み立て。クエリパラメータ: `q`, `language`, `dialect`, `max_candidates`, `response_language`。
  - [ ] `build_request_echo(request: SearchRequest) -> RequestEcho` — sanitized echo を組み立て。
- [ ] `src/api/_response_meta.py` (新規) を作成:
  - [ ] `build_response_meta(*, request_id: str, request: SearchRequest | None, base_url: str | None, engine_version: str, data_versions: DataVersions, ruleset_versions: dict[str, str]) -> ResponseMeta` — REST/MCP 両方から呼ぶ共通ヘルパー。
  - [ ] このモジュールは `api._models` と `phonology.profiles` のみ import し、`api.main` を import しない (circular import 回避)。

#### 1.5 Request-ID middleware
- [ ] `src/api/main.py` に `@app.middleware("http")` の `_add_request_id` を追加。
  - [ ] 受信 header `X-Request-ID` を尊重 (validation: hex, length 8–64)。
  - [ ] 無い/不正なら `uuid.uuid4().hex` で生成。
  - [ ] `request.state.request_id = ...` に格納。
  - [ ] response header に `X-Request-ID` を必ず付ける。
- [ ] middleware 順序: security headers の前に置く (エラー時にも `X-Request-ID` が付くように)。
- [ ] `search()` 内の request_id はこの middleware の結果を取り出す: `request_id = fastapi_request.state.request_id`。

#### 1.6 Engine / ruleset version 解決
- [ ] `_build_ruleset_versions(language: str) -> dict[str, str]` を `src/api/main.py` に追加。
  - [ ] 入力 language の `profile.rules_dir` から `get_rules_version()` を呼ぶ (`src/phonology/explainer.py`)。
  - [ ] 結果は `{language_id: "<max_version>"}` 形式。
  - [ ] 例外時は空 dict (warn ログ)。
- [ ] `_APP_VERSION` (`src/api/main.py:111`) を `engine_version` として `ResponseMeta` に流す。

#### 1.7 `search()` の組み立てを書き換え
- [ ] `SearchResponse` 生成時に `meta=build_response_meta(...)` を必ず渡すよう修正 (`src/api/main.py:632-650`)。
- [ ] `meta.request_echo` には sanitized request を入れる。
- [ ] PII リスクは古典語クエリでは低いが、`PROTEUS_LOG_RAW_SEARCH_QUERY` が false の場合 echo は出すがログには出ないことをコメントで明記。

#### 1.8 Constants & API version
- [x] `src/api/_constants.py` (新規) を作成:
  - [x] `API_VERSION = "1.0"`
  - [x] `SCHEMA_VERSION = "1.0.0"`
- [x] `SCHEMA_VERSION` の bump policy を docstring に書く。`SearchResponse` / `LanguagesResponse` の互換性が崩れない追加は minor、追加+互換性破壊は major。
- [ ] 章 1 実装時は、章 3 で確立済みの `API_VERSION` / `SCHEMA_VERSION` constants を参照する。`ResponseMeta.api_version` は `src/api/_constants.py` から `API_VERSION` を import し、schema/version fields が必要なら `SCHEMA_VERSION` も同じ path から import する。`VersionInfo.api_version` と同じ import path を使い、runtime で `ResponseMeta.api_version` と `VersionInfo.api_version` が同一定数を参照している状態を維持する。

#### 1.9 OpenAPI export
- [ ] `scripts/export_openapi.py` (新規) を作成。
  - [ ] `from api.main import app` してから `app.openapi()` を取り、`json.dump` で `docs/api/openapi.json` に書き出す。
  - [ ] CLI 引数: `--output docs/api/openapi.json --indent 2`。
  - [ ] `--check` モード: 生成結果と commit 済み JSON を diff し非ゼロ exit (CI drift 検知用)。
- [ ] `docs/api/openapi.json` を repo にコミット (確定済み設計判断)。

#### 1.10 Tests
- [ ] `tests/test_api_search_meta.py` (新規):
  - [ ] `test_search_response_includes_meta_envelope`: `meta.engine_version`, `meta.api_version`, `meta.schema_version` が存在し非空。
  - [ ] `test_search_response_meta_includes_request_id_uuid4_hex`: `meta.request_id` が 32 桁 hex。
  - [ ] `test_search_response_meta_includes_verification_url`: `meta.verification_url` が `query` `language` `dialect` を含む URL。
  - [ ] `test_search_response_meta_includes_timestamp_iso8601`: `datetime.fromisoformat` で parse 可能。
  - [ ] `test_search_response_meta_includes_ruleset_versions_for_language`: `meta.ruleset_versions["ancient_greek"]` が `data_versions.rules` と一致。
  - [ ] `test_search_response_meta_data_versions_mirrors_top_level`: `meta.data_versions == data_versions`。
  - [ ] `test_search_response_meta_includes_request_echo`: `meta.request_echo.query_form == request.query_form` 等。
  - [ ] `test_search_response_meta_engine_version_matches_app_version`: `meta.engine_version == api_main._APP_VERSION`。
- [ ] `tests/test_api_request_id.py` (新規):
  - [ ] `test_x_request_id_header_set_when_absent`: client が送らないとき response header `X-Request-ID` が hex string。
  - [ ] `test_x_request_id_header_respected_when_valid`: client が `X-Request-ID: abc123...` を送ると同値が response header に返る。
  - [ ] `test_x_request_id_header_replaced_when_invalid`: 不正値 (空 / 非 hex / 異常長) は server-generated value で上書き。
  - [ ] `test_request_id_present_on_error_responses`: 400/503 でも `X-Request-ID` が付く。
- [ ] `tests/test_api_main.py` の既存 `TestSearchEndpoint` で response shape を assert している箇所が壊れないことを確認。新規 `meta` の存在を明示的にテストする項目を 1 つ追加。

### Files to create / modify
- Create: `src/api/_request_context.py`
- Create: `src/api/_response_meta.py`
- Create: `src/api/_constants.py`
- Create: `scripts/export_openapi.py`
- Create: `tests/test_api_search_meta.py`
- Create: `tests/test_api_request_id.py`
- Create: `docs/api/openapi.json` (生成物、コミット)
- Modify: `src/api/_models.py` — `ResponseMeta`, `RequestEcho`, `SearchResponse.meta` 追加。
- Modify: `src/api/main.py` — request id middleware、`_build_ruleset_versions`、`search()` の response 組み立て。

### Test commands
- `uv run pytest tests/test_api_search_meta.py -q`
- `uv run pytest tests/test_api_request_id.py -q`
- `uv run pytest tests/test_api_main.py -q`
- `uv run python scripts/export_openapi.py --output docs/api/openapi.json`

### Acceptance
- `POST /search` レスポンスが `meta.engine_version`, `meta.ruleset_versions`, `meta.request_id`, `meta.verification_url`, `meta.request_echo`, `meta.timestamp`, `meta.api_version`, `meta.schema_version` を含む。
- `X-Request-ID` response header が付与される。
- 既存 `data_versions` トップレベルが互換維持。
- `python -m scripts.export_openapi --output docs/api/openapi.json` が成功し JSON が valid。

---

## 2. GET /languages Endpoint

### Goal
登録済みの `LanguageProfile` を機械可読で公開し、Phase 5 以降の多言語拡張で API 仕様を変えずに済む基盤を作る。

Staged migration note: 章 2 の done state は partial/temporary done とする。`LanguagesResponse.meta` は章 1 完了まで `VersionInfo` を一時再利用し、章 1 の PR で `ResponseMeta` へ置換する。

### Current gap
- `src/phonology/profiles.py` には enumeration API がない (`_REGISTRY` は private dict)。`register_language_profile` / `get_language_profile` のみ public (`src/phonology/profiles.py:45-106`)。
- `GET /languages` が未実装。

### Atomic checklist

#### 2.1 Registry 公開関数
- [x] `src/phonology/profiles.py` に `list_language_profiles() -> tuple[LanguageProfile, ...]` を追加。
  - [x] `_REGISTRY_LOCK` を取得して `_REGISTRY.values()` を snapshot し、`tuple()` で凍結。
  - [x] 戻り値は `language_id` で昇順ソート。
  - [x] docstring: registry の current snapshot を返す。後続の register 操作は反映されない。
- [x] `list_language_profiles` を `__all__` に追加 (`src/phonology/profiles.py:167-173`)。

#### 2.2 Pydantic モデル: `LanguageInfo` / `LanguagesResponse`
- [x] `src/api/_models.py` に `LanguageInfo(BaseModel)` を追加。フィールド:
  - [x] `language_id: str`
  - [x] `display_name: str`
  - [x] `default_dialect: str`
  - [x] `supported_dialects: list[str]`
  - [x] `status: Literal["pilot", "experimental", "stable"]` — API 層で `language_id` ベースに決める。`ancient_greek` は `"pilot"`、他 default `"experimental"`。
  - [x] `ruleset_version: str` — 該当言語の `get_rules_version` 集計 max。エラー時は `"unknown"`。
  - [x] `lexicon_schema_version: str` — `_load_lexicon_document(language).schema_version`。エラー時は `"unknown"`。
  - [x] `matrix_version: str` — `_load_distance_matrix_with_meta(language)[1]["version"]`。エラー時は `"unknown"`。
  - [x] `description: str` — API 層で `language_id` ベースの固定マッピング。未知の language は `""`。
- [x] `LanguagesResponse(BaseModel)` を追加:
  - [x] `languages: list[LanguageInfo]`
  - [x] `meta: VersionInfo` — partial/temporary done。章 1 の `ResponseMeta` は未実装のため、今回の章 2 では既存 `/version` と同じ `VersionInfo` metadata を一時再利用する。
  - [ ] follow-up: 章 1 完了時に `LanguagesResponse.meta` を `VersionInfo` から `ResponseMeta` へ置換し、`scripts/export_openapi.py`, `docs/api/openapi.json`, 関連 tests / CI expectation を同じ PR で更新する。
  - [x] docs note: 章 1 で `meta` が `ResponseMeta` に置換される際も互換性を保つため、`LanguageInfo` / `LanguagesResponse.meta` と `/languages` endpoint を読む client は JSON object の未知 field を許容する。これは既存 client config の unknown-field tolerance 方針と揃える。
- [x] `LanguageInfo`, `LanguagesResponse`, `VersionInfo` を module-level `__all__` に追加。章 1 で `ResponseMeta` / `RequestEcho` を追加する時も、同じ `__all__` に追加する。

#### 2.3 `/languages` 実装
- [x] `src/api/main.py` に `@app.get("/languages", response_model=LanguagesResponse)` を追加。
  - [x] `phonology.profiles.list_language_profiles()` を呼ぶ。
  - [x] 各 `LanguageProfile` に対して `LanguageInfo` を組み立てる。
  - [x] エラー (lexicon/matrix が無い等) は当該 field を `"unknown"` にしてレスポンスを 200 で返す。
  - [x] 全 language の build に失敗した場合のみ 503。実装上は登録 profile が 0 件の場合のみ 503 とし、個別 asset 失敗は `"unknown"` に degrade。
- [x] HEAD `/languages` を追加 (frontend 用 lightweight probe)。
  - [ ] follow-up: 章 1 で error envelope を正式策定した際に、`/languages` GET の 503 と HEAD 503 の body 形式を共通 envelope と整合させる (現在は FastAPI default `{"detail": ...}`)。

#### 2.4 言語別 LanguageInfo helper
- [x] `_build_language_info(profile: LanguageProfile) -> LanguageInfo` を `src/api/main.py` に追加。
  - [x] 内部で `get_rules_version(profile.rules_dir)` を try/except で実行。
  - [x] `_load_distance_matrix_with_meta(profile.language_id)` を try/except で実行。
  - [x] `_load_lexicon_document(profile.language_id)` を try/except で実行。
- [x] `description` の i18n は Phase 2 では英語のみ。`response_language` をクエリで受けるかは Phase 3 以降の課題 (TODO docstring)。

#### 2.5 Tests
- [x] `tests/test_api_languages.py` (新規):
  - [x] `test_languages_endpoint_lists_ancient_greek`: `LanguagesResponse.languages` に `language_id == "ancient_greek"` が含まれる。
  - [x] `test_languages_endpoint_returns_supported_dialects`: `attic` と `koine` が含まれる。
  - [x] `test_languages_endpoint_returns_ruleset_version`: `ruleset_version` が semver 文字列。
  - [x] `test_languages_endpoint_includes_toy_language_when_registered`: `isolated_language_registry` を使い、toy profile を register したら enumeration に含まれる。
  - [x] `test_languages_endpoint_sorted_by_language_id`: 戻り値が `language_id` で昇順。
  - [x] `test_languages_endpoint_includes_response_meta`: `meta.engine_version` 等が章 1 と同じく含まれる。
  - [x] `test_languages_endpoint_returns_unknown_on_missing_assets`: monkeypatch で `get_rules_version` を raise させて `"unknown"` になる。
- [x] `tests/test_language_profiles.py` に `list_language_profiles` の単体テストを追加:
  - [x] `test_list_language_profiles_snapshot_returns_registered_profiles`
  - [x] `test_list_language_profiles_sorted_by_language_id`
  - [x] `test_list_language_profiles_returns_tuple`

### Files to create / modify
- Modify: `src/phonology/profiles.py` — `list_language_profiles` 追加。
- Modify: `src/api/_models.py` — `LanguageInfo`, `LanguagesResponse`。
- Modify: `src/api/main.py` — `_build_language_info`, `GET/HEAD /languages`。
- Create: `tests/test_api_languages.py`.
- Modify: `tests/test_language_profiles.py`.

### Test commands
- `uv run pytest tests/test_api_languages.py -q`
- `uv run pytest tests/test_language_profiles.py -q`

### Acceptance
- `GET /languages` が 200 で `LanguagesResponse` を返す。
- `ancient_greek` が必ず含まれ、`supported_dialects` に `attic` と `koine` が含まれる。
- 新言語登録で API spec を変更せず enumeration に反映される。

---

## 3. GET /version Endpoint

### Goal
エンジン本体のバージョンと API スキーマのバージョンを軽量に返すエンドポイントを公開する。per-language の ruleset version は `/languages` で返す方針と分離。

### Current gap
- `_load_app_version()` (`src/api/main.py:82-108`) はあるが、外向きエンドポイントが存在しない。

### Atomic checklist

#### 3.1 Pydantic モデル: `VersionInfo`
- [x] `src/api/_models.py` に `VersionInfo(BaseModel)` を追加。フィールド (確定済み: ミニマル):
  - [x] `engine_version: str` — `_APP_VERSION`。
  - [x] `api_version: str` — `API_VERSION` (`"1.0"`)。
  - [x] `schema_version: str` — `SCHEMA_VERSION` (`"1.0.0"`)。
  - [x] `rule_schema_version: str` — `data/schemas/phonology_rule_file.schema.json` の `$id` 文字列。schema が無ければ `""`。
  - [x] `build_timestamp: str` — env `PROTEUS_BUILD_TIMESTAMP` (未設定なら `""`)。
  - [x] `git_sha: str` — env `PROTEUS_GIT_SHA` (未設定なら `""`)。
  - [x] `python_version: str` — `sys.version_info` から `"X.Y.Z"` 形式。
  - [x] `mcp_server_version: str` — 章 5 の MCP server version。Phase 2 初期は `engine_version` と同値。
- [x] `VersionInfo` を module-level `__all__` に追加する。`ResponseMeta` は未実装のため現時点では含めず、章 1 でモデル追加と同じ PR で export する。

#### 3.2 Rule schema version helper
- [x] `src/api/main.py` に `_load_rule_schema_version() -> str` を追加。
  - [x] `data/schemas/phonology_rule_file.schema.json` を `@lru_cache` で読む。
  - [x] `$id` 文字列 (URL) を返す。schema が無ければ `""`。
- [x] URL のまま返す (一意性確保)。

#### 3.3 `/version` 実装
- [x] `src/api/main.py` に `@app.get("/version", response_model=VersionInfo)` を追加。
- [x] `HEAD /version` を追加。
- [x] `VersionInfo` instance はリクエスト毎に組み立てる (lightweight)。

#### 3.4 Tests
- [x] `tests/test_api_version.py` (新規):
  - [x] `test_version_endpoint_returns_engine_version`: `engine_version == api_main._APP_VERSION`。
  - [x] `test_version_endpoint_returns_api_version`: `api_version == API_VERSION`。
  - [x] `test_version_endpoint_returns_schema_version`: semver pattern。
  - [x] `test_version_endpoint_returns_rule_schema_id`: `rule_schema_version` startswith `"https://"` and ends with `.schema.json`。
  - [x] `test_version_endpoint_python_version_format`: `re.fullmatch(r"\d+\.\d+\.\d+", value)`。
  - [x] `test_version_endpoint_build_timestamp_env_override`: `monkeypatch.setenv("PROTEUS_BUILD_TIMESTAMP", "2026-05-15T00:00:00Z")` で値が反映。
  - [x] `test_version_endpoint_git_sha_env_override`: `monkeypatch.setenv("PROTEUS_GIT_SHA", "abcdef12")` で値が反映。
  - [x] `test_version_endpoint_head_returns_204`。

### Files to create / modify
- Modify: `src/api/_models.py` — `VersionInfo`。
- Modify: `src/api/_constants.py` (章 1.8 で作成) — `API_VERSION`, `SCHEMA_VERSION` がここから供給される。
- Modify: `src/api/main.py` — `/version` endpoint, `_load_rule_schema_version`。
- Create: `tests/test_api_version.py`.

### Test commands
- `uv run pytest tests/test_api_version.py -q`

### Acceptance
- `GET /version` が 200 で `VersionInfo` を返す。
- `engine_version` が `pyproject.toml` または env 由来。
- `rule_schema_version` が `data/schemas/phonology_rule_file.schema.json` の `$id`。

---

## 4. Reproducibility & Verification URL

### Goal
レスポンスから再現可能な検索条件をクライアントが復元できるようにし、LLM への根拠提示 (MCP) でも同じ URL を提供する。

### Current gap
- `meta.request_echo` / `verification_url` / `request_id` は章 1 で `SearchResponse` に追加するが、独立の test surface として「再現可能性」を確認する設計が未整備。

### Atomic checklist

#### 4.1 Verification URL の確定挙動
- [ ] base URL の解決順序: env `PROTEUS_PUBLIC_BASE_URL` > `fastapi_request.base_url`。
- [ ] URL query パラメータ: `q=<query_form>&language=<language>&dialect=<dialect>&max_candidates=<n>&response_language=<en|ja>`。
- [ ] URL から `orthography_hint` を除外 (deprecated)。
- [ ] base URL に trailing slash があってもなくても結合結果が `<base>/?q=...` の形に揃うようにする。

#### 4.2 Helper (章 1.4 と統合)
- [ ] `src/api/_request_context.py` の以下を最終形で実装:
  - [ ] `resolve_public_base_url(fastapi_request: Request) -> str`
  - [ ] `build_verification_url(base: str, request: SearchRequest) -> str`
  - [ ] `build_request_echo(request: SearchRequest) -> RequestEcho`

#### 4.3 MCP response でも同じ verification_url
- [ ] 章 5 で MCP 出力に `meta.verification_url` をそのまま使い回す前提で本章 helper を再利用する。
- [ ] MCP context には `fastapi_request` が無いため、env `PROTEUS_PUBLIC_BASE_URL` 未設定時の verification_url は `""` (warn ログ)。

#### 4.4 Tests
- [ ] `tests/test_api_verification.py` (新規):
  - [ ] `test_verification_url_uses_public_base_url_env`: monkeypatch で `PROTEUS_PUBLIC_BASE_URL=https://proteus.example/` を set すると返却 URL がそれを含む。
  - [ ] `test_verification_url_uses_request_base_url_when_env_unset`: env 未設定で `http://testserver/` を含む。
  - [ ] `test_verification_url_includes_query_form_urlencoded`: 非 ASCII クエリ (`λόγος`) が percent-encode される。
  - [ ] `test_verification_url_includes_language_and_dialect`。
  - [ ] `test_verification_url_excludes_orthography_hint_deprecated`。
  - [ ] `test_verification_url_deterministic_across_calls`: 同じリクエストを 2 回送ると `verification_url` が同じ (request_id は変わる)。
  - [ ] `test_request_echo_mirrors_validated_request_fields`: `request_echo.dialect_hint == "attic"` (default 適用後)。
  - [ ] `test_request_echo_excludes_internal_legacy_flag`: `legacy_language_alias_used` が echo に出ない。
  - [ ] `test_request_id_is_uuid4_hex_lowercase`。

### Files to create / modify
- Modify: `src/api/_request_context.py` (章 1.4 で作成済み想定)。
- Create: `tests/test_api_verification.py`.

### Test commands
- `uv run pytest tests/test_api_verification.py -q`

### Acceptance
- `meta.verification_url` が deterministic に組み立てられ、env override が効く。
- 同じクエリに対して同じ verification_url が返る (`request_id` は毎回変わるが URL は変わらない)。

---

## 5. MCP Server Prototype

### Goal
Python MCP SDK (FastMCP) を使い `proteus-mcp` という別エントリポイントを追加し、`ancient_phonology.search` ツールで既存検索エンジンを呼べるようにする。出力は REST `SearchResponse` と同じ構造化データ + meta envelope を返す。

### Current gap
- `mcp` 依存も `src/mcp_server/` パッケージも存在しない。
- `pyproject.toml` の hatch packages に `src/mcp_server` 未登録。
- `[project.scripts]` セクションが pyproject に未定義。

### Atomic checklist

#### 5.1 依存関係
- [ ] `pyproject.toml` の `[project] dependencies` に `mcp>=1.0.0,<2.0.0` を追加。
- [ ] `[project.optional-dependencies] dev` には MCP テスト用の追加依存は基本不要 (`mcp.shared.memory.create_connected_server_and_client_session` が標準で利用可能)。
- [ ] `[project.scripts]` セクションを `pyproject.toml` に追加:
  - [ ] `proteus-mcp = "mcp_server.server:main"`
- [ ] `[tool.hatch.build.targets.wheel] packages` に `src/mcp_server` を追加。
- [ ] mypy / pyright include パスに `src/mcp_server` が含まれることを確認 (`mypy_path = "src"` で自動的に含まれる)。

#### 5.2 パッケージ骨格
- [ ] 新規ディレクトリ `src/mcp_server/`。
- [ ] `src/mcp_server/__init__.py`:
  - [ ] バージョン定数 `__version__` を定義 (`api.main._load_app_version` を再利用)。
  - [ ] `__all__ = ["__version__"]`。
- [ ] `src/mcp_server/server.py`:
  - [ ] `from mcp.server.fastmcp import FastMCP` を import。
  - [ ] `app = FastMCP("proteus")` を構築。
  - [ ] `register_default_profiles()` を import 時に呼ぶ (API と同様)。
  - [ ] tools をローカルで register。
  - [ ] `def main() -> None:` で `app.run(transport="stdio")` を呼ぶ。
  - [ ] ロガー `logger = logging.getLogger("proteus.mcp")` を設定。
- [ ] `src/mcp_server/tools/__init__.py` (空、または `from .search import register_search_tool`)。
- [ ] `src/mcp_server/tools/search.py`:
  - [ ] `@app.tool("ancient_phonology.search")` で関数を定義。
  - [ ] 入力 Pydantic モデル `McpSearchInput`:
    - [ ] `query_form: str`
    - [ ] `source_language: str = "ancient_greek"`
    - [ ] `dialect_hint: str | None = None`
    - [ ] `max_candidates: int = 10`
    - [ ] `response_language: Literal["en", "ja"] = "en"`
  - [ ] 出力 Pydantic モデル `McpSearchOutput`:
    - [ ] `candidates: list[SearchHit]` — REST と同じ `SearchHit`。
    - [ ] `query: str`
    - [ ] `query_ipa: str`
    - [ ] `query_mode: Literal["Full-form", "Short-query", "Partial-form"]`
    - [ ] `truncated: bool`
    - [ ] `meta: ResponseMeta`。
  - [ ] handler 関数で `_run_search_for_mcp` adapter を呼ぶ。
- [ ] `src/mcp_server/_search_adapter.py`:
  - [ ] `_run_search_for_mcp(input: McpSearchInput) -> McpSearchOutput`。
  - [ ] 内部で REST の `search()` パスと同じ手順:
    - [ ] `request = SearchRequest(query_form=..., language=..., dialect_hint=..., max_candidates=..., response_language=...)` を組み立て validator を通す。
    - [ ] `deps = api_main._load_search_dependencies(request.language)` を呼ぶ。
    - [ ] 直接 `phonology_search.search_execution(...)` を呼ぶ (`api/main.py:586-606` と同じ kwargs)。
    - [ ] hits は `_build_search_hit` (`src/api/_hit_formatting.py`) を再利用。
    - [ ] `ResponseMeta` を `build_response_meta(...)` (章 1.4) で再利用。
    - [ ] `verification_url` は MCP では request の base_url が無いため env `PROTEUS_PUBLIC_BASE_URL` 必須化、未設定なら `""` (log warn)。

#### 5.3 Tool schema export
- [ ] `scripts/export_mcp_schema.py` (新規) — MCP tool I/O schema を JSON で `docs/mcp/tools.json` に書き出す。
  - [ ] `from mcp_server.server import app` してから `app.list_tools()` の結果をシリアライズ。
  - [ ] `--check` モードで commit 済み JSON との diff を返す (CI drift 検知)。

#### 5.4 Example config
- [ ] `docs/mcp/example-claude-desktop.json` を作成:
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

#### 5.5 Logging / observability
- [ ] `src/mcp_server/server.py` でロガー `logger = logging.getLogger("proteus.mcp")`。
- [ ] tool invocation 時に request_id (UUID4) + sanitized query (`_summarize_query_for_logs` を再利用) をログ。
- [ ] MCP server から `phonology_search.search_execution` を呼ぶときに raw query をログするかは REST と同じ env flag `PROTEUS_LOG_RAW_SEARCH_QUERY` で制御。

#### 5.6 Tests
- [ ] `tests/test_mcp_search_tool.py` (新規):
  - [ ] `test_mcp_server_lists_ancient_phonology_search_tool`: `app.list_tools()` に `ancient_phonology.search` が含まれる。
  - [ ] `test_mcp_search_tool_returns_candidates`: 既知クエリ (`λόγος` など) で `candidates` が 1 件以上。
  - [ ] `test_mcp_search_tool_returns_confidence_and_distance`。
  - [ ] `test_mcp_search_tool_returns_applied_rules`: Koine query で `rules_applied` に CCH-009 を含む (or 同等の Phase 1 で確立した assertion)。
  - [ ] `test_mcp_search_tool_returns_meta_envelope`: `meta.engine_version`, `meta.ruleset_versions`, `meta.request_id`, `meta.verification_url`。
  - [ ] `test_mcp_search_tool_respects_max_candidates`: `max_candidates=2` で 2 件以内。
  - [ ] `test_mcp_search_tool_validates_unsupported_language`: 未登録 language で MCP error。
  - [ ] `test_mcp_search_tool_validates_dialect_hint`: 未対応 dialect で error。
  - [ ] `test_mcp_search_tool_uses_default_dialect_when_omitted`。
  - [ ] `test_mcp_search_tool_response_language_ja`: ja 指定で日本語 prose が返る。
  - [ ] `test_mcp_search_tool_verification_url_uses_env_when_set`。
  - [ ] `test_mcp_search_tool_verification_url_empty_when_env_unset`。
- [ ] `tests/test_mcp_server_init.py` (新規):
  - [ ] `test_mcp_server_module_importable`: `import mcp_server.server` が succeeds。
  - [ ] `test_mcp_server_entrypoint_script_resolves`: `importlib.metadata.entry_points()` に `proteus-mcp` が出る (`pip install -e .` 後)。
  - [ ] `test_mcp_server_version_matches_app_version`: `mcp_server.__version__ == api.main._APP_VERSION`。
- [ ] MCP test fixture は `app` を直接呼ぶ in-process tests とする (stdio をモックしない)。Python MCP SDK の `create_connected_server_and_client_session` を使うのが標準。

### Files to create / modify
- Create: `src/mcp_server/__init__.py`
- Create: `src/mcp_server/server.py`
- Create: `src/mcp_server/tools/__init__.py`
- Create: `src/mcp_server/tools/search.py`
- Create: `src/mcp_server/_search_adapter.py`
- Modify: `pyproject.toml` — `[project] dependencies`, `[project.scripts]`, `[tool.hatch.build.targets.wheel] packages`。
- Create: `scripts/export_mcp_schema.py`
- Create: `docs/mcp/example-claude-desktop.json`
- Create: `tests/test_mcp_search_tool.py`
- Create: `tests/test_mcp_server_init.py`

### Test commands
- `uv run pytest tests/test_mcp_search_tool.py -q`
- `uv run pytest tests/test_mcp_server_init.py -q`
- `uv run python -c "from mcp_server.server import app; print([t.name for t in app._tool_manager.list_tools()])"` (smoke)

### Acceptance
- `proteus-mcp` がインストール後に CLI として起動できる (stdio)。
- `ancient_phonology.search` が `query_form="λόγος"` で REST `/search` と同等の構造化結果 + meta を返す。
- `tests/test_mcp_search_tool.py` 全 pass。

---

## 6. Documentation

### Goal
REST API と MCP の使用方法、出力スキーマ、deprecation policy をドキュメント化する。Phase 2 受け入れ基準 "API and MCP output schemas are documented" を満たす。

### Current gap
- `docs/API.md` 不在。
- `docs/MCP.md` 不在。
- `docs/api/openapi.json` 不在 (章 1.9 で生成)。
- `README.md` に API / MCP セクションが薄い。

### Atomic checklist

#### 6.1 `docs/API.md`
- [ ] H1 タイトル `# Proteus REST API` を付け、Phase 2 v1.0 として明記。
- [ ] セクション構成:
  - [ ] `## Versioning` — `api_version`, `schema_version`, deprecation policy。
  - [ ] `## Endpoints`:
    - [ ] `POST /search` (Request / Response / errors / examples / curl)。
    - [ ] `GET /languages`。
    - [ ] `GET /version`。
    - [ ] `GET /health`, `GET /ready`。
    - [ ] `GET /` (frontend HTML), `GET /changelog`。
  - [ ] `## Request / Response Models` — `SearchRequest`, `SearchResponse`, `SearchHit`, `RuleStep`, `OrthographicNote`, `DataVersions`, `ResponseMeta`, `RequestEcho`, `LanguageInfo`, `LanguagesResponse`, `VersionInfo` を semantic table で。
  - [ ] `## Error Responses` — 400, 422, 503 と detail 文字列。
  - [ ] `## Reproducibility` — `request_id`, `verification_url`, `request_echo` の使い方。
  - [ ] `## Deprecation Policy` — `Deprecation: true` header, `Link: deprecation` header, `X-Proteus-Migration`, `orthography_hint` の deprecated 注記。
  - [ ] `## Environment Variables` — `PROTEUS_ALLOWED_ORIGINS`, `PROTEUS_PUBLIC_BASE_URL`, `PROTEUS_APP_VERSION`, `PROTEUS_ENABLE_API_DOCS`, `PROTEUS_LOG_RAW_SEARCH_QUERY`, `PROTEUS_DISABLE_STARTUP_WARMUP`, `PROTEUS_GIT_SHA`, `PROTEUS_BUILD_TIMESTAMP`。
  - [ ] `## OpenAPI Schema` — `docs/api/openapi.json` への参照と再生成コマンド。
  - [ ] `## Examples` — curl, httpie, Python (`httpx`) でクエリする 3 例。
- [ ] `docs/API.md` は英語、重要 section は日本語要約を付ける (REQUIREMENTS / ROADMAP と一貫)。

#### 6.2 `docs/MCP.md`
- [ ] H1 `# Proteus MCP Server`。
- [ ] セクション:
  - [ ] `## Overview` — `ancient_phonology.search` ツールが提供する機能。
  - [ ] `## Installation` — `uv pip install proteus` または `pipx install proteus`。
  - [ ] `## Running` — `proteus-mcp` (stdio transport)。
  - [ ] `## Claude Desktop config` — `docs/mcp/example-claude-desktop.json` 参照。
  - [ ] `## Tool Reference: ancient_phonology.search` — input / output schema、例。
  - [ ] `## Output Schema` — `McpSearchOutput` の全フィールド表。
  - [ ] `## Verification & Reproducibility` — `meta.verification_url`, `meta.request_id`。
  - [ ] `## Limitations` — 単一 transport (stdio)、認証なし、Phase 2 prototype と明記。
  - [ ] `## Future Tools` — `ancient_phonology.explain_rule`, `ancient_phonology.list_languages`, `ancient_phonology.list_rules`, `ancient_phonology.compare_candidates` (`docs/REQUIREMENTS.md` §4.4)。
- [ ] `docs/mcp/example-claude-desktop.json` を作成し MCP.md からリンク。
- [ ] `docs/mcp/tools.json` (章 5.3 で生成) へのリンクを記載。

#### 6.3 OpenAPI artifact
- [ ] `docs/api/openapi.json` を `scripts/export_openapi.py` (章 1.9) で生成し commit する。
- [ ] `docs/API.md` の `## OpenAPI Schema` セクションで `uv run python scripts/export_openapi.py --output docs/api/openapi.json` を再生成コマンドとして文書化。
- [ ] CI で `scripts/export_openapi.py --check` を実行し、生成結果と commit 済み JSON を diff し非ゼロ exit させる (確定済み設計判断)。

#### 6.4 README 更新
- [ ] `README.md` の Table of Contents に "REST API" と "MCP Server" を追加。
- [ ] `## REST API` セクションを追加: 簡単な curl 例 + `docs/API.md` へのリンク。
- [ ] `## MCP Server` セクションを追加: `proteus-mcp` 起動例 + `docs/MCP.md` へのリンク。
- [ ] 既存の日英併記スタイルを継承する。

#### 6.5 CODEMAPS 更新
- [ ] `docs/CODEMAPS/api.md` を更新:
  - [ ] 新規モデル (`ResponseMeta`, `RequestEcho`, `LanguageInfo`, `LanguagesResponse`, `VersionInfo`) を追加。
  - [ ] 新規 endpoint (`GET /languages`, `GET /version`) を追加。
  - [ ] middleware (`_add_request_id`) を追加。
- [ ] `docs/CODEMAPS/INDEX.md` に MCP entry を追加 (`docs/CODEMAPS/mcp.md` 新規)。
- [ ] `docs/CODEMAPS/mcp.md` (新規):
  - [ ] `src/mcp_server/server.py`, `src/mcp_server/tools/search.py`, `src/mcp_server/_search_adapter.py`, `tests/test_mcp_search_tool.py` の関係図。

#### 6.6 CHANGELOG
- [ ] `CHANGELOG.md` に `## [0.3.0] Phase 2: REST API and MCP Prototype` を追加:
  - [ ] Added: `/languages`, `/version`, `meta` envelope, `X-Request-ID`, `verification_url`, `proteus-mcp` script, `ancient_phonology.search` MCP tool。
  - [ ] Documentation: `docs/API.md`, `docs/MCP.md`, `docs/api/openapi.json`。
  - [ ] Compatibility: 既存 `data_versions` トップレベルは維持。

### Files to create / modify
- Create: `docs/API.md`
- Create: `docs/MCP.md`
- Create: `docs/mcp/example-claude-desktop.json`
- Create: `docs/mcp/tools.json` (生成物)
- Create: `docs/api/openapi.json` (生成物)
- Create: `docs/CODEMAPS/mcp.md`
- Modify: `docs/CODEMAPS/api.md`
- Modify: `docs/CODEMAPS/INDEX.md`
- Modify: `README.md`
- Modify: `CHANGELOG.md`

### Acceptance
- `docs/API.md` が全 endpoint と全モデルを説明している。
- `docs/MCP.md` が tool reference と Claude Desktop config を含む。
- `docs/api/openapi.json` が `app.openapi()` の出力と一致 (drift check)。

---

## 7. CI / Test Coverage

### Goal
Phase 2 で追加した endpoints / MCP server をテストスイートに統合し、CI で常に検証する。

### Current gap
- `.github/workflows/ci.yml` には `validate-matrix`, `type-check`, `test` の 3 ジョブのみ。
- OpenAPI drift / MCP schema drift / openapi spec validity の自動チェックがない。

### Atomic checklist

#### 7.1 既存 test 拡張
- [ ] `tests/test_api_main.py` の `TestSearchEndpoint` に下記を 1 件ずつ追加:
  - [ ] `test_search_response_includes_meta_envelope` (章 1 と重複 OK — defence in depth)。
  - [ ] `test_search_response_x_request_id_header_present`。
- [ ] `tests/test_packaging.py` に下記を追加:
  - [ ] `test_proteus_mcp_script_installed`: `importlib.metadata.entry_points` (group `console_scripts`) に `proteus-mcp` が含まれる。
  - [ ] `test_mcp_server_module_in_wheel`: built wheel に `mcp_server/server.py` が含まれる (`uv build --wheel` 後の ZipFile を確認、既存 `tests/test_packaging.py` のパターンに合わせる)。
- [ ] `tests/test_data_files.py` に下記を追加 (任意):
  - [ ] `test_rule_schema_id_is_resolvable_url`: `data/schemas/phonology_rule_file.schema.json` の `$id` が URL 形式。

#### 7.2 新規 test ファイル (章 1–5 で列挙)
- [ ] `tests/test_api_search_meta.py` (章 1)
- [ ] `tests/test_api_request_id.py` (章 1)
- [ ] `tests/test_api_languages.py` (章 2)
- [x] `tests/test_api_version.py` (章 3)
- [ ] `tests/test_api_verification.py` (章 4)
- [ ] `tests/test_mcp_search_tool.py` (章 5)
- [ ] `tests/test_mcp_server_init.py` (章 5)
- [ ] `tests/test_openapi.py` (新規 — 7.3 参照)

#### 7.3 OpenAPI / MCP schema drift 検出
- [ ] `tests/test_openapi.py`:
  - [ ] `test_openapi_schema_is_valid_jsonschema`: `app.openapi()` の結果が OpenAPI 3.1 として妥当 (`jsonschema` で簡易検証)。
  - [ ] `test_openapi_schema_contains_search_endpoint`。
  - [ ] `test_openapi_schema_contains_languages_endpoint`。
  - [ ] `test_openapi_schema_contains_version_endpoint`。
  - [ ] `test_openapi_artifact_matches_app_openapi`: `docs/api/openapi.json` と `app.openapi()` が一致 (drift 検知)。fail 時には修復コマンド (`uv run python scripts/export_openapi.py --output docs/api/openapi.json`) を message に含める。
- [ ] `tests/test_mcp_schema.py` (任意):
  - [ ] `test_mcp_tools_artifact_matches_runtime`: `docs/mcp/tools.json` が `app.list_tools()` と一致。
  - [ ] `test_mcp_search_tool_input_schema_includes_required_fields`。
  - [ ] `test_mcp_search_tool_output_schema_includes_meta_envelope`。

#### 7.4 CI workflow 更新
- [ ] `.github/workflows/ci.yml` の `test` ジョブで `uv pip install -e .[dev]` が `mcp>=1.0.0` を含む依存をインストールすることを確認。
- [ ] `mcp` の wheel が Python 3.11/3.12 両方で available か PyPI で確認。
- [ ] CI に新しいステップを追加:
  - [ ] `Validate OpenAPI artifact`: `uv run python scripts/export_openapi.py --check` (drift がないことを確認)。
  - [ ] `Validate MCP schema artifact`: `uv run python scripts/export_mcp_schema.py --check`。

#### 7.5 Linting / typing
- [ ] `src/mcp_server/` 配下を mypy / pyright が型チェックすることを確認 (`tool.mypy.files = ["src"]` で含まれる)。
- [ ] `uv run mypy` / `uv run pyright` が green であること。

### Files to create / modify
- Modify: `.github/workflows/ci.yml`
- Modify: `tests/test_api_main.py` (additions)
- Modify: `tests/test_packaging.py` (additions)
- Create: `tests/test_openapi.py`
- Create: `tests/test_mcp_schema.py` (任意)

### Test commands
- `uv run pytest -q` (全体 green を確認)
- `uv run python scripts/export_openapi.py --check`
- `uv run mypy`
- `uv run pyright`

### Acceptance
- 全 test 通過 (`uv run pytest -q`)。
- CI が openapi drift / MCP schema drift を fail する。

---

## 8. Acceptance Verification

### Goal
ROADMAP Phase 2 受け入れ基準を、各々具体的なテストまたは手動検証ステップに紐付けて緑にし、ROADMAP に "✅ COMPLETE" マークを反映する。

### Atomic checklist

#### 8.1 Acceptance criterion mapping
- [ ] **API returns structured candidates**
  - [ ] 検証: `uv run pytest tests/test_api_main.py::TestSearchEndpoint -q` が pass。
  - [ ] 検証: `curl -s -X POST http://localhost:8000/search -H 'Content-Type: application/json' -d '{"query_form": "λόγος"}' | jq '.hits | length' >= 1`。
- [ ] **API returns applied rules and explanations**
  - [ ] 検証: 上記レスポンスで `.hits[0].rules_applied` が配列、`.hits[0].explanation` が非空。
  - [ ] 検証: `tests/test_search.py` / `tests/test_search_annotation.py` の Phase 1 受け入れテストが緑。
- [ ] **MCP server can answer a query through the search engine**
  - [ ] 検証: `uv run pytest tests/test_mcp_search_tool.py::test_mcp_search_tool_returns_candidates -q`。
  - [ ] 手動検証: Claude Desktop で `docs/mcp/example-claude-desktop.json` を設定し、対話で `ancient_phonology.search` を呼び結果が返る。
- [ ] **MCP response includes candidates, confidence, applied rules, and metadata**
  - [ ] 検証: `tests/test_mcp_search_tool.py::test_mcp_search_tool_returns_meta_envelope` が pass。
  - [ ] 検証: `tests/test_mcp_search_tool.py::test_mcp_search_tool_returns_applied_rules` が pass。
- [ ] **API and MCP output schemas are documented**
  - [ ] 検証: `docs/API.md` 存在し全 endpoint / 全モデルを記載。
  - [ ] 検証: `docs/MCP.md` 存在し tool reference を記載。
  - [ ] 検証: `docs/api/openapi.json` が `app.openapi()` と一致 (`scripts/export_openapi.py --check`)。

#### 8.2 Phase 2 完了前の最終チェック
- [ ] フル test を 1 度通す: `uv run pytest -q` (Phase 2 開始時の baseline を超える pass 数になっていること)。
- [ ] `uv run mypy` / `uv run pyright` 共に green。
- [ ] `uv build --wheel` が成功し、`mcp_server` package が wheel 内に含まれる。
- [ ] `uv run proteus-mcp` が立ち上がる (stdio で hang up しない、SIGINT で終了する) — 手動 smoke。
- [ ] Phase 2 の release tag を打つ: `v0.3.0` (additive changes のみ、minor bump)。
- [ ] `docs/ROADMAP.md` の Phase 2 section を `## 4. Phase 2: REST API and MCP Prototype ✅ COMPLETE` に更新し、各受け入れ基準にチェックを入れる。
- [ ] `CHANGELOG.md` を `## [0.3.0]` として release date 付きで確定。
- [ ] `pyproject.toml` の `version` を `0.3.0` に bump (Phase 1 までは `0.2.1`)。

### Files to modify (final)
- Modify: `docs/ROADMAP.md`
- Modify: `CHANGELOG.md`
- Modify: `pyproject.toml` (version bump)

### Acceptance
- ROADMAP Phase 2 が ✅ COMPLETE に更新できる状態。
- `git diff` で本 plan の Files to create / modify 一覧と一致する変更が repository に存在する。

---

## Suggested Implementation Order

実装はこの順番で進める。各ステップは独立 PR にできる粒度。
順序依存: step 2 の `/version` で `VersionInfo` と `API_VERSION` / `SCHEMA_VERSION` を確立し、step 3 の `/languages` は章 1 が入るまで `LanguagesResponse.meta` でその `VersionInfo` を一時再利用する。step 4 で `ResponseMeta` を導入した後、`LanguagesResponse.meta` は `VersionInfo` から関連する別モデルの `ResponseMeta` へ staged migration するため、章 2 と章 1 後で `/languages` response schema が変わる。この schema drift は `scripts/export_openapi.py` と `docs/api/openapi.json`、および関連 tests / CI expectation を同じ PR で更新して検知・処理する。

1. **章 0 baseline** — test 数値を記録し、Phase 2 のコミットメントを README/CHANGELOG に予約。
2. **章 3 `/version`** — `_load_app_version()` を活用するだけの最小 endpoint。ここで `API_VERSION` / `SCHEMA_VERSION` 定数を確立。
3. **章 2 `/languages`** — `list_language_profiles()` 追加。新言語追加への基盤。完了。
4. **章 1 + 章 4 `meta` envelope + verification URL + request_id** — `SearchResponse` 変更。Pydantic モデル追加と middleware 追加を 1 PR。
5. **章 1.9 + 章 6.3 OpenAPI artifact** — `scripts/export_openapi.py` + `docs/api/openapi.json` を commit + CI drift 検知。
6. **章 5 MCP server prototype** — 依存追加 + パッケージ作成 + `ancient_phonology.search` tool + test。
7. **章 6 documentation** — `docs/API.md`, `docs/MCP.md`, README / CODEMAPS / CHANGELOG。
8. **章 7 CI 統合** — openapi / mcp schema drift 検知の CI step を追加。
9. **章 8 acceptance** — ROADMAP 更新と version bump (`0.3.0`)、release tag。

---

## Definition of Done

- [ ] `GET /version`, `GET /languages` が green-on-CI で動作する。
- [ ] `POST /search` レスポンスに `meta.engine_version`, `meta.ruleset_versions`, `meta.request_id`, `meta.verification_url`, `meta.request_echo` が含まれる。
- [ ] `X-Request-ID` レスポンスヘッダが全 endpoint で常に付与される。
- [ ] `proteus-mcp` script がインストール可能で stdio transport で起動する。
- [ ] `ancient_phonology.search` MCP tool が REST `/search` と同等の構造化レスポンス + meta envelope を返す。
- [ ] `docs/API.md`, `docs/MCP.md`, `docs/api/openapi.json`, `docs/mcp/tools.json`, `docs/mcp/example-claude-desktop.json` がコミットされている。
- [ ] CI が openapi drift と MCP schema drift を検出して fail させる。
- [ ] Phase 2 開始時の baseline `uv run pytest -q` pass 数を、新規追加 test 分だけ上回って green。
- [ ] `docs/ROADMAP.md` Phase 2 が ✅ COMPLETE。
- [ ] `pyproject.toml` の version が `0.3.0`。

---

## Verification

実装後、Phase 2 が完了したことを以下のコマンドで確認する。

```bash
# 1. 全 test green
uv run pytest -q

# 2. 型チェック green
uv run mypy
uv run pyright

# 3. OpenAPI drift がない
uv run python scripts/export_openapi.py --check

# 4. MCP schema drift がない
uv run python scripts/export_mcp_schema.py --check

# 5. REST endpoints が動く
uv run uvicorn api.main:app --reload &
curl -s http://localhost:8000/version | jq .
curl -s http://localhost:8000/languages | jq .
curl -s -X POST http://localhost:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query_form": "λόγος"}' | jq '.meta'

# 6. MCP server が起動する (smoke test)
uv build --wheel
uv pip install dist/proteus-*.whl
proteus-mcp &
# stdio transport なので別の MCP client (Claude Desktop or `mcp` CLI) で接続確認

# 7. Wheel に mcp_server が含まれる
python -c "import zipfile; z=zipfile.ZipFile('dist/proteus-0.3.0-py3-none-any.whl'); print([n for n in z.namelist() if 'mcp_server' in n])"
```

### Critical Files for Implementation

- `src/api/main.py`
- `src/api/_models.py`
- `src/phonology/profiles.py`
- `pyproject.toml`
- `tests/test_api_main.py`
- `src/mcp_server/server.py` (新規 — MCP work の中心)
