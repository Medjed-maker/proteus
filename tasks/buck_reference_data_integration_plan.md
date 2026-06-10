# Buck Reference Data Integration Implementation Plan

作成日: 2026-06-09  
対象: `data/languages/ancient_greek/rules/buck/`  
目的: Buck 正規化データを、まず参照データとして安全に利用可能にし、その後、検索・説明・方言対応へ段階的に接続する。

## 背景

`data/languages/ancient_greek/rules/buck/` には Buck 由来の暫定正規化データがある。

- `grammar_rules.yaml`: Buck 由来の規則サマリー
- `dialects.yaml`: 方言カタログと dialect-to-rule 参照
- `glossary.yaml`: 実例語彙と rule/dialect 参照

現状では `src/phonology/languages/ancient_greek/buck.py` の `load_buck_data()` で 3 YAML を読み込み、参照整合性を検証できる。一方、通常の検索・説明ルールローダーは `rules/` 直下の YAML だけを対象にするため、`buck/` サブディレクトリのデータは検索・説明 API の通常フローでは使われていない。

この計画では、Buck データをいきなり検索スコアや候補生成に混ぜず、以下の順に進める。

1. 参照データとして内部 service / MCP から安全に取得できるようにする。
2. 検索結果に Buck 由来の根拠注釈を付与する。
3. 実行可能な一部ルールだけを既存 rule schema に変換する（後続 epic）。
4. 方言対応と glossary 拡張を、レビュー境界を保ったまま進める（後続 epic）。

## 確定事項 (Decisions)

実装着手前に固定した契約レベルの決定。以前 Open Questions に残っていた項目を確定済み。

1. **公開面は MCP + 内部 service を先行**する。REST public API は MCP の契約が安定してから出す。
2. **検索結果に付与する Buck 参照フィールド名は `buck_references`** で固定する。
3. **`citation_ready` と `review_status` は、レスポンス全体 metadata と各 item の両方**に持たせる。
4. **ドキュメント構造は MVP + 後続 epic 分割**とする。本書は MVP（Phase 0-2 + MCP exposure + 内部 service）を扱い、実行可能ルール変換・方言拡張・glossary 拡充は別 epic 書に分離する。
   - 実行可能ルール変換: [buck_executable_rule_conversion_epic.md](buck_executable_rule_conversion_epic.md)
   - 方言対応拡張: [buck_dialect_expansion_epic.md](buck_dialect_expansion_epic.md)
   - glossary 拡充 pipeline: [buck_glossary_expansion_epic.md](buck_glossary_expansion_epic.md)

## 進捗凡例

- [ ] 未着手
- [x] 完了
- [~] 実装中または要再検討

## Acceptance Criteria

- [x] Buck データを参照する内部 service と MCP tool がある（REST public は契約安定後）。
- [x] Buck の rule / dialect / glossary を ID・section・dialect・word で検索できる。
- [ ] 既存検索 ranking、distance、rule explanation の挙動を不用意に変更しない。
  - [ ] 固定クエリ集合に対する `/search` golden/snapshot テストで、`buck_references` など追加注釈 field を除外した ranking、distance、candidate ordering、`rules_applied`、`matched_rules` が Buck 統合 ON/OFF で不変であることを保証する。
- [ ] 検索結果に Buck 参照を付ける場合、音韻ルール適用とは別の注釈として扱う。
- [ ] `citation_ready: false` と expert-review boundary が API/docs で明示される。
- [ ] Buck glossary の増補は OCR 本文からの自動投入ではなく、抽出・正規化・レビュー状態を分ける。
- [ ] 方言を増やす場合、`LanguageProfile.supported_dialects`、API validation、UI、テストを同時に更新する。
- [ ] wheel/sdist に Buck データと追加 schema/docs が含まれる。

## Non-Goals

- [ ] Buck 全文 Markdown をそのまま runtime data として同梱しない。
- [ ] Buck の全規則を一括で実行可能ルール化しない。
- [ ] 未レビューの OCR 抽出結果を citation-ready として表示しない。
- [ ] 検索 ranking や距離行列を Buck データで即時変更しない。
- [ ] 商用・権利上制約のある本文テキストを無許諾で再配布しない。

## Phase 0: Baseline & Boundary Confirmation

- [ ] 現在の Buck データ件数を確認する。
  - [ ] `grammar_rules.yaml` の rule 件数
  - [ ] `dialects.yaml` の dialect 件数
  - [ ] `glossary.yaml` の word 件数
- [ ] `load_buck_data()` の現行 validation 範囲を整理する。
  - [ ] top-level mapping validation
  - [ ] required list key validation
  - [ ] duplicate rule id validation
  - [ ] duplicate dialect id validation
  - [ ] dialect-to-rule reference validation
  - [ ] glossary-to-rule reference validation
  - [ ] glossary-to-dialect reference validation
  - [ ] grammar rule dialect reference validation
- [ ] Buck データの status と review boundary を確認する。
  - [ ] `status: provisional`
  - [ ] `review_status: not_expert_reviewed`
  - [ ] `citation_ready: false`
- [ ] `DATA_LICENSE.md` の Buck / source-derived data 方針を確認する。
- [ ] `pyproject.toml` の package inclusion を確認する。
- [ ] 既存 `tests/test_buck_loader.py` と `tests/test_buck_data_files.py` の責務を整理する。
- [ ] 既存 `tests/test_packaging.py` の Buck asset coverage を確認する。
- [ ] 変更対象外の既存挙動を明文化する。
  - [ ] `/search` ranking
  - [ ] phonological distance
  - [ ] `rules_applied`
  - [ ] `matched_rules`
  - [ ] orthographic notes
  - [ ] corpus source references

## Phase 1: Buck Reference Domain Model

- [x] Buck 参照用の内部 model を設計する。
  - [x] `BuckRule`
  - [x] `BuckDialect`
  - [x] `BuckGlossaryEntry`
  - [x] `BuckReference`
  - [x] `BuckReviewStatus`
- [x] phonology 層の index/public モデルは `frozen=True` dataclass に変換する（CLAUDE.md 不変性原則）。
  - [x] `load_buck_data()` のキャッシュ済み可変 dict から frozen dataclass へ変換し、可変構造への参照漏れを防ぐ。
  - [ ] API 層で Pydantic model に変換する。
- [x] `grammar_rules.yaml` の rule field を整理する。
  - [x] `id`
  - [x] `buck_section`
  - [x] `category`
  - [x] `description`
  - [x] `transformation`
  - [x] `affected_dialects`
  - [x] `variants`
  - [x] `notes`
- [x] `dialects.yaml` の dialect field を整理する。
  - [x] `id`
  - [x] `name`
  - [x] `kind`
  - [x] `group`
  - [x] `parent`
  - [x] `rules`
- [x] `glossary.yaml` の word field を整理する。
  - [x] `word`
  - [x] `standard_form`
  - [x] `dialect`
  - [x] `rule_id`
  - [x] `definition`
  - [x] `inscription_no`
  - [x] `buck_ref`
  - [x] `notes`
- [ ] public response 用に返す field と内部専用 field を分ける。
- [ ] `citation_ready` と `review_status` は、レスポンス全体 metadata と各 item の両方に持たせる（確定事項 3）。

## Phase 2: Buck Reference Service

- [x] service の配置先を決める。
  - [x] 候補: `src/phonology/languages/ancient_greek/buck_service.py`
  - [ ] 候補: `src/phonology/languages/ancient_greek/buck.py` に read-only query helper を追加
  - [x] 推奨: `buck.py` は loader、service は別 module に分離する。
- [x] `load_buck_data()` の戻り値から read-only index を作る。
- [x] index の cache 方針（確定）。
  - [x] 2つ目の `lru_cache` は持たない。`load_buck_data()` のキャッシュ済み結果から index を遅延構築する。
  - [x] env override（`PROTEUS_TRUSTED_BUCK_DIR`）時の cache clear は単一関数に集約し、`_load_buck_data_cached.cache_clear()` と index 再構築を同一手順で扱う。
  - [x] cache clear 手順を docs/test に書く。
- [x] section 正準化関数を service 層に1つだけ定義する。
  - [x] `"41.4"`（string）と numeric YAML 値の揺れをここで吸収する。
  - [x] Phase 3（`buck_section` / `buck_ref.section`）と Phase 10（型揺れ検証）はこの関数を共有し、層ごとに再実装しない。
- [x] rule id index を作る。
  - [x] `get_rule(rule_id: str)`
  - [x] `list_rules(category: str | None = None, dialect: str | None = None)`
- [x] Buck section index を作る。
  - [x] `get_rules_by_section(section: str)`
  - [x] section の正規化は上記の単一正準化関数を使用する。
- [x] dialect index を作る。
  - [x] `get_dialect(dialect_id: str)`
  - [x] `list_dialects(kind: str | None = None)`
  - [x] `get_dialect_rules(dialect_id: str, include_inherited: bool = True)`
- [x] dialect inheritance 解決を実装する。
  - [x] parent chain を上方向に辿る。
  - [ ] cycle 検出は load/schema 検証（Phase 10）を単一の真実とし、ここでは「検証済み・非循環」を前提とする。
  - [x] 重複 rule id は first-seen order で dedupe する。
- [x] glossary index を作る。
  - [x] `list_glossary_entries(dialect: str | None = None, rule_id: str | None = None)`
  - [x] `find_glossary_by_word(word: str)`
  - [x] `find_glossary_by_standard_form(standard_form: str)`
- [x] Unicode normalization 方針を決める。
  - [ ] Greek diacritics を厳密一致にするか
  - [x] NFC/NFD 正規化だけ行うか
  - [ ] accent-insensitive search を別 option にするか
- [x] service が source YAML を mutation しないことをテストする。

## Phase 3: API Models

- [ ] `src/api/_models.py` に Buck response model を追加する。
  - [ ] `BuckMetadata`
  - [ ] `BuckRuleInfo`
  - [ ] `BuckDialectInfo`
  - [ ] `BuckGlossaryEntryInfo`
  - [ ] `BuckRulesResponse`
  - [ ] `BuckDialectsResponse`
  - [ ] `BuckGlossaryResponse`
  - [ ] `BuckRuleResponse`
  - [ ] `BuckDialectResponse`
- [ ] Field descriptions を英語で追加する。
- [ ] `citation_ready` と `review_status` を全体 metadata と各 item の両方に持たせる（確定事項 3）。
- [ ] `citation_ready` の意味を description に含める。
- [ ] `review_status` の許可値を Literal 化するか決める。
- [ ] `buck_section` / `buck_ref.section` は Phase 2 の単一 section 正準化関数を通して string で返す。
- [ ] OpenAPI 生成結果の差分方針（確定）。
  - [ ] endpoint 追加は後方互換のため major schema version bump は不要。
  - [ ] `docs/api/openapi.json` を再生成してコミットし、`tests/test_packaging.py` の対象を更新する。

## Phase 4: REST API Endpoints （後続: MCP 契約安定後）

> 確定事項 1 により公開面は MCP 先行。本 Phase の endpoint 設計は残すが、実装は MCP 契約が安定してから着手する。

- [ ] endpoint の URL 設計を決める。
  - [ ] 候補: `/languages/{language}/buck/rules`
  - [ ] 候補: `/languages/{language}/buck/rules/{rule_id}`
  - [ ] 候補: `/languages/{language}/buck/dialects`
  - [ ] 候補: `/languages/{language}/buck/dialects/{dialect_id}`
  - [ ] 候補: `/languages/{language}/buck/glossary`
  - [ ] 推奨: `language` path parameter を入れ、初期は `ancient_greek` のみ対応。
- [ ] unsupported language の扱いを決める。
  - [ ] `404 Not Found`
  - [ ] または `422 Validation Error`
  - [ ] 推奨: registered language だが Buck 非対応なら `404`。
- [ ] `/buck/rules` query params を設計する。
  - [ ] `category`
  - [ ] `dialect`
  - [ ] `section`
  - [ ] `limit`
  - [ ] `offset`
- [ ] `/buck/dialects` query params を設計する。
  - [ ] `kind`
  - [ ] `group`
  - [ ] `include_rules`
  - [ ] `include_inherited`
- [ ] `/buck/glossary` query params を設計する。
  - [ ] `word`
  - [ ] `standard_form`
  - [ ] `dialect`
  - [ ] `rule_id`
  - [ ] `accent_insensitive`
  - [ ] `limit`
  - [ ] `offset`
- [ ] pagination の default と上限を決める。
  - [ ] 推奨: `limit=50`, max `200`
- [ ] response meta に engine/api/schema version を入れるか確認する。
- [ ] API error messages を英語で統一する。
- [ ] endpoint 単位で cache 可能性を検討する。
- [ ] `/languages` response に Buck availability を追加するか決める。
  - [ ] `has_buck_reference_data: true`
  - [ ] `buck_reference_status: "provisional"`
  - [ ] 初期実装では endpoint discovery docs だけに留めるか検討する。

## Phase 5: MCP Tool Exposure （MVP: 公開面の先行実装）

> 確定事項 1 により、Buck 参照は内部 service + MCP tool を先に公開する。本 Phase は MVP の中核。

- [x] MCP に Buck 参照 tool を追加する（確定）。
  - [x] `search_buck_rules`
  - [x] `get_buck_dialect`
  - [x] `search_buck_glossary`
  - [x] 内部 service の response を直接利用する（REST API には依存しない）。
- [x] LLM 向けの system/tool description に review boundary を含める。
- [x] 長文引用を返さないことを tool schema に明記する。
- [x] `citation_ready: false` のとき、断定的な回答を避ける注意文を含める。
- [x] `docs/MCP.md` と `docs/mcp/tools.json` を更新する。
- [x] MCP tests を追加する。

## Phase 6: Search Result Annotation

> 初期スコープは rule_id 対応に限定する。glossary は現状 15 件のため `word` / `standard_form` 一致注釈はほぼ発火せず、費用対効果が低い。glossary `word` / `standard_form` ベースの注釈は [buck_glossary_expansion_epic.md](buck_glossary_expansion_epic.md) 完了後に拡張する。

- [ ] Buck 参照を検索結果に付ける対象（初期）。
  - [ ] applied/matched rule id が Buck `rule_id` と対応（初期スコープ）。
  - [ ] candidate dialect と Buck dialect の一致だけでは初期 annotation 対象にしない。
  - [ ] query form が glossary `word` と一致（glossary 拡充後・NFC 正規化前提）。
  - [ ] candidate headword が glossary `standard_form` と一致（glossary 拡充後・NFC 正規化前提）。
- [ ] annotation model を設計する。
  - [ ] `kind: "buck_reference"`
  - [ ] `rule_id`
  - [ ] `buck_section`
  - [ ] `dialect`
  - [ ] `example_word`
  - [ ] `standard_form`
  - [ ] `message`
  - [ ] `citation_ready`
  - [ ] `review_status`
- [ ] 既存の `orthographic_notes` には入れず、別 field `buck_references` に出す（確定事項 2）。
  - [ ] 理由: Buck は表記補助だけでなく、方言・音韻・形態・統語情報を含む。
- [ ] `citation_ready` と `review_status` を `buck_references` の各 item に含める（確定事項 3）。
- [ ] `rules_applied` / `matched_rules` と混ぜない方針を固定する。
- [ ] ranking score を変更しない。
- [ ] candidate bucket を変更しない。
- [ ] annotation 生成失敗時は検索全体を落とさず warning log にする。
- [ ] MCP の response に `buck_references` field を出す（REST は契約安定後に同 field を追加）。
- [ ] API docs に field の意味と provisional status を書く。

## Phase 7-9: 後続 Epic（別書に分離）

> 確定事項 4 により、以下は MVP 完了後の独立 epic として別ドキュメントに分離した。本書では概要のみ参照する。
>
> - **Phase 7 相当**: Executable Rule Conversion Pilot → [buck_executable_rule_conversion_epic.md](buck_executable_rule_conversion_epic.md)
> - **Phase 8 相当**: Dialect Support Expansion → [buck_dialect_expansion_epic.md](buck_dialect_expansion_epic.md)
> - **Phase 9 相当**: Buck Glossary Expansion Pipeline → [buck_glossary_expansion_epic.md](buck_glossary_expansion_epic.md)

## Phase 10: Data Schema & Validation

- [ ] Buck 専用 JSON Schema を追加するか決める。
  - [ ] `data/schemas/buck_grammar_rules.schema.json`
  - [ ] `data/schemas/buck_dialects.schema.json`
  - [ ] `data/schemas/buck_glossary.schema.json`
- [ ] 既存 Python validation と JSON Schema の責務を分ける。
  - [ ] schema: shape validation
  - [ ] Python: cross-file reference validation
- [ ] `tests/test_buck_data_files.py` を schema validation に拡張する。
- [ ] dialect parent cycle validation を追加する（cycle 検出の単一の真実。Phase 2 の継承解決はこれを前提とする）。
- [ ] `affected_dialects` と `variants[].dialects` の unknown dialect 検証を維持する。
- [ ] glossary `buck_ref.section` 型の揺れは Phase 2 の単一 section 正準化関数で正規化する。
- [ ] glossary `rule_id: null` の許可条件を明記する。
- [ ] `meta.current_entries` と実際の `words` 件数一致を検証する。
- [ ] `meta.total_entries_original` は実件数 validation 対象外にする。
- [ ] 未レビュー OCR draft の一括 merge を防ぐ guard を設ける（OCR draft から `glossary.yaml` へ自動 merge する場合は未承認 entry を拒否する。既存 provisional reference data は `citation_ready: false` として許可する。詳細は glossary epic）。
- [ ] `citation_ready: true` を許可する条件を docs に書く。

## Phase 11: Documentation

- [ ] `docs/ARCHITECTURE.md` に Buck reference layer を追記する。
- [ ] `docs/API.md` に Buck endpoints を追記する。
- [ ] `docs/MCP.md` に Buck tools を追記する。
- [ ] `docs/phonology_rules.md` に Buck executable conversion boundary を追記する。
- [ ] `docs/REQUIREMENTS.md` または `docs/requirements.md` に研究根拠としての Buck 参照を追記する。
- [ ] `DATA_LICENSE.md` に Buck structured reference data の扱いを明確化する。
- [ ] `README.md` に利用例を追加する。
- [ ] `docs/CODEMAPS/api.md` を更新する。
- [ ] `docs/CODEMAPS/phonology.md` を更新する。
- [ ] `docs/ROADMAP.md` から本計画書へリンクする。

## Phase 12: Web UI

- [ ] Buck 参照を UI に出すか決める。
  - [ ] 初期は API only
  - [ ] 検索結果詳細に Buck reference panel
  - [ ] 独立した Buck reference browser
- [ ] UI に出す場合の表示項目を決める。
  - [ ] Rule id
  - [ ] Buck section
  - [ ] Dialect
  - [ ] Example word
  - [ ] Standard form
  - [ ] Review status
- [ ] `citation_ready: false` を小さく明示する。
- [ ] 長文引用を表示しない。
- [ ] `src/web/static/translations.json` に翻訳キーを追加する。
- [ ] `src/web/index.html` fallback translations を更新する。
- [ ] mobile layout で Buck reference panel が崩れないことを確認する。

## Phase 13: Tests

- [x] Buck service tests を追加する。
  - [x] rule id lookup
  - [x] section lookup
  - [x] category filter
  - [x] dialect lookup
  - [x] dialect inherited rules
  - [x] glossary word lookup
  - [x] glossary standard_form lookup
  - [x] defensive copy / no mutation
- [ ] Buck API tests を追加する。
  - [ ] `/languages/ancient_greek/buck/rules`
  - [ ] `/languages/ancient_greek/buck/rules/{rule_id}`
  - [ ] `/languages/ancient_greek/buck/dialects`
  - [ ] `/languages/ancient_greek/buck/dialects/{dialect_id}`
  - [ ] `/languages/ancient_greek/buck/glossary`
  - [ ] unsupported language
  - [ ] unknown rule id
  - [ ] unknown dialect id
  - [ ] pagination limit validation
- [ ] Buck search annotation tests を追加する。
  - [ ] rule id 対応がある hit に `buck_references` が付く（初期スコープ）。
  - [ ] annotation failure は検索 200 を維持する。
  - [ ] 固定クエリ集合の `/search` 出力から `buck_references` など追加注釈 field を除外した ranking、distance、candidate ordering、`rules_applied`、`matched_rules` が Buck ON/OFF で不変（ranking 不変の回帰 snapshot）。
  - [ ] glossary word/standard_form 一致注釈は glossary 拡充 epic 後に追加。
- [ ] Executable conversion pilot tests は別 epic で扱う（[buck_executable_rule_conversion_epic.md](buck_executable_rule_conversion_epic.md)）。
- [ ] Data schema tests を追加する。
  - [ ] valid Buck YAML を受け入れる。
  - [ ] duplicate ids を拒否する。
  - [ ] unknown refs を拒否する。
  - [ ] parent cycle を拒否する。
  - [ ] `current_entries` mismatch を拒否する。
- [ ] Packaging tests を更新する。
  - [ ] new schema files
  - [ ] new docs where required
  - [ ] generated OpenAPI if committed
- [ ] Web UI tests を追加する場合の範囲を決める。

## Phase 14: Verification Commands

- [ ] `uv run pytest tests/test_buck_loader.py -q`
- [ ] `uv run pytest tests/test_buck_data_files.py -q`
- [x] `uv run pytest tests/test_packaging.py -q`
- [ ] `uv run pytest tests/test_api_languages.py -q`
- [ ] `uv run pytest tests/test_api_main.py -q`
- [x] `uv run pytest tests/test_mcp_search_tool.py -q`
- [x] `uv run pytest -q`
- [ ] `uv build --wheel`

## Suggested Implementation Order

> これが唯一の正系列。上記 Phase 群は機能カタログであり、Phase 番号順 = 実行順ではない。各 Step は test-first（テストを実装と同 Step 内で先行）で進める。

**MVP（本書）:**

- [x] Step 1: Buck service tests を先に書き（rule/dialect/glossary lookup・no mutation）、既存 `load_buck_data()` の上に read-only index（frozen dataclass）を構築する。
- [x] Step 2: MCP tests を先に書き、Buck 参照 MCP tool（`search_buck_rules` / `get_buck_dialect` / `search_buck_glossary`）を内部 service 直結で公開する。
- [ ] Step 3: 検索結果への `buck_references` annotation を rule_id 限定で追加（回帰 snapshot test で ranking 不変を先に固定）。
- [ ] Step 4: MCP docs（`docs/MCP.md` / `docs/mcp/tools.json`）と data schema / validation を更新する。

**契約安定後:**

- [ ] Step 5: REST API model と Buck endpoints を追加し、`docs/API.md` と `docs/api/openapi.json` を再生成・コミットする。

**後続 epic（別書）:**

- [ ] Step 6: glossary expansion pipeline → [buck_glossary_expansion_epic.md](buck_glossary_expansion_epic.md)
- [ ] Step 7: executable rule conversion pilot → [buck_executable_rule_conversion_epic.md](buck_executable_rule_conversion_epic.md)
- [ ] Step 8: 方言対応拡張 → [buck_dialect_expansion_epic.md](buck_dialect_expansion_epic.md)

## Open Questions

> 公開面（MCP 先行）・field 名（`buck_references`）・status 配置（全体 metadata + 各 item）・ドキュメント構造（MVP + epic）は「確定事項」で解決済み。残る未解決項目のみを以下に残す。

- [ ] Buck の section/page 参照を API/MCP response でどこまで公開するか。
- [ ] `glossary.yaml` の 1400 件増補は、どの dialect / section から着手するか（→ glossary epic）。
- [ ] 方言を `/search` request の正式 dialect として増やす最初の候補をどれにするか（→ dialect epic）。
- [ ] executable conversion の review owner を誰にするか（→ executable epic）。

## Deferred Risks

- [ ] OCR 由来 Markdown は誤認識が多く、機械抽出だけでは citation-ready にできない。
- [ ] Buck の summary rule は既存の `input` / `output` / `context` schema へ単純変換できないものが多い。
- [ ] 方言を増やすと IPA conversion、distance matrix、rule matching、UI validation の影響範囲が広い。
- [ ] `buck/` を通常 `load_rules()` に再帰的に読ませると、`dialects.yaml` / `glossary.yaml` が通常 rule file ではないため破壊的変更になりうる。
- [ ] 未レビュー Buck annotation を検索候補の支持根拠として強く見せると、学術的信頼性を過大表示するリスクがある。
