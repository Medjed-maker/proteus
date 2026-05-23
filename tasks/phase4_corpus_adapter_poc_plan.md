# Phase 4 Corpus Adapter Proof of Concept Implementation Plan

作成日: 2026-05-23
対象: `docs/ROADMAP.md` Phase 4 — Corpus Adapter Proof of Concept
目的: 検索結果に外部 source metadata を付与する。本文テキストは保持せず、source id、短い citation、外部リンク、license note のみを扱う。

## 進捗凡例

- [ ] 未着手
- [x] 完了
- [~] 実装中または要再検討

## Acceptance Mapping

- [x] Search result can include source references: `SearchHit.source_references` を追加。
- [x] Restricted corpora are linked rather than redistributed: `SourceReference` と YAML schema が本文 field を禁止。
- [x] Data source attribution is documented: `docs/CORPUS_ADAPTERS.md`, `DATA_LICENSE.md`, API/MCP docs を更新。
- [x] Corpus adapter logic is separate from core search logic: `phonology.corpus` adapter を post-search enrichment として接続。
- [x] pre-403/2 BCE Attic orthographic notes are not inferred from papyri.info metadata alone: docs に human review boundary を明記。
- [x] Automatically ingested source metadata is not treated as citation-ready runtime note data without review: `citation_ready` と docs で固定。

## Implementation Checklist

### 0. Baseline & Boundaries

- [x] Phase 4 acceptance criteria を本計画書にマッピングする。
- [x] `DATA_LICENSE.md` の corpus data 方針を Phase 4 の制約として明記する。
- [x] restricted corpus text、長文引用、未レビュー papyri.info / PHI / AIO 自動 ingest 結果を runtime data に入れない方針を固定する。
- [x] 既存検索 ranking、distance、rule explanation、orthographic note 挙動は変更しない。

### 1. Public API Models

- [x] `SourceReference` モデルを追加する。
- [x] 必須 field を定義する: `source_id`, `corpus`, `short_citation`, `external_url`, `license_note`, `access_policy`, `citation_ready`。
- [x] `access_policy` は `open_metadata`, `linked_restricted_text`, `expert_review_required` を許可する。
- [x] `SourceReference` は本文 excerpt を持たない。
- [x] `SearchHit` に `source_references: list[SourceReference] = []` を追加する。
- [x] OpenAPI と `docs/API.md` / `docs/MCP.md` に新 field を追記する。
- [x] `schema_version` を `1.1.0` に bump する。

### 2. Corpus Adapter Interface

- [x] `phonology.corpus` package を追加する。
- [x] `CorpusAdapter` Protocol を定義する。
- [x] Interface は `lookup(entry_id: str, headword: str, language: str) -> tuple[SourceReference, ...]` とする。
- [x] adapter は core search ranking に依存せず、検索完了後の enrichment として呼ぶ。
- [x] adapter 例外は検索全体を落とさず、warning log のうえ空 list に degrade する。
- [x] 複数 adapter を合成できる `CompositeCorpusAdapter` を用意する。
- [x] 未登録言語では empty adapter を返す。

### 3. Static Perseus / Scaife PoC Adapter

- [x] `data/languages/ancient_greek/corpus_sources/perseus_scaife_sources.yaml` を追加する。
- [x] YAML は lexicon `entry_id` を key にし、公開可能な source metadata のみを保持する。
- [x] `λόγος` と `ἄνθρωπος` の実例を追加する。
- [x] `source_id` には Perseus identifier を入れる。
- [x] `external_url` は source landing page へのリンクに限定する。
- [x] `license_note` に provider attribution と redistribution boundary を書く。
- [x] YAML schema を `data/schemas/corpus_source_reference.schema.json` に追加する。
- [x] validator は URL scheme、必須 field、本文混入禁止 field を検査する。

### 4. Runtime Integration

- [x] `LanguageProfile` に `corpus_adapter_factory` を追加する。
- [x] Ancient Greek profile に static Perseus / Scaife adapter を登録する。
- [x] `_load_search_dependencies()` に corpus adapter を追加する。
- [x] `SearchDependencies` に adapter を保持する。
- [x] `run_search()` で `_build_search_hit()` 後に `source_references` を付与する。
- [x] `SearchResult.entry_id` が無い場合は `source_references=[]` とする。
- [x] adapter lookup は候補ごとに一度だけ呼び、結果順は YAML 定義順を保持する。
- [x] MCP は REST と同じ `SearchResponse` / `SearchHit` を通して field を返す。

### 5. Licensing & Documentation

- [x] `docs/CORPUS_ADAPTERS.md` を追加する。
- [x] adapter interface、static metadata format、license boundary、restricted corpus handling を説明する。
- [x] papyri.info / PHI / AIO は candidate generation までに留め、human review なしで citation-ready runtime note にしないと明記する。
- [x] `DATA_LICENSE.md` に Phase 4 PoC の具体例を追記する。
- [x] `docs/API.md` に `source_references` の利用上の注意を追記する。
- [x] `docs/MCP.md` に LLM client 向けの attribution 注意を追記する。
- [x] `docs/ROADMAP.md` Phase 4 からこの task plan へリンクする。

### 6. Tests

- [x] `tests/test_corpus_adapter_schema.py` を追加する。
- [x] valid YAML fixture を受け入れる。
- [x] 必須 field 欠落を拒否する。
- [x] `external_url` 以外の本文 URL 混入ポリシー違反を拒否する。
- [x] `evidence_excerpt`, `source_text`, `passage_text`, `quote` など本文 field を拒否する。
- [x] `tests/test_corpus_adapters.py` を追加する。
- [x] static adapter が `entry_id` から `SourceReference` を返す。
- [x] 未登録 entry は empty tuple を返す。
- [x] malformed adapter data は startup validation error になる。
- [x] `tests/test_api_search_sources.py` を追加する。
- [x] `/search` hit に `source_references` が存在する。
- [x] fixture 対象候補では external link と license note が返る。
- [x] fixture 非対象候補では `source_references=[]` になる。
- [x] adapter failure 時に検索レスポンス自体は 200 のままになる。
- [x] MCP search response にも `source_references` が含まれることを確認する。
- [x] `tests/test_packaging.py` に corpus source YAML と schema の wheel/sdist 同梱確認を追加する。

## Verification Commands

- [x] `uv run pytest tests/test_corpus_adapter_schema.py -q`
- [x] `uv run pytest tests/test_corpus_adapters.py -q`
- [x] `uv run pytest tests/test_api_search_sources.py -q`
- [x] `uv run pytest tests/test_mcp_search_tool.py -q`
- [x] `uv run pytest tests/test_packaging.py -q`
- [x] `uv run pytest -q`
- [x] `uv build --wheel`
