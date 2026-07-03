# Phase 3 Scholarly Validation and Hard Query Collection Plan

作成日: 2026-05-22
対象: `docs/ROADMAP.md` Phase 3 — Scholarly Validation and Hard Query Collection
目的: 検索品質を実研究の hard query で検証し、失敗例、false positive / false negative、専門家レビュー、orthographic-note seed 昇格を追跡できる運用基盤を作る。

## 進捗凡例

- [ ] 未着手
- [x] 完了
- [~] 実装中または要再検討

## Summary

- 公開 repo には schema、template、公開可能 seed のみを置く。
- 20件以上の実ケース収集は、公開可否を確認したものだけ repo に昇格し、未公開・機微データは非公開ログで管理する。
- Phase 3 の評価データは検索品質用であり、`tools/benchmark_search_latency.py` の性能ベンチとは別に管理する。
- REST / MCP の wire schema は Phase 3 では変更しない。

## ROADMAP Acceptance Mapping

- [x] At least 20 real or semi-real hard query cases documented
  - 公開 seed `data/evaluation/hard_queries/public_seed_cases.yaml` のみで 20 件到達(hq-ag-0001〜0020、2026-07-04)。
  - `tools/validate_hard_queries.py --min-cases 20` で確認済み。
  - hq-ag-0015(χώρη → χώρα)は evaluator 実測で false negative を記録した known-miss ケースとして保持し、ルール改善 PR の材料にする。
- [x] Each case includes input form, expected candidate, reasoning, and source notes
  - `data/schemas/hard_query_case.schema.json` で必須化済み。
- [x] Student-facing inscriptional orthography aid is clearly marked provisional
  - `docs/VALIDATION.md` と既存 orthographic-note docs で明記済み。
- [~] The first reviewed orthographic-note seed has a recorded reviewer decision before citation-ready
  - `παιδίο -> παιδίου` を first reviewed seed pilot として接続済み。
  - reviewer decision と `citation_ready: true` 昇格は未完了。
- [x] Expert feedback is recorded separately from public code if needed
  - template と validation docs で public/private 境界を明記済み。
- [x] Sensitive or unpublished research data is not committed to the public repository
  - `--public-only` validator が `private_collaborator` / `embargoed` を拒否する。

## Implementation Checklist

### 0. Baseline & Boundaries

- [x] `docs/ROADMAP.md` Phase 3 の Required Work / Acceptance Criteria を計画書冒頭にマッピングする。
- [x] 公開 repo に置ける情報と置けない情報を明文化する。
- [x] `tasks/orthographic_notes_citation_ready_plan.md` Gate B/C と Phase 3 の関係を明記する。
- [x] `tools/benchmark_search_latency.py` の性能ベンチと、Phase 3 の品質ベンチを別物として扱う。

### 1. Hard Query Case Schema

- [x] `data/schemas/hard_query_case.schema.json` を追加する。
- [x] 必須項目を定義する: `case_id`, `visibility`, `input_form`, `language`, `dialect_hint`, `expected_candidates`, `reasoning`, `source_notes`, `existing_search_failure`, `review_status`, `false_positive_notes`, `false_negative_notes`。
- [x] `visibility` は `public_seed`, `public_anonymized`, `private_collaborator`, `embargoed` を許可する。
- [x] `private_collaborator` / `embargoed` のケースが公開 dataset に混入したら validation error にする。
- [x] source text の長文引用を禁止し、source identifiers / short citations / notes に限定する。

### 2. Public Dataset Skeleton

- [x] `data/evaluation/hard_queries/README.md` を追加する。
- [x] `data/evaluation/hard_queries/public_seed_cases.yaml` を追加する。
- [x] 公開 seed は少数でよいが、schema・validation・docs の実例として成立させる。
- [x] 20件到達は `public_seed_cases.yaml` だけでなく、非公開 collector log を含む運用目標として扱う。
- [x] `tests/fixtures/` には機微情報を置かない。

### 3. Collection Templates

- [x] `docs/hard_query_collection_template.md` を追加する。
- [x] 1ケースごとに、入力語形、期待候補、手検索の結果、既存検索の失敗、Proteus結果、判断理由を記録できる形式にする。
- [x] reviewer / collaborator identity は handle または initials を推奨し、実名は本人合意なしに公開しない。
- [x] 外部 reviewer comment は原文ではなく公開可能な要約だけを repo に残す。

### 4. Validation Tooling

- [x] `tools/validate_hard_queries.py` を追加する。
- [x] YAML / JSON の両方を読み、schema validation と公開境界チェックを行う。
- [x] `--public-only` では `private_collaborator` / `embargoed` を拒否する。
- [x] `--min-cases 20` を用意し、非公開収集ログ込みの Phase 3 完了確認に使う。
- [x] CI では公開 seed の schema validation のみ実行する想定を docs に記録する。

### 5. Search Quality Runner

- [x] `tools/evaluate_hard_queries.py` を追加する。
- [x] hard query case を読み、既存 `/search` 相当の search runner を直接呼ぶ。
- [x] expected candidate が top-N に入ったか、rank、confidence、applied rules、orthographic notes を記録する。
- [x] false positive / false negative の候補を machine-readable に出力する。
- [x] 出力先は指定パスにし、公開可否未確認の結果を repo に自動追加しない。

### 6. Orthographic Note Review Integration

- [x] `παιδίο -> παιδίου` を Phase 3 の first reviewed seed pilot として扱う。
- [x] `docs/orthographic_notes_review_template.md` を使い、source identifiers, short references, reviewer metadata, reviewer decision を記録する。
- [x] `citation_ready: true` への昇格条件を再掲する。
- [x] source evidence が不十分な場合は `needs_expert_review` のままにし、citation-ready と記述しない。
- [x] `pre_403_2_attic` 系 note は inscriptional source または明示的 expert judgment がない限り昇格しない。

### 7. Documentation

- [x] `docs/VALIDATION.md` を追加する。
- [x] hard query collection workflow、public/private data boundary、manual vs tool-assisted comparison 手順を書く。
- [x] `docs/API.md` / `docs/MCP.md` に、Phase 3 dataset が API schema 変更ではなく評価運用であることを補足する。
- [x] `docs/ROADMAP.md` Phase 3 に、計画書へのリンクと進捗管理方法を追記する。
- [x] `README.md` に協力者向けの短い導線を追加する。

### 8. Tests

- [x] `tests/test_hard_query_schema.py` を追加する。
- [x] valid public seed を受け入れる。
- [x] 必須項目不足を拒否する。
- [x] 公開 dataset 内の `private_collaborator` / `embargoed` を拒否する。
- [x] source text の長文引用・URL混入ポリシー違反を拒否する。
- [x] `tools/validate_hard_queries.py --public-only` の CLI smoke test を追加する。
- [x] `tools/evaluate_hard_queries.py` は小さな fixture で expected candidate hit / miss を検証する。

## Verification Commands

- [x] `uv run pytest tests/test_hard_query_schema.py -q`
- [x] `uv run python tools/validate_hard_queries.py --public-only data/evaluation/hard_queries/public_seed_cases.yaml`
- [x] `uv run python tools/evaluate_hard_queries.py --cases data/evaluation/hard_queries/public_seed_cases.yaml --output-json /tmp/proteus-hard-query-eval.json`
- [x] `uv run pytest -q`

## Remaining Phase 3 Work

- [x] Collect at least 20 real or semi-real cases across public and private logs.
  - 公開 seed のみで 20 件到達(2026-07-04)。非公開 collaborator ログの追加収集は任意の継続タスク。
- [ ] Complete expert review for the first orthographic-note seed.
- [ ] Promote the first seed to `citation_ready: true` only after complete source and reviewer metadata are recorded.
- [ ] Use evaluator reports to decide rule-set improvements in separate PRs.
