# Buck Dialect Support Expansion Epic

前提: MVP（[buck_reference_data_integration_plan.md](buck_reference_data_integration_plan.md) の Phase 0-2 + MCP exposure）完了後に着手する。dialect は個別に計画・実装する。

目的: Buck dialect カタログを段階的に `/search` の正式 dialect へ拡張する。方言を増やすと IPA conversion、distance matrix、rule matching、UI validation の影響範囲が広いため、dialect 単位でレビュー境界を保って進める。

注記: 距離行列は `attic_doric` だが、`doric` は現行 `supported_dialects`（`attic`, `koine`）に未収載。matrix 存在と supported dialect は別概念である点に注意。

## 進捗凡例

- [ ] 未着手
- [x] 完了
- [~] 実装中または要再検討

## Tasks

- [ ] Buck dialect と現行 `supported_dialects` の差分を一覧化する。
  - [ ] 現行: `attic`
  - [ ] 現行: `koine`
  - [ ] Buck: `ionic`
  - [ ] Buck: `arcadian`
  - [ ] Buck: `cyprian`
  - [ ] Buck: `lesbian`
  - [ ] Buck: `thessalian`
  - [ ] Buck: `boeotian`
  - [ ] Buck: `elean`
  - [ ] Buck: `locrian`
  - [ ] Buck: `laconian`
  - [ ] Buck: `cretan`
  - [ ] Buck: その他
- [ ] 方言を API request で受けるか、Buck endpoint の filter に限定するか決める。
  - [ ] 推奨: 初期は Buck endpoint filter に限定。
- [ ] `/search` の `dialect` として追加する条件を決める。
  - [ ] IPA converter が対応する。
  - [ ] distance matrix が妥当。
  - [ ] dialect skeleton builder がある。
  - [ ] rules/explanations が最低限ある。
  - [ ] expected behavior tests がある。
- [ ] `LanguageProfile.supported_dialects` 更新手順を作る（`profile.py:55`）。
- [ ] `data/schemas/hard_query_case.schema.json` の enum 更新が必要か確認する。
- [ ] Web UI の dialect selector 更新が必要か確認する。
- [ ] MCP schema validation 更新が必要か確認する。
- [ ] dialect 追加ごとに acceptance examples を作る。

## Open Questions

- [ ] 方言を `/search` request の正式 dialect として増やす最初の候補をどれにするか。

## Risks

- [ ] 方言を増やすと IPA conversion、distance matrix、rule matching、UI validation の影響範囲が広い。
