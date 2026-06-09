# Buck Executable Rule Conversion Epic

前提: MVP（[buck_reference_data_integration_plan.md](buck_reference_data_integration_plan.md) の Phase 0-2 + MCP exposure）完了後に着手する。

目的: Buck rule のうち実行可能な一部だけを、feature flag 下で既存 rule schema へ変換するパイロット。検索 ranking・距離行列・既存 rule explanation の挙動は変更しない。

## 進捗凡例

- [ ] 未着手
- [x] 完了
- [~] 実装中または要再検討

## Tasks

- [ ] Buck rule を実行可能ルールへ変換する条件を決める。
  - [ ] `transformation.from` がある。
  - [ ] `transformation.to` がある。
  - [ ] context が既存 matcher で表現できる。
  - [ ] affected dialect が既存 `supported_dialects` に含まれる、または feature flag 下で扱う。
  - [ ] expert review status が実行可能として許可されている。
- [ ] 変換対象外カテゴリを明確にする。
  - [ ] alphabet
  - [ ] syntax
  - [ ] broad summary only
  - [ ] semantic peculiarity
  - [ ] insufficient transformation
- [ ] 変換結果の出力先を決める。
  - [ ] build-time generated YAML
  - [ ] runtime in-memory conversion
  - [ ] 推奨: build-time generated candidate file + review diff。
- [ ] generated rule id 命名規則を決める。
  - [ ] Buck original id を保持するか
  - [ ] `BUCK-*` prefix を付けるか
  - [ ] 既存 `VSH` / `CCH` / `MPH` と衝突しないようにする。
- [ ] 既存 rule schema との mapping を作る。
  - [ ] Buck `description` -> `name_en` / `description`
  - [ ] Buck `transformation.from` -> `input`
  - [ ] Buck `transformation.to` -> `output`
  - [ ] Buck `transformation.context` -> `context`
  - [ ] Buck `affected_dialects` -> `dialects`
  - [ ] Buck `buck_section` -> `references`
  - [ ] Buck `notes` -> `note`
- [ ] 変換できない rule の report を出す。
- [ ] pilot 対象 rule を最小数に絞る。
  - [ ] 母音変化 1-3 件
  - [ ] 子音変化 1-3 件
  - [ ] glossary example がある rule を優先
- [ ] feature flag を設計する。
  - [ ] `PROTEUS_ENABLE_BUCK_EXECUTABLE_RULES`
  - [ ] default は disabled。
- [ ] feature flag disabled で既存テストが完全に通ることを確認する。
- [ ] feature flag enabled の dedicated tests を追加する。

## Tests

- [ ] feature flag disabled で既存挙動不変。
- [ ] feature flag enabled で pilot rule が load される。
- [ ] 変換不能 rule は report に出る。

## Open Questions

- [ ] executable conversion の review owner を誰にするか。

## Risks

- [ ] Buck の summary rule は既存の `input` / `output` / `context` schema へ単純変換できないものが多い。
