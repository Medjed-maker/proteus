# Buck Glossary Expansion Pipeline Epic

前提: MVP（[buck_reference_data_integration_plan.md](buck_reference_data_integration_plan.md) の Phase 0-2 + MCP exposure）完了後に着手する。

目的: `glossary.yaml`（現状 `current_entries: 15`、`total_entries_original: 1400`）を、OCR 本文からの自動投入ではなく、抽出・正規化・人手レビューを分けた pipeline で段階的に拡充する。

## 進捗凡例

- [ ] 未着手
- [x] 完了
- [~] 実装中または要再検討

## Tasks

- [ ] repo 外の OCR/Markdown source candidate を指定する方針を決める。
  - [ ] 候補: `PROTEUS_BUCK_GLOSSARY_SOURCE` 環境変数。
  - [ ] 候補: extraction script の明示的な入力 path 引数。
  - [ ] ローカル絶対パスを計画書・コード・設定に固定しない。
- [ ] OCR/Markdown 本文を repo に入れるか決める。
  - [ ] 推奨: そのまま runtime data として入れない。
  - [ ] 抽出済み structured data と出典 metadata のみを入れる。
- [ ] extraction script の配置先を決める。
  - [ ] 候補: `tools/extract_buck_glossary.py`
  - [ ] 候補: `scripts/extract-buck-glossary.sh`
- [ ] extraction output の intermediate format を決める。
  - [ ] JSONL
  - [ ] YAML draft
  - [ ] TSV review sheet
- [ ] glossary entry schema を明確化する。
  - [ ] `word`
  - [ ] `standard_form`
  - [ ] `dialect`
  - [ ] `rule_id`
  - [ ] `definition`
  - [ ] `buck_ref.section`
  - [ ] `buck_ref.page`
  - [ ] `inscription_no`
  - [ ] `source_line`
  - [ ] `review_status`
  - [ ] `citation_ready`
- [ ] duplicate detection を設計する。
  - [ ] exact word + dialect + standard_form
  - [ ] accent-insensitive key
  - [ ] normalized Unicode key
- [ ] rule_id 自動推定をするか決める。
  - [ ] 初期は自動推定しない。
  - [ ] human review で rule_id を付ける。
- [ ] dialect abbreviation mapping を作る。
  - [ ] `Att.` -> `attic`
  - [ ] `Arc.` -> `arcadian`
  - [ ] `Cypr.` -> `cyprian`
  - [ ] `Lesb.` -> `lesbian`
  - [ ] `Boeot.` -> `boeotian`
  - [ ] `El.` -> `elean`
  - [ ] `Lac.` -> `laconian`
- [ ] extraction confidence を付与する。
  - [ ] `low`
  - [ ] `medium`
  - [ ] `high`
- [ ] human review workflow を決める。
  - [ ] draft file
  - [ ] review checklist
  - [ ] approved entries only merge
- [ ] **未レビュー一括投入ガードを設ける**。
  - [ ] OCR draft から `glossary.yaml` へ自動 merge する場合、`review_status: approved` ではない entry を拒否する CI チェック / test。
  - [ ] 既存 provisional reference data は `citation_ready: false` として許可する。
  - [ ] Non-Goal「未レビュー OCR 抽出結果を citation-ready として表示しない」と、未承認 entry の `citation_ready: true` 昇格禁止をテストで強制する。
- [ ] `glossary.yaml` の `current_entries` 更新を自動検証する（`meta.current_entries == len(words)`）。
- [ ] 1400 件全投入ではなく、方言別・rule別に小分けで PR 化する。

## Open Questions

- [ ] `glossary.yaml` の 1400 件増補は、どの dialect / section から着手するか。

## Risks

- [ ] OCR 由来 Markdown は誤認識が多く、機械抽出だけでは citation-ready にできない。
