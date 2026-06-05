# 移行計画: core から Ancient Greek を引き剥がす

**目的**: `proteus` の core（`src/phonology/` のうち `languages/` を除く層）を完全に言語独立にし、Ancient Greek を 1 プラグインへ閉じ込める。

**ステータス**: 全フェーズ（0〜5）完了 — core 言語独立ゲートが許可リストなしで緑
**作成日**: 2026-06-02
**最終更新**: 2026-06-03
**関連**: `docs/ARCHITECTURE.md`, `CLAUDE.md`（設計原則）

---

## 北極星（不変条件）

> `src/phonology/` のうち `languages/` を除いた core に、`attic | doric | ionic | koine | ancient_greek` が出現したら CI 失敗。

この grep ゲートがゼロになった時点で「言語独立フレームワーク」が看板倒れでなくなる。各フェーズは独立 PR、前後で `uv run pytest` グリーン必須。

---

## 進捗オーバービュー

| Phase | 内容 | リスク | 状態 |
|---|---|---|---|
| 0 | 言語独立ゲートを許可リスト付きで先に立てる | 最小 | ☑ 完了 |
| 1 | 依存逆転の解消（core→具象プラグイン import を撤廃） | 高 | ☑ 完了 |
| 2 | 計算層への Greek トークナイザ注入漏れを塞ぐ | 高 | ☑ 完了 |
| 3 | core デフォルト値の脱 Greek | 中 | ☑ 完了 |
| 4 | 誤配置モジュールのプラグイン移送・重複解消 | 中 | ☑ 完了 |
| 5 | 仕上げ（ゲート締め・docstring・ports 集約） | 低 | ☑ 完了 |

進捗記法: ☐ 未着手 / ◐ 進行中 / ☑ 完了

---

## 結合マップ（剥がす対象の正体）

| 分類 | 中身 | 対象（確認済み） |
|---|---|---|
| **A. 依存逆転違反** | core が具象プラグインを名指し import | ☑ 解消済み（Phase 1: `profiles.py` を entry point discovery に置換） |
| **B. 計算層への注入漏れ** | エンジンが Greek トークナイザを直接呼ぶ | ☑ 解消済み（Phase 2: `LanguageProfile.phone_inventory` + `core/ipa.py` tokenizer へ置換） |
| **C. core デフォルト直書き** | シグネチャ・定数に Greek リテラル | ☑ 解消済み（Phase 3: `_paths.py` / `search/` の default を profile 解決へ） |
| **D. 誤配置モジュール** | 本来 Greek 固有なのに core 直下 | ☑ 解消済み（Phase 4: `lsj/`・`*_generator`・`betacode` 等を `languages/ancient_greek/` へ移送。`log_odds.py` は汎用のため core 残置） |
| **E. docstring 言及のみ** | 例示で attic / koine 言及 | ☑ 解消済み（Phase 5: `distance.py` / `search/_indexing.py` / `_explainer_rule_loader.py` の docstring を一般化） |

既存の良い土台: `LanguageProfile` 契約・registry（`profiles.py`）・`languages/ancient_greek/`・`data/languages/ancient_greek/`。作り直しではなく **始まっているプラグイン化を「core 側の配線断ち」まで完遂する**。

---

## Phase 0 — 言語独立ゲートを先に立てる

**目的**: 「core に Greek 語彙が出たら落ちる」テストを許可リスト付きで導入し、各フェーズの成果を自動で可視化する。

**対象**: `tests/test_core_language_independence.py`（新規）

### タスク
- [x] core 対象ファイル集合を定義する（`src/phonology/**` から `src/phonology/languages/**` を除外）
- [x] 禁止語彙パターンを定義（`attic`, `doric`, `ionic`, `koine`, `ancient_greek`、大文字小文字無視）
- [x] 各ファイルを走査し、ヒット箇所（ファイル:行）を収集するテストを実装
- [x] 現状ヒット箇所を `ALLOWLIST`（既知の負債リスト）として明示列挙し、スナップショット化
- [x] 「`ALLOWLIST` 外の新規ヒット」が出たら失敗するアサーションを実装
- [x] docstring のみのヒット（分類 E）も `ALLOWLIST` に含め、Phase 5 で消す前提でコメント付与
- [x] `uv run pytest tests/test_core_language_independence.py` がグリーンであることを確認

### 完了判定
- [x] テストが追加され全体グリーン、挙動変更ゼロ
- [x] `ALLOWLIST` が現状の全ヒットを正確に反映している（漏れがあれば即失敗するはず）

---

## Phase 1 — 依存逆転の解消（キーストーン）

**目的**: core registry が具象プラグイン（Greek）を名指しで import している唯一最大の違反を断つ。プラグイン自己登録方式へ移行。

**対象**: `pyproject.toml`, `src/phonology/profiles.py`, `src/phonology/_paths.py`（default 解決のみ）

### タスク — entry points 化
- [x] `pyproject.toml` に `[project.entry-points."proteus.languages"]` を追加
- [x] `ancient_greek = "phonology.languages.ancient_greek.profile:build_profile"` を登録
- [x] entry point の解決可否を確認（`importlib.metadata.entry_points(group="proteus.languages")`）

### タスク — registry の脱具象
- [x] `profiles.py:_build_ancient_greek_profile()` を削除
- [x] `register_default_profiles()` を entry points discovery 実装に置換（`languages.ancient_greek` 直接 import を撤廃）
- [x] discovery 失敗時の挙動を定義（プラグイン 0 件のときのエラーメッセージ）
- [x] trusted-dir 登録（`register_trusted_rules_dir` / `register_trusted_matrices_dir`）が discovery 経由でも実行されることを確認

### タスク — default 言語の一般化
- [x] `get_default_language_profile()` の default 決定ロジックを設計
  - [x] `PROTEUS_DEFAULT_LANGUAGE` 環境変数を最優先
  - [x] 未指定かつ登録 1 件ならそれを default
  - [x] 未指定かつ複数件なら明示エラー
- [x] `ancient_greek` を「設定された default」として温存し、既存挙動を不変に保つ
- [x] `_paths.py:DEFAULT_LANGUAGE_ID` への依存箇所を洗い出し（Phase 3 で本体撤去するため暫定維持の可否を判断）

### タスク — テスト追従
- [x] `_reset_language_registry_for_tests()` が discovery を再実行できることを確認
- [x] registry 関連テスト（`test_profiles.py` 等）を新方式に追従
- [x] `language="ancient_greek"` を前提とする既存テストが全て緑のまま通ることを確認

### 完了判定
- [x] `profiles.py` から `languages.ancient_greek` への import が消滅
- [x] Phase 0 ゲートの `profiles.py` 該当エントリを `ALLOWLIST` から削除しても緑
- [x] 全テストグリーン、公開挙動（default=Greek）不変

---

## Phase 2 — 計算層への Greek トークナイザ注入漏れを塞ぐ

**目的**: core 計算（distance / explainer）が Greek の `tokenize_ipa` を直接 import している深い結合を断ち、profile 経由の注入に変える。

**対象**: `src/phonology/profiles.py`（契約拡張なし）, `distance.py`, `_explainer_context.py`, `_explainer_rule_tokenize.py`

### 方式決定（確定済み: 2026-06-02）
- [x] 供給方式を決定 → **案 B（`phone_inventory` 導出）採用**。詳細は末尾「決定ログ」参照
- [x] スパイクは不要（汎用 tokenizer が既に `core/ipa.py:51` に存在するため）

### タスク — 契約・実装（案 B）
- [x] `LanguageProfile` への新フィールド追加は**行わない**（`phone_inventory` を流用）
- [x] `distance.py:28` の `from .ipa_converter import tokenize_ipa` を撤廃し、`core/ipa.py:tokenize_ipa(text, phone_inventory=...)` 経由に
- [x] `distance.py` の純関数シグネチャに `phone_inventory`（または解決済み tokenizer）を引数で渡す（副作用隔離・design by contract）
- [x] `_explainer_context.py:10` の同 import を撤廃し context 経由で `phone_inventory` を注入
- [x] `_explainer_rule_tokenize.py:11` の同 import を撤廃
- [x] `search/__init__.py` の `to_ipa` 利用箇所（368-375 付近）を profile.converter 経由に統一（non-default profile は `search/compat.py` で converter を補完。default の `to_ipa` seam は Phase 2 では互換維持）
- [x] `search/_tokenization.py:18` の重複 `tokenize_ipa` を `core/ipa.py` 版へ集約（既存実装が委譲済みであることを確認）

### タスク — テスト
- [x] distance のテストがモック tokenizer / モック行列で動くことを確認（純粋性の検証）
- [x] explainer のテストを注入方式に追従
- [x] 全テストグリーン

### 完了判定
- [x] core compute / explainer から `ipa_converter` への import が消滅
- [x] Phase 0 ゲートの該当エントリを `ALLOWLIST` から削除しても緑（Phase 2 対象 import は禁止語彙 ALLOWLIST ではなく import grep で検証）

---

## Phase 3 — core デフォルト値の脱 Greek

**目的**: シグネチャ・定数に直書きされた `attic` / `ancient_greek` を profile 解決・設定値へ置換。

**対象**: `_paths.py`, `search/__init__.py`, `search/_registry.py`, `search/_orchestration.py`

### タスク — `_paths.py`
- [x] `DEFAULT_LANGUAGE_ID = "ancient_greek"` を撤去
- [x] legacy `data/<subdir>` フォールバック（Greek 専用ハック、64-68・91-97 行）を撤去
- [x] 明示パスは plugin の `build_profile` が `resolve_language_data_dir` 経由で渡す形に統一
- [x] `resolve_repo_data_dir` の `{lexicon, matrices, rules}` 特例（91 行）を見直し

### タスク — `search/` 公開 API
- [x] `search/__init__.py` の `dialect: str = "attic"`（429 行）を `dialect: str | None = None` に
- [x] None 時に `profile.default_dialect` を解決する内部ロジックを実装
- [x] `language: str | Path = "ancient_greek"`（572・697 行）を default 解決経由に
- [x] `_legacy_to_ipa`（151 行）の `dialect="attic"` 直書きを解消
- [x] docstring の「Defaults to "attic" / "ancient_greek"」記述を一般化
- [x] `search/_registry.py`（55・110 行）の `language="ancient_greek"` 直書きを解消
- [x] `search/_orchestration.py`（305・309 行）の `dialect="attic"` / `language="ancient_greek"` を解消

### タスク — テスト
- [x] default 言語/方言を明示しない呼び出しが従来通り Greek/attic に解決されることを確認
- [x] 明示指定のパスも引き続き機能することを確認
- [x] 全テストグリーン

### 完了判定
- [x] `search/` と `_paths.py` から Greek リテラルが消滅
- [x] Phase 0 ゲートの該当エントリを `ALLOWLIST` から削除しても緑
- [x] 公開 API の挙動（default=Greek/attic）不変

---

## Phase 4 — 誤配置モジュールのプラグイン移送・重複解消

**目的**: 本来 Greek 固有なのに core 直下にあるモジュールを `languages/ancient_greek/` へ移送し、`ipa_converter.py` の重複を解消する。**compat shim は作らない**（負債を増やさない / YAGNI）。

**対象**: 下記モジュール群と、その import 元

### タスク — `ipa_converter` 重複解消
- [x] legacy `src/phonology/ipa_converter.py` と `languages/ancient_greek/ipa.py` の差分を突合
- [x] 正典を `languages/ancient_greek/ipa.py` に一本化
- [x] 全 import 元（`distance.py`, `lsj/__init__.py`, `_explainer_*`, `search/`）を新版へ配線（Phase 2 で大半が注入化済みの想定）
- [x] legacy `ipa_converter.py` を削除

### タスク — モジュール移送（各々 import 追従＋テスト緑を確認）
- [x] `lsj/`（サブパッケージ丸ごと）→ `languages/ancient_greek/lsj/`
- [x] `lsj_extractor.py` → `languages/ancient_greek/`
- [x] `build_lexicon.py` → `languages/ancient_greek/`
- [x] `matrix_generator.py` → `languages/ancient_greek/`
- [x] `buck.py` → `languages/ancient_greek/`
- [x] `betacode.py` → `languages/ancient_greek/`
- [x] `transliterate.py` → `languages/ancient_greek/`
- [x] `_phones.py` → `languages/ancient_greek/`
- [x] `log_odds.py` → 配置を判断（Greek 固有なら移送 / 汎用なら core 残置）

### タスク — 周辺追従
- [x] `pyproject.toml` の `[project.scripts]`（`proteus-mcp` 等）やデータ force-include への影響を確認
- [x] `build_lexicon.py:29` 等のハードコードパス（`Path("src/phonology/ipa_converter.py")`）を更新
- [x] 移送後に dead import / 循環 import がないことを確認
- [x] 全テストグリーン

### 完了判定
- [x] core 直下から Greek 固有モジュールが消滅
- [x] Phase 0 ゲートの該当エントリ群を `ALLOWLIST` から削除しても緑

---

## Phase 5 — 仕上げ・ゲート締め

**目的**: 残りの docstring 言及を片付け、ゲートを厳格化し、ports 層を整理する。

**対象**: `tests/test_core_language_independence.py`, `distance.py`(docstring), `search/_indexing.py`(docstring), 任意で ports 集約

### タスク — ゲート締め
- [x] `distance.py` の docstring 内 `attic_doric.json` 言及を一般化（`<matrix>.json` へ）
- [x] `search/_indexing.py` の koine / Ancient Greek 言及を一般化（`_phones.py` は Phase 4 で移送済みのため対象外）
- [x] `_explainer_rule_loader.py` の docstring 内 `ancient_greek` 言及を一般化（`<language_id>` へ）
- [x] core 唯一の実コード結合 `OrthographicNoteKind` の `pre_403_2_attic` を core から除去し、Greek プラグインの `AncientGreekNoteKind` へ移管（core payload の `kind` は `str` に拡幅）
- [x] Phase 0 の `ALLOWLIST` を空にする
- [x] ゲートが厳格モード（許可リストなし）で緑であることを確認

### タスク — 任意の構造仕上げ
- [x] top-level の契約モジュール（`profiles.py`, `corpus/`, `orthography_notes.py`）を `core/ports/` へ集約（全 import 元を追従、~30 ファイル）
- [x] `_explainer_*`（7 ファイル）を `explain/` サブパッケージへ集約。公開 facade は `explainer.py` のまま温存し公開 import パス（`phonology.explainer`）を不変に保持（外部 import / テスト無改変）

### タスク — ドキュメント
- [x] `docs/ARCHITECTURE.md` を新構造に更新（ソースレイアウト・マッピング節を追加）
- [x] `CLAUDE.md` のソースレイアウト記述を更新
- [x] 本計画書の進捗オーバービューを全て ☑ に更新
- [x] `docs/CODEMAPS/*` は `/update-codemaps` で別途再生成（Phase 1〜4 から既に陳腐化しており Phase 5 の手作業対象外）

### 完了判定
- [x] 言語独立ゲートが許可リストなしで緑
- [x] core を grep して Greek 語彙が 0 件
- [x] 全テストグリーン（公開 API 挙動・OpenAPI schema 不変。API 層の `pre_403_2_attic` は gate スコープ外のため現状維持）

---

## リスクと緩和

| リスク | 緩和策 |
|---|---|
| Phase 1: 多数テストが `language="ancient_greek"` default に依存 | default を「設定値」として温存し挙動を保つ（ハードコード位置だけ移動） |
| Phase 2: tokenizer 注入方式が設計判断を含む | 先に小スパイクで案 A/B を比較し決定ログに記録 |
| Phase 4: 大量の import 追従で循環 import | モジュール単位の小 PR に分割、各々で `pytest` 緑を確認 |
| 全体: 順序依存（1→2→3→4） | Phase 0 のみ先行可。以降は依存順を厳守 |

## PR 分割方針

1 フェーズ = 1 PR（各 PR 単独で緑・単独レビュー可能）。Phase 4 はモジュールごとにさらに細分化する。

## 決定ログ

| 日付 | 項目 | 決定 | 根拠 |
|---|---|---|---|
| 2026-06-02 | Phase 2: tokenizer 供給方式（案 A フィールド注入 / 案 B inventory 導出） | **案 B（`phone_inventory` 導出）を採用**。必要時に optional な `tokenizer` override を後付けする | 下記詳細参照 |
| 2026-06-03 | Phase 4: `log_odds.py` の帰属（Greek 固有 / 汎用） | core 残置 | IPA 配列の Needleman-Wunsch / log-odds 行列計算であり、Greek 固有語彙・データパス・プラグイン import を持たないため |
| 2026-06-03 | Phase 5: `OrthographicNoteKind` の `pre_403_2_attic` の扱い | core から除去し Greek プラグイン `AncientGreekNoteKind` へ移管。core payload の `kind` は `str` に拡幅 | core は note kind 値で分岐しない純データ運搬。種別語彙は言語固有なのでプラグイン所有が正しい。バリデーションはプラグインのロード時 `_ALLOWED_KINDS` と API 境界の Literal が担保。`kind` の `str` 化により API 境界（`_hit_formatting.py`）で `OrthographicNote` の Literal への `cast` を 1 箇所追加（pydantic が実行時に再検証） |
| 2026-06-03 | Phase 5: API 層（`src/api/_models.py`）の `pre_403_2_attic` | 現状維持 | gate スコープは core（`src/phonology/`）のみ。API は Greek パイロットの公開契約であり、一般化は OpenAPI 破壊変更を伴う別タスク |
| 2026-06-03 | Phase 5: explainer の公開 facade | `explainer.py` を facade として温存し、private 実装のみ `explain/` へ集約 | 公開 import パス `phonology.explainer` は API・テスト ~20 箇所が依存。リネームは言語デカップリングに無関係な破壊変更のため、散在解消（実目的）のみ達成し公開パスは不変に保つ |

### 決定詳細: Phase 2 tokenizer 供給方式（2026-06-02）

**現状認識**: トークン化アルゴリズムは既に言語非依存で `core/ipa.py:51` に存在する（`tokenize_ipa(ipa_text, *, phone_inventory)`、純関数・在庫を引数で受ける）。`languages/ancient_greek/ipa.py:387` の Greek 版は、それを `_IPA_PHONE_INVENTORY` を埋め込んで呼ぶだけのラッパー。**Greek 固有なのは「音素在庫データ」のみで、アルゴリズムではない**。書記素→IPA 変換（`to_ipa`）は別途 Greek 固有ロジックだが、これは既に `LanguageProfile.converter` として分離済み。今回の対象は IPA→トークンの部分だけ。

**両案の差**: 案 A = アルゴリズムの注入（プラグインが `tokenizer` callable を提供）/ 案 B = データの注入（プラグインは `phone_inventory` のみ提供し、core の汎用 tokenizer を使う）。

| 観点 | 案 A | 案 B（採用） |
|---|---|---|
| アルゴリズムの所在 | プラグイン所有（core はブラックボックス呼び出し） | core 所有（`core/ipa.py` に既存） |
| 契約の表面積 | `tokenizer` フィールドを新規追加 | `phone_inventory` が既存、追加ゼロ |
| プラグイン作者の負担 | 各言語が tokenize 実装（実装ドリフトの温床） | tuple[str] を出すだけ |
| キャッシュ/純粋性 | callable はキャッシュキーにしづらい | core が hashable 在庫をキー化（`sorted_phone_inventory` の決定的ソートは lru_cache 用設計の痕跡） |
| 表現力 | 任意の tokenize 可能 | longest-match + 在庫モデル専用 |
| 重複解消との整合 | 各言語が持つを追認 | `core/ipa.py` 1実装へ集約（Phase 4 と同ベクトル） |

**採用理由**:
1. 汎用 tokenizer が既に core に存在し、Greek 固有なのは在庫だけ、という事実が B と自然に噛み合う
2. `phone_inventory` を使うだけで契約拡張ゼロ。`distance.py` 等は `core.tokenize_ipa(text, phone_inventory=profile.phone_inventory)` 形に
3. キャッシュ・純粋性・重複解消が一貫（CLAUDE.md の副作用隔離・design by contract と整合）
4. 表現力の限界（longest-match に収まらない正書法）は YAGNI。2 言語目で実需が出たら optional な `tokenizer` override を後付けすればよい（B をデフォルト、A を例外として後付けが綺麗な順序）

**Phase 2 への反映**: 「方式決定（先にスパイク）」のスパイクは不要。案 B 確定として実装に進む。`LanguageProfile` への新フィールド追加は行わず、`phone_inventory` 経由で `core/ipa.py:tokenize_ipa` に配線する。

---

## フォローアップ課題（2026-06-05 レビュー由来）

Phase 0〜5 完了後のコードレビューで挙がった、**マージブロッカーではないが将来対応すべき課題**。
現状の挙動には影響しないが、2 言語目のプラグイン追加時や保守時に表面化しうる負債としてここに記録する。
レビュー時点で `uv run pytest` = **1934 passed**（言語独立ゲート `test_core_language_independence.py` 含む）。
本リファクタ自体は完了扱いで問題ない。

| # | 優先度 | 課題 | 該当箇所 | 状態 |
|---|---|---|---|---|
| F1 | MEDIUM | `_extract_consonant_skeleton` の `vowel_phones=()` 既定がサイレント劣化を招く | `search/_query.py` | ☐ 未着手 |
| F2 | MEDIUM | API Literal とプラグイン kind の二重管理（`pre_403_2_attic`） | `api/_hit_formatting.py`, `api/_models.py` | ☐ 未着手（gate スコープ外・意識的保留） |
| F3 | LOW | `_get_trusted_matrices_dir` の profile 解決失敗が warning ログ止まり | `distance.py` | ☐ 未着手（記録のみ） |
| F4 | LOW | `core.ports.profiles` ⇄ `distance` の循環回避ローカル import | `distance.py` | ☐ 未着手（記録のみ） |
| F5 | LOW | `explainer.py` の `import X as X` redundant alias 多用 | `explainer.py` | ☐ 未着手 |

### F1 — `_extract_consonant_skeleton` の母音既定値（MEDIUM）

- **内容**: 旧実装は Greek の `VOWEL_PHONES` 定数で母音を除外していたが、新実装は
  `vowel_phones: Iterable[str] = ()`（空タプル）が既定。呼び出し側が `vowel_phones` を
  渡し忘れると母音が子音スケルトンに混入し、**例外にならず検索品質が静かに劣化**する。
- **現状**: 全呼び出し経路（`_execute_search` → `_finalize_*`）で profile から backfill した
  `vowel_phones` を渡しており動作は正しい。ただし「明示的に渡すべき」契約が docstring 依存で、
  新しい呼び出し経路を追加する開発者が見落としやすい。
- **推奨対応**: `vowel_phones` を必須引数化する、もしくは未指定時に明示エラー／profile 解決を
  強制するガードを入れて design by contract を明文化する。

### F2 — note kind の二重管理（MEDIUM）

- **内容**: `pre_403_2_attic` を含む note kind の Literal が `_hit_formatting.py` の `cast` と
  `_models.py` の `OrthographicNote.kind` の 2 箇所に重複。API 層が依然 Greek 固有 kind に密結合。
- **現状**: 「決定ログ 2026-06-03」で gate スコープ外（OpenAPI 破壊変更を伴う別タスク）として
  意識的に保留済み。両箇所にコメントで「新言語プラグイン追加時は両方更新せよ」と明記済み。
- **推奨対応**: kind をプラグインから動的に集約する仕組みを検討し、独立タスク化。

### F3 — matrices フォールバックの可視性（LOW）

- **内容**: `_get_trusted_matrices_dir` は profile 解決失敗時に repo matrices へフォールバックしつつ
  `logger.warning` のみで続行。本番で profile 0 件なら検索自体が機能しないはずで、後段で
  より不明瞭なエラーになる可能性。
- **現状**: repo レイアウト開発時の利便性とのトレードオフとして許容範囲。
- **推奨対応**: 将来 fail-fast 方針へ寄せるか要検討（現時点では記録のみ）。

### F4 — 循環回避のローカル import（LOW）

- **内容**: `distance.py` が `core.ports.profiles` を関数内 import して循環依存
  （`profiles` → `distance` の `register_trusted_matrices_dir`）を回避。理由コメントあり。
- **現状**: 妥当な対処だが、循環の根本構造（registry 登録が `register_trusted_matrices_dir` を呼ぶ）は残存。
- **推奨対応**: ports 層が肥大化した際の構造再整理候補として記録。

### F5 — explainer facade の redundant alias（LOW）

- **内容**: `explainer.py` が re-export 明示（F401 回避 + 公開 facade 維持）のため
  `import X as X` を 50 行超機械的に並べている。
- **現状**: 公開 import パス `phonology.explainer` 維持のための既存 facade パターン踏襲。
- **推奨対応**: `__all__` で公開を明示する方式への置換を検討（現状維持でも可）。
