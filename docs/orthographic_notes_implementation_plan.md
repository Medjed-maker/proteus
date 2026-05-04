# Orthographic Note Implementation Plan

作成日: 2026-05-02  
実装完了日: 2026-05-05  
対象: Proteus Ancient Greek pilot  
目的: 音韻ルールとは別に、表記体系・綴字慣習・初学者向け補助説明を候補ごとに表示できるようにする。

## 背景

東京大学 松浦高志先生から、方言碑文だけでなく、前403/2年以前のアッティカ碑文表記に苦手意識を持つ学生にも有用である、という指摘があった。特に `παιδίο = παιδίου (paidiou)` のように、初心者が辞書形・正規化形との対応を理解できる補助表示が必要である。

この変更では、既存の `Applied rules` / `Matched rules` に表記説明を混ぜず、独立した `Orthographic note` / `表記体系コメント` を追加する。

## 進捗凡例

- [ ] 未着手
- [x] 完了
- [~] 実装中または要再検討

## 方針

- [x] `Applied rules` は音韻・形態音韻ルールの説明に限定する。
- [x] `Orthographic note` は表記体系、綴字慣習、辞書形・正規化形との対応、初学者向け読み替え補助に限定する。
- [x] 前403/2年以前のアッティカ碑文表記は、一般的な表記ゆれではなく独立カテゴリとして扱う。
- [x] コア検索エンジンには Ancient Greek 固有ロジックを入れず、言語固有の note builder として分離する。
- [x] 初期実装では断定を避け、`may correspond to` / `対応する可能性があります` のような控えめな文言にする。
- [x] 将来的に時代・コーパス・出典メタデータが入るまで、`pre-403/2 BCE Attic` の判定は低信頼度または明示的ヒント付きにする。

## 対象ファイル

- [x] `src/api/_models.py`
- [x] `src/api/_hit_formatting.py`
- [x] `src/api/main.py`
- [x] `src/phonology/profiles.py`
- [x] `src/phonology/languages/ancient_greek/`
- [x] `src/web/index.html`
- [x] `src/web/static/translations.json`
- [x] `tests/test_api_main.py`
- [x] `tests/test_i18n.py`
- [x] `tests/test_web_assets.py`
- [x] `tests/test_search_annotation.py` または新規テストファイル
- [x] `docs/REQUIREMENTS.md`
- [x] `docs/ARCHITECTURE.md`
- [x] `docs/CODEMAPS/api.md`
- [x] `README.md`

## Phase 0: 仕様確定

- [x] 表示欄名を決める。
  - [x] 英語: `Orthographic note`
  - [x] 日本語: `表記体系コメント`
- [x] UI上の配置順を決める。
  - [x] `Difference summary`
  - [x] `Matched rules`
  - [x] `Orthographic note`
  - [x] `Alignment`
  - [x] `Why this candidate?`
  - [x] `Notes`
- [x] 表示対象を決める。
  - [x] 候補カードごとに表示する。
  - [x] 注記がない候補では欄ごと非表示にする。
  - [x] `Notes` とは統合しない。
- [x] 初期対応する note 種別を決める。
  - [x] `orthographic_correspondence`
  - [x] `historical_spelling_system` は初期実装から除外し、履歴的表記体系の初期カテゴリは `pre_403_2_attic` に限定する。
  - [x] `beginner_aid`
  - [x] `pre_403_2_attic`
- [x] API互換性方針を決める。
  - [x] `orthographic_notes` はデフォルト空配列にして既存クライアントを壊さない。
  - [x] `explanation` は既存の音韻説明として残す。

## Phase 1: APIモデル追加

- [x] `src/api/_models.py` に `OrthographicNote` Pydantic model を追加する。
- [x] `kind` を Literal で定義する。
  - [x] `"orthographic_correspondence"`
  - [x] `"beginner_aid"`
  - [x] `"pre_403_2_attic"`
- [x] `label: str` を追加する。
- [x] `messages: list[str]` を追加する。
- [x] `normalized_form: str | None` を追加する。
- [x] `romanization: str | None` を追加する。
- [x] `period_label: str | None` を追加する。
- [x] `references: list[str]` を追加する。
- [x] `confidence: Literal["low", "medium", "high"]` を追加する。
- [x] `SearchHit` に `orthographic_notes: list[OrthographicNote] = Field(default_factory=list)` を追加する。
- [x] Field description を英語で追加する。
- [x] Pydantic validation テストを追加する。
  - [x] `orthographic_notes` のデフォルトが空配列になる。
  - [x] 不正な `kind` を拒否する。
  - [x] 不正な `confidence` を拒否する。
  - [x] `messages` が list 以外なら拒否する。

## Phase 2: Note Builder の設計

- [x] Ancient Greek 固有ロジックの配置先を決める。
  - [x] 推奨: `src/phonology/languages/ancient_greek/orthography_notes.py`
- [x] API層で使う公開関数を定義する。

```python
def build_orthographic_notes(
    *,
    query_form: str,
    candidate_headword: str,
    candidate_ipa: str,
    query_ipa: str,
    applied_rule_ids: list[str],
    response_language: Literal["en", "ja"],
    orthography_hint: str | None = None,
) -> list[OrthographicNotePayload]:
    ...
```

- [x] `api._models.OrthographicNote` に直接依存しない内部 payload 型を作るか決める。
  - [x] 推奨: phonology 層では dataclass または dict を返し、API層で Pydantic model に変換する。
- [x] 言語非依存性を保つため、`LanguageProfile` に optional hook を追加するか決める。
  - [x] 案A: `LanguageProfile.orthographic_note_builder`
  - [ ] 案B: API層で `language_id == "ancient_greek"` のときだけ builder を呼ぶ。
  - [x] 推奨: 案A。将来のラテン語・コプト語にも拡張しやすい。
- [x] `LanguageProfile` に hook を追加する場合の後方互換性を確認する。
  - [x] dataclass にデフォルト `None` を設定する。
  - [x] 既存テスト用プロファイルの初期化が壊れないことを確認する。

## Phase 3: 初期ルールセット

### 3.1 Orthographic correspondence

- [x] `original -> regularized` 対応を扱う最小データ構造を決める。
- [x] まずは手 curated な小規模対応表を追加するか決める。
  - [x] 候補: `data/languages/ancient_greek/orthography/orthographic_correspondences.yaml`
- [x] 対応表の schema を決める。

```yaml
entries:
  - original: "παιδίο"
    normalized: "παιδίου"
    romanization: "paidiou"
    kind: "orthographic_correspondence"
    tags: ["beginner_aid", "inscriptional"]
    confidence: "medium"
    references: []
```

- [x] `παιδίο -> παιδίου` を初期データとして入れるか確認する。
- [x] 正規化形が候補見出し語と一致しない場合の扱いを決める。
  - [x] 候補の `headword`
  - [x] 入力語形の `query_form`
  - [x] 正規化・読み替え形 `normalized_form`
  - [x] この3つを区別して表示する。
- [x] transliteration は既存 `phonology.transliterate.transliterate()` を使う。
- [x] `romanization` がデータにない場合は自動生成する。

### 3.2 Historical spelling system

- [x] `pre_403_2_attic` の trigger 条件を決める。
  - [x] 明示的 `orthography_hint == "pre_403_2_attic"`
  - [x] または対応表に `tags: ["pre_403_2_attic"]` がある場合
- [x] 時代ヒントなしの自動判定は初期実装では避ける。
- [x] 表示文言を決める。
  - [x] EN: `This form may reflect a pre-403/2 BCE Attic inscriptional spelling.`
  - [x] JA: `この形は、紀元前403/2年以前のアッティカ碑文表記を反映している可能性があります。`
- [x] Buck データの `grc_orth_4_*` を参照に使えるか確認する。
- [x] 外部参照リンクを docs に記録する。

### 3.3 Beginner aid

- [x] 初学者向け文言を決める。
  - [x] EN: `Reading aid: this form may correspond to ...`
  - [x] JA: `読み替え補助: この形は ... に対応する可能性があります。`
- [x] `beginner_aid` は常時表示するか決める。
  - [x] 初期実装: 対応表に `beginner_aid` tag がある場合だけ表示。
  - [ ] 将来: UIで beginner mode を切り替え。
- [x] beginner mode の UI トグルは初期実装から外すか決める。
  - [x] 推奨: 初期実装では外す。欄そのものを学生にも読める文体にする。

## Phase 4: API整形への組み込み

- [x] `_build_search_hit()` の引数に必要な情報を追加する。
  - [x] `query_form`
  - [x] `language_profile` または note builder
  - [x] `orthography_hint`
- [x] `src/api/main.py` から `_build_search_hit()` へ `request.query_form` を渡す。
- [x] `SearchRequest` に `orthography_hint` を追加するか決める。
  - [x] 初期候補: `None` デフォルトの optional field
  - [x] 許可値: `"standard"`, `"inscriptional"`, `"pre_403_2_attic"`
- [x] `orthography_hint` を追加する場合、バリデーションを追加する。
- [x] `legacy_language_alias_used` と同様、後方互換性を壊さないことを確認する。
- [x] `_build_search_hit()` 内で note builder を呼ぶ。
- [x] note builder がない言語では空配列を返す。
- [x] `rules_applied` とは別に `orthographic_notes` を SearchHit に詰める。
- [x] `to_prose(explanation)` の文言には混ぜない。
- [x] `orthographic_correspondence` がある候補は `Supported` 表示グループに昇格する。
  - [x] これはランキングスコアや候補順序の変更ではなく、UI上の Supported/Exploratory グルーピング整理として扱う。

## Phase 5: Web UI

- [x] `src/web/static/translations.json` に翻訳キーを追加する。
  - [x] `sectionOrthographicNote`
  - [x] `orthographicNoteEmpty` は原則不要。注記なしなら欄を出さない。
- [x] `src/web/index.html` 内蔵 fallback translations に同じキーを追加する。
- [x] `renderOrthographicNotes(notes)` を追加する。
- [x] note の `messages` を paragraph または list として描画する。
- [x] `normalized_form` がある場合は強調表示する。
- [x] `romanization` がある場合は括弧で表示する。
- [x] `references` は初期実装では非表示または小さく表示する。
- [x] `confidence` は初期UIでは表示しないか決める。
  - [x] 推奨: 表示しない。文言を `may` にして不確実性を表現する。
- [x] `appendCardBody()` の `Matched rules` 後、`Alignment` 前に挿入する。
- [x] compact card での表示方針を決める。
  - [x] 推奨: compact card でも details 展開内には表示する。
- [x] スマホ幅で文章がはみ出さないことを確認する。
- [x] 既存 `Notes` details とは別セクションとして表示する。

## Phase 6: データ

- [x] `data/languages/ancient_greek/orthography/` ディレクトリを追加するか決める。
- [x] 対応表 YAML を追加する。
- [x] 対応表のローダーを追加する。
- [x] trusted path 方針を確認する。
  - [x] パッケージ内データなので `resolve_language_data_dir()` 経由で読む。
- [x] `pyproject.toml` の package data 対象に含まれるか確認する。
- [x] データファイル schema テストを追加する。
- [x] 最初の seed examples を決める。
  - [x] `παιδίο -> παιδίου`
  - [x] 前403/2年以前の Attic 表記で安全に扱える例
- [x] 出典・レビュー状態を `_meta` に入れる。
  - [x] `status: provisional`
  - [x] `review_status: not_expert_reviewed`
  - [x] `citation_ready: false`

## Phase 7: EpiDoc choice データとの接続

- [x] `scripts/extract_epidoc_choices.py` の出力を note builder に直接使うか検討する。
- [x] 初期実装では学習データとランタイム注記データを分ける。
  - [x] `data/training/epidoc_choices.json`: 学習・行列生成用
  - [x] `data/languages/ancient_greek/orthography/*.yaml`: ランタイム表示用
- [x] 将来の変換ジョブを設計する。
  - [x] EpiDoc choice JSON から高頻度対応を抽出
  - [x] 人手レビュー
  - [x] ランタイム YAML に昇格
- [x] `tm_id` や `source_file` を将来の references に使う設計を残す。
- [x] papyri.info 由来データと Attic inscription 由来データを混同しない。
- [x] 前403/2年以前の Attic 専用データソースを別途調査する。

Phase 7 方針: papyri.info EpiDoc choice 抽出は、phonological matrix training のための候補ペア生成に限定する。`tm_id` と `source_file` は将来の出典レビュー・references 昇格候補として保持するが、未レビューのまま runtime orthographic note YAML へ投入しない。前403/2年以前の Attic inscription evidence は別ソースとして扱い、papyri.info 由来メタデータと混同しない。

## Phase 8: テスト

### 8.1 API model tests

- [x] `SearchHit` が `orthographic_notes=[]` を持つ。
- [x] 正常な `OrthographicNote` を受け入れる。
- [x] 不正な `kind` を拒否する。
- [x] 不正な `confidence` を拒否する。
- [x] `messages` が list でない場合は拒否する。

### 8.2 Builder tests

- [x] `παιδίο` 入力で `παιδίου` 対応注記を返す。
- [x] `romanization` が `paidiou` になる。
- [x] `lang="en"` で英語文言を返す。
- [x] `lang="ja"` で日本語文言を返す。
- [x] 対応表にない語形では空配列を返す。
- [x] `orthography_hint="pre_403_2_attic"` で historical note を返す。
- [x] `orthography_hint=None` では前403/2年注記を出しすぎない。
- [x] builder が phonological `applied_rules` を変更しない。

### 8.3 API integration tests

- [x] `/search` の hit に `orthographic_notes` が含まれる。
- [x] note がない場合は空配列で返る。
- [x] `response_language="ja"` で日本語 note が返る。
- [x] `orthography_hint` を渡せる場合はレスポンスに反映される。
- [x] 既存の `rules_applied` / `explanation` テストが壊れない。

### 8.4 Web tests

- [x] `translations.json` の英日キー整合性テストを更新する。
- [x] `WEB_ASSET_KEYS` に新キーを追加する。
- [x] `Orthographic note` セクションが描画される。
- [x] note が空の場合はセクションを描画しない。
- [x] `Matched rules` と `Alignment` の間に表示される。
- [x] 日本語表示で `表記体系コメント` が出る。

### 8.5 Regression tests

- [x] 既存の full-form search が同じランキングを保つ。
- [x] short-query search が同じランキングを保つ。
- [x] partial-form search が同じランキングを保つ。
- [x] `uv run pytest tests/test_api_main.py` が通る。
- [x] `uv run pytest tests/test_i18n.py` が通る。
- [x] `uv run pytest tests/test_web_assets.py` が通る。
- [x] `uv run pytest` 全体が通る。

## Phase 9: ドキュメント更新

- [x] `README.md` の概要に「研究者・学生支援」を追加する。
- [x] `README.md` の Status に、表記注記は provisional であることを明記する。
- [x] `docs/REQUIREMENTS.md` の利用者に初学者・学生を追加する。
- [x] `docs/REQUIREMENTS.md` に `表記体系コメント` 要件を追加する。
- [x] `docs/ARCHITECTURE.md` に orthographic note builder の位置づけを追加する。
- [x] `docs/CODEMAPS/api.md` の `SearchHit` に `orthographic_notes` を追加する。
- [x] `docs/phonology_rules.md` に「音韻ルールと表記注記は分ける」方針を追記する。
- [x] `docs/ROADMAP.md` に「student-facing inscriptional orthography aid」を追加する。

## Phase 10: プロダクト文言

- [x] Web subtitle を更新する。
  - [x] EN候補: `Explainable phonological search and inscriptional spelling aid, beginning with Ancient Greek`
  - [x] JA候補: `説明可能な音韻検索と碑文表記読解支援。まずは古代ギリシャ語から`
  - [x] 方針: 将来の多言語展開を示しつつ、現MVPの対象が古代ギリシャ語であることは明記する。
- [x] `Supported candidates` の説明を必要なら調整する。
- [x] `Orthographic note` の文言を研究者にも学生にも読める温度にする。
- [x] 断定的な `is` ではなく `may reflect` / `可能性があります` を使う。
- [x] `beginner` というラベルをUIに直接出すか検討する。
  - [x] 推奨: 初期UIでは出さず、本文で自然に補足する。

## Phase 11: リリース前確認

- [x] API schema の差分を確認する。
- [x] 既存クライアントが壊れないことを確認する。
- [x] UIスクリーンショットを確認する。
- [x] 日本語・英語両方の表示を確認する。
- [x] `παιδίο` のデモケースを手動確認する。
- [x] `orthography_hint` なしで過剰注記が出ないことを確認する。
- [x] `pre_403_2_attic` ヒントありで注記が出ることを確認する。
- [x] README に provisional status が残っていることを確認する。
- [x] changelog または PR description に変更点を書く。

## 推奨実装順

- [x] 1. API model に `OrthographicNote` と `SearchHit.orthographic_notes` を追加する。
- [x] 2. テストで API schema の期待値を固める。
- [x] 3. Ancient Greek note builder を空実装で追加する。
- [x] 4. `LanguageProfile` に optional hook を追加する。
- [x] 5. `_build_search_hit()` から hook を呼ぶ。
- [x] 6. Web UI に `Orthographic note` セクションを追加する。
- [x] 7. 最小対応表として `παιδίο -> παιδίου` を追加する。
- [x] 8. builder で対応表を読む。
- [x] 9. `pre_403_2_attic` ヒントを追加する。
- [x] 10. i18n と docs を更新する。
- [x] 11. focused tests を通す。
- [x] 12. full test suite を通す。

## 最小実装スコープ

最初のPRで狙う最小スコープ:

- [x] `SearchHit.orthographic_notes` を追加する。
- [x] `παιδίο -> παιδίου (paidiou)` の固定または小規模 YAML 対応表を追加する。
- [x] `Orthographic note` セクションを UI に表示する。
- [x] 英日文言を追加する。
- [x] `pre_403_2_attic` は `kind` と文言だけ用意し、自動判定はしない。
- [x] focused tests を追加する。

最初のPRではやらないこと:

- [ ] 大規模な EpiDoc choice データのランタイム取り込み
- [ ] PHI / AIO からの自動抽出
- [ ] 時代推定の自動化
- [ ] beginner mode トグル
- [ ] ランキングスコア・候補順序への反映
- [ ] 音韻距離行列の再生成

## 確認済みリスクと対策

- リスク: 音韻ルールと表記注記が混ざる。
  - [x] 対策: `rules_applied` と `orthographic_notes` をAPI上もUI上も分離する。
- リスク: 前403/2年以前の Attic 表記を過剰に自動判定する。
  - [x] 対策: 初期実装では明示ヒントまたは curated data のみに限定する。
- リスク: `παιδίο -> παιδίου` のような屈折形対応を lemma と混同する。
  - [x] 対策: `headword` と `normalized_form` を別フィールドにする。
- リスク: 学生向け文言が研究用途で軽く見える。
  - [x] 対策: 見出しは学術的に `Orthographic note` とし、本文だけ平易にする。
- リスク: Ancient Greek 固有処理がコアに混入する。
  - [x] 対策: `LanguageProfile` hook または `languages/ancient_greek` モジュールに閉じる。
- リスク: 出典未確認の注記が権威づけされる。
  - [x] 対策: `_meta.status: provisional` と `citation_ready: false` をデータに持たせる。

## 参考情報

- Attic Inscriptions Online は Attic inscriptions を IG I up to 403/2 BC と IG II after 403/2 BC に分けている。
- AtticGreek / Digital Greek では、403 BCE 以前のアテナイ公文書表記について、eta/omega がない、spurious diphthongs が単一の epsilon/omicron で綴られる、アクセントがない、などの差異が整理されている。
- Phase 3 初期データでは Buck `grc_orth_4_*` を個別 seed entry の直接出典にせず、Buck section 4 orthography notes を broad reference として YAML `_meta.references` に記録する。

参考リンク:

- https://www.atticinscriptions.com/
- https://neelsmith.github.io/DigitalGreek/news/attic/
- https://neelsmith.quarto.pub/atticgreek/concepts/
