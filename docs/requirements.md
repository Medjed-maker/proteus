# Proteus (HPSI) 要件定義書

更新日: 2026-04-24  
対象リポジトリ: proteus  
対象フェーズ: Phase 0〜Phase 2 初期実装  
版: v0.1  
位置づけ: 修正版事業計画書および統合実装ロードマップを反映した開発要件定義

---

## 1. システム導入の背景・目的

### 1.1 背景

古代言語資料、特に碑文・パピルス・写本・文学テキストでは、以下の理由により既存の文字列一致型検索だけでは必要な証拠に到達できない。

- 表記ゆれ
- 方言差
- 時代差
- 音韻変化
- 欠損・断片化
- 異体字・転写差
- コーパス間の分断
- 既存検索の完全一致依存
- 候補推定過程の非再現性
- LLM出力の根拠不足・ハルシネーションリスク

従来の主要ツールは、文学コーパス、碑文コーパス、パピルスコーパス、形態素解析、AI復元など、それぞれに強みを持つ一方で、崩れた語形・方言形・非標準表記から候補見出し語を根拠付きで逆引きする機能を十分に提供していない。

### 1.2 目的

本システムの目的は、古代言語資料における「検索不能」を解消し、研究者が不明語形・崩れた綴り・断片的表記から、候補語・適用音韻ルール・方言仮説・出典・検索履歴を取得できる、説明可能で再現可能な研究インフラを構築することである。

### 1.3 事業上の目的

本システムは、単なる古代ギリシャ語検索ツールではなく、歴史音韻変化を扱う言語非依存フレームワークとして設計する。

初期実装では古代ギリシャ語 PHI 碑文をパイロット対象とするが、アーキテクチャ上は以下への拡張を前提とする。

- ラテン語碑文
- パピルス資料
- コプト語
- 古代セム語群
- アッカド語・楔形文字資料
- 古代エジプト語
- サンスクリット語
- 中英語・古高ドイツ語など歴史言語学一般

### 1.4 開発方針

現行実装は、古代ギリシャ語専用アプリとして固定せず、以下の形に再設計する。

```text
language-independent phonological search framework
+ ancient_greek pilot plugin
```

すなわち、古代ギリシャ語は本体ではなく、最初の言語プラグインとして扱う。

---

## 2. システムの概要

### 2.1 システム名称

正式名称候補:

```text
Historical Phonological Search Infrastructure
```

略称:

```text
HPSI
```

実装リポジトリ名:

```text
Proteus
```

### 2.2 システムコンセプト

古代言語の崩れた綴り・方言形・音韻変異形を、言語ごとの音韻ルールセットと音韻距離行列に基づいて候補見出し語へ逆引きし、その根拠を説明可能な形で返す研究インフラ。

### 2.3 基本アーキテクチャ

本システムは以下の3層構造を基本とする。

#### Layer 1: 言語非依存フレームワーク

役割:

- 音韻距離計算
- 置換行列の解釈
- 候補生成
- 候補ランキング
- アラインメント
- 説明生成
- 検索履歴記録
- APIレスポンス整形
- MCPツール応答生成

言語依存性:

- 原則として持たない
- `ancient_greek`, `latin` などの特定言語名に直接依存しない

#### Layer 2: 言語プラグイン / ルールセット層

役割:

- 表記から IPA / 音素列への変換
- 音素インベントリ定義
- 母音・子音分類
- 方言定義
- 音韻変化ルール定義
- 音韻距離行列
- 語彙データ
- 言語固有の前処理・正規化

初期対象:

- `ancient_greek`

将来対象:

- `latin`
- `coptic`
- `ancient_semitic`
- `akkadian`
- `egyptian`
- その他

#### Layer 3: コーパスアダプター層

役割:

- 各コーパスへの接続
- データ正規化
- 出典情報の保持
- 外部DBへのリンク生成
- ライセンス条件に応じたデータ保持方式の切り替え

対象候補:

- PHI Greek Inscriptions
- Perseus / Scaife
- papyri.info / DDbDP
- TLG
- EAGLE / EpiDoc
- CIL / EDCS
- CDLI

### 2.4 利用者

主な利用者は以下とする。

- 碑文学者
- パピルス学者
- 古典語研究者
- 比較言語学者
- デジタルヒューマニティーズ研究者
- 大学教員
- 大学院生
- 古典語・碑文学を学ぶ初学者
- 学部・大学院授業で碑文表記を扱う学生
- 研究プロジェクト運営者
- LLM / RAG / MCP 経由で古代言語分析を行う開発者

### 2.5 提供インターフェース

本システムは段階的に以下を提供する。

- CLI
- Web UI
- REST API
- MCP Server
- 将来的な Scaife / Perseus ウィジェット
- 将来的なバッチアノテーションAPI

---

## 3. 業務要件

### 3.1 対象業務

本システムは、古代言語研究における以下の業務を支援する。

#### 3.1.1 不明語形の候補探索

研究者が碑文・パピルス・写本等で確認した非標準語形を入力し、候補見出し語を探索する。

#### 3.1.2 表記ゆれ・方言差の分析

入力語形と候補見出し語の差分について、適用可能な音韻変化・方言対応・表記体系差を確認する。

#### 3.1.3 研究根拠の整理

候補ごとに、以下を記録・確認できるようにする。

- 入力語形
- 候補見出し語
- 適用ルール
- 方言仮説
- スコア
- 類例
- 出典
- ルールの参照文献
- 検索条件
- システムバージョン
- ルールセットバージョン

#### 3.1.4 LLM回答のグラウンディング

LLMが古代言語の不明語形について回答する際、本システムを MCP / API 経由で参照し、根拠付き候補を取得できるようにする。

#### 3.1.5 ルールセットの改善

研究者・協力者が GitHub Pull Request 等を通じて、音韻ルール・用例・参考文献・例外条件を追加・修正できるようにする。

### 3.2 対象外業務

初期フェーズでは以下を対象外とする。

- 文レベル翻訳
- 自動校訂の最終判断
- 欠損復元の自動確定
- OCR / HTR / 画像認識
- 全古代言語への同時対応
- 学術的真偽の自動確定
- 商用クローズドコーパスの無許諾再配布
- LLM単独による候補生成

### 3.3 業務フロー

#### 3.3.1 単発検索フロー

1. 利用者が語形を入力する
2. 利用者が対象言語を選択する
3. 必要に応じて方言ヒント・時代ヒント・資料種別を指定する
4. システムが入力を正規化する
5. システムが言語プラグインに基づき音素列へ変換する
6. システムが候補語彙を検索する
7. システムが音韻距離・ルール適用可能性を評価する
8. システムが候補をランキングする
9. システムが適用ルール・根拠・出典を返す
10. 利用者が候補を確認し、必要に応じて保存・引用する

#### 3.3.2 ルールセット改善フロー

1. 研究者または開発者が既存ルールの問題を発見する
2. ルールYAML / JSONを修正する
3. テストケースを追加する
4. Pull Requestを作成する
5. CIでスキーマ検証・検索テストを実行する
6. レビュー後にマージする
7. ルールセットのバージョンを更新する
8. 将来的に DOI / release note に反映する

#### 3.3.3 MCP利用フロー

1. LLMクライアントが MCP Server に問い合わせる
2. `query_form`, `source_language`, `dialect_hint`, `max_candidates` を送信する
3. MCP Server が検索エンジンを呼び出す
4. 候補・根拠・出典・検証URLを構造化して返す
5. LLMはその結果を根拠として回答を生成する

---

## 4. システム要件

### 4.1 全体構成

本システムは以下のコンポーネントで構成する。

```text
proteus/
  src/
    phonology/
      search/
      distance.py
      alignment.py
      explainer.py
      indexing.py
      query.py
      types.py
      scoring.py

    phonology/languages/
      registry.py
      base.py
      ancient_greek/
        profile.py
        converter.py
        phones.py
        normalizer.py
      latin/
        profile.py
        converter.py

    proteus_api/
      main.py
      models.py
      dependencies.py

    proteus_mcp/
      server.py
      tools.py

    proteus_web/
      static/
      templates/

  data/
    languages/
      ancient_greek/
        profile.yaml
        rules/
        matrices/
        lexicon/
        examples/
      toy_language/
        profile.yaml
        rules/
        matrices/
        lexicon/

  tests/
    core/
    languages/
    api/
    mcp/
    fixtures/
```

### 4.2 コア設計要件

#### 4.2.1 コアは特定言語に依存しないこと

`phonology` は以下に直接依存してはならない。

- 古代ギリシャ語
- アッティカ方言
- コイネー
- ギリシャ文字
- ラテン語
- 特定コーパス名
- 特定辞書ファイル名

#### 4.2.2 言語依存処理は LanguageProfile 経由で注入すること

各言語は `LanguageProfile` として登録する。

必須フィールド:

```python
language_id: str
display_name: str
default_dialect: str | None
supported_dialects: tuple[str, ...]
converter: Callable
phone_inventory: frozenset[str]
vowel_phones: frozenset[str]
lexicon_path: Path
matrix_path: Path
rules_dir: Path
```

#### 4.2.3 新言語追加時にコア変更を不要にすること

新言語の追加は原則として以下のみで完了すること。

- `data/languages/{language_id}/profile.yaml`
- `data/languages/{language_id}/rules/*.yaml`
- `data/languages/{language_id}/matrices/*.json`
- `data/languages/{language_id}/lexicon/*.json`
- `src/proteus_languages/{language_id}/converter.py`
- `src/proteus_languages/{language_id}/profile.py`

### 4.3 API要件

#### 4.3.1 REST API

初期APIとして以下を提供する。

```http
POST /search
GET /languages
GET /languages/{language_id}
GET /health
GET /version
```

#### 4.3.2 SearchRequest

```json
{
  "query_form": "ΔΑΜΟΣΘΕΝΑΣ",
  "language": "ancient_greek",
  "dialect_hint": "attic",
  "max_candidates": 10,
  "include_explanations": true,
  "include_alignment": true,
  "include_citations": true
}
```

#### 4.3.3 SearchResponse

```json
{
  "query": {
    "query_form": "ΔΑΜΟΣΘΕΝΑΣ",
    "language": "ancient_greek",
    "dialect_hint": "attic",
    "normalized_form": "δαμοσθενας",
    "ipa": ["d", "a", "m", "o", "s", "tʰ", "e", "n", "a", "s"]
  },
  "candidates": [
    {
      "lemma": "Δημοσθένης",
      "score": 0.91,
      "confidence": "high",
      "dialect_attribution": ["attic"],
      "applied_rules": [
        {
          "rule_id": "attic_ionic_a_to_eta",
          "description": "Long ā corresponds to Attic-Ionic ē",
          "input": "a",
          "output": "ɛː",
          "position": 1,
          "references": ["Buck §...", "Smyth §..."]
        }
      ],
      "alignment": {
        "query": ["d", "a", "m", "o", "s", "tʰ", "e", "n", "a", "s"],
        "candidate": ["d", "ɛː", "m", "o", "s", "tʰ", "e", "n", "ɛː", "s"]
      },
      "attestations": [],
      "explanation": "..."
    }
  ],
  "metadata": {
    "engine_version": "0.1.0",
    "ruleset_version": "ancient_greek-0.1.0",
    "matrix_version": "attic_doric-0.1.0",
    "request_id": "..."
  }
}
```

### 4.4 MCP要件

MCP Server は以下のツールを提供する。

```text
ancient_phonology.search
```

将来的には以下を追加する。

```text
ancient_phonology.explain_rule
ancient_phonology.list_languages
ancient_phonology.list_rules
ancient_phonology.compare_candidates
```

MCP応答には以下を含める。

- 候補語
- スコア
- 適用ルール
- 方言仮説
- 参考文献
- 出典
- 検証URL
- エンジンバージョン
- ルールセットバージョン

### 4.5 データ管理要件

#### 4.5.1 ルールセット

ルールセットは YAML または JSON で管理する。

必須フィールド:

```yaml
id: string
name_en: string
name_ja: string
input: string
output: string
context: string | object
dialects: list[string]
period: string | object
direction: string
confidence: string
references: list[string]
examples: list[object]
notes: string
```

#### 4.5.2 語彙データ

語彙データは言語ごとに管理する。

必須フィールド:

```json
{
  "lemma": "string",
  "language": "string",
  "ipa": ["string"],
  "gloss": "string",
  "pos": "string",
  "source": "string",
  "source_id": "string",
  "dialect": ["string"],
  "period": "string"
}
```

#### 4.5.3 距離行列

距離行列は言語・方言ペア・時代単位で管理可能とする。

例:

```text
data/languages/ancient_greek/matrices/attic_doric.json
data/languages/ancient_greek/matrices/koine_variants.json
data/languages/latin/matrices/early_late_latin.json
```

---

## 5. 機能要件

### 5.1 言語プロファイル管理機能

#### FR-001: 言語一覧取得

システムは登録済み言語一覧を返せること。

受け入れ基準:

- `GET /languages` が利用可能である
- `ancient_greek` が含まれる
- 将来的に `latin` などを追加しても API 仕様を変更しない

#### FR-002: 言語プロファイル解決

システムは `language` パラメータから対応する `LanguageProfile` を解決できること。

受け入れ基準:

- `language="ancient_greek"` で既存の古代ギリシャ語検索が動作する
- 未対応言語の場合は明確なエラーを返す
- コア検索処理は言語IDに直接分岐しない

### 5.2 入力正規化機能

#### FR-003: 入力語形の正規化

システムは入力語形に対して、言語ごとの正規化処理を実行できること。

対象例:

- 大文字・小文字差
- ダイアクリティカルマーク
- ポリトニック表記
- 碑文体大文字
- 語末記号
- 空白・句読点

#### FR-004: 言語別IPA変換

システムは言語プラグインの converter を用いて入力語形を IPA / 音素列に変換できること。

受け入れ基準:

- Greek converter は `ΔΑΜΟΣΘΕΝΑΣ` を音素列に変換できる
- IPA tokenization は言語非依存モジュールとして再利用できる
- 新言語 converter 追加時に core を変更しない

### 5.3 検索機能

#### FR-005: 候補生成

システムは入力語形から候補見出し語を生成できること。

検索方式:

- 子音骨格によるシード候補取得
- 音韻距離行列による候補評価
- アラインメントに基づく差分抽出
- ルール適用可能性による補正
- スコア順ランキング

#### FR-006: BLAST型3段階検索

検索は原則として以下の3段階で実行する。

1. Seed: 高速な候補絞り込み
2. Extend: 音素列アラインメント
3. Rank / Explain: スコアリングと説明生成

#### FR-007: 候補ランキング

システムは候補をスコア順に返すこと。

スコア要素:

- 音韻距離
- 適用可能ルール数
- ルール信頼度
- 方言ヒント一致
- 時代ヒント一致
- 語彙頻度または出典信頼度
- 完全一致・既知変異形の優先

### 5.4 説明生成機能

#### FR-008: 適用ルール表示

候補ごとに適用された音韻ルールを表示すること。

表示項目:

- ルールID
- ルール名
- 入力音
- 出力音
- 位置
- 方言
- 時代
- 参考文献
- 用例
- 信頼度

#### FR-009: アラインメント表示

入力語形と候補語形の音素アラインメントを表示できること。

#### FR-010: 候補理由の自然文説明

システムは候補が提示された理由を自然文で説明できること。

注意:

- LLMによる自由生成を中核根拠にしない
- 構造化されたルール適用結果をもとに説明文を生成する
- 説明文にはルールID・出典・バージョンを紐づける

#### FR-010-ORTH: 候補ごとの表記体系コメント

システムは候補ごとに、音韻ルールとは別の `表記体系コメント`
を表示できること。

対象:

- 表記体系・綴字慣習
- 辞書形・正規化形との対応
- 初学者向けの読み替え補助
- 前403/2年以前のアッティカ碑文表記など、明示的な時代・表記体系ヒント

受け入れ基準:

- 注記がない候補では欄ごと表示しない
- `rules_applied` / `Applied rules` には音韻・形態音韻ルールだけを入れる
- 表記注記は `orthographic_notes` として構造化して返す
- 初期データは provisional とし、expert review 前の研究引用可能データとして扱わない
- EpiDoc choice 抽出データは未レビューのまま runtime 表記注記へ投入しない

### 5.5 検索履歴・再現性機能

#### FR-011: 検索メタデータ記録

検索ごとに以下を記録可能とする。

- request_id
- query_form
- normalized_form
- language
- dialect_hint
- engine_version
- ruleset_version
- matrix_version
- timestamp
- candidate_ids
- applied_rule_ids

#### FR-012: 再実行可能な検索条件

将来的に、検索条件を保存し、同じ条件で再実行できるようにする。

### 5.6 ルールセット管理機能

#### FR-013: ルールスキーマ検証

ルールセットは CI でスキーマ検証されること。

#### FR-014: ルール単位テスト

各ルールは、最低1件以上の positive example を持つことが望ましい。

#### FR-015: ルールバージョン管理

ルールセットにはバージョンを付与する。

例:

```text
ancient_greek-ruleset-v0.1.0
```

### 5.7 MCP連携機能

#### FR-016: MCP検索ツール

MCP Server は LLM クライアントからの検索要求を受け、構造化された候補を返すこと。

#### FR-017: LLMグラウンディング用メタデータ

MCP応答には以下を含める。

- verification_url
- citation_info
- engine_version
- ruleset_version
- confidence
- applied_rules

### 5.8 アノテーション機能

初期フェーズでは必須ではないが、Phase 2以降で実装する。

#### FR-018: コーパス語形のレベル分類

コーパス内語形を以下に分類できること。

- Level 0: 辞書完全一致
- Level 1: 既知変異形
- Level 2: 規則的変異形
- Level 3: 不確定形
- Level 4: 未同定形

#### FR-019: バッチ処理

大量語形に対して、検索・分類・注釈付けをバッチ処理できること。

---

## 6. 非機能要件

### 6.1 性能要件

#### NFR-001: 検索応答時間

Phase 0〜1:

- 小規模語彙データで 3秒以内
- ローカルまたは無料枠デプロイで動作可能

Phase 2以降:

- p50: 3秒以内
- p95: 8秒以内を目標とする

#### NFR-002: 候補生成効率

全語彙に対する総当たり計算を避けるため、シードインデックスを利用する。

### 6.2 可用性

Phase 0では高可用性を要求しない。

Phase 2以降:

- API停止時にもルールセット・CLIは利用可能であること
- GitHub上のルールセットは独立して取得可能であること

### 6.3 拡張性

#### NFR-003: 言語追加の容易性

新言語追加時に core の変更を最小化する。

合格基準:

- `toy_language` を追加して core 変更なしで検索テストが通る
- `ancient_greek` は言語プラグインとして登録される
- APIは `language` パラメータで言語を切り替えられる

#### NFR-004: コーパス追加の容易性

新コーパス追加時に検索コアを変更しない。

### 6.4 保守性

#### NFR-005: モジュール分離

以下を明確に分離する。

- core search
- language plugins
- API
- MCP
- web UI
- data loaders
- tests

#### NFR-006: テスト容易性

各コンポーネントに単体テストを設ける。

対象:

- IPA tokenization
- language converter
- distance calculation
- alignment
- search execution
- explanation generation
- API request / response
- LanguageProfile registry
- toy_language integration

### 6.5 信頼性

#### NFR-007: 学術的説明可能性

システムは候補だけでなく、候補生成根拠を返すこと。

#### NFR-008: 不確実性の明示

確信度が低い候補は high / medium / low 等で明示する。

#### NFR-009: ルール由来の明示

候補には適用ルールIDを紐づける。

### 6.6 セキュリティ

Phase 0では個人情報を保存しない。

将来的にユーザー管理を行う場合:

- 認証
- APIキー
- レート制限
- ログの匿名化
- クエリハッシュ化
- 秘密情報の環境変数管理

を行う。

### 6.7 監査性

#### NFR-010: バージョン追跡

検索結果には以下を含める。

- engine_version
- ruleset_version
- matrix_version
- data_version

### 6.8 国際化

初期UIは英語を優先するが、日本語説明・日本語README補助を許容する。

将来的には以下に対応する。

- 英語UI
- 日本語UI
- 多言語説明ラベル
- ルール名の日英併記

---

## 7. 技術要件

### 7.1 使用言語

- Python 3.11 以上を推奨
- 型ヒントを使用する
- 可能な範囲で `dataclasses` または `pydantic` を使用する

### 7.2 バックエンド

- FastAPI
- Uvicorn
- Pydantic
- PyYAML
- 標準ライブラリ中心
- 必要に応じて LingPy を利用

### 7.3 フロントエンド

Phase 0:

- HTML
- JavaScript
- Tailwind CDN 等の軽量構成

Phase 1以降:

- React または Vue
- ただし、検索コアとは独立させる

### 7.4 データ格納

Phase 0:

- JSON
- YAML
- SQLite任意

Phase 1以降:

- SQLite
- PostgreSQL検討

Phase 2以降:

- PostgreSQL
- Redis cache 検討

### 7.5 API

- REST API
- OpenAPI / Swagger 自動ドキュメント
- CORS設定
- `/health`
- `/version`
- `/search`
- `/languages`

### 7.6 MCP

- Python MCP SDK
- tool name: `ancient_phonology`
- 将来的に `historical_phonology` への名称拡張を検討

### 7.7 CI/CD

GitHub Actions により以下を実行する。

- pytest
- ruff lint
- ruff format check
- ルールセットスキーマ検証
- JSON/YAML構文検証
- toy_language integration test

### 7.8 ディレクトリ構成要件

初期リファクタ後の推奨構成:

```text
src/
  proteus_core/
  proteus_languages/
  proteus_api/
  proteus_mcp/

data/
  languages/
    ancient_greek/
    toy_language/

tests/
  core/
  languages/
  api/
  mcp/
```

### 7.9 LanguageProfile実装要件

最低限、以下を実装する。

```python
@dataclass(frozen=True)
class LanguageProfile:
    language_id: str
    display_name: str
    default_dialect: str | None
    supported_dialects: tuple[str, ...]
    converter: Callable
    phone_inventory: frozenset[str]
    vowel_phones: frozenset[str]
    lexicon_path: Path
    matrix_path: Path
    rules_dir: Path
```

### 7.10 後方互換性

既存の古代ギリシャ語検索は壊さない。

受け入れ基準:

- 既存の `/search` 呼び出しが動く
- `language` 未指定時は `ancient_greek` として扱う
- 既存テストが通る
- 既存デモ入力が同等の結果を返す

---

## 8. 制約条件

### 8.1 事業上の制約

- 初期市場は小さいため、VC型の急拡大を前提にしない
- 研究インフラとしての信頼性・引用可能性を優先する
- 助成金・B2C・B2B・機関契約のハイブリッドモデルを前提とする
- UI単体ではなく、ルールセット・API・MCP・研究基盤として価値を持たせる

### 8.2 学術上の制約

- 候補は学術的確定判断ではなく、根拠付き仮説として提示する
- 出典・ルール・参考文献のない候補を過度に高信頼として扱わない
- 音韻変化ルールは専門家レビューが必要である
- 方言・時代・地域の不確実性を表示する
- LLM出力を一次根拠として扱わない

### 8.3 データライセンス上の制約

- PHI Greek Inscriptions は商用利用・再配布条件に注意する
- TLG はクローズドデータとして扱う
- Perseus / papyri.info / DDbDP / Morpheus 等は各ライセンスを確認する
- コーパス本文を保持せず、リンク・インデックス・メタデータのみ保持する設計を優先する場合がある
- データソースごとに Layer A / B / C の統合方針を分ける

### 8.4 技術上の制約

- Phase 0では過度な最適化をしない
- Elasticsearch等の重い構成は初期導入しない
- 認証・課金は初期対象外
- OCR / HTR / 画像処理は初期対象外
- 大規模DB運用は Phase 1 以降に判断する
- まずは単体検索とルール説明を優先する

### 8.5 アーキテクチャ上の制約

以下は禁止または避ける。

- `proteus_core` 内で `ancient_greek` に直接分岐する
- Greek converter に他言語処理を追記していく
- APIで `attic`, `koine` などを固定 enum として増殖させる
- ルールをPythonコード内にハードコードする
- 言語追加のたびに search / distance / explainer を修正する
- UI表示文言に事業コンセプトを固定し、フレームワーク性を失わせる

### 8.6 初期実装上の制約

Phase 0〜1では以下を優先する。

1. 現行古代ギリシャ語検索を壊さない
2. `LanguageProfile` を導入する
3. `ancient_greek` を plugin 化する
4. `toy_language` を追加し、言語非依存性をテストする
5. APIに `language` フィールドを追加する
6. READMEを「framework + Ancient Greek pilot」として書き換える
7. MCP PoCに進む前に core / plugin 境界を固定する

---

## 9. 受け入れ基準

### 9.1 Phase 0 リファクタ完了基準

- `LanguageProfile` が実装されている
- `ancient_greek` が言語プラグインとして登録されている
- `/search` が `language` パラメータを受け取る
- `language` 未指定時は `ancient_greek` として動作する
- 既存の古代ギリシャ語テストが通る
- `toy_language` が core 変更なしで動作する
- `proteus_core` に Greek 固有分岐がない
- READMEに「language-independent framework」と明記されている

### 9.2 Phase 1 完了基準

- 古代ギリシャ語ルールセット v0.1 が YAML / JSON で管理されている
- 検索結果に適用ルールIDが表示される
- 候補ランキングが音韻距離行列に基づいている
- APIドキュメントが自動生成される
- ルールセットスキーマ検証が CI に入っている
- 最低5件以上の代表テストケースが通る

### 9.3 Phase 2 完了基準

- MCP Server PoC が動作する
- LLMクライアントから検索結果を取得できる
- 検索結果に engine_version / ruleset_version / verification_url が含まれる
- アノテーション Level 0〜4 の設計が実装または仕様化されている
- Scaife / Perseus 統合に向けたAPI仕様が整理されている

---

## 10. 今回コミットに含めるべき変更

### 10.1 ドキュメント

- `docs/REQUIREMENTS.md`
- `docs/ARCHITECTURE.md`
- `docs/DECISIONS.md`
- `docs/ROADMAP.md`
- `README.md` の位置づけ修正

### 10.2 コード

- `LanguageProfile` 導入
- `language registry` 導入
- `ancient_greek` plugin 化
- `toy_language` fixture 追加
- API `language` パラメータ追加
- 既存 Greek converter の責務分離

### 10.3 テスト

- 既存 Ancient Greek テスト
- LanguageProfile registry test
- toy_language integration test
- API backward compatibility test
- core が Greek に依存していないことを確認するテスト

---

## 11. 設計判断メモ

### 11.1 なぜゼロから作り直さないか

現行実装には、検索・距離計算・説明生成・API・テストの基礎が存在するため、ゼロから作り直すよりも、言語依存部分をプラグイン層へ移動するリファクタリングの方が合理的である。

### 11.2 なぜ古代ギリシャ語を本体にしないか

修正版事業計画書上の中核価値は、古代ギリシャ語専用ツールではなく、歴史音韻変化の言語非依存フレームワークであるため。

### 11.3 なぜ toy_language が必要か

実在言語を追加すると言語学的複雑性に引きずられるため、最小限の架空言語で「core を変更せずに新言語を追加できるか」を検証する。

### 11.4 なぜ MCP を重視するか

本システムは、LLMと競合するのではなく、LLMが参照する権威的な音韻ルール・候補検索・根拠提示基盤になることを目指すため。

---

## 12. 一文要約

Proteus / HPSI は、古代ギリシャ語専用検索アプリではなく、古代ギリシャ語を初期プラグインとして搭載した、言語非依存の歴史音韻変異検索・説明・グラウンディング基盤である。

