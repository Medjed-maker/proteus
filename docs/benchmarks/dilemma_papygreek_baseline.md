# Dilemma 実測ベースライン (PapyGreek Treebanks)

**測定日**: 2026-08-10
**対象**: `dilemma-nlp` 1.2.0 (`open-greek/dilemma`, MIT)
**目的**: Proteus の UVP —「崩れた綴り・方言形・音韻変異形を候補見出し語へ根拠付きで逆引きする」— が
既存の最有力ツールに対して実測でどれだけ空いているかを確定する。

> **測定世代**: 本文の件数と率は 2026-08-10 の修正前データ
> （`variant_ortho` 2,160件）を保存した歴史的測定である。2026-08-13 の
> Leiden マークアップ修正後、検索可能な対象は2,143件になった。修正後の
> 比較値と結論への影響は `preregistration_3rd.md` の「対応済み」を参照。

事前の文献調査では「Layer 1 は既製部品、空いているのは Layer 2」と結論していたが、
それはあくまで推定だった。本ドキュメントはその推定を実データで検証した記録である。

---

## 1. 結論

| 主張 | 実測 |
|---|---|
| Dilemma 公称 99.7%（古典期）は文書パピルスを説明しない | 標準綴りのパピルスでも strict **84.6%** / lenient **90.4%** |
| 非標準綴りからの逆引きが UVP の空白である | 同一語彙で **31.2% 対 83.3%（52.1ポイント差）** |
| Dilemma の正規化機能はこの空白を埋めていない | `normalize=True` の寄与は **+3.6ポイント**、`dialect=koine` / `convention=lsj` は **±0** |
| Dilemma は候補集合を返す逆引きシステムではない | 変異層の平均候補数 **1.13**、gold が候補内に存在する率 **34.1%**（top-1 31.0% とほぼ同値） |
| 説明可能性は Proteus 要件を満たさない | `via` は `normalize` / `stripped` / `mono` 等の機構名のみ。ルール名・方言帰属・出典は返らない |

**最重要**: 変異層で top-1 を外した場合、正解は候補集合のどこにも現れない（31.0% → 34.1%）。
下流にどんな再ランキングを足しても **34.1% が天井**になる。
Proteus の逆引きは「Dilemma の後段」ではなく「Dilemma が持たない前段」として成立する。

---

## 2. 評価データ

[PapyGreek Treebanks v1.01](https://doi.org/10.5281/zenodo.5074307)
(Vierros & Henriksson, ヘルシンキ大学, CC BY-SA 4.0)。
文書パピルス 395点 / 3,102文 / 44,036トークンの人手係り受け・形態論アノテーション。

このデータを選んだ理由は、各語が **3層** を同時に持つことにある。

| 属性 | 内容 |
|---|---|
| `form_orig` | パピルスに実際に書かれている綴り |
| `form_reg` | 校訂者による正規化形 |
| `lemma_reg` | 正規化形に対する正解見出し語 |

つまり「崩れた綴り → 研究者が必要とする見出し語」の正解ペアが人手で付与されている。
事前調査の段階で「ベンチマークとルールセット検証データの供給源」として
想定していたデータ源そのものである。

### 前処理

Leiden/papyri.info の校訂記号（不確実文字の下点 U+0323、正規化括弧 `∼∽`、
略記展開 `❨❩`、行分割 `∤`、補入 `⸢⸣` 等）と、レンマ内の `|num:11|` 形式の注記を除去。
記号除去後に層別化した。

| 層 | n | 定義 |
|---|---:|---|
| `clean` | 30,131 | 記号除去後 `form_orig == form_reg`。パピルスの綴りがそのまま標準形 |
| `variant_ortho` | **2,160** | 綴りが異なり、かつ `lemma_orig == lemma_reg`。**Proteus の UVP 領域** |
| `variant_lex` | 145 | 綴りもレンマも異なる。校訂者による語自体の訂正、または crasis のトークン分割 |
| `abbrev` | 1,179 | 略記の展開（`❨στρα(τηγῷ)❩`）。音韻現象ではないため分離 |
| 合計 | 33,615 | 句読点・非ギリシャ語・欠損 `[...]` を除外 |

`variant_lex` は `τἆλλα → τὰ` のようなトークン境界の不一致を含むため、
主要指標からは分離した（数値は参考値）。

### 採点

- **strict**: NFC 完全一致
- **lenient**: 同綴り番号（`ἄν1`→`ἄν`）、語末シグマ、アクセント・気息記号を無視した一致。
  Dilemma README の "equivalence-adjusted scoring" に相当する扱い。
- **echo**: 予測が入力そのもの。呼び出し側からは正解の同一形レンマ化と区別できないため別計上。
- **abstain**: `None` を返した（`guess=False` 時のみ発生）。

---

## 3. 精度（strict / lenient, %）

`Dilemma(lang="grc", resolve_articles=True, ...)`。
`resolve_articles=True` は AGDT 系 treebank 評価時に README が推奨する設定。

| 設定 | clean (30,131) | **variant_ortho (2,160)** | abbrev (1,179) | variant_lex (145) |
|---|---|---|---|---|
| A. デフォルト・原綴り | 84.6 / 90.4 | **27.6 / 30.2** | 87.8 / 92.1 | 1.4 / 2.8 |
| C. `normalize=True, period="hellenistic"`・原綴り | 84.5 / 90.3 | **31.2 / 34.0** | 87.6 / 91.9 | 2.1 / 3.4 |
| F. C + `convention="lsj"` | 84.5 / 90.3 | **31.2 / 34.0** | 87.6 / 91.9 | 2.1 / 3.4 |
| **B. デフォルト・校訂済み綴りを入力（上限値）** | 84.6 / 90.4 | **83.3 / 86.6** | 89.3 / 93.6 | 70.3 / 81.4 |

*`dialect="koine"` は予備測定（2,000トークン）で C と有意差なしのため本測定から除外。*

### 3.1 綴りだけで 52.1 ポイント

A/C と B の差は入力が原綴りか校訂済み綴りかだけである。語彙も設定も同一。
**31.2% → 83.3%**。落ちている原因は辞書の不足ではなく、
**非標準綴りから見出し語への逆引き経路が無いこと**である。

### 3.2 正規化機能の実効値

| 機能 | variant_ortho への寄与 |
|---|---|
| `normalize=True` (+ `period`) | **+3.6 pt** (27.6 → 31.2) |
| `dialect="koine"` | ±0 |
| `convention="lsj"` | ±0（clean 層でも ±0） |

Dilemma README が itacism・iota subscript・spirantization・geminate simplification に
対応すると記載する機能群の、文書パピルス上での実効値は +3.6 ポイントである。

### 3.3 失敗が沈黙しない

| 設定 | 層 | echo | abstain |
|---|---|---|---|
| C (`guess=True`) | variant_ortho | **35.8%** | 0% |
| E (`guess=False`) | variant_ortho | 8.2% | **27.8%** |
| C (`guess=True`) | clean | 3.9% | 0% |
| E (`guess=False`) | clean | 1.6% | 2.4% |

デフォルトでは変異層の失敗の過半（35.8/68.8 ≈ 52%）が入力のオウム返しとして返る。
`None` ではなく文字列が返るため、**呼び出し側は失敗を検出できない**。
`guess=False` にして初めて棄権率 27.8% として可視化される。
辞書引き用途では `guess=False` が実質的に必須である。

---

## 4. 候補再現率と説明可能性

`lemmatize_verbose()` による測定（設定 C 相当）。

| 層 | n | top-1 | **gold が候補内に存在** | 平均候補数 |
|---|---:|---|---|---|
| clean | 30,131 | 84.3% | 90.3% | 1.25 |
| **variant_ortho** | 2,160 | **31.0%** | **34.1%** | **1.13** |
| abbrev | 1,179 | 87.6% | 92.0% | 1.13 |
| variant_lex | 145 | 2.1% | 3.4% | 1.18 |

**再ランキングの余地が無い。** top-1 と候補内存在率の差は変異層でわずか 3.1 ポイント。
平均候補数 1.13 が示すとおり、Dilemma は候補集合を返さず単一解を返す。

### 候補の由来（`source|via`, variant_ortho 層）

| `source\|via` | 割合 |
|---|---|
| `lookup\|exact` | 31.4% |
| `identity\|` (＝解無し) | **28.4%** |
| `normalize\|normalize` | 11.2% |
| `lookup\|mono` | 6.8% |
| `lookup\|stripped` | 5.4% |
| `lookup\|exact+case_alt` | 4.8% |
| `article\|` | 3.0% |
| `byzantine_norm\|` | 1.7% |

`via` が返すのは `normalize` / `stripped` / `mono` といった内部機構名である。
`docs/API.md` の `SearchHit` が定義する `rules_applied`（適用音韻ルール）、
`dialect_attribution`（方言帰属）、`source_references`（出典）に相当する情報は返らない。
**説明の一級市民化は行われていない。**

---

## 5. 未達成の音韻現象（variant_ortho 層 2,160件の内訳）

設定 C での失敗率。Proteus の音韻ルールセットが埋めるべき対象の優先順位に対応する。

| 交替 | トークン数 | 失敗 | 失敗率 | 例 |
|---|---:|---:|---:|---|
| ε 挿入（ει/ι イオタ化） | 261 | 187 | 72% | `ἰσώρακεν → εἰσώρακεν` (εἰσοράω) |
| 重子音 | 271 | 159 | 59% | `ἔρροσσο → ἔρρωσο` (ῥώννυμι) |
| ε 削除 | 186 | 127 | 68% | `ἡμεῖν → ἡμῖν` (ἐγώ) |
| その他の置換 | 213 | 97 | 46% | — |
| ε/αι | 105 | 66 | 63% | `πεδίων → παιδίων` (παιδίον) |
| ο/ω 長短 | 145 | 62 | 43% | `ὀψόνια → ὀψώνια` (ὀψώνιον) |
| **鼻音同化** | 71 | 57 | **80%** | `συνπροσγενόμενος → συμ-` (σύν-προσγίγνομαι) |
| ι 削除 | 78 | 42 | 54% | `χαίριν → χάριν` (χάρις) |
| **itacism η/ι/ει** | 41 | 32 | **78%** | `ἠ → εἰ` (εἰ) |
| **itacism οι/υ/ι** | 36 | 30 | **83%** | `κοίριος → κύριος` (κύριος) |
| ι 挿入 | 61 | 28 | 46% | `ὑγαίνῃς → ὑγιαίνῃς` (ὑγιαίνω) |
| 有声化 κ/γ π/β τ/δ | 36 | 23 | 64% | `ἐνέκκαι → ἐνέγκαι` (φέρω) |
| **複数交替の重畳** | 25 | 23 | **92%** | `καλιόοτερεν → καλλιότερον` (καλός) |
| **σ 挿入** | 13 | 13 | **100%** | `εὐχαριτῶμεν → εὐχαριστῶμεν` (εὐχαριστέω) |
| **σ 削除** | 13 | 13 | **100%** | `πράξσω → πράξω` (πράσσω) |
| υ 挿入 | 22 | 18 | 82% | `σεατοῦ → σεαυτοῦ` (σαυτοῦ) |

いずれも標準的な後期ギリシャ語音韻論の記述対象であり、
`docs/phonology_rules.md` の適用範囲に収まる。
特に **鼻音同化（80%）・itacism οι/υ（83%）・複数交替の重畳（92%）** は失敗率が高く、
かつ規則性が明瞭であるため、ルールベース逆引きの費用対効果が最も高い。

---

## 6. 性能

33,615トークンのバッチ処理時間（CPU, macOS x86_64, onnxruntime 1.19.2）。

| 設定 | 時間 | スループット |
|---|---|---|
| A. デフォルト | 1,513s | 22 tok/s |
| C. `normalize=True` | 933s | 36 tok/s |
| **E. `guess=False`** | **13s** | **2,585 tok/s** |

`guess=False` は transformer フォールバックを飛ばすため **72倍速い**。
実行時間のほぼ全量が、変異層で 34.1% しか当たらない transformer 推論に費やされている。

初期化コスト: `lookup.db` 774MB + `spell_index.db` 414MB。
データ総量 3.7GB（`python -m dilemma download`）。

---

## 7. 測定の限界

1. **正解データのノイズ**: PapyGreek は AGDT/Morpheus のレンマ規約に従うため、
   Dilemma（Wiktionary 見出し語規約）との差が全層で誤りとして計上される。
   実例: gold `σου`（`σύ` ではない）、gold `ὅλοξ`（誤記）、gold `μῂ`、
   gold `Ταησις` に対し Dilemma `Ταήσεις`。
   `convention="lsj"` を試したが差は 0 だった。
   このノイズは **clean 層と variant 層に等しくかかる**ため、両者の差（52.1pt）は頑健である。
   絶対値としての clean 84.6/90.4% は、Dilemma 自身が報告する
   Gorman treebank 94.0%（equivalence-adjusted）と同族の水準と解釈すべきで、
   本測定は Dilemma を過小評価している可能性がある。

2. **訓練データ混入の可能性**: Dilemma は `glaux_pairs.json` を同梱する。
   GLAUx (Keersmaekers) は文書パピルスを含み、
   PapyGreek の形態論事前アノテーションも Keersmaekers 提供である（README 記載）。
   したがって clean 層の数値は楽観側に振れている可能性があり、
   **variant 層との差は保守的な見積もり**である。

3. **文脈非依存**: `lemmatize_batch()` は語単独で評価しており、
   Dilemma が持つ POS 指定 API（`lemmatize_batch_pos`）や前語による曖昧性解消は使っていない。
   README は gold POS 併用で Byzantine が 89.2% → 91.8% に改善すると報告しており、
   文脈を与えれば clean 層は数ポイント改善しうる。ただし変異層の失敗の 52% は
   echo（候補生成そのものの失敗）であり、POS では解けない。

4. **単一コーパス**: 文書パピルス（BCE 300 – CE 700, エジプト）のみ。
   碑文・写本の崩れ、および非ギリシャ語への一般化は未測定。

---

## 8. Proteus への含意

1. **UVP は実測で裏付けられた。** 「非標準綴りからの逆引き」という空白は
   最有力の既存ツールで 31.2%、候補集合を含めても 34.1% しか埋まっていない。

2. **勝負の土俵は精度ではなく候補生成である。** Dilemma の変異層 top-1 を上回るだけでは弱い。
   平均候補数 1.13 に対し、Proteus が **根拠付きの候補集合**を返せることが差分である。
   Proteus の受入基準は「top-1 精度」ではなく「gold が top-N に入る率（recall@N）」で
   設定すべきである（Dilemma の実効天井 = 34.1%）。

3. **失敗の可視化そのものが差別化になる。** Dilemma は変異層の失敗の 52% を
   入力のオウム返しとして返す。Proteus が「解無し」と「同一形レンマ」を
   区別して返すだけで、辞書引き用途では実用上の差が出る。

4. **ルール実装の優先順位**（§5 より）: 鼻音同化 → itacism（οι/υ, η/ι/ει）→
   重子音 → ε/αι → ο/ω 長短。σ 挿入・削除は件数は少ないが失敗率 100%。

5. **Dilemma は競合ではなく補完たりうる。** clean 層 90.4% は十分に強い。
   `guess=False` で棄権させた 27.8% を Proteus の逆引きに回す構成は、
   事前調査を踏まえた差別化の再定義と整合する。
   ライセンスは MIT。

---

## 9. 再現手順

```bash
python -m venv .venv && .venv/bin/pip install \
  "dilemma-nlp[onnx] @ git+https://github.com/open-greek/dilemma.git@f82f15a62ddce5d55c19b299c34a6c89476af5ce"
.venv/bin/python -m dilemma download      # 3.7GB, 約90分

curl -sL "https://zenodo.org/api/records/5074307/files/ezhenrik/papygreek-treebanks-v1.01.zip/content" \
  -o papygreek.zip && unzip -q papygreek.zip -d papygreek

.venv/bin/python build_dataset.py         # 33,615トークンへ層別化
.venv/bin/python run_eval.py              # 精度（5設定, 約75分）
.venv/bin/python run_candidates.py        # 候補再現率・由来
.venv/bin/python categorize.py            # 音韻カテゴリ別失敗率
```

評価スクリプト（`build_dataset.py`, `clean.py`, `run_eval.py`, `run_candidates.py`,
`categorize.py`）は本リポジトリ未収録。再利用する場合は `tools/benchmarks/dilemma/` へ
移設したうえで、PapyGreek の CC BY-SA 4.0 帰属表示を `DATA_LICENSE.md` に追加すること。

## 参照

- Dilemma: <https://github.com/open-greek/dilemma> (MIT)
- PapyGreek Treebanks v1.01: <https://doi.org/10.5281/zenodo.5074307> (CC BY-SA 4.0)
- Vierros & Henriksson, *PapyGreek Treebanks: A Dataset of Linguistically Annotated
  Greek Documentary Papyri*, Journal of Open Humanities Data,
  <https://doi.org/10.5334/johd.55>
