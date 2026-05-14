# Orthographic Notes Citation-Ready Plan

作成日: 2026-05-05  
対象: Ancient Greek orthographic note data  
目的: provisional な表記体系コメントを、出典確認済み・専門家レビュー可能・将来的に citation-ready な runtime data に引き上げる。

## 進捗凡例

- [ ] 未着手
- [x] 完了
- [~] 実装中または要再検討

## この plan の使い方

この文書は、citation-ready 化の設計メモであると同時に、issue / PR checklist として使う。実装順は topic 番号ではなく、下部の `Initial implementation order` を正とする。

PR でこの checklist を使う場合は、各項目を次の基準で更新する。

- [x] 完了: 方針、対象ファイル、判断結果、または実装結果がこの文書か code / data / test に記録され、次の実装者が追加判断なしで作業できる。
- [~] 実装中または要再検討: 調査済みだが source evidence、専門家判断、または実装方針がまだ固定できない。
- [ ] 未着手: 調査、設計、実装、検証のいずれもまだ作業単位として完了していない。

今回の実装範囲は `Initial implementation order` の 1-4 のみ。runtime YAML、API/UI wire shape、loader validation、tests、`pre_403_2_attic` tag の維持/削除判断、`orthography_hint="pre_403_2_attic"` fallback 削除は 5 以降に残す。

## 目的と成功条件

- [ ] `data/languages/ancient_greek/orthography/orthographic_correspondences.yaml` の各 seed entry について、根拠、レビュー状態、表示可否を追跡できるようにする。
- [ ] `παιδίο -> παιδίου` を最初の review pilot として扱い、entry 単位の確認手順を確立する。
- [ ] `pre_403_2_attic` note を出す根拠を、一般的な印象ではなく Attic inscription 専用ソースに寄せる。
- [ ] papyri.info / EpiDoc choice 由来データと Attic inscription 由来データを混同しない。
- [ ] runtime YAML に入れる entry は、未レビューでも provisional であることが docs から誤解されない状態を保つ。
  - [ ] 初回 PR では API/UI の wire shape は変えないため、API consumer から entry-level `citation_ready` は見えない。
  - [ ] API/UI に provisional / citation-ready 状態を出すかどうかは Gate D で別途決める。
  - [ ] それまでは docs で `orthographic_notes` が citation-ready とは限らないことを明記する。
- [ ] `citation_ready: true` に上げる条件を明文化し、実装者が判断を追加で作らなくてよい状態にする。

## Phase 0: 現状固定と範囲確認

- [x] 現在の runtime orthography data を確認する。
  - [x] 対象ファイル: `data/languages/ancient_greek/orthography/orthographic_correspondences.yaml`
  - [x] `_meta.status == provisional` を確認する。
  - [x] `_meta.review_status == not_expert_reviewed` を確認する。
  - [x] `_meta.citation_ready == false` を確認する。
  - [x] seed entry が `παιδίο -> παιδίου` のみか確認する。
- [x] 現在の builder 挙動を確認する。
  - [x] `orthographic_correspondence` note を返す条件を確認する。
  - [x] `beginner_aid` note を返す条件を確認する。
  - [x] `pre_403_2_attic` note を返す条件を確認する。
  - [x] `orthography_hint="pre_403_2_attic"` のみで historical note が出ることを確認する。
- [x] 今回の citation-ready 化で変更しない範囲を明記する。
  - [x] ランキングスコア・候補順序は変更しない。
  - [x] 音韻距離行列は再生成しない。
  - [x] EpiDoc choice データを runtime YAML に自動投入しない。
  - [x] PHI / AIO から自動抽出する scraper は初期スコープに入れない。
  - [x] UI に references を表示するかどうかは、データ設計完了後の別判断にする。

Phase 0 note: current builder behavior is documented here as the starting point only. Removing the `orthography_hint="pre_403_2_attic"` fallback remains an Initial implementation order 9 task.

## Phase 1: Citation-ready 判定基準の定義

- [x] entry 単位の `citation_ready` 判定基準を決める。
  - [x] `false`: 根拠未確認、または broad reference のみ。
  - [x] `true`: source evidence と専門家レビューの両方が揃っている。
- [x] entry 単位の `review_status` 値を決める。
  - [x] `not_expert_reviewed`
  - [x] `source_located`
  - [x] `needs_expert_review`
  - [x] `expert_reviewed`
  - [x] `rejected`
  - [x] `rejected` は runtime YAML では配信対象にしない。初回 PR では committed runtime YAML 内の `rejected` entry を loader error にする。
  - [x] reject された候補を記録したい場合は、runtime YAML ではなく docs / issue / review log に移す。
- [x] `confidence` と `review_status` の役割分担を明文化する。
  - [x] `confidence` は note 内容の確からしさを示す。
  - [x] `review_status` は人間レビュー工程の進捗を示す。
  - [x] `citation_ready` は UI/API で citation として提示可能かを示す。
  - [x] 初回 PR では `citation_ready` は runtime YAML 内の判定値であり、API/UI にはまだ露出しない。
- [x] `pre_403_2_attic` tag を付ける条件を明文化する。
  - [x] 明示的に pre-403/2 BCE Attic inscription として同定できる source がある。
  - [x] または専門家が pre-403/2 Attic 表記として扱えると判断した。
  - [x] 単なる一般的な表記ゆれ、papyri.info 由来の choice、後代 Attic inscription だけでは付けない。
  - [x] `secondary_literature` 単独で tag を付ける場合は、その文献内で specific inscription への明示的 citation (例: 該当ページから IG I^3 番号を引いている) がある場合のみ許可する。
  - [x] Buck section 4 のような broad な dialect orthography 概説の単独参照では tag を付けない。
  - [x] source が未確認の entry には committed runtime YAML で `pre_403_2_attic` tag を付けない。既存 seed で source が見つからない場合は tag を外し、`review_status: needs_expert_review` にする。
- [x] `beginner_aid` tag を付ける条件を明文化する。
  - [x] 初学者が辞書形・正規化形に戻す補助として有用である。
  - [x] 形態論的説明と表記体系説明を混同しない。
  - [x] 断定文ではなく `may correspond to` / `対応する可能性があります` の温度を維持する。

## Phase 2: Evidence schema の設計

- [x] YAML entry に追加する review metadata の最小形を決める。
  - [x] `review_status`
  - [x] `citation_ready`
  - [x] `source_type` (list)
  - [x] `source_ids`
  - [x] `references`
  - [x] `reference_urls` (optional)
  - [x] `review_notes`
  - [x] `reviewed_by`
  - [x] `reviewed_at`
- [x] committed runtime YAML の必須 / optional を固定する。
  - [x] Direct presence required: `review_status`, `citation_ready`, `source_type`, `source_ids`, `references`。
  - [x] Optional: `reference_urls`, `review_notes`, `reviewed_by`, `reviewed_at`。
  - [x] `review_status == not_expert_reviewed` または `review_status == needs_expert_review` の entry では、`source_type`, `source_ids`, `references` は空 list を許容する。
  - [x] `review_status == source_located` または `review_status == expert_reviewed` では、`source_type`, `source_ids`, `references` を非空にする。
  - [x] `review_status == expert_reviewed` では、`reviewed_by` と `reviewed_at` を必須にする。
  - [x] `citation_ready: true` では、`review_status == expert_reviewed`, `source_type` 非空, `source_ids` 非空, `references` 非空, `reviewed_by` 非空, `reviewed_at` ISO date を必須にする。
- [x] `reviewed_by` / `reviewed_at` は初期 schema に含めるが、`citation_ready: true` または `review_status: expert_reviewed` になるまでは空・未設定を許容する。
- [ ] `reviewed_by` の表記ガイドラインを決める。
  - [ ] 公開 repo にコミットされる前提で、ASCII handle または initials を推奨する。
  - [ ] フルネームを書く場合は本人合意済みであることを前提にする。
  - [ ] reviewer 識別子と実名の対応表が必要なら、entry 単位ではなく非公開の対応表で管理する選択肢も検討する。
- [x] `source_type` の表現を `list[str]` で統一する。
  - [x] 許可値: `aio` / `phi` / `ig` / `secondary_literature` / `expert_note`。
  - [x] 単一値時も list 形式で書く (例: `["aio"]`)。
  - [x] 複数 source の重ね合わせは list で表現する (例: `["aio", "expert_note"]`)。
  - [x] `mixed` は撤廃する (list 表現で代替可能)。
- [ ] `expert_note` の使い方を定義する。
  - [ ] `expert_note` は、単一の primary source がない named expert の判断、または専門的・慣用的知識に基づく判断に使う。
  - [ ] AIO / PHI / IG / secondary literature の根拠がある場合は、それぞれ `aio` / `phi` / `ig` / `secondary_literature` を併記する。
  - [ ] 専門家解釈と一次資料を併用する場合は list で両方記録する (例: `["aio", "expert_note"]`)。
- [x] `source_ids` の表現を決める (Phase 7 と同期)。
  - [x] AIO: AIO record identifier を保存する。URL は `reference_urls` に分離する。
  - [x] PHI: PHI region/publication/text reference を保存する。URL は `reference_urls` に分離する。
  - [x] IG: `IG I^3 123` のような人間可読 citation を保存する。
  - [x] secondary literature: 著者、書名、節、ページを保存する。
  - [x] link rot 耐性のため、`source_ids` は canonical identifier のみで再解決可能であること。
- [x] `references` の用途を決める (Phase 7 と同期)。
  - [x] API で返せる短い citation 文字列を入れる。
  - [x] 生データ再配布が危ない場合は、本文全文ではなく identifier と citation のみにする。
  - [x] URL は `references` に混ぜず、`reference_urls` に optional フィールドとして持つ。
- [x] `review_notes` の用途を決める。
  - [x] editor 向けの非 UI 注記に限定する。
  - [x] 表示文言そのものは `messages` builder 側で生成する。
  - [x] source の Greek text を長く引用しない。
- [ ] schema 変更時の後方互換性を決める。
  - [ ] 暫定決定: committed runtime YAML では entry-level の `review_status` と `citation_ready` を必須とする。
  - [ ] `_meta` はファイル全体の表示用 status のみに使い、entry-level metadata の継承元としては使わない。
  - [ ] entry 単位の direct presence validation を loader に実装する (post-inheritance ではない)。
  - [ ] 開発中・未 commit の fixture YAML には metadata 不足を許容するが、test fixture と runtime YAML は同じ validation を通す。
  - [ ] API `OrthographicNote` の wire shape は初期段階では変えない。
  - [ ] `_meta.status` は API response には出ないため、API consumer が provisional 状態を判断する根拠としては使わない。

## Phase 3: Source reconnaissance

- [x] AIO の使い方を整理する。
  - [x] pre-403/2 BCE Attic の候補は IG I 系を優先的に見る。
  - [x] AIO が提供する translation / commentary / dating を確認する。
  - [x] AIO record から IG 番号、年代、本文確認先を記録する。
  - [x] AIO を runtime data の直接全文ソースとして再配布しない。
- [x] PHI Greek Inscriptions の使い方を整理する。
  - [x] PHI は searchable Greek inscriptions の照合先として使う。
  - [x] PHI で inscription text と publication reference を確認する。
  - [x] PHI の再配布条件に注意し、本文全文を YAML に取り込まない。
  - [x] PHI record identifier / publication reference を evidence として記録する。
- [x] IG citation の扱いを整理する。
  - [x] `IG I` は 403/2 BCE 以前の Attic inscription 候補として扱う。
  - [x] `IG II` は 403/2 BCE 以後の Attic inscription として、pre_403_2_attic の直接根拠にはしない。
  - [x] 年代境界が 405/4 and 403/2 BC のようにまたがる場合は `needs_expert_review` にする。
- [x] secondary literature の扱いを整理する。
  - [x] Buck section 4 は broad reference として保持する。
  - [x] 個別 seed entry の直接根拠にする場合は、該当節・ページまで確認する。
  - [x] Attic orthography の一般説明と個別 inscription evidence を分ける。
- [ ] Digital Greek / AtticGreek 系資料の扱いを整理する。
  - [ ] 403 BCE 以前の Athenian public documents の orthography 概説として使う。
  - [ ] 個別 entry の直接根拠ではなく、system-level background reference として扱う。

### Source reconnaissance notes for initial order 3

- AIO は Attic inscriptions の translation / commentary / dating を確認する入口として使う。AIO の説明では、Greek text は AIO 内または外部 open-access site へのリンクとして提供されるため、runtime YAML には AIO 本文ではなく record identifier、IG 番号、短い citation、必要なら `reference_urls` を記録する。
- PHI Greek Inscriptions は inscription text と publication reference の照合先として使う。PHI の利用条件に合わせ、Greek text の全文や大きな excerpt は runtime YAML に保存しない。保存対象は PHI text id / publication reference / short citation / URL に限定する。
- IG は source identity の canonical citation として扱う。`pre_403_2_attic` の直接根拠は原則 `IG I` / `IG I^3` 系に寄せ、`IG II` / `IG II^2` / `IG II^3` 系は 403/2 BCE 以後として扱う。
- Buck, Introduction to the Study of the Greek Dialects, section 4 は broad background reference として保持する。個別 entry の direct evidence に使う場合は、該当 section / page が specific inscription または該当 spelling を明示していることを再確認する。
- Initial reconnaissance conclusion: `παιδίο -> παιδίου` を `citation_ready: true` に上げる直接 source id は、この初回調査では確定しない。pilot entry は `needs_more_evidence` として扱い、expert review と source id 確定後まで `citation_ready: false` を維持する。

## Phase 4: `παιδίο -> παιδίου` review pilot

- [x] review packet を作る。
  - [x] `original: παιδίο`
  - [x] `normalized: παιδίου`
  - [x] `candidate_headwords: [παιδίον]`
  - [x] `romanization: paidiou`
  - [x] 現在の `kind`
  - [x] 現在の `tags`
  - [x] 現在の `confidence`
  - [x] 現在の `references`
- [x] 形態・語形関係を確認する。
  - [x] `παιδίον` が candidate headword であることを確認する。
  - [x] `παιδίου` が normalized/read-as form であり、headword ではないことを確認する。
  - [x] API/UI 文言が headword と normalized form を混同していないことを確認する。
- [ ] orthographic correspondence として妥当か確認する。
  - [ ] `παιδίο` が inscriptional spelling として現れる evidence を探す。
  - [ ] `παιδίου` に対応する根拠を探す。
  - [ ] 対応が単なる final nu deletion や形態音韻ルールだけでは説明できないか確認する。
  - [ ] 表記体系コメントとして出す価値があるか確認する。
- [ ] `pre_403_2_attic` tag の妥当性を確認する。
  - [ ] source が pre-403/2 BCE Attic inscription であるか確認する。
  - [ ] source が post-403/2 BCE または非 Attic なら tag を外す計画にする。
  - [ ] source が不明なら tag は維持せず `needs_expert_review` に落とす計画にする。
- [ ] `beginner_aid` tag の妥当性を確認する。
  - [ ] 初学者が `παιδίο` を `παιδίου` と読む補助として有用か確認する。
  - [ ] 表示文言が断定的でないことを確認する。
- [x] pilot の結論を記録する。
  - [ ] `accepted`
  - [ ] `accepted_with_lower_confidence`
  - [ ] `accepted_without_pre_403_2_attic_tag`
  - [x] `needs_more_evidence`
  - [ ] `rejected`

### Review packet: `παιδίο -> παιδίου`

Current runtime seed entry (before this PR):

```yaml
original: "παιδίο"
normalized: "παιδίου"
candidate_headwords: ["παιδίον"]
romanization: "paidiou"
kind: "orthographic_correspondence"
tags: ["beginner_aid", "inscriptional", "pre_403_2_attic"]
confidence: "medium"
references: []
```

Review interpretation:

- `παιδίον` is the candidate headword.
- `παιδίου` is the normalized/read-as form used by the orthographic note; it is not the candidate headword.
- Existing API/UI messages distinguish the query form from the normalized form by saying that `παιδίο` may correspond to normalized form `παιδίου (paidiou)`.
- `beginner_aid` is provisionally useful because it helps a learner read `παιδίο` as corresponding to `παιδίου`.
- `pre_403_2_attic` remains unconfirmed in this packet. If a specific pre-403/2 BCE Attic inscription source is not found in the later evidence pass, Initial implementation order 8 should remove this tag and keep the entry as `needs_expert_review`.

Pilot decision: `needs_more_evidence`. This packet is ready for issue / PR review, but it is not expert-reviewed and does not make the entry citation-ready.

## Phase 5: Review packet template の作成

- [x] `docs/orthographic_notes_review_template.md` を作るか決める。
- [x] template に entry metadata 欄を用意する。
  - [x] original
  - [x] normalized
  - [x] candidate_headwords
  - [x] romanization (`normalized` 形に対応するよう統一する。`original` の音表記ではない。builder の現挙動と一致させる。)
  - [x] kind
  - [x] tags
  - [x] confidence
- [x] template に evidence 欄を用意する。
  - [x] source_type (list 形式)
  - [x] source_ids (canonical identifier のみ。URL は混ぜない)
  - [x] reference_urls (optional)
  - [x] dates
  - [x] place
  - [x] dialect / region
  - [x] publication reference
- [x] template に reviewer decision 欄を用意する。
  - [x] keep entry
  - [x] change normalized form
  - [x] change candidate headword
  - [x] change tags
  - [x] change confidence
  - [x] reject entry
  - [x] needs another source
- [x] template に implementation action 欄を用意する。
  - [x] YAML metadata update required
  - [x] tests update required
  - [x] docs update required
  - [x] no runtime change required

Phase 5 result: `docs/orthographic_notes_review_template.md` now defines the
review packet template. It follows the Phase 6 runtime YAML review metadata
schema, keeps `source_ids` separate from `reference_urls`, records
`romanization` as the normalized-form romanization, and treats rejected entries
as non-runtime review-log material.

## Phase 6: YAML schema 実装計画

- [x] loader の現在の validation を確認する。
  - [x] `_meta` validation の範囲を確認する。
  - [x] entry validation の範囲を確認する。
  - [x] unknown keys を許容しているか確認する。
- [x] entry-level review metadata を追加する。
  - [x] `review_status`
  - [x] `citation_ready`
  - [x] `source_type` (list)
  - [x] `source_ids`
  - [x] `references`
  - [x] `reference_urls` (optional)
  - [x] `review_notes`
  - [x] `reviewed_by`
  - [x] `reviewed_at`
- [x] `reviewed_by` / `reviewed_at` は Phase 2 と同じく初期 schema に含め、expert review 前は空・未設定を許容する。
- [x] `_meta` と entry-level metadata の関係を決める (Phase 2 と同期)。
  - [x] `_meta` はファイル全体の表示用 status とし、entry-level metadata の継承元としては使わない。
  - [x] entry-level metadata (`review_status` / `citation_ready` / `source_type` / `source_ids` / `references`) は committed runtime YAML で direct presence required とする。
  - [x] validation は entry-level の direct presence check で行う (post-inheritance ではない)。
- [x] builder の `orthography_hint="pre_403_2_attic"` fallback パスを処理する。
  - [x] 現状 `src/phonology/languages/ancient_greek/orthography_notes.py` の `build_orthographic_notes` は YAML エビデンスを経由せず引数だけで historical note を生成する。
  - [x] このパスは Phase 1 の tag ルール (source 由来でのみ tag 付与) と矛盾する。
  - [x] 決定: 初回 PR でこの fallback パスを削除する。
  - [x] 削除後、`pre_403_2_attic` historical note は YAML 由来 entry のみが生成する。
  - [x] この変更は `OrthographicNote` の wire shape を変えないが、これまで provisional で出ていた orthography_hint 由来 note が出なくなる observability 上の変化を伴う。docs と CHANGELOG に明記する。
- [x] validation tests を追加する。
  - [x] valid review metadata を受け入れる。
  - [x] unsupported `review_status` を拒否する。
  - [x] unsupported `source_type` 値を拒否する。
  - [x] `source_type` が list でない (旧スキーマ) なら拒否する。
  - [x] entry-level の `review_status` / `citation_ready` が欠けていたら拒否する。
  - [x] `citation_ready: true` なのに `references` が空なら拒否する。
  - [x] `citation_ready: true` なのに `source_type` / `source_ids` が空なら拒否する。
  - [x] `citation_ready: true` なのに `review_status != expert_reviewed` なら拒否する。
  - [x] `citation_ready: true` なのに `reviewed_by` / `reviewed_at` が空なら拒否する。
  - [x] `reviewed_at` が ISO date 形式でなければ拒否する。
  - [x] `review_status == expert_reviewed` なのに `reviewed_by` / `reviewed_at` が空なら拒否する。
  - [x] `review_status == rejected` の entry は committed runtime YAML では拒否する。
  - [x] `pre_403_2_attic` tag があるのに `source_type` / `source_ids` / `references` が空なら test failure にする。
- [x] runtime payload への反映方針を決める。
  - [x] 初期段階では API `OrthographicNote.references` だけを維持する。
  - [x] `review_status` と `citation_ready` は API に出さない。
  - [x] UI 表示は変更しない。
  - [x] `citation_ready: false` の entry も既存どおり API note として配信する (filter しない)。
  - [x] API consumer は `_meta.status: provisional` を見られないため、初回 PR では docs で API note が citation-ready とは限らないことを明記する。
  - [x] citation_ready 化と filter 判断は別 PR で行う (Phase 12 Gate D)。

Minimal schema decision for Initial implementation order 4:

```yaml
review_status: not_expert_reviewed
citation_ready: false
source_type: []
source_ids: []
references: []
reference_urls: []  # optional
review_notes: ""    # optional
reviewed_by: ""     # optional until expert_reviewed / citation_ready
reviewed_at: ""     # optional until expert_reviewed / citation_ready; ISO date when set
```

Implemented validation policy: the five direct fields `review_status`, `citation_ready`, `source_type`, `source_ids`, and `references` are required on every committed runtime YAML entry. `_meta` does not supply defaults for entries. `citation_ready: true` requires `review_status: expert_reviewed`, non-empty source fields, non-empty reviewer fields, and ISO-date `reviewed_at`. Public API / UI wire shape does not change in this schema step.

## Phase 7: References 実運用

- [x] citation 文字列の形式を決める。
  - [x] `IG I^3 000`
  - [x] `AIO, IG I^3 000`
  - [x] `PHI Greek Inscriptions, [publication reference]`
  - [x] `Buck, section 4`
- [x] URL の扱いを決める。
  - [x] `source_ids` には canonical identifier (IG citation / AIO record id / PHI text id / 著者+書名+節) を保存する。
  - [x] URL は optional な `reference_urls` に分離する。
  - [x] link rot 時は canonical id 優先で再解決可能にする。
  - [x] `references` には人間可読な短い citation 文字列を入れ、URL は混ぜない。
  - [x] UI に出す可能性がある文字列は短く保つ。
- [x] source text の扱いを決める。
  - [x] YAML に長い Greek text を保存しない。
  - [x] `evidence_excerpt` フィールドは初期スコープでは導入しない (drop)。`references` + canonical id + 外部再確認で運用する。
  - [x] 将来的に excerpt が必要になった場合は、文字数上限・著作権・利用条件を含めた別 RFC で議論する。
  - [x] 原則として source ID と citation で再確認できる形にする。
- [x] reviewer note と public reference を分ける。
  - [x] `review_notes` は内部用。
  - [x] `references` は API/UI に出してもよい短い出典。

Phase 7 implementation result: runtime YAML validation now rejects URL-like
strings in `references` and `source_ids`; URLs must be stored in
`reference_urls`, and `reference_urls` accepts only `http` / `https` URLs. The
loader also rejects `evidence_excerpt` in runtime data so source text is not
introduced accidentally. Citation string shape remains deliberately light:
short, human-readable, URL-free labels such as `IG I^3 000`, `AIO, IG I^3 000`,
`PHI Greek Inscriptions, [publication reference]`, and `Buck, section 4`.

## Phase 8: Documentation 更新

- [x] `docs/ARCHITECTURE.md` を更新する。
  - [x] runtime orthographic note data と evidence metadata の関係を説明する。
  - [x] papyri.info training data と runtime note data を分ける理由を説明する。
- [x] `docs/REQUIREMENTS.md` を更新する。
  - [x] citation-ready note の要件を追加する。
  - [x] provisional note の表示上の制約を明記する。
- [x] `docs/CODEMAPS/api.md` を更新する。
  - [x] `OrthographicNote.references` の意味を明記する。
  - [x] `review_status` が API に出ない場合は、その理由を明記する。
- [x] `README.md` を更新する。
  - [x] current status は provisional のまま維持する。
  - [x] citation-ready 化は roadmap / data review task として説明する。
- [x] `docs/ROADMAP.md` を更新する。
  - [x] expert review workflow を短期 roadmap に追加する。
  - [x] source ingestion automation は中長期に残す。

Phase 8 implementation result: architecture, requirements, API codemap,
README, and roadmap docs now describe citation-ready orthographic notes as a
reviewed data state rather than a public API guarantee. The docs distinguish
runtime Attic inscription evidence from papyri.info / EpiDoc-derived
candidate-generation data, keep current packaged notes provisional, and explain
that `references` are short citation labels while review metadata and
`reference_urls` remain outside the current API shape.

## Phase 9: Expert review workflow

- [x] reviewer に渡す単位を決める。
  - [x] 初期 default は 1 entry ずつとする。
  - [x] 同じ source に属する小バッチは、同じ source evidence と同じ review 判断で処理できる場合のみ許可する。
  - [x] 同じ orthographic pattern に属する小バッチは、各 entry の source identity が明示され、pattern-level 判断だけで entry-level evidence を代替しない場合のみ許可する。
  - [x] `παιδίο -> παιδίου` pilot は単独 entry review として扱う。
- [x] review request に含める内容を決める。
  - [x] entry metadata
  - [x] source reference / source ids / reference URLs
  - [x] proposed note messages の EN / JA 文言
  - [x] proposed tags
  - [x] uncertainty / confidence
  - [x] reviewer decision options
  - [x] implementation action
  - [x] final review state
- [x] reviewer decision の反映ルールを決める。
  - [x] `expert_reviewed` に上げる条件: reviewer が keep / change の判断を返し、runtime に残す entry について必要な source metadata と reviewer metadata を repo に記録できる状態になっている。
  - [x] `citation_ready: true` に上げる条件: `review_status: expert_reviewed`, 非空の `source_type` / `source_ids` / `references`, 非空の `reviewed_by`, ISO date の `reviewed_at` が揃っている。
  - [x] `confidence` を上げる条件: reviewer が entry、normalized form、tags、source evidence を肯定し、source が entry-level evidence として十分であると判断した場合。
  - [x] `confidence` を下げる条件: reviewer が evidence を challenge した、source が後代 / 非 Attic と判明した、tag を外した、broad reference 単独に格下げされた、または normalized form / headword の判断に不確実性が残る場合。
  - [x] `pre_403_2_attic` tag を外す条件: pre-403/2 BCE Attic inscription source、または明示的な expert judgment が記録できない場合。
  - [x] entry を削除する条件: reviewer が reject した、source evidence が entry の表示内容を支えない、または normalized form / headword / tag の修正では安全に配信できない場合。
  - [x] `needs another source` の場合は `expert_reviewed` に上げず、`needs_expert_review` または `source_located` のまま追加 evidence を待つ。
  - [x] rejected entry は runtime YAML に残さず、review log / docs / issue / PR discussion にのみ記録する。
- [x] 多レビュア時の合意ルール。
  - [x] 初期は単一 reviewer 前提で進める。
  - [x] 複数 reviewer 間の合意 / 不一致解消ルールは初回 citation-ready PR では out of scope とし、別フェーズで設計する。
- [x] review log の保存場所を決める。
  - [x] YAML entry の `review_notes` は短い実装向け要約だけにする。
  - [x] docs 配下の review log は、公開可能な判断理由や entry-level decision の要約を残す場所として使える。
  - [x] issue / PR discussion は、実装判断、reviewer decision、follow-up task の履歴を残す場所として使える。
  - [x] 外部 reviewer comment は要約だけを repo に残し、個人情報・未公開コメント・実名は本人合意なしに公開 repo へ入れない。

Phase 9 implementation result: expert review starts with one entry per review
request. Small batches are allowed only when source identity and entry-level
evidence remain explicit. Review requests must include metadata, source
identifiers, proposed EN / JA note messages, proposed tags, confidence /
uncertainty, decision options, implementation actions, and final review state.
Promotion to `citation_ready: true` requires `expert_reviewed` plus complete
source and reviewer metadata. Rejections stay out of runtime YAML, and external
reviewer comments are summarized only when they are safe to record publicly.

## Phase 10: Expansion strategy

- [ ] 追加 seed 候補の優先順位を決める。
  - [ ] 授業・デモで頻出する forms。
  - [ ] pre-403/2 Attic orthography の特徴を示す forms。
  - [ ] normalized form への読み替えが初学者に有用な forms。
  - [ ] source evidence が明確な forms。
- [ ] source-first 追加ルールを決める。
  - [ ] 先に AIO / PHI / IG source を確認する。
  - [ ] その後に runtime YAML entry を追加する。
  - [ ] evidence なしの entry は追加しない。
- [ ] corpus automation の境界を決める。
  - [ ] 自動抽出は candidate generation まで。
  - [ ] runtime YAML への昇格は人手レビュー後。
  - [ ] papyri.info と Attic inscriptions の source lineage を分ける。

## Phase 11: Test plan

- [x] focused tests を追加または更新する。
  - [x] `tests/test_orthography_notes.py`
  - [x] `tests/test_packaging.py`
  - [x] `tests/test_api_main.py`
- [x] YAML schema tests を追加する。
  - [x] review metadata の正常系 (`source_type` を list で表現したケース、`["aio", "expert_note"]` のような併用ケースを含む)。
  - [x] invalid `review_status`。
  - [x] `review_status == rejected` が committed runtime YAML では拒否される。
  - [x] invalid `source_type` 値 (許可値外)。
  - [x] `source_type` が list ではなく単一文字列のとき拒否される (旧スキーマ拒否)。
  - [x] entry-level の `review_status` が欠けたら拒否される (`_meta` 継承に依存しない direct presence check)。
  - [x] entry-level の `citation_ready` が欠けたら拒否される。
  - [x] entry-level の `source_type` が欠けたら拒否される。
  - [x] entry-level の `source_ids` が欠けたら拒否される。
  - [x] entry-level の `references` が欠けたら拒否される。
  - [x] `citation_ready: true` without references。
  - [x] `citation_ready: true` without source identifiers or source types。
  - [x] `citation_ready: true` without `expert_reviewed`。
  - [x] `citation_ready: true` without `reviewed_by` / `reviewed_at`。
  - [x] invalid `reviewed_at` date format。
  - [x] `pre_403_2_attic` tag without source evidence is rejected.
- [x] builder regression tests を追加する。
  - [x] review metadata が note messages を変えない。
  - [x] API wire shape が変わらない。
  - [x] references は既存どおり `OrthographicNote.references` に入る。
  - [x] `orthography_hint="pre_403_2_attic"` だけが指定されて該当 YAML entry がない場合、historical note が生成されないことを検証する (fallback 削除の regression)。
- [ ] docs consistency tests を検討する。
  - [ ] README と YAML の provisional status が矛盾しない。
  - [ ] source_type 許可値が docs と loader で一致する。
- [x] verification command を決める。
  - [x] `rtk uv run pytest tests/test_orthography_notes.py`
  - [x] `rtk uv run pytest tests/test_packaging.py`
  - [x] `rtk uv run pytest tests/test_api_main.py`
  - [x] `rtk uv run pytest`
  - [x] `rtk uv build --wheel`

## Phase 12: Release gates

- [x] Gate A: review metadata schema ready (target: 短期 / 初回 citation-ready PR で達成)
  - [x] YAML schema が決まっている。
  - [x] loader validation がある。
  - [x] tests が通っている。
  - [x] builder の `orthography_hint="pre_403_2_attic"` fallback パスが削除されている。
  - [x] source evidence のない `pre_403_2_attic` tag が runtime YAML に残っていない。
  - [x] `rejected` entry が runtime YAML では loader error になる。
- [ ] Gate B: first reviewed seed ready (target: 短期〜中期 / pilot 完走時)
  - [ ] `παιδίο -> παιδίου` の source evidence が記録されている。
  - [ ] reviewer decision が記録されている。
  - [ ] tag / confidence / references が decision と一致している。
- [ ] Gate C: citation-ready promotion ready (target: 中期 / pilot を citation-ready 化するとき)
  - [ ] `review_status == expert_reviewed`
  - [ ] `citation_ready == true`
  - [ ] `references` が空でない。
  - [ ] `source_type` と `source_ids` が空でない。
  - [ ] `reviewed_by` と `reviewed_at` が記録されている。
  - [ ] source identifiers が再確認可能。
  - [ ] docs が provisional / citation-ready の違いを説明している。
- [ ] Gate D: UI/API publication ready (target: 中長期 / 別 PR)
  - [ ] references を UI に出すか決定済み。
  - [ ] `citation_ready: false` の entry を API から filter するか決定済み。
  - [ ] `review_status` / `citation_ready` / provisional status を API/UI に露出するか決定済み。
  - [ ] API schema 変更が必要なら migration note がある。
  - [ ] 既存クライアント互換性が確認済み。
- [ ] target release のマッピングは plan 採択時に決める (本 plan ではバージョン番号を固定しない)。

## Initial implementation order

> phase 番号は topic ベースの章立てであり、実装順とは異なる。実装はこのセクションを正とする。

- [x] 1. この plan を issue / PR checklist として使える状態にする。
- [x] 2. `παιδίο -> παιδίου` の review packet を作る。
- [x] 3. AIO / PHI / IG / Buck で source evidence を調査する。
- [x] 4. entry-level review metadata の schema を最小決定する (Phase 6 → 後で Phase 5 template に反映)。
- [x] 5. review packet template を schema に整合させる (Phase 5)。
- [x] 6. YAML loader validation と tests を追加する。
- [x] 7. seed entry に review metadata を追加する。
- [x] 8. source evidence が見つからない場合は seed entry から `pre_403_2_attic` tag を外す。source evidence が見つかった場合のみ `source_type` / `source_ids` / `references` を埋めて tag を維持する。
- [x] 9. builder の `orthography_hint="pre_403_2_attic"` fallback パスを削除し、regression test を追加する。
- [x] 10. docs に citation-ready workflow を反映する (CHANGELOG に fallback 削除と未確認 `pre_403_2_attic` tag 除去の observability 変化を明記)。
- [x] 11. focused tests を通す。
- [x] 12. full test と wheel build を通す。
- [ ] 13. reviewer に pilot entry を渡す。
- [x] 14. API `orthography_hint` を OpenAPI で `deprecated: true` 化し、将来削除のシグナルを送る。
- [x] 15. `ReviewStatus` / `SourceType` Literal 型を導入し runtime entry の型安全性を強化する。

## 初回 PR の観測可能変化

- [x] schema 拡張、loader validation、tests の追加が中心。
- [x] API `OrthographicNote` の wire shape は変わらない。
- [x] UI 表示は変わらない。
- [x] note messages の文言は変わらない。
- [x] ranking score / candidate order は変わらない。
- [x] Observable changes:
  - [x] `orthography_hint="pre_403_2_attic"` だけで生成されていた YAML 由来でない historical note が出なくなる。これは Phase 1 の tag ルールに合わせた意図的な変更で、CHANGELOG に明記する。
  - [x] 既存 seed の `pre_403_2_attic` source evidence が確認できない場合は、`παιδίο -> παιδίου` から `pre_403_2_attic` note が出なくなる。これは未確認 historical note の配信を止めるための意図的な変更で、CHANGELOG に明記する。
  - [x] OpenAPI schema で `orthography_hint` が `deprecated: true` になる。behavior change ではなく documentation hint であり、正常リクエストは引き続き 200 を返す。

## Out of scope for first citation-ready PR

- [ ] AIO / PHI scraping。
- [ ] PHI text の大量保存。
- [ ] papyri.info choice から runtime YAML への自動昇格。
- [ ] UI の references 表示。
- [ ] API `OrthographicNote` への `review_status` / `citation_ready` 追加。
- [ ] `citation_ready: false` の entry を API から filter する変更 (Gate D で再検討)。
- [ ] `evidence_excerpt` フィールド (drop。将来必要になれば別 RFC)。
- [ ] 多レビュア合意 / 不一致解消ルール (初期は単一 reviewer 前提)。
- [ ] `mixed` source_type (list 化により不要)。
- [ ] ranking score / candidate order への反映。
- [ ] distance matrix regeneration。

## Reference sources to verify

- [x] Attic Inscriptions Online: https://www.atticinscriptions.com/
  - [x] Attica inscriptions の translation / commentary / dating を確認する。
  - [x] IG I up to 403/2 BC と IG II after 403/2 BC の区分を確認する。
- [x] PHI Greek Inscriptions: https://inscriptions.packhum.org/
  - [x] inscription text と publication reference の照合に使う。
  - [x] 再配布ではなく source identifier 記録を基本にする。
- [ ] Digital Greek Attic orthography notes: https://neelsmith.github.io/DigitalGreek/news/attic/
  - [ ] pre-403 BCE Athenian public document orthography の概説として使う。
- [x] Buck, Introduction to the Study of the Greek Dialects, section 4
  - [x] broad reference として保持する。
  - [x] 個別 entry の direct citation にする場合は該当箇所を再確認する。
