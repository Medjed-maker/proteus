# レビュー指摘 (MEDIUM) フォローアップ

Phase 0/1 リファクタのセルフレビューで提起された MEDIUM 指摘のうち、構造変更を伴うため fix-pack には含めず別 PR に持ち越した 2 件。

- **対象ブランチ**: `main` (Phase 0/1 完了直後の状態)
- **記録日**: 2026-05-09
- **fix-pack 実装記録**: `~/.claude/plans/shiny-booping-harbor.md`

---

## MEDIUM #4 — テスト ⇄ 内部実装の密結合

### 概要

Phase 0/1 で公開 API (`seed_stage` / `build_lexicon_map` 等) に Ancient Greek デフォルト補填層 (`_public_compatibility_search_defaults`) が挿入された。これにより、補填層を「飛ばして」内部 core を直接モックする多数のテストが、`seed_stage` → `_seed_stage_core`、`build_lexicon_map` → `_build_lexicon_map_for_inventory` への参照差し替えに移行している。

### 該当箇所

`tests/test_search.py`、`tests/test_search_partial.py`、`tests/test_search_short_query.py`、`tests/test_search_unigram_fallback.py` ほか多数。

代表的な変更パターン (各テストで反復):

```python
# Before
monkeypatch.setattr(search_module, "seed_stage", lambda *_args, **_kwargs: ["L1"])
monkeypatch.setattr(search_module, "build_lexicon_map", fake_build_lexicon_map)

# After
monkeypatch.setattr(search_module, "_seed_stage_core", lambda *_args, **_kwargs: ["L1"])
monkeypatch.setattr(
    search_module,
    "_build_lexicon_map_for_inventory",
    fake_build_lexicon_map,
)
```

### 問題

1. `_` 始まり (private) シンボルへのテスト依存が広範囲に固定される。今後 internal core をリネーム/再構成するたびに数十テストを追従更新する必要がある。
2. 「公開 API 経路でデフォルト補填が起動する挙動」自体が失われたわけではないが、テストはバイパスしているため、補填層の回帰は専用テスト (`test_public_compatibility_search_defaults` 系数件) にしか検出されない。

### 推奨対処

A. **テスト基盤の整備**: 公開 API のフェイクラッパー (補填層を含む) を `tests/_helpers/` に提供し、各テストが `from tests._helpers.fakes import fake_seed_stage` のように使えるようにする。

```python
# tests/_helpers/fakes.py
def fake_seed_stage_factory(returns: list[str]) -> Callable:
    """補填層を通る公開 seed_stage と差し替えても安全な fake."""
    def _stage(query_ipa: str, index, *, k=2,
               phone_inventory=None, language="ancient_greek") -> list[str]:
        return list(returns)
    return _stage
```

B. **デフォルト補填の必須引数化**: `phone_inventory: Iterable[str] | None = None` を `phone_inventory: Iterable[str]` に変更し、補填層を「default 引数の解決層」ではなく「explicit factory 関数」として呼び出し側に強制する。core 関数のシグネチャから "未解決の None" を排除すれば、テストが core を直叩きしても意味のある assertion になる。

C. **段階的に B → A**: B で `_seed_stage_core` 等の `phone_inventory` を必須化したうえで、A の fake ヘルパーを公開 API 経由に統一すれば、`_seed_stage_core` を直接モックするテストはほぼ消える。

### 影響範囲

- 直接修正対象: `src/phonology/search/__init__.py` (公開/内部関数のシグネチャ変更)、`tests/_helpers/` (新規ヘルパー)、上記 4 テストファイル群 (合計 50+ 箇所のモック更新)
- 振る舞い変更: なし (リファクタ純粋)
- リスク: 大量テスト同時更新による review 負荷。コミットは「ヘルパー導入」「補填必須化」「テスト移行 (ファイル単位)」と分割するのが望ましい。

---

## MEDIUM #6 — `_public_compatibility_search_defaults` の境界統合

### 概要

Ancient Greek デフォルト (phone_inventory + dialect_skeleton_builders) の補填が公開 API 関数 6 箇所で個別に呼ばれている。新しい公開関数を増やすたびに呼び忘れる構造が残っている。

### 該当箇所

`src/phonology/search/__init__.py` 内:

| 関数 | 行 (現状) | 補填呼び出し |
|------|-----------|--------------|
| `build_kmer_index` | ~180 | 1 回 |
| `build_lexicon_map` | ~833 | 1 回 |
| `prepare_query_ipa` | ~988 | 1 回 |
| `seed_stage` | ~1151 | 1 回 |
| `extend_stage` | ~1445 | 1 回 (`isinstance(language, str)` ガード付き) |
| `search_execution` | ~2032 | 1 回 |

冪等な実装 (`phone_inventory is None` のときのみ補填) のため重複呼び出しは害がないが、新規公開 API で漏らしても**実行時には沈黙**するため検出が難しい。

### 推奨対処

A. **デコレータ化** (最小侵襲):

```python
def _backfill_public_defaults(
    func: Callable[..., T]
) -> Callable[..., T]:
    """language/phone_inventory/dialect_skeleton_builders kwargs をラップ前に補填."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        kwargs["phone_inventory"], kwargs["dialect_skeleton_builders"] = (
            _public_compatibility_search_defaults(
                language=kwargs.get("language", DEFAULT_LANGUAGE_ID),
                phone_inventory=kwargs.get("phone_inventory"),
                dialect_skeleton_builders=kwargs.get("dialect_skeleton_builders"),
            )
        )
        return func(*args, **kwargs)
    return wrapper

@_backfill_public_defaults
def seed_stage(...): ...
```

公開関数すべてに同じデコレータを貼ることで、補填忘れがコードレビューで自明になる。

B. **シグネチャ統一 + 単一エントリ層**:
公開関数を `compat/` モジュールにまとめて切り出し、内部 core (`_seed_stage_core` 等) は `phone_inventory` 必須に。`compat/` 層のみが `_public_compatibility_search_defaults` を呼ぶ。MEDIUM #4 の方針 (B) と整合。

C. **型による強制** (将来):
`PhoneInventory = NewType("PhoneInventory", tuple[str, ...])` を導入し、core 関数は `PhoneInventory` を要求、公開層のみが `Optional` を扱う。pyright/mypy で補填忘れがコンパイルエラー扱いになる。

### 影響範囲

- 直接修正対象: `src/phonology/search/__init__.py` (デコレータ追加 or `compat/` 抽出)
- 振る舞い変更: なし (補填層自体は冪等)
- リスク: A は局所的で安全。B/C は MEDIUM #4 と同時に進めるのが効率的。

---

## 実装順序の推奨

1. MEDIUM #6 (A) のデコレータ化を**先に**入れる (低リスク・即効性)
2. MEDIUM #4 (B) で core 関数の `phone_inventory` を必須化
3. MEDIUM #4 (A) のフェイクヘルパー導入とテスト移行
4. (任意) MEDIUM #6 (B/C) で完全な型安全化

各ステップは独立してマージ可能。1 だけでも「補填忘れ」回帰を抑止できる。

## 参考

- レビュー全文: 2026-05-09 セッション (`/review` コマンド出力)
- fix-pack: `~/.claude/plans/shiny-booping-harbor.md`
- 関連コミット: Phase 0/1 マージ前の未コミット変更 (29 ファイル / 1,486 +)
