# Codex Review — Cycle 3: explainer.py split

**Date:** 2026-05-14 (split) / 2026-05-15 (review)
**Target:** working-tree diff (1720-line module split into six sibling files)
**Reviewer:** `/codex:review --wait` (codex-companion 1.0.4)

## Scope

- `src/phonology/explainer.py`: 1720 → 200 lines (facade only)
- New `src/phonology/_explainer_rule_paths.py` (179): trusted-directory registry + rules-base resolution; `_RULES_BASE_DIR_OVERRIDE` continues to live in the facade so test monkeypatches keep working, with `_get_rules_base_dir()` looking the override up via the package.
- New `src/phonology/_explainer_rule_loader.py` (240): `load_rules`, `get_rules_version`, YAML version-node extraction helpers.
- New `src/phonology/_explainer_rule_tokenize.py` (132): `TokenizedRule`, `tokenize_rules_for_matching`, `_rule_specificity`, `_tokenize_rule_side`, `_tokenize_context_tail`, `_tokenize_rules`, `_ALWAYS_MATCH_CONTEXTS`, `Rule` alias.
- New `src/phonology/_explainer_types.py` (129): `Alignment`, `RuleApplication`, `Explanation` dependencies `_MismatchBlock`/`_WordFinalSuffixMatch`/`_RuleMatchResult`, `POSITION_UNKNOWN`, `RuleMetadata` alias.
- New `src/phonology/_explainer_context.py` (201): vowel/consonant predicates, lemma/query token lookahead, `_matches_context`, `_matches_following_set`, `_matches_same_word_lookahead`, plus the local `_NASAL_PHONES` / `_AFTER_E_I_R_PHONES` constants.
- New `src/phonology/_explainer_rule_match.py` (707): mismatch-block iteration, rule-application building, `explain` / `explain_with_tokenized_rules` entry points.
- New `src/phonology/_explainer_prose.py` (75): `Explanation` dataclass and `to_prose`.
- All new modules use `logging.getLogger("phonology.explainer")` so existing `caplog.set_level(logger="phonology.explainer")` blocks keep capturing diagnostics.

## Outcome

Codex flagged one P3 issue and otherwise confirmed the split is behaviour-preserving:

> [P3] Re-export `_rule_specificity` from the facade — `src/phonology/explainer.py`.
> The facade docstring claims private symbols remain importable, but
> `_rule_specificity` was the one private helper not re-exported.

**Fix applied in-cycle:** added `_rule_specificity` to the facade's
re-export list from `_explainer_rule_tokenize`. Verified with
`uv run python -c "from phonology.explainer import _rule_specificity; print(_rule_specificity)"`.

## Local Verification

- `uv run pytest -q` — **1621 passed**
- `uv run mypy` — **0 issues across 59 source files**
- API smoke (`POST /search` for `λόγος`) returns the expected `lóɡos` (0.00) /
  `ἄλογος` (0.17) / `ἄλγος` (0.30) ranking.
- File sizes (post-refactor):
  - `_explainer_rule_match.py` 707
  - `_explainer_rule_loader.py` 240
  - `_explainer_context.py` 201
  - `explainer.py` 200 (facade)
  - `_explainer_rule_paths.py` 179
  - `_explainer_rule_tokenize.py` 132
  - `_explainer_types.py` 129
  - `_explainer_prose.py` 75

## Test-Seam and Compatibility Notes

- `phonology.explainer.RULES_BASE_DIR` continues to resolve lazily via the
  facade `__getattr__`; `_get_rules_base_dir()` in `_explainer_rule_paths`
  reads `_RULES_BASE_DIR_OVERRIDE` from the facade so
  `monkeypatch.setattr(phonology.explainer, "_RULES_BASE_DIR_OVERRIDE", ...)`
  still takes effect.
- `tests/test_explainer.py` reaches `_MismatchBlock`,
  `_find_matching_rule_candidate`, `_build_observed_application_for_column`,
  `_RuleMatchResult`, `TokenizedRule`, `_advance_block_cursors`,
  `_resolve_and_validate_rules_dir` via `explainer_module.<name>`; all are
  re-exported from the facade.
- Downstream importers (`phonology.search._types`, `phonology.search._scoring`,
  `phonology.search._annotation`, `phonology.log_odds`,
  `api/_hit_formatting`, `api/main`, multiple tests) continue to use the
  documented public surface (`Alignment`, `RuleApplication`, `Explanation`,
  `explain`, `explain_alignment`, `load_rules`, `tokenize_rules_for_matching`,
  `get_rules_version`, `register_trusted_rules_dir`,
  `clear_trusted_external_rules_dirs`, `to_prose`) unmodified.

## Follow-ups

None.
