# Phonology Rule Notation

Proteus stores Ancient Greek sound changes in YAML files under
`data/languages/ancient_greek/rules/`.
The current Ancient Greek rule inventory is split into three categories:

## Schema and Validation

Rule files must validate against the JSON Schema at `data/schemas/phonology_rule_file.schema.json`.
This schema defines the machine-readable structure for all phonology rule files across languages.

Validate rule files locally:
```bash
uv run python tools/validate_rule_files.py
```

Validate against a specific rules directory or schema:
```bash
uv run python tools/validate_rule_files.py --rules-dir path/to/rules --schema path/to/schema.json
```

Rule validation is also run in CI via `tests/test_data_files.py::test_rule_file_validates_against_schema`.

## Current Rule Status

The Ancient Greek rules are **provisional research data** and **not citation-ready**.
They are provided as a pilot implementation for the Proteus phonological search framework.
Expert review is required before these rules can be considered suitable for scholarly citation.

## File Categories

- `consonant_changes.yaml` for segmental consonant developments
- `vowel_shifts.yaml` for vowel quantity, quality, and contraction rules
- `morphophonemic_alternations.yaml` for recurrent ending-level alternations

Use `consonant_changes.yaml` when the target is an individual consonant segment and the trigger is a local phonological environment such as `V_V`, `#_V`, or a nearby consonant set.
Use `vowel_shifts.yaml` when the target is an individual vowel or vowel sequence and the change is primarily segmental, quantitative, qualitative, or a local contraction pattern.
Use `morphophonemic_alternations.yaml` when the target is an ending or recurrent alternation pattern spanning a suffix-sized slice, especially when the condition is a regular paradigm-level alternation rather than a local sound environment.

Short decision flow:

1. If the rule changes an individual consonant in a local environment, place it in `consonant_changes.yaml`.
2. If the rule changes an individual vowel or contraction pattern in a local environment, place it in `vowel_shifts.yaml`.
3. If the rule describes a recurring ending-level alternation such as `-eus` -> `-eos` or `-as` -> `-ɛːs`, place it in `morphophonemic_alternations.yaml`.

Examples:

- `pʰ` -> `f` in Koine is a segment-level consonant development, so it belongs in `consonant_changes.yaml`.
- `eus` -> `eos` as a recurring word-final ending alternation belongs in `morphophonemic_alternations.yaml`.

The runtime `to_ipa()` converter accepts `dialect="koine"` for query-side
Koine consonant normalization, so searches can compare a Koine-style query IPA
form against the existing Attic-oriented lexicon IPA and surface the matching
Koine consonant rules during explanation.

The runtime profile does not currently expose Doric, Ionic, or Aeolic as public
API dialect hints. Rules labelled with those dialects are explanation metadata
for cross-dialect correspondences; they do not imply that `dialect_hint` accepts
those labels.

The `dialects` field is restricted to a canonical controlled vocabulary, enforced
by the `dialects.items.enum` constraint in the JSON Schema (and mirrored by the
`ALLOWED_DIALECTS` allowlist in `tests/test_data_files.py`). The currently
allowed labels are:

`attic`, `cretan`, `cyprian`, `doric`, `elean`, `ionic`, `ionic_east`, `koine`,
`lesbian`, `northwest_greek`, `severe_doric`, `west_greek`.

Only `attic` and `koine` are accepted as public `dialect_hint` values at the API;
the remaining labels are explanation metadata only. Adding a new dialect label
requires updating both the schema enum and the test allowlist (kept in sync by
`test_dialect_allowlist_matches_schema_enum`).

Phonological rule YAML files are limited to phonological and morphophonemic
explanations. Writing-system comments, spelling conventions, normalized-form
correspondences, and beginner reading aids belong in `orthographic_notes`
runtime data and are displayed separately as `Orthographic note` /
`表記体系コメント`. Do not add a spelling-system note to a phonological rule
just to make it visible in `Applied rules`.

Consonant rules generally use a compact notation in the `context` field:

- `_` marks the position of the segment being transformed.
- Example: `#_V` means "the target segment sits at word start before a vowel", e.g. a rule applying to the initial consonant in `pa`.
- `#` marks a word boundary.
- Example: `#_V` uses `#` to mark the left edge of a word before an initial vowel, e.g. `#a`.
- Example: `_#` means "the matched suffix runs to word end", e.g. a word-final ending rule matching the tail of `po.lis`.
- `...` means any intervening span within the same word.
- Example: `V...V` means "between two vowels somewhere in the same word", e.g. `axta` matches the span from the first `a` to the final `a`.
- `V` means any vowel.
- Example: `#_V` means "word-initial before a vowel", e.g. the vowel `a` at the start of `a.na`.
- Brace sets such as `{a,o}` mean "one of these alternatives".
- Example: `{p,t,k}_` means "immediately after p, t, or k", e.g. the target in `pa`, `ta`, or `ka`.

By default, `...` denotes an arbitrary intervening span within the same word. It does not cross `#`, and it is not used to cross an unmarked morpheme boundary.

The Grassmann-style rules `CCH-001` and `CCH-002` follow this general rule directly. Their `...` notation is not a special exception; it is the standard same-word span used everywhere in these YAML rule files.

Deletion rules must set `output: ""` and `change_type: deletion`. Non-deletion
rules must use a non-empty `output`; the shared JSON Schema enforces this so
empty outputs cannot be introduced accidentally. By convention, place
`change_type` between `period:` and `references:` (see CCH-003 in
`data/languages/ancient_greek/rules/consonant_changes.yaml`).

Insertion rules must set `input: ""` and `is_insertion: true`. This explicit
flag is required because empty-input rules can otherwise match too broadly.
For example, the digamma rule models Doric query-side `w` against an Attic
lemma where the inherited sound is absent:

```yaml
input: ""
output: "w"
is_insertion: true
context: "all environments"
```

`is_insertion` is intentionally restricted to the literal value `true`: it is an
explicit opt-in gate for empty-input rules, not a boolean toggle. Non-insertion
rules simply omit the field entirely — there is no `is_insertion: false`. The
shared JSON Schema enforces this bidirectionally (`input: ""` requires
`is_insertion: true`, and `is_insertion: true` requires `input: ""`).

Use `change_type: retention` when the rule records that a segment is preserved
in a context where it might otherwise be expected to change or delete. The
simplest case is an identity mapping, where `input` equals `output`:

```yaml
input: p
output: p
context: "#_V"
change_type: retention
examples:
  - standard: pa
    dialect: pa
```

Retention rules may also use `input ≠ output`. These encode a **reverse
cross-dialect mapping**: the `input` is the form taken by the innovating dialect
(often Attic-Ionic), and the `output` is the conservatively retained form. For
example, `VSH-025` records that Attic-Ionic fronted inherited `/u/` to `/y/`
while Doric retained `/u/`, so the rule maps `input: y` → `output: u`. The
existing `VSH-002` (Doric long-alpha retention) and `MPH-004`/`MPH-005`/`MPH-006`
(Doric ending retentions) follow the same reverse-mapping convention. In all
cases the rule documents which segment a conservative dialect kept, expressed in
the runtime IPA token space so it can be matched against the innovating-dialect
lemma.

The shared JSON Schema currently restricts `change_type` to `retention` or
`deletion`. Other change classes (assimilation, palatalization, etc.) should
omit `change_type` entirely; the field is intentionally minimal until a
broader classification scheme is reviewed.

## Vowel Rules

Vowel rules may instead use short descriptive English phrases such as `all environments`, `after e, i, or r`, or `vowel contraction across hiatus` when a prose description is clearer than the compact notation.

For Doric secondary long mid vowels, the committed runtime rules distinguish
severe Doric as the open-vowel pattern (`eː` -> `ɛː`, `oː` -> `ɔː`). Mild Doric
may pattern closer to Attic `ει`/`ου`, but that subdialect distinction is not a
separate public converter mode yet.

For an all-environments vowel rule, the prose maps directly onto the `context` field:

```yaml
rule: VSH-001
type: vowel
context: all environments
```

For a conditioned vowel rule, use the descriptive environment exactly as written in the rule:

```yaml
rule: VSH-010
type: vowel
context: after e, i, or r
```

## Morphophonemic Rules

Morphophonemic rules use the same `input`/`output` schema, but they model
larger suffix-sized alternations rather than a single segment in isolation.
They are written in the same runtime IPA token space produced by `to_ipa()`,
which converts orthographic Greek input into the runtime IPA representation,
and consumed by `tokenize_ipa()`, which splits an IPA string into runtime phone
tokens for comparison. This section uses IPA notation such as `ɛː`; readers new
to IPA may want a short primer such as the International Phonetic Alphabet
overview at https://en.wikipedia.org/wiki/International_Phonetic_Alphabet.
Because they are meant to capture recurring ending patterns, they
typically use `_#` to mark a word-final suffix slice and may start at the
first differing token rather than at the beginning of the orthographic ending.

Example:

```yaml
rule: MPH-001
type: morphophonemic
input: "as"
output: "ɛːs"
context: "_#"
```

These ending-level rules may intentionally overlap with broader vowel rules.
The Explainer component, a module that selects, ranks, and emits explanation-ready
rule matches, resolves that overlap by preferring the longest matching rule.

## Accent Scope

Accent-related patterns are not currently encoded as executable rules.
`tokenize_ipa()` strips accent marks before phoneme comparison, so adding
accent rules to YAML alone would not make them fire in the present engine.
This remains a known limitation and future support is not yet scheduled.
Supporting accent-sensitive rules would likely require either a
`tokenize_ipa()` mode that preserves accent marks or changes to the comparison
logic so accent-bearing tokens survive into rule matching.
