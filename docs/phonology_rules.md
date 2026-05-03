# Phonology Rule Notation

Proteus stores Ancient Greek sound changes in YAML files under
`data/languages/ancient_greek/rules/`.
The current Ancient Greek rule inventory is split into three categories:

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

## Vowel Rules

Vowel rules may instead use short descriptive English phrases such as `all environments`, `after e, i, or r`, or `vowel contraction across hiatus` when a prose description is clearer than the compact notation.

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
