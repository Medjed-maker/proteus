# Phonology Rule Notation

Proteus stores Ancient Greek sound changes in YAML files under `data/rules/ancient_greek/`.
Consonant rules generally use a compact notation in the `context` field:

- `_` marks the position of the segment being transformed.
- Example: `#_V` means "the target segment sits at word start before a vowel", e.g. a rule applying to the initial consonant in `pa`.
- `#` marks a word boundary.
- Example: `#_V` uses `#` to mark the left edge of a word before an initial vowel, e.g. `#a`.
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
