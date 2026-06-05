"""Private implementation package for the phonological-rule explainer.

The public facade lives in :mod:`phonology.explainer`, which re-exports the
symbols defined across the modules in this package:

    - ``_rule_paths``: trusted-directory registry and rule-path resolution.
    - ``_rule_loader``: YAML rule loading and version-metadata extraction.
    - ``_rule_tokenize``: ``TokenizedRule`` plus tokenization / sorting.
    - ``_rule_match``: ``Alignment`` / ``RuleApplication`` / mismatch-block
      matching state machine.
    - ``_context``: rule-context predicates and token-lookup helpers.
    - ``_types``: shared public/internal data types.
    - ``_prose``: ``Explanation`` plus ``to_prose``.

This package is language-independent; rule data is supplied by language plugins.
"""
