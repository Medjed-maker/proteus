"""Language-independent boundary contracts (ports) for the phonology core.

This package gathers the core's outward-facing contracts that language plugins
and adapters implement or supply:

    - ``profiles``: ``LanguageProfile`` / ``IpaConverter`` and the language
      profile registry.
    - ``orthography_notes``: orthographic-note payload types and the
      ``OrthographicNoteBuilder`` protocol.
    - ``corpus``: corpus source-metadata models and the ``CorpusAdapter``
      protocol.

None of these contracts reference any specific language; concrete behavior is
provided by plugins under ``phonology.languages``.
"""
