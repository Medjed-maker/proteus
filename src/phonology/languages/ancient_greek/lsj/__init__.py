"""LSJ extractor package.

This package replaces the old monolithic ``phonology.languages.ancient_greek.lsj_extractor`` module.
The original ``lsj_extractor.py`` is preserved as a thin shim that re-exports
the public surface from this package; tests and downstream callers that reach
``phonology.languages.ancient_greek.lsj_extractor.<X>`` continue to work unchanged.

Logger name: every internal module uses
``logging.getLogger("phonology.languages.ancient_greek.lsj_extractor")`` so ``caplog`` blocks pointed
at that logger keep capturing diagnostics after the split.
"""

from __future__ import annotations

import logging

from ..ipa import to_ipa
from ._document import (
    build_lexicon_document,
    validate_document,
    validate_supplemental_lemma_entries,
)
from ._extract import extract_entry
from ._xml_iter import extract_all, find_xml_files, iter_xml_entries

logger = logging.getLogger("phonology.languages.ancient_greek.lsj_extractor")

# main and run_cli live in the parent phonology.languages.ancient_greek.lsj_extractor shim so test
# monkeypatches on the shim namespace take effect.

__all__ = [
    "extract_all",
    "extract_entry",
    "find_xml_files",
    "iter_xml_entries",
    "build_lexicon_document",
    "validate_document",
    "validate_supplemental_lemma_entries",
    "logger",
    "to_ipa",
]
