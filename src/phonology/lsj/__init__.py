"""LSJ extractor package.

This package replaces the old monolithic ``phonology.lsj_extractor`` module.
The original ``lsj_extractor.py`` is preserved as a thin shim that re-exports
the public surface from this package; tests and downstream callers that reach
``phonology.lsj_extractor.<X>`` continue to work unchanged.

Logger name: every internal module uses
``logging.getLogger("phonology.lsj_extractor")`` so ``caplog`` blocks pointed
at that logger keep capturing diagnostics after the split.
"""

from __future__ import annotations

import logging

# Re-export ``to_ipa`` and ``logger`` so tests that monkeypatch
# ``phonology.lsj_extractor.to_ipa`` continue to take effect.
from ..ipa_converter import to_ipa

logger = logging.getLogger("phonology.lsj_extractor")

from ._constants import (
    _DIALECT_MAP,
    _GENDER_MAP,
    _GENDER_REQUIRED_POS,
    _POS_MAP,
)
from ._xml import (
    _elem_text,
    _find_gen_text,
    _find_text,
    _find_text_deep,
    _find_texts,
    _local_name,
)
from ._normalize import (
    _has_descendant,
    _headword_itypes,
    _looks_like_adjective_itypes,
    _looks_like_known_numeral_key,
    _looks_like_verb_key,
    _normalize_beta_token,
    _normalize_headword_key,
    _normalize_intro_text,
)
from ._intro import (
    _append_intro_child_text,
    _entry_intro_text,
    _has_attic_inline_context,
    _inline_pos_candidates,
    _sense_intro_texts,
    _sense_post_gloss_pos_texts,
)
from ._heading import (
    _DialectDecisionContext,
    _DialectLabelDecision,
    _HeadingContext,
    _backward_scan_heading_context,
    _build_dialect_decision_context,
    _decide_dialect_label,
    _has_attic_dialect_label,
    _has_dialect_variant_chain,
    _has_distinct_following_heading_surface_form,
    _has_distinct_nominal_surface_variant,
    _has_following_heading_headword_context,
    _has_heading_prose_headword_context,
    _has_multiple_intro_greek_forms,
    _has_nearby_variant_context,
    _has_nominal_morphology_continuation,
    _has_plural_neuter_article,
    _has_prior_heading_headword_context,
    _heading_spelling_form,
    _is_attic_without_prior,
    _is_heading_dialect_gramgrp,
    _is_heading_gen_marker,
    _is_heading_itype_marker,
    _is_heading_surface_form,
    _is_single_dialect_surface_variant,
    _is_variant_only,
    _leading_dialect_labels,
    _mapped_heading_dialect,
    _qualifies_by_context,
    _qualifies_by_gen_marker,
    _qualifies_by_nearby_variant_note,
    _scan_heading_context,
    _should_keep_heading_dialect_label,
)
from ._fields import (
    _PosInferenceContext,
    _extract_dialect,
    _extract_gender,
    _extract_gloss,
    _extract_headword,
    _extract_pos,
    _infer_adjective_itype_pos,
    _infer_adverb_ending_pos,
    _infer_explicit_pos,
    _infer_final_participle_pos,
    _infer_gender_based_pos,
    _infer_inline_prose_pos,
    _infer_known_numeral_pos,
    _infer_participle_intro_marker,
    _infer_post_gloss_pos,
    _infer_verb_indicator_pos,
)
from ._extract import extract_entry
from ._xml_iter import extract_all, find_xml_files, iter_xml_entries
from ._document import (
    _document_dialect,
    _document_dialect_label,
    build_lexicon_document,
    validate_document,
)
# main and run_cli live in the parent phonology.lsj_extractor shim so test
# monkeypatches on the shim namespace take effect.

__all__ = [
    "extract_all",
    "extract_entry",
    "find_xml_files",
    "iter_xml_entries",
    "build_lexicon_document",
    "validate_document",
]
