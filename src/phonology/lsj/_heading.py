"""LSJ extractor helper module.

Logger uses the literal ``phonology.lsj_extractor`` name so existing
``caplog.set_level(logger="phonology.lsj_extractor")`` blocks in
``tests/test_lsj_extractor.py`` keep capturing diagnostics from this module
after the split.
"""

from __future__ import annotations

import logging
import unicodedata
from typing import Any, NamedTuple

from ._constants import (
    _DIALECT_MAP,
    _HEADING_ENTRY_LEVEL_CONSTRAINT_RE,
    _HEADING_GREEK_FORM_TAGS,
    _HEADING_HEADWORD_CONTEXT_TAGS,
    _HEADING_LEXICOGRAPHIC_CONTEXT_RE,
    _HEADING_NEARBY_VARIANT_CONTEXT_RE,
    _HEADING_NONPROSE_CONTEXT_TAGS,
    _HEADING_SURFACE_FORM_TAGS,
)
from ._intro import _inline_pos_candidates
from ._normalize import _normalize_beta_token, _normalize_intro_text
from ._xml import _elem_text, _find_gen_text, _local_name

logger = logging.getLogger("phonology.lsj_extractor")


def _is_heading_surface_form(child: Any) -> bool:
    """Return True when a heading child presents an alternate surface form."""
    return (
        _local_name(child) in _HEADING_SURFACE_FORM_TAGS
        and child.get("lang") == "greek"
        and bool(_elem_text(child))
    )


def _heading_spelling_form(child: Any) -> str | None:
    """Return a normalized Greek spelling form for ``orth``/``foreign`` tags only."""
    if _local_name(child) not in {"orth", "foreign"} or child.get("lang") != "greek":
        return None
    text = _elem_text(child)
    if not text:
        return None
    return _normalize_beta_token(text)


def _is_heading_gen_marker(child: Any) -> bool:
    """Return True when a heading child carries Greek gender morphology only."""
    return (
        _local_name(child) == "gen"
        and child.get("lang") == "greek"
        and bool(_elem_text(child))
    )


def _is_heading_itype_marker(child: Any) -> bool:
    """Return True when a heading child carries Greek inflectional morphology."""
    return _local_name(child) == "itype" and bool(_elem_text(child))


def _is_heading_dialect_gramgrp(child: Any) -> bool:
    """Return True when ``child`` is a heading ``<gramGrp>`` with a dialect label."""
    if _local_name(child) != "gramGrp":
        return False
    for descendant in child.iter():
        if descendant is child:
            continue
        if (
            _local_name(descendant) == "gram"
            and descendant.get("type", "") == "dialect"
        ):
            return True
    return False


class _HeadingContext(NamedTuple):
    """Pre-computed heading context for variant-only dialect label decisions.

    All fields are computed by :func:`_scan_heading_context` via a single
    forward + backward pass over the heading's children list.
    """

    following_greek_form_count: (
        int  # Wide zone: number of Greek surface forms before next dialect gramGrp
    )
    has_following_form: bool  # Narrow zone: any surface form before the first gramGrp
    has_following_surface_form: bool  # Wide zone: any orth/foreign/pron with lang=greek
    has_following_gen_marker: bool  # Wide zone: any <gen lang=greek> marker
    has_following_itype_marker: bool  # Wide zone: any <itype> inflectional marker
    has_non_dialect_gramgrp_before_gen: (
        bool  # Wide zone: a non-dialect gramGrp appeared before a gen marker
    )
    has_preceding_form: bool  # Backward scan: surface form before start_index
    has_preceding_itype_marker: bool  # Backward scan: <itype> marker before start_index
    has_prior_dialect_label: (
        bool  # Backward scan: a dialect gramGrp exists before start_index
    )
    has_following_dialect_label: (
        bool  # Full zone: any dialect gramGrp after start_index (before sense)
    )
    has_following_attic_label: (
        bool  # Full zone: a dialect gramGrp with Attic label after start_index
    )


class _DialectDecisionContext(NamedTuple):
    """Inputs needed to decide whether a heading dialect label is entry-level."""

    mapped_dialect: str
    heading: _HeadingContext
    variant_context: bool
    entry_level_constraint: bool
    prior_headword_context: bool
    following_headword_context: bool
    has_distinct_following_surface_form: bool
    has_extra_following_forms: bool


class _DialectLabelDecision(NamedTuple):
    """Decision for a mapped dialect label encountered in the entry heading."""

    mapped_dialect: str
    is_variant_only: bool


def _has_attic_dialect_label(gramgrp: Any) -> bool:
    """Return True when a dialect ``<gramGrp>`` contains an Attic label."""
    for descendant in gramgrp.iter():
        if _local_name(descendant) != "gram":
            continue
        if descendant.get("type", "") != "dialect":
            continue
        text = _elem_text(descendant) or ""
        if _DIALECT_MAP.get(text.strip()) == "attic":
            return True
    return False


def _backward_scan_heading_context(
    children: list[Any], start_index: int
) -> tuple[bool, bool, bool]:
    """Scan backward from *start_index* and return context flags.

    Returns:
        A tuple of ``(has_preceding_form, has_preceding_itype_marker,
        has_prior_dialect_label)``.
    """
    has_preceding_form = False
    has_preceding_itype_marker = False

    for child in reversed(children[:start_index]):
        if _is_heading_dialect_gramgrp(child):
            return has_preceding_form, has_preceding_itype_marker, True
        if _is_heading_surface_form(child):
            has_preceding_form = True
        if _is_heading_itype_marker(child):
            has_preceding_itype_marker = True

    return has_preceding_form, has_preceding_itype_marker, False


def _scan_heading_context(children: list[Any], start_index: int) -> _HeadingContext:
    """Compute all heading context predicates in a single forward + backward pass.

    Three forward scan zones with progressively wider stopping conditions:

    - **Narrow**: stops at any ``<gramGrp>`` or ``<sense>`` — feeds
      ``has_following_form``.
    - **Wide**: stops at dialect ``<gramGrp>`` or ``<sense>`` — feeds
      ``following_greek_form_count``, ``has_following_surface_form``,
      ``has_following_gen_marker``, ``has_following_itype_marker``, and
      ``has_non_dialect_gramgrp_before_gen``.
    - **Full**: stops only at ``<sense>`` — feeds ``has_following_dialect_label``
      and ``has_following_attic_label``.

    The backward scan uses two zones:

    - **Narrow**: stops at dialect ``<gramGrp>`` — feeds ``has_preceding_form``
      and ``has_preceding_itype_marker``.
    - **Full**: stops at the first dialect ``<gramGrp>`` when computing
      ``has_prior_dialect_label`` (an intentional optimization). Feeds
      ``has_preceding_form``, ``has_preceding_itype_marker``, and
      ``has_prior_dialect_label``.
    """
    # -- Forward scan --
    following_greek_form_count = 0
    has_following_form = False
    has_following_surface_form = False
    has_following_gen_marker = False
    has_following_itype_marker = False
    has_non_dialect_gramgrp_before_gen = False
    has_following_dialect_label = False
    has_following_attic_label = False

    narrow_done = False
    wide_done = False
    wide_seen_non_dialect_gramgrp = False

    for child in children[start_index + 1 :]:
        local = _local_name(child)
        if local == "sense":
            break

        is_dialect_gramgrp = _is_heading_dialect_gramgrp(child)

        # Narrow zone: stops at ANY gramGrp
        if not narrow_done:
            if local == "gramGrp":
                narrow_done = True
            elif _is_heading_surface_form(child):
                has_following_form = True

        # Wide zone: stops at dialect gramGrp only
        if not wide_done:
            if is_dialect_gramgrp:
                wide_done = True
            else:
                if _is_heading_surface_form(child):
                    following_greek_form_count += 1
                    has_following_surface_form = True
                if _is_heading_gen_marker(child):
                    has_following_gen_marker = True
                    # non_dialect_gramgrp_before_gen is only true if we saw
                    # a non-dialect gramGrp BEFORE this gen marker
                    if wide_seen_non_dialect_gramgrp:
                        has_non_dialect_gramgrp_before_gen = True
                if _is_heading_itype_marker(child):
                    has_following_itype_marker = True
                if local == "gramGrp":
                    wide_seen_non_dialect_gramgrp = True

        # Full zone: stops only at sense (handled at top of loop)
        if is_dialect_gramgrp:
            has_following_dialect_label = True
            if _has_attic_dialect_label(child):
                has_following_attic_label = True

    # -- Backward scan --
    has_preceding_form, has_preceding_itype_marker, has_prior_dialect_label = (
        _backward_scan_heading_context(children, start_index)
    )

    return _HeadingContext(
        following_greek_form_count=following_greek_form_count,
        has_following_form=has_following_form,
        has_following_surface_form=has_following_surface_form,
        has_following_gen_marker=has_following_gen_marker,
        has_following_itype_marker=has_following_itype_marker,
        has_non_dialect_gramgrp_before_gen=has_non_dialect_gramgrp_before_gen,
        has_preceding_form=has_preceding_form,
        has_preceding_itype_marker=has_preceding_itype_marker,
        has_prior_dialect_label=has_prior_dialect_label,
        has_following_dialect_label=has_following_dialect_label,
        has_following_attic_label=has_following_attic_label,
    )


def _has_prior_heading_headword_context(children: list[Any], start_index: int) -> bool:
    """Return True when the heading already established the main headword context.

    Counts explicit heading POS tags and only prose that looks like
    grammatical/lexicographic context. Bare English glosses such as ``gift``
    must not qualify because they would incorrectly mark entry-level dialect
    labels as variant-only.
    """
    if start_index <= 0:
        return False
    prior_parts: list[str] = []
    for child in children[:start_index]:
        local = _local_name(child)
        if local in _HEADING_HEADWORD_CONTEXT_TAGS and _elem_text(child):
            return True
        if local not in _HEADING_NONPROSE_CONTEXT_TAGS:
            child_text = _elem_text(child)
            if child_text:
                prior_parts.append(child_text)
        prior_parts.append(child.tail or "")
    text = _normalize_intro_text(prior_parts)
    return _has_heading_prose_headword_context(text)


def _has_following_heading_headword_context(
    children: list[Any], start_index: int
) -> bool:
    """Return True when following heading content establishes the headword context.

    Scans forward until the next dialect ``<gramGrp>`` or top-level ``<sense>``.
    Explicit heading POS tags count immediately. Otherwise, only prose that
    looks like grammatical/lexicographic context counts; bare gloss words do
    not.
    """
    if start_index >= len(children) - 1:
        return False
    following_parts: list[str] = []
    for child in children[start_index + 1 :]:
        local = _local_name(child)
        if local == "sense":
            break
        if _is_heading_dialect_gramgrp(child):
            break
        if local in _HEADING_HEADWORD_CONTEXT_TAGS and _elem_text(child):
            return True
        if local not in _HEADING_NONPROSE_CONTEXT_TAGS:
            child_text = _elem_text(child)
            if child_text:
                following_parts.append(child_text)
        following_parts.append(child.tail or "")
    text = _normalize_intro_text(following_parts)
    return _has_heading_prose_headword_context(text)


def _has_heading_prose_headword_context(text: str) -> bool:
    """Return True when heading prose carries grammatical or variant context."""
    if not text:
        return False
    return bool(
        _inline_pos_candidates(text) or _HEADING_LEXICOGRAPHIC_CONTEXT_RE.search(text)
    )


def _has_distinct_following_heading_surface_form(
    children: list[Any], start_index: int
) -> bool:
    """Return True when a following Greek spelling differs from prior heading forms."""
    prior_forms = {
        form
        for child in children[:start_index]
        if (form := _heading_spelling_form(child)) is not None
    }
    if not prior_forms:
        return False
    for child in children[start_index + 1 :]:
        local = _local_name(child)
        if local == "sense":
            break
        if _is_heading_dialect_gramgrp(child):
            break
        form = _heading_spelling_form(child)
        if form and form not in prior_forms:
            return True
    return False


def _has_nearby_variant_context(children: list[Any], start_index: int) -> bool:
    """Return True when nearby heading prose marks a dialect label as variant-only.

    Checks both the preceding sibling's tail (text after the previous element)
    and the current element's own tail (text after this ``<gramGrp>``) for
    variant context markers: "so in", "mostly", "rarely", "strengthd.",
    "before vowels", and "before consonants".
    """
    if start_index > 0:
        preceding_tail = children[start_index - 1].tail or ""
        if _HEADING_NEARBY_VARIANT_CONTEXT_RE.search(preceding_tail):
            return True
    current_tail = children[start_index].tail or ""
    return bool(_HEADING_NEARBY_VARIANT_CONTEXT_RE.search(current_tail))


def _build_dialect_decision_context(
    children: list[Any], start_index: int, mapped_dialect: str
) -> _DialectDecisionContext:
    """Collect heading-derived inputs for a dialect label decision."""
    heading = _scan_heading_context(children, start_index)
    return _DialectDecisionContext(
        mapped_dialect=mapped_dialect,
        heading=heading,
        variant_context=_has_nearby_variant_context(children, start_index),
        entry_level_constraint=bool(
            _HEADING_ENTRY_LEVEL_CONSTRAINT_RE.search(children[start_index].tail or "")
        ),
        prior_headword_context=_has_prior_heading_headword_context(
            children, start_index
        ),
        following_headword_context=_has_following_heading_headword_context(
            children, start_index
        ),
        has_distinct_following_surface_form=_has_distinct_following_heading_surface_form(
            children, start_index
        ),
        has_extra_following_forms=heading.following_greek_form_count > 1,
    )


def _is_attic_without_prior(context: _DialectDecisionContext) -> bool:
    """Return True when an Attic label is likely the primary entry dialect."""
    ctx = context.heading
    return context.mapped_dialect == "attic" and not ctx.has_prior_dialect_label


def _is_single_dialect_surface_variant(context: _DialectDecisionContext) -> bool:
    """Return True for a lone dialect label attached to one surface variant."""
    ctx = context.heading
    return (
        not ctx.has_prior_dialect_label
        and not ctx.has_following_dialect_label
        and ctx.has_following_surface_form
        and (context.prior_headword_context or context.following_headword_context)
    )


def _has_dialect_variant_chain(context: _DialectDecisionContext) -> bool:
    """Return True when a multi-dialect heading enumerates surface variants."""
    ctx = context.heading
    return (
        ctx.has_following_dialect_label
        and not ctx.has_following_attic_label
        and ctx.has_following_form
        and (ctx.has_following_surface_form or ctx.has_following_gen_marker)
    )


def _has_nominal_morphology_continuation(context: _DialectDecisionContext) -> bool:
    """Return True when a dialect label continues nominal morphology markers."""
    ctx = context.heading
    return (
        ctx.has_following_gen_marker
        and not ctx.has_following_surface_form
        and (ctx.has_preceding_itype_marker or ctx.has_following_itype_marker)
    )


def _has_distinct_nominal_surface_variant(context: _DialectDecisionContext) -> bool:
    """Return True for a distinct surface spelling followed by nominal gender."""
    ctx = context.heading
    return (
        not ctx.has_prior_dialect_label
        and not ctx.has_following_dialect_label
        and context.has_distinct_following_surface_form
        and ctx.has_following_gen_marker
    )


def _qualifies_by_context(
    context: _DialectDecisionContext,
    *,
    has_dialect_variant_chain: bool,
    has_distinct_nominal_surface_variant: bool,
    is_single_dialect_surface_variant: bool,
) -> bool:
    """Return True when surrounding context marks the label as variant-only."""
    ctx = context.heading
    return (
        context.prior_headword_context
        or context.has_extra_following_forms
        or ctx.has_prior_dialect_label
        or has_dialect_variant_chain
        or has_distinct_nominal_surface_variant
        or is_single_dialect_surface_variant
        or (context.variant_context and ctx.has_following_surface_form)
    )


def _qualifies_by_nearby_variant_note(context: _DialectDecisionContext) -> bool:
    """Return True when nearby prose cues make an adjacent form variant-only."""
    ctx = context.heading
    return context.variant_context and (
        (ctx.has_preceding_form and context.prior_headword_context)
        or ctx.has_following_surface_form
    )


def _qualifies_by_gen_marker(
    context: _DialectDecisionContext,
    *,
    has_nominal_morphology_continuation: bool,
) -> bool:
    """Return True when a following gender marker belongs to a variant note."""
    ctx = context.heading
    return (
        ctx.has_following_gen_marker
        and not ctx.has_following_surface_form
        and (
            has_nominal_morphology_continuation
            or (
                context.prior_headword_context
                and not ctx.has_non_dialect_gramgrp_before_gen
            )
        )
    )


def _is_variant_only(
    context: _DialectDecisionContext,
    *,
    is_attic_without_prior: bool,
    qualifies_by_context: bool,
    qualifies_by_gen_marker: bool,
    qualifies_by_nearby_variant_note: bool,
) -> bool:
    """Return the final variant-only decision for a dialect label."""
    ctx = context.heading
    return (
        not is_attic_without_prior
        and not context.entry_level_constraint
        and (
            (ctx.has_following_form and qualifies_by_context)
            or qualifies_by_gen_marker
            or qualifies_by_nearby_variant_note
        )
    )


def _decide_dialect_label(
    context: _DialectDecisionContext,
) -> _DialectLabelDecision:
    """Return whether a heading dialect label is variant-only."""
    is_attic_without_prior = _is_attic_without_prior(context)
    is_single_dialect_surface_variant = _is_single_dialect_surface_variant(context)
    has_dialect_variant_chain = _has_dialect_variant_chain(context)
    has_nominal_morphology_continuation = _has_nominal_morphology_continuation(context)
    has_distinct_nominal_surface_variant = _has_distinct_nominal_surface_variant(
        context
    )
    qualifies_by_context = _qualifies_by_context(
        context,
        has_dialect_variant_chain=has_dialect_variant_chain,
        has_distinct_nominal_surface_variant=has_distinct_nominal_surface_variant,
        is_single_dialect_surface_variant=is_single_dialect_surface_variant,
    )
    qualifies_by_nearby_variant_note = _qualifies_by_nearby_variant_note(context)
    qualifies_by_gen_marker = _qualifies_by_gen_marker(
        context,
        has_nominal_morphology_continuation=has_nominal_morphology_continuation,
    )
    variant_only = _is_variant_only(
        context,
        is_attic_without_prior=is_attic_without_prior,
        qualifies_by_context=qualifies_by_context,
        qualifies_by_gen_marker=qualifies_by_gen_marker,
        qualifies_by_nearby_variant_note=qualifies_by_nearby_variant_note,
    )
    return _DialectLabelDecision(
        mapped_dialect=context.mapped_dialect,
        is_variant_only=variant_only,
    )


def _mapped_heading_dialect(descendant: Any) -> str | None:
    """Return a mapped dialect label for a heading ``<gram>`` descendant."""
    if _local_name(descendant) != "gram":
        return None
    if descendant.get("type", "") != "dialect":
        return None
    text = _elem_text(descendant) or ""
    return _DIALECT_MAP.get(text.strip())


def _should_keep_heading_dialect_label(
    children: list[Any], start_index: int, mapped_dialect: str
) -> bool:
    """Return True when a heading dialect label belongs to the entry."""
    decision_context = _build_dialect_decision_context(
        children, start_index, mapped_dialect
    )
    decision = _decide_dialect_label(decision_context)
    return not decision.is_variant_only


def _leading_dialect_labels(entry: Any) -> list[str]:
    """Return dialect labels attached before the first top-level ``<sense>``."""
    labels: list[str] = []
    children = list(entry)
    for index, child in enumerate(children):
        local = _local_name(child)
        if local == "sense":
            break
        if local == "gramGrp":
            for descendant in child.iter():
                mapped_dialect = _mapped_heading_dialect(descendant)
                if mapped_dialect is None:
                    continue
                if not _should_keep_heading_dialect_label(
                    children, index, mapped_dialect
                ):
                    continue
                labels.append(mapped_dialect)
    return labels


def _has_plural_neuter_article(entry: Any) -> bool:
    """Return True when the entry heading marks a neuter plural article."""
    gen_text = _find_gen_text(entry)
    normalized = unicodedata.normalize("NFC", gen_text.strip())
    return normalized in {"ta/", "τὰ", "τά"}


def _has_multiple_intro_greek_forms(entry: Any) -> bool:
    """Return True when the entry intro contains multiple Greek form markers."""
    count = 0
    for child in entry:
        local = _local_name(child)
        if local == "sense":
            break
        if child.get("lang") == "greek" and local in _HEADING_GREEK_FORM_TAGS:
            if _elem_text(child):
                count += 1
        if count > 1:
            return True
    return False

