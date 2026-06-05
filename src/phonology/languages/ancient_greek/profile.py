"""Ancient Greek language profile factory for phonological search.

This module provides the built-in Ancient Greek language profile
with all language-specific configuration including IPA conversion,
phone inventory, dialect skeleton builders, and orthographic notes.
"""

from __future__ import annotations

from ..._paths import resolve_language_data_dir
from ...core.ports.profiles import LanguageProfile
from ...core.ports.corpus import load_static_corpus_adapter
from .ipa import (
    apply_attic_sigma_sigma_to_tau_tau_shift,
    apply_koine_consonant_shifts,
    get_known_phones,
    to_ipa,
)
from .phones import VOWEL_PHONES, phones_match
from .orthography_notes import (
    build_orthographic_notes,
    prepare_orthographic_data,
)


def build_profile(
    matrix_name: str = "attic_doric",
    matrix_format: str = "json",
) -> LanguageProfile:
    """Build the built-in Ancient Greek profile from packaged data paths.

    Args:
        matrix_name: Base name of the distance matrix file (default: "attic_doric").
        matrix_format: File extension without dot (default: "json").

    Returns:
        Configured Ancient Greek LanguageProfile instance.

    Raises:
        FileNotFoundError: If language data directories cannot be resolved.
    """
    language_id = "ancient_greek"
    lexicon_dir = resolve_language_data_dir(language_id, "lexicon")
    matrix_dir = resolve_language_data_dir(language_id, "matrices")
    rules_dir = resolve_language_data_dir(language_id, "rules")
    corpus_sources_dir = resolve_language_data_dir(language_id, "corpus_sources")

    matrix_path = matrix_dir / f"{matrix_name}.{matrix_format}"
    corpus_sources_path = corpus_sources_dir / "perseus_scaife_sources.yaml"

    return LanguageProfile(
        language_id=language_id,
        display_name="Ancient Greek",
        default_dialect="attic",
        supported_dialects=("attic", "koine"),
        converter=to_ipa,
        phone_inventory=get_known_phones(),
        lexicon_path=lexicon_dir / "greek_lemmas.json",
        matrix_path=matrix_path,
        rules_dir=rules_dir,
        status="pilot",
        description=(
            "Ancient Greek pilot profile with Attic and Koine search support."
        ),
        vowel_phones=tuple(sorted(VOWEL_PHONES)),
        phone_matcher=phones_match,
        dialect_skeleton_builders=(
            apply_koine_consonant_shifts,
            apply_attic_sigma_sigma_to_tau_tau_shift,
        ),
        orthographic_note_builder=build_orthographic_notes,
        orthographic_data_preparer=prepare_orthographic_data,
        corpus_adapter_factory=lambda: load_static_corpus_adapter(corpus_sources_path),
        always_match_contexts=(
            "vowel contraction across hiatus",
            "quantitative metathesis environments",
        ),
    )
