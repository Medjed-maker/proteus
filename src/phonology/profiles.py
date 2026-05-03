"""Language profile registry for phonological search assets."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Protocol

from ._paths import DEFAULT_LANGUAGE_ID, resolve_language_data_dir
from .languages.ancient_greek.ipa import (
    apply_koine_consonant_shifts,
    get_known_phones,
    to_ipa,
)
from .languages.ancient_greek.orthography_notes import (
    build_orthographic_notes as build_ancient_greek_orthographic_notes,
    prepare_orthographic_data as prepare_ancient_greek_orthographic_data,
)
from .orthography_notes import OrthographicNoteBuilder


class IpaConverter(Protocol):
    """Protocol for language-specific IPA conversion functions."""

    def __call__(self, text: str, *, dialect: str) -> str:
        """Convert text to IPA sequence."""
        ...


@dataclass(frozen=True, slots=True)
class LanguageProfile:
    """Runtime configuration for a language-specific phonological plugin."""

    language_id: str
    display_name: str
    default_dialect: str
    supported_dialects: tuple[str, ...]
    converter: IpaConverter
    phone_inventory: tuple[str, ...]
    lexicon_path: Path
    matrix_path: Path
    rules_dir: Path
    dialect_skeleton_builders: tuple[Callable[[list[str]], list[str]], ...] = ()
    orthographic_note_builder: OrthographicNoteBuilder | None = None
    orthographic_data_preparer: Callable[[], None] | None = None


_REGISTRY: dict[str, LanguageProfile] = {}
_REGISTRY_LOCK = threading.Lock()


def register_language_profile(profile: LanguageProfile) -> None:
    """Register or replace a language profile by ``language_id``.

    Validation:
        - ``language_id`` must already be lowercase and non-empty (no normalisation
          is applied — the caller must strip and lowercase before constructing the
          profile).
        - ``default_dialect`` must appear in ``supported_dialects``.
        - ``phone_inventory`` must be non-empty.

    Side effects:
        Registers ``profile.rules_dir`` and ``profile.matrix_path.parent`` as
        trusted directories via ``explainer.register_trusted_rules_dir`` and
        ``distance.register_trusted_matrices_dir``. Subsequent matrix and rule
        loads from those directories are permitted without additional validation.
        To undo this, call ``_reset_language_registry_for_tests()`` (tests only).

    Thread safety:
        Registry mutation is guarded by ``_REGISTRY_LOCK``. Trusted-directory
        registration uses its own locks inside ``explainer`` and ``distance``.

    Raises:
        ValueError: When any validation rule is violated, or when the rules/matrix
            directories cannot be registered as trusted paths.
    """
    language_id = profile.language_id.strip().lower()
    if not language_id:
        raise ValueError("language_id must be a non-empty string")
    if profile.language_id != language_id:
        raise ValueError("language_id must already be lowercase and stripped")
    if profile.default_dialect not in profile.supported_dialects:
        raise ValueError(
            "default_dialect must be included in supported_dialects for "
            f"{profile.language_id!r}"
        )
    if not profile.phone_inventory:
        raise ValueError(
            f"phone_inventory must be non-empty for {profile.language_id!r}"
        )

    from .explainer import register_trusted_rules_dir
    from .distance import register_trusted_matrices_dir

    register_trusted_rules_dir(profile.rules_dir)
    register_trusted_matrices_dir(profile.matrix_path.parent)

    with _REGISTRY_LOCK:
        _REGISTRY[language_id] = profile


def get_language_profile(language_id: str = DEFAULT_LANGUAGE_ID) -> LanguageProfile:
    """Return a registered language profile.

    Raises:
        ValueError: If no profile is registered for ``language_id``.
    """
    normalized = language_id.strip().lower()
    with _REGISTRY_LOCK:
        try:
            return _REGISTRY[normalized]
        except KeyError as exc:
            raise ValueError(f"Unsupported language profile: {language_id!r}") from exc


def _build_ancient_greek_profile() -> LanguageProfile:
    """Build the built-in Ancient Greek profile from packaged data paths."""
    language_id = DEFAULT_LANGUAGE_ID
    lexicon_dir = resolve_language_data_dir(language_id, "lexicon")
    matrix_dir = resolve_language_data_dir(language_id, "matrices")
    rules_dir = resolve_language_data_dir(language_id, "rules")
    return LanguageProfile(
        language_id=language_id,
        display_name="Ancient Greek",
        default_dialect="attic",
        supported_dialects=("attic", "koine"),
        converter=to_ipa,
        phone_inventory=get_known_phones(),
        lexicon_path=lexicon_dir / "greek_lemmas.json",
        matrix_path=matrix_dir / "attic_doric.json",
        rules_dir=rules_dir,
        dialect_skeleton_builders=(apply_koine_consonant_shifts,),
        orthographic_note_builder=build_ancient_greek_orthographic_notes,
        orthographic_data_preparer=prepare_ancient_greek_orthographic_data,
    )


DEFAULT_LANGUAGE_PROFILE: LanguageProfile | None = None
_DEFAULT_PROFILE_LOCK = threading.Lock()


def get_default_language_profile() -> LanguageProfile:
    """Get the default Ancient Greek language profile with lazy initialization.

    Builds and caches the profile on first use to avoid import-time errors.

    Returns:
        The default Ancient Greek language profile.

    Raises:
        FileNotFoundError: If language data directories cannot be resolved.
    """
    global DEFAULT_LANGUAGE_PROFILE

    if DEFAULT_LANGUAGE_PROFILE is None:
        with _DEFAULT_PROFILE_LOCK:
            # Double-check pattern to avoid building profile twice
            if DEFAULT_LANGUAGE_PROFILE is None:
                try:
                    DEFAULT_LANGUAGE_PROFILE = _build_ancient_greek_profile()
                except Exception:
                    DEFAULT_LANGUAGE_PROFILE = None
                    raise

    return DEFAULT_LANGUAGE_PROFILE


def register_default_profiles() -> None:
    """Register built-in language profiles explicitly."""
    register_language_profile(get_default_language_profile())


def _reset_language_registry_for_tests() -> None:
    """TEST-ONLY: Clear registered language profiles and trusted directory state for tests.

    Do not use in production code.
    """
    from .explainer import clear_trusted_external_rules_dirs
    from .distance import clear_trusted_external_matrix_dirs

    global DEFAULT_LANGUAGE_PROFILE

    with _REGISTRY_LOCK:
        _REGISTRY.clear()
    with _DEFAULT_PROFILE_LOCK:
        DEFAULT_LANGUAGE_PROFILE = None
    clear_trusted_external_rules_dirs()
    clear_trusted_external_matrix_dirs()


__all__ = [
    "LanguageProfile",
    "get_default_language_profile",
    "get_language_profile",
    "register_default_profiles",
    "register_language_profile",
]
