"""Language profile registry for phonological search assets."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata
import os
from pathlib import Path
import threading
from typing import Literal, Protocol

from .corpus import CorpusAdapter
from .orthography_notes import OrthographicNoteBuilder
from .trusted_matrices import (
    clear_trusted_external_matrix_dirs,
    register_trusted_matrices_dir,
)


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
    status: Literal["pilot", "experimental", "stable"] = "experimental"
    description: str = ""
    vowel_phones: tuple[str, ...] = ()
    phone_matcher: Callable[[str, str], bool] | None = None
    dialect_skeleton_builders: tuple[Callable[[list[str]], list[str]], ...] = ()
    orthographic_note_builder: OrthographicNoteBuilder | None = None
    deprecated_orthography_hints: tuple[str, ...] = ()
    orthographic_data_preparer: Callable[[], None] | None = None
    corpus_adapter_factory: Callable[[], CorpusAdapter] | None = None
    always_match_contexts: tuple[str, ...] = ()


_REGISTRY: dict[str, LanguageProfile] = {}
_REGISTRY_LOCK = threading.Lock()
_DISCOVERY_LOCK = threading.RLock()

_LANGUAGE_ENTRY_POINT_GROUP = "proteus.languages"
_DEFAULT_LANGUAGE_ENV_VAR = "PROTEUS_DEFAULT_LANGUAGE"


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
        ``core.ports.trusted_matrices.register_trusted_matrices_dir``.
        Subsequent matrix and rule
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
    for hint in profile.deprecated_orthography_hints:
        normalized_hint = hint.strip().lower()
        if hint != normalized_hint or not normalized_hint:
            raise ValueError(
                "deprecated_orthography_hints must contain lowercase, "
                f"non-empty strings for {profile.language_id!r}"
            )

    from ...explainer import register_trusted_rules_dir
    register_trusted_rules_dir(profile.rules_dir)
    register_trusted_matrices_dir(profile.matrix_path.parent)

    with _REGISTRY_LOCK:
        _REGISTRY[language_id] = profile


def get_language_profile(language_id: str | None = None) -> LanguageProfile:
    """Return a registered language profile.

    Raises:
        ValueError: If no profile is registered for ``language_id``.
    """
    if language_id is None:
        return get_default_language_profile()

    _ensure_default_profiles_registered()
    return _get_language_profile_by_id(language_id)


def _get_language_profile_by_id(language_id: str) -> LanguageProfile:
    """Return a registered profile by id without default-profile fallback."""
    normalized = language_id.strip().lower()
    with _REGISTRY_LOCK:
        try:
            return _REGISTRY[normalized]
        except KeyError as exc:
            raise ValueError(f"Unsupported language profile: {language_id!r}") from exc


def list_language_profiles() -> tuple[LanguageProfile, ...]:
    """Return a sorted snapshot of the currently registered language profiles.

    The returned tuple is a snapshot: language profiles registered after this
    call are not reflected in the previously returned value.
    """
    with _REGISTRY_LOCK:
        profiles = tuple(_REGISTRY.values())

    return tuple(sorted(profiles, key=lambda profile: profile.language_id))


def _language_profile_entry_points() -> tuple[metadata.EntryPoint, ...]:
    """Return configured language profile entry points sorted by name."""
    return tuple(
        sorted(
            metadata.entry_points(group=_LANGUAGE_ENTRY_POINT_GROUP),
            key=lambda entry_point: entry_point.name,
        )
    )


def _load_language_profile_from_entry_point(
    entry_point: metadata.EntryPoint,
) -> LanguageProfile:
    """Load and build a language profile from an entry point."""
    try:
        build_profile = entry_point.load()
    except Exception as exc:
        raise RuntimeError(
            "Failed to load language profile entry point "
            f"{entry_point.name!r} from group {_LANGUAGE_ENTRY_POINT_GROUP!r}: {exc}"
        ) from exc

    try:
        profile = build_profile()
    except Exception as exc:
        raise RuntimeError(
            "Failed to build language profile from entry point "
            f"{entry_point.name!r} in group {_LANGUAGE_ENTRY_POINT_GROUP!r}: {exc}"
        ) from exc

    if not isinstance(profile, LanguageProfile):
        raise TypeError(
            "Language profile entry point "
            f"{entry_point.name!r} returned {type(profile).__name__}, "
            "expected LanguageProfile"
        )
    return profile


def register_default_profiles() -> None:
    """Discover and register packaged language profiles via entry points.

    Raises:
        ValueError: If no language profile entry points are configured.
        RuntimeError: If an entry point cannot be loaded or built.
        TypeError: If an entry point returns a non-``LanguageProfile`` object.
    """
    with _DISCOVERY_LOCK:
        entry_points = _language_profile_entry_points()
        if not entry_points:
            raise ValueError(
                "No language profile entry points registered for group "
                f"{_LANGUAGE_ENTRY_POINT_GROUP!r}"
            )
        for entry_point in entry_points:
            register_language_profile(
                _load_language_profile_from_entry_point(entry_point)
            )


def _ensure_default_profiles_registered() -> None:
    """Run entry point discovery when no profile is currently registered."""
    with _REGISTRY_LOCK:
        if _REGISTRY:
            return

    with _DISCOVERY_LOCK:
        with _REGISTRY_LOCK:
            if _REGISTRY:
                return
        register_default_profiles()


def _configured_default_language_id() -> str | None:
    """Return the configured default language id, if set."""
    configured = os.environ.get(_DEFAULT_LANGUAGE_ENV_VAR)
    if configured is None:
        return None
    normalized = configured.strip().lower()
    return normalized or None


def get_default_language_profile() -> LanguageProfile:
    """Return the configured or unambiguous default language profile.

    Raises:
        ValueError: If no profiles are registered, the configured default is
            missing, or multiple profiles are registered without an explicit
            ``PROTEUS_DEFAULT_LANGUAGE`` value.
    """
    _ensure_default_profiles_registered()

    configured_language_id = _configured_default_language_id()
    if configured_language_id is not None:
        try:
            # Use the non-recursive lookup here so a configured default id never
            # delegates back through get_default_language_profile().
            return _get_language_profile_by_id(configured_language_id)
        except ValueError as exc:
            raise ValueError(
                f"{_DEFAULT_LANGUAGE_ENV_VAR}={configured_language_id!r} does not "
                "match a registered language profile"
            ) from exc

    profiles = list_language_profiles()
    if len(profiles) == 1:
        return profiles[0]
    if not profiles:
        raise ValueError("No language profiles are registered")
    registered_ids = ", ".join(profile.language_id for profile in profiles)
    raise ValueError(
        "Multiple language profiles are registered "
        f"({registered_ids}); set {_DEFAULT_LANGUAGE_ENV_VAR} to choose a default"
    )


def _reset_language_registry_for_tests() -> None:
    """TEST-ONLY: Clear registered language profiles and trusted directory state for tests.

    Do not use in production code.
    """
    from ...explainer import clear_trusted_external_rules_dirs

    with _REGISTRY_LOCK:
        _REGISTRY.clear()
    clear_trusted_external_rules_dirs()
    clear_trusted_external_matrix_dirs()


__all__ = [
    "LanguageProfile",
    "get_default_language_profile",
    "get_language_profile",
    "list_language_profiles",
    "register_default_profiles",
    "register_language_profile",
]
