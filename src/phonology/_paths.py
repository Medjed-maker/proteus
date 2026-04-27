"""Shared path utilities for locating repository data directories."""

from pathlib import Path

DEFAULT_LANGUAGE_ID = "ancient_greek"


def _validate_single_segment(value: str, *, name: str) -> str:
    """Validate a single path segment used for internal data lookup."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")

    path_value = Path(value)
    if (
        path_value.is_absolute()
        or not value.strip()
        or "/" in value
        or "\\" in value
        or len(path_value.parts) != 1
        or path_value.parts[0] in {"", ".", ".."}
    ):
        raise ValueError(
            f"{name} must be a single relative path segment without traversal"
        )
    return path_value.parts[0]


def _iter_data_roots() -> list[Path]:
    """Return candidate data roots from repo and packaged layouts."""
    module_path = Path(__file__).resolve()
    roots: list[Path] = []
    for parent in module_path.parents:
        data_root = parent / "data"
        if (parent / "pyproject.toml").is_file() and data_root.is_dir():
            roots.append(data_root)
    for parent in module_path.parents:
        data_root = parent / "data"
        if data_root.is_dir() and data_root not in roots:
            roots.append(data_root)
    return roots


def resolve_language_data_dir(language_id: str, subdirectory: str) -> Path:
    """Find a language-scoped data directory.

    The canonical repo layout is ``data/languages/<language_id>/<subdirectory>``.
    For the default Ancient Greek profile, legacy ``data/<subdirectory>`` is
    also accepted when present so older development checkouts and generated
    temporary projects keep working.
    """
    validated_language_id = _validate_single_segment(language_id, name="language_id")
    validated_subdirectory = _validate_single_segment(
        subdirectory,
        name="subdirectory",
    )

    for data_root in _iter_data_roots():
        candidate = (
            data_root / "languages" / validated_language_id / validated_subdirectory
        )
        if candidate.is_dir():
            return candidate

    if validated_language_id == DEFAULT_LANGUAGE_ID:
        for data_root in _iter_data_roots():
            candidate = data_root / validated_subdirectory
            if candidate.is_dir():
                return candidate

    raise FileNotFoundError(
        "Could not resolve an existing data directory for "
        f"language {validated_language_id!r}, subdirectory {validated_subdirectory!r}"
    )


def resolve_repo_data_dir(subdirectory: str) -> Path:
    """Find a repository data directory without relying on fixed parent depth.

    Accepts only a single relative path segment for ``subdirectory``.
    Walks up from this module's location, first preferring an existing
    ``data/<subdirectory>`` beneath a parent containing ``pyproject.toml``.
    If no repo-rooted match exists, returns the first existing
    ``data/<subdirectory>`` directory it finds. If none exists, raises
    ``FileNotFoundError`` instead of returning a non-existent path.
    """
    validated_subdirectory = _validate_single_segment(
        subdirectory,
        name="subdirectory",
    )

    if validated_subdirectory in {"lexicon", "matrices", "rules"}:
        try:
            return resolve_language_data_dir(
                DEFAULT_LANGUAGE_ID, validated_subdirectory
            )
        except FileNotFoundError:
            pass

    for data_root in _iter_data_roots():
        candidate = data_root / validated_subdirectory
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        "Could not resolve an existing data directory for "
        f"subdirectory {validated_subdirectory!r} from module path {Path(__file__).resolve()}"
    )
