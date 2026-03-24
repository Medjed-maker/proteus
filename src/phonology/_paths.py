"""Shared path utilities for locating repository data directories."""

from pathlib import Path


def resolve_repo_data_dir(subdirectory: str) -> Path:
    """Find a repository data directory without relying on fixed parent depth.

    Accepts only a single relative path segment for ``subdirectory``.
    Walks up from this module's location, first preferring an existing
    ``data/<subdirectory>`` beneath a parent containing ``pyproject.toml``.
    If no repo-rooted match exists, returns the first existing
    ``data/<subdirectory>`` directory it finds. If none exists, raises
    ``FileNotFoundError`` instead of returning a non-existent path.
    """
    if not isinstance(subdirectory, str):
        raise TypeError("subdirectory must be a string")

    path_subdirectory = Path(subdirectory)
    if (
        path_subdirectory.is_absolute()
        or not subdirectory.strip()
        or "/" in subdirectory
        or "\\" in subdirectory
        or len(path_subdirectory.parts) != 1
        or path_subdirectory.parts[0] in {"", ".", ".."}
    ):
        raise ValueError(
            "subdirectory must be a single relative path segment without traversal"
        )

    validated_subdirectory = path_subdirectory.parts[0]
    module_path = Path(__file__).resolve()

    for parent in module_path.parents:
        candidate = parent / "data" / validated_subdirectory
        if (parent / "pyproject.toml").is_file() and candidate.is_dir():
            return candidate

    for parent in module_path.parents:
        candidate = parent / "data" / validated_subdirectory
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        "Could not resolve an existing data directory for "
        f"subdirectory {validated_subdirectory!r} from module path {module_path}"
    )
