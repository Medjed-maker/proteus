"""Tests for phonology._paths."""

from pathlib import Path

import pytest

from phonology import _paths as paths_module
from phonology._paths import resolve_repo_data_dir


def test_resolve_repo_data_dir_returns_existing_repo_data_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    module_file = repo_root / "src" / "phonology" / "_paths.py"
    data_dir = repo_root / "data" / "rules"
    data_dir.mkdir(parents=True)
    (repo_root / "pyproject.toml").write_text(
        "[project]\nname = 'proteus'\n", encoding="utf-8"
    )
    monkeypatch.setattr(paths_module, "__file__", str(module_file))

    resolved = resolve_repo_data_dir("rules")

    assert resolved == data_dir
    assert resolved.exists()


@pytest.mark.parametrize(
    "subdirectory",
    ["", "../rules", "/abs/path", "nested/rules", ".", "..", "rules/../etc"],
)
def test_resolve_repo_data_dir_rejects_unsafe_subdirectory_values(
    subdirectory: str,
) -> None:
    with pytest.raises(
        ValueError, match="single relative path segment without traversal"
    ):
        resolve_repo_data_dir(subdirectory)


def test_resolve_repo_data_dir_rejects_none_subdirectory() -> None:
    with pytest.raises(TypeError, match="subdirectory must be a string"):
        resolve_repo_data_dir(None)  # type: ignore[arg-type]


def test_resolve_repo_data_dir_skips_repo_marker_without_existing_data_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True)
    module_file = repo_root / "src" / "phonology" / "_paths.py"
    repo_candidate = repo_root / "data" / "rules"
    (repo_root / "pyproject.toml").write_text(
        "[project]\nname = 'proteus'\n", encoding="utf-8"
    )
    monkeypatch.setattr(paths_module, "__file__", str(module_file))

    with pytest.raises(
        FileNotFoundError,
        match=r"subdirectory 'rules'.*module path",
    ):
        resolve_repo_data_dir("rules")

    assert not repo_candidate.exists()
