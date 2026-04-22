"""Tests for trusted directory override helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from phonology._trusted_paths import (
    TRUSTED_DIR_OVERRIDES_OPT_IN_ENV_VAR,
    resolve_trusted_dir_override,
    trusted_dir_overrides_enabled,
)

_TEST_ENV_VAR = "PROTEUS_TEST_TRUSTED_DIR"


@pytest.fixture(autouse=True)
def clear_trusted_override_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_TEST_ENV_VAR, raising=False)
    monkeypatch.delenv(TRUSTED_DIR_OVERRIDES_OPT_IN_ENV_VAR, raising=False)


@pytest.mark.parametrize("env_value", [" yes ", "1", "true", "TRUE", "on"])
def test_trusted_dir_overrides_enabled_accepts_truthy_string(
    monkeypatch: pytest.MonkeyPatch,
    env_value: str,
) -> None:
    monkeypatch.setenv(TRUSTED_DIR_OVERRIDES_OPT_IN_ENV_VAR, env_value)

    assert trusted_dir_overrides_enabled() is True


@pytest.mark.parametrize(
    ("env_value",),
    [
        (None,),
        ("0",),
        ("false",),
        ("",),
        ("FALSE",),
        ("no",),
        ("off",),
    ],
)
def test_trusted_dir_overrides_enabled_rejects_falsy_values(
    monkeypatch: pytest.MonkeyPatch,
    env_value: str | None,
) -> None:
    if env_value is None:
        monkeypatch.delenv(TRUSTED_DIR_OVERRIDES_OPT_IN_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(TRUSTED_DIR_OVERRIDES_OPT_IN_ENV_VAR, env_value)

    assert trusted_dir_overrides_enabled() is False


def test_resolve_trusted_dir_override_returns_none_when_unset() -> None:
    assert resolve_trusted_dir_override(
        env_var=_TEST_ENV_VAR,
        description="test data",
    ) is None


def test_resolve_trusted_dir_override_rejects_override_without_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    override_dir = tmp_path / "override"
    override_dir.mkdir()
    monkeypatch.setenv(_TEST_ENV_VAR, str(override_dir))

    with pytest.raises(
        ValueError,
        match=(
            f"{_TEST_ENV_VAR} requires "
            f"{TRUSTED_DIR_OVERRIDES_OPT_IN_ENV_VAR}=1"
        ),
    ):
        resolve_trusted_dir_override(env_var=_TEST_ENV_VAR, description="test data")


def test_resolve_trusted_dir_override_returns_resolved_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    override_dir = tmp_path / "override"
    override_dir.mkdir()
    monkeypatch.setenv(TRUSTED_DIR_OVERRIDES_OPT_IN_ENV_VAR, "1")
    monkeypatch.setenv(_TEST_ENV_VAR, str(override_dir))

    resolved = resolve_trusted_dir_override(
        env_var=_TEST_ENV_VAR,
        description="test data",
    )

    assert resolved == override_dir.resolve()


def test_resolve_trusted_dir_override_expands_tilde(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home_dir = tmp_path / "home"
    override_dir = home_dir / "trusted"
    override_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setenv(TRUSTED_DIR_OVERRIDES_OPT_IN_ENV_VAR, "1")
    monkeypatch.setenv(_TEST_ENV_VAR, "~/trusted")

    resolved = resolve_trusted_dir_override(
        env_var=_TEST_ENV_VAR,
        description="test data",
    )

    assert resolved == override_dir.resolve()


def test_resolve_trusted_dir_override_rejects_symlink_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    link_dir = tmp_path / "link"
    link_dir.symlink_to(real_dir)
    monkeypatch.setenv(TRUSTED_DIR_OVERRIDES_OPT_IN_ENV_VAR, "1")
    monkeypatch.setenv(_TEST_ENV_VAR, str(link_dir))

    with pytest.raises(ValueError, match="must not contain a symlink"):
        resolve_trusted_dir_override(env_var=_TEST_ENV_VAR, description="test data")


def test_resolve_trusted_dir_override_rejects_symlink_in_parent_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    (real_parent / "trusted").mkdir()
    link_parent = tmp_path / "link-parent"
    link_parent.symlink_to(real_parent)
    override_dir = link_parent / "trusted"
    monkeypatch.setenv(TRUSTED_DIR_OVERRIDES_OPT_IN_ENV_VAR, "1")
    monkeypatch.setenv(_TEST_ENV_VAR, str(override_dir))

    with pytest.raises(ValueError, match="must not contain a symlink"):
        resolve_trusted_dir_override(env_var=_TEST_ENV_VAR, description="test data")


def test_resolve_trusted_dir_override_reports_missing_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_dir = tmp_path / "missing"
    monkeypatch.setenv(TRUSTED_DIR_OVERRIDES_OPT_IN_ENV_VAR, "1")
    monkeypatch.setenv(_TEST_ENV_VAR, str(missing_dir))

    with pytest.raises(FileNotFoundError, match="Could not find test data directory"):
        resolve_trusted_dir_override(env_var=_TEST_ENV_VAR, description="test data")


def test_resolve_trusted_dir_override_rejects_file_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    override_file = tmp_path / "override.yaml"
    override_file.write_text("rules: []\n", encoding="utf-8")
    monkeypatch.setenv(TRUSTED_DIR_OVERRIDES_OPT_IN_ENV_VAR, "1")
    monkeypatch.setenv(_TEST_ENV_VAR, str(override_file))

    with pytest.raises(NotADirectoryError, match="test data path is not a directory"):
        resolve_trusted_dir_override(env_var=_TEST_ENV_VAR, description="test data")
