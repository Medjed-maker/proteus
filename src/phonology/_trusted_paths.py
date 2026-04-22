"""Private helpers for trusted runtime directory overrides.

These helpers reject obvious symlink usage for environment-provided override
paths, but the checks are not atomic and cannot fully eliminate TOCTOU races.
Callers operating in untrusted environments should prefer OS-level protections
such as ``O_NOFOLLOW``/file-descriptor-based validation, or verify ownership
and permissions immediately before use if they need stronger guarantees.
"""

from __future__ import annotations

import os
from pathlib import Path

TRUSTED_DIR_OVERRIDES_OPT_IN_ENV_VAR = "PROTEUS_ALLOW_TRUSTED_DIR_OVERRIDES"
_TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on"})


def trusted_dir_overrides_enabled() -> bool:
    """Return True when trusted directory overrides are explicitly enabled."""
    raw_value = os.environ.get(TRUSTED_DIR_OVERRIDES_OPT_IN_ENV_VAR)
    if raw_value is None:
        return False
    return raw_value.strip().lower() in _TRUTHY_ENV_VALUES


def _raise_if_symlink_present(*, candidate: Path, env_var: str, raw_override: str) -> None:
    """Reject override paths that currently contain symlinked components.

    This is a best-effort path walk and is subject to TOCTOU races between the
    symlink check and subsequent use. For stronger guarantees in untrusted
    environments, callers should use OS-level atomic checks such as opening
    with ``O_NOFOLLOW`` or validating the resulting file descriptor, and may
    also need to re-check ownership/permissions immediately before use.
    """
    for part in (candidate, *candidate.parents):
        if part == part.parent:
            break
        if part.is_symlink():
            raise ValueError(
                f"{env_var} must not contain a symlink, "
                f"got {raw_override} (symlink at {part})"
            )


def resolve_trusted_dir_override(*, env_var: str, description: str) -> Path | None:
    """Resolve and validate a trusted directory override from the environment.

    Returns ``None`` when ``env_var`` is unset. When it is set, the override is
    only honored if ``PROTEUS_ALLOW_TRUSTED_DIR_OVERRIDES`` is enabled.

    Raises:
        ValueError: When ``env_var`` value is empty or contains a symlinked path
            component, or when trusted directory overrides are not enabled.
        FileNotFoundError: When the resolved path does not exist.
        NotADirectoryError: When the resolved path exists but is not a directory.
    """
    raw_override = os.environ.get(env_var)
    if raw_override is not None:
        raw_override = raw_override.strip()
    if not raw_override:
        return None

    if not trusted_dir_overrides_enabled():
        raise ValueError(
            f"{env_var} requires {TRUSTED_DIR_OVERRIDES_OPT_IN_ENV_VAR}=1"
        )

    override_path = Path(raw_override).expanduser()
    _raise_if_symlink_present(
        candidate=override_path,
        env_var=env_var,
        raw_override=raw_override,
    )

    # Resolving and then checking still leaves a TOCTOU window before use; use
    # OS-level atomic primitives if stronger guarantees are required.
    resolved_path = override_path.resolve(strict=False)
    _raise_if_symlink_present(
        candidate=resolved_path,
        env_var=env_var,
        raw_override=raw_override,
    )
    if not resolved_path.exists():
        raise FileNotFoundError(f"Could not find {description} directory {resolved_path}")
    if not resolved_path.is_dir():
        raise NotADirectoryError(
            f"{description} path is not a directory: {resolved_path}"
        )
    return resolved_path
