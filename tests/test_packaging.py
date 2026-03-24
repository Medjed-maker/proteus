"""Packaging regression tests for wheel runtime assets."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
EXPECTED_WHEEL_ASSETS = {
    "phonology/data/lexicon/greek_lemmas.json",
    "phonology/data/matrices/attic_doric.json",
    "phonology/data/rules/ancient_greek/consonant_changes.yaml",
    "phonology/data/rules/ancient_greek/vowel_shifts.yaml",
    "web/index.html",
    "web/static/styles.css",
}


def test_wheel_force_include_config_and_packaged_layout_support_runtime_loaders(
    tmp_path: Path,
) -> None:
    """Verify wheel force-include assets and packaged layout both satisfy runtime loaders."""
    pyproject = tomllib.loads((ROOT_DIR / "pyproject.toml").read_text(encoding="utf-8"))
    force_include = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    assert force_include == {
        "data/lexicon": "phonology/data/lexicon",
        "data/matrices": "phonology/data/matrices",
        "data/rules": "phonology/data/rules",
        "src/web/index.html": "web/index.html",
        "src/web/static": "web/static",
    }

    packaged_root = tmp_path / "packaged-root"
    shutil.copytree(ROOT_DIR / "src" / "api", packaged_root / "api")
    shutil.copytree(ROOT_DIR / "src" / "phonology", packaged_root / "phonology")

    for source, destination in force_include.items():
        source_path = ROOT_DIR / source
        destination_path = packaged_root / destination
        if source_path.is_dir():
            shutil.copytree(source_path, destination_path)
        else:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)

    for asset_path in EXPECTED_WHEEL_ASSETS:
        assert (packaged_root / asset_path).is_file()

    script = """
from pathlib import Path
import sys

packaged_root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(packaged_root))

import api.main as api_module
import phonology.distance as distance_module
import phonology.explainer as explainer_module

assert Path(api_module.__file__).resolve().is_relative_to(packaged_root)
assert Path(distance_module.__file__).resolve().is_relative_to(packaged_root)
assert Path(explainer_module.__file__).resolve().is_relative_to(packaged_root)

lexicon_entries = api_module.load_lexicon_entries()
matrix = distance_module.load_matrix("attic_doric.json")
rules = explainer_module.load_rules("ancient_greek")
legacy_matrix = distance_module.load_matrix("data/matrices/attic_doric.json")
legacy_rules = explainer_module.load_rules("data/rules/ancient_greek")

assert lexicon_entries
assert any(entry.get("headword") == "ἄνθρωπος" for entry in lexicon_entries)
assert matrix["a"]["a"] == 0.0
assert legacy_matrix["a"]["a"] == 0.0
assert "CCH-001" in rules
assert "VSH-001" in rules
assert "CCH-001" in legacy_rules
assert "VSH-001" in legacy_rules
print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(packaged_root)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"Subprocess failed with stderr:\n{result.stderr}"
    assert result.stdout.strip() == "ok"


def _create_fake_binary(path: Path, script_body: str) -> None:
    """Write *script_body* to *path* and mark it executable."""
    path.write_text(script_body, encoding="utf-8")
    path.chmod(0o755)


def _setup_css_build_project(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Set up a minimal fake project for testing ``build-css.sh``.

    Returns ``(project_dir, web_dir, bin_dir)`` paths.
    """
    project_dir = tmp_path / "project"
    scripts_dir = project_dir / "scripts"
    web_dir = project_dir / "src" / "web"
    bin_dir = project_dir / ".tailwind"

    scripts_dir.mkdir(parents=True)
    web_dir.mkdir(parents=True)
    bin_dir.mkdir(parents=True)

    shutil.copy2(ROOT_DIR / "scripts" / "build-css.sh", scripts_dir / "build-css.sh")
    (project_dir / "tailwind.config.js").write_text("module.exports = {};\n", encoding="utf-8")
    (web_dir / "input.css").write_text("@tailwind utilities;\n", encoding="utf-8")

    return project_dir, web_dir, bin_dir


def _create_fake_uname(directory: Path, real_uname_path: str) -> None:
    """Create a fake ``uname`` shim that reports Darwin/x86_64.

    The shim handles ``-s`` and ``-m`` as separate invocations, which
    matches how ``build-css.sh`` calls ``uname``.  Combined flags
    (e.g. ``uname -sm``) are delegated to the real binary.
    """
    quoted_real_uname_path = shlex.quote(real_uname_path)
    _create_fake_binary(
        directory / "uname",
        "#!/usr/bin/env bash\n"
        'result=""\n'
        'for arg in "$@"; do\n'
        '  case "$arg" in\n'
        '    -s) result="${result:+$result }Darwin" ;;\n'
        '    -m) result="${result:+$result }x86_64" ;;\n'
        f'    *)  exec {quoted_real_uname_path} "$@" ;;\n'
        "  esac\n"
        "done\n"
        'if [ -n "$result" ]; then\n'
        '  printf "%s\\n" "$result"\n'
        "else\n"
        f'  exec {quoted_real_uname_path} "$@"\n'
        "fi\n",
    )


def test_create_fake_uname_delegates_to_real_binary_with_spaces_in_path(
    tmp_path: Path,
) -> None:
    """Verify the uname shim safely delegates to a real binary in a spaced path."""
    real_bin_dir = tmp_path / "real bin"
    fake_bin_dir = tmp_path / "fake-bin"
    real_bin_dir.mkdir()
    fake_bin_dir.mkdir()

    real_uname_path = real_bin_dir / "uname"
    _create_fake_binary(
        real_uname_path,
        "#!/usr/bin/env bash\n"
        'printf "delegated:%s\\n" "$1"\n',
    )
    _create_fake_uname(fake_bin_dir, str(real_uname_path))

    result = subprocess.run(
        [str(fake_bin_dir / "uname"), "--kernel-name"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"Shim failed with stderr:\n{result.stderr}"
    assert result.stdout == "delegated:--kernel-name\n"


def test_build_css_script_uses_platform_specific_cached_binary(tmp_path: Path) -> None:
    """Verify the CSS build script ignores legacy shared cache names."""
    project_dir, web_dir, bin_dir = _setup_css_build_project(tmp_path)
    fake_bin_dir = tmp_path / "fake-bin"
    fake_bin_dir.mkdir()

    _create_fake_binary(
        bin_dir / "tailwindcss-v3.4.17",
        "#!/usr/bin/env bash\n"
        "echo legacy-binary-should-not-run >&2\n"
        "exit 99\n",
    )
    _create_fake_binary(
        bin_dir / "tailwindcss-v3.4.17-macos-x64",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'output=""\n'
        'while [ "$#" -gt 0 ]; do\n'
        '  if [ "$1" = "-o" ]; then\n'
        '    output="$2"\n'
        "    shift 2\n"
        "    continue\n"
        "  fi\n"
        "  shift\n"
        "done\n"
        '[ -n "$output" ]\n'
        'printf "generated by platform binary\\n" > "$output"\n',
    )

    real_uname = shutil.which("uname") or "/usr/bin/uname"
    _create_fake_uname(fake_bin_dir, real_uname)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}"

    result = subprocess.run(
        ["bash", str(project_dir / "scripts" / "build-css.sh")],
        cwd=project_dir,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, f"Script failed with stderr:\n{result.stderr}"
    assert "legacy-binary-should-not-run" not in result.stderr
    assert (web_dir / "static" / "styles.css").read_text(encoding="utf-8") == (
        "generated by platform binary\n"
    )
