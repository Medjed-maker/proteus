"""Packaging regression tests for wheel runtime assets."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tomllib
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
EXPECTED_WHEEL_ASSETS = {
    "proteus/phonology/data/lexicon/greek_lemmas.json",
    "proteus/phonology/data/matrices/attic_doric.json",
    "proteus/phonology/data/rules/ancient_greek/consonant_changes.yaml",
    "proteus/phonology/data/rules/ancient_greek/vowel_shifts.yaml",
}


def test_wheel_force_include_config_and_packaged_layout_support_runtime_loaders(
    tmp_path: Path,
) -> None:
    """Verify wheel force-include assets and packaged layout both satisfy runtime loaders."""
    pyproject = tomllib.loads((ROOT_DIR / "pyproject.toml").read_text(encoding="utf-8"))
    force_include = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    assert force_include == {
        "data/lexicon": "proteus/phonology/data/lexicon",
        "data/matrices": "proteus/phonology/data/matrices",
        "data/rules": "proteus/phonology/data/rules",
    }

    packaged_root = tmp_path / "packaged-root"
    shutil.copytree(ROOT_DIR / "src" / "proteus", packaged_root / "proteus")

    for source, destination in force_include.items():
        shutil.copytree(
            ROOT_DIR / source,
            packaged_root / destination,
        )

    for asset_path in EXPECTED_WHEEL_ASSETS:
        assert (packaged_root / asset_path).is_file()

    script = """
from pathlib import Path
import sys

packaged_root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(packaged_root))

import proteus.api.main as api_module
import proteus.phonology.distance as distance_module
import proteus.phonology.explainer as explainer_module

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
