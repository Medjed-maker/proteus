"""Packaging regression tests for wheel runtime assets."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import tomllib
import zipfile
import importlib
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from phonology import build_lexicon


ROOT_DIR = Path(__file__).resolve().parents[1]
EXPECTED_WHEEL_ASSETS = {
    "phonology/data/languages/ancient_greek/lexicon/greek_lemmas.json",
    "phonology/data/languages/ancient_greek/matrices/attic_doric.json",
    "phonology/data/languages/ancient_greek/rules/consonant_changes.yaml",
    "phonology/data/languages/ancient_greek/rules/buck/dialects.yaml",
    "phonology/data/languages/ancient_greek/rules/buck/glossary.yaml",
    "phonology/data/languages/ancient_greek/rules/buck/grammar_rules.yaml",
    "phonology/data/languages/ancient_greek/rules/morphophonemic_alternations.yaml",
    "phonology/data/languages/ancient_greek/rules/vowel_shifts.yaml",
    "web/changelog.html",
    "web/index.html",
    "web/static/styles.css",
    "web/static/translations.json",
}

EXPECTED_BUCK_RULE_ASSETS = {
    asset for asset in EXPECTED_WHEEL_ASSETS
    if asset.startswith("phonology/data/languages/ancient_greek/rules/buck/")
}


def _minimal_lexicon_document() -> dict[str, object]:
    """Return a minimal lexicon document that satisfies runtime loaders."""
    return {
        "schema_version": "2.0.0",
        "_meta": {
            "source": "LSJ (test fixture)",
            "encoding": "Unicode NFC",
            "ipa_system": "scholarly Ancient Greek IPA",
            "dialect": "attic",
            "version": "test",
            "last_updated": "2026-03-29T00:00:00Z",
            "license": "CC-BY-SA-4.0",
            "contributors": ["Proteus maintainers"],
            "data_schema_ref": "data/languages/ancient_greek/lexicon/greek_lemmas.schema.json",
            "description": "Packaging test fixture.",
            "note": "Generated during packaging tests.",
        },
        "lemmas": [
            {
                "id": "LSJ-000001",
                "headword": "ἄνθρωπος",
                "transliteration": "anthrōpos",
                "ipa": "ántʰrɔːpos",
                "pos": "noun",
                "gender": "common",
                "gloss": "person",
                "dialect": "attic",
            }
        ],
    }


def _write_fake_lsj_checkout(project_dir: Path) -> None:
    """Write a tiny LSJ checkout that the build hook can extract from."""
    _write_fake_lsj_checkout_at(
        project_dir / "data" / "external" / "lsj",
        key="a)/nqrwpos",
        orth="a)/nqrwpos",
        gen="h(",
        gloss="person",
    )


def _write_fake_lsj_checkout_at(
    repo_dir: Path,
    *,
    key: str,
    orth: str,
    gen: str,
    gloss: str,
) -> None:
    """Write a tiny LSJ checkout rooted at ``repo_dir``."""
    xml_dir = (
        repo_dir
        / "CTS_XML_TEI"
        / "perseus"
        / "pdllex"
        / "grc"
        / "lsj"
    )
    xml_dir.mkdir(parents=True, exist_ok=True)
    (xml_dir / "grc.lsj.perseus-eng1.xml").write_text(
        "<TEI><text><body>"
        f'<entryFree id="n1" key="{key}" type="main">'
        f'<orth extent="full" lang="greek">{orth}</orth>'
        f'<gen lang="greek">{gen}</gen>'
        f'<sense id="s1" n="A" level="1"><tr>{gloss}</tr></sense>'
        "</entryFree>"
        "</body></text></TEI>\n",
        encoding="utf-8",
    )


def _metadata_path_for_lexicon(lexicon_path: Path) -> Path:
    """Return the sidecar metadata path used for a generated lexicon."""
    return lexicon_path.with_name(f"{lexicon_path.stem}.meta.json")


def _uv_build_env(tmp_path: Path) -> dict[str, str]:
    """Return an environment for isolated uv builds inside tests."""
    env = os.environ.copy()
    env["UV_CACHE_DIR"] = str(tmp_path / "uv-cache")
    env["UV_PYTHON"] = sys.executable
    return env


def _uv_executable() -> str:
    """Return the absolute path to the uv executable used by packaging tests."""
    uv_executable = shutil.which("uv")
    if uv_executable is None:
        raise FileNotFoundError("uv CLI not available")
    return uv_executable


def _skip_known_uv_build_instability(result: subprocess.CompletedProcess[str]) -> None:
    """Skip tests when uv itself crashes in this runtime."""
    stderr = result.stderr.strip()
    if (
        "Attempted to create a NULL object." in stderr
        or "Tokio executor failed" in stderr
        or "system-configuration" in stderr
    ):
        pytest.skip(f"uv build is unstable in this runtime: {stderr}")


def _assert_uv_build_succeeded_or_skip(result: subprocess.CompletedProcess[str]) -> None:
    """Assert ``uv build`` succeeded, skipping known environment-specific crashes."""
    if result.returncode == 0:
        return
    _skip_known_uv_build_instability(result)
    assert result.returncode == 0, f"uv build failed with stderr:\n{result.stderr}"


def _assert_uv_build_failed_with_message_or_skip(
    result: subprocess.CompletedProcess[str],
    message: str,
) -> None:
    """Assert ``uv build`` failed for the expected reason, or skip uv runtime crashes."""
    _skip_known_uv_build_instability(result)
    if result.returncode == 0:
        pytest.fail(
            f"uv build unexpectedly succeeded (returncode=0)\n"
            f"Expected stderr to contain: {message!r}\n"
            f"Actual stderr:\n{result.stderr}"
        )
    if message not in result.stderr:
        pytest.fail(
            f"Expected message not found in stderr\n"
            f"Expected: {message!r}\n"
            f"Returncode: {result.returncode}\n"
            f"Actual stderr:\n{result.stderr}"
        )


def _copy_build_project(tmp_path: Path) -> Path:
    """Copy the minimum files required for an isolated wheel build test."""
    # Validate required directories exist
    required_dirs = [
        ROOT_DIR / "src",
        ROOT_DIR / "data" / "languages" / "ancient_greek" / "lexicon",
        ROOT_DIR / "data" / "languages" / "ancient_greek" / "matrices",
        ROOT_DIR / "data" / "languages" / "ancient_greek" / "rules",
        ROOT_DIR / "scripts",
    ]
    for dir_path in required_dirs:
        if not dir_path.is_dir():
            raise FileNotFoundError(
                f"_copy_build_project: required directory not found: {dir_path}"
            )

    # Validate required files exist
    required_files = [
        ROOT_DIR / ".gitignore",
        ROOT_DIR / "DATA_LICENSE.md",
        ROOT_DIR / "LICENSE",
        ROOT_DIR / "NOTICE",
        ROOT_DIR / "pyproject.toml",
        ROOT_DIR / "README.md",
        ROOT_DIR / "hatch_build.py",
    ]
    for file_path in required_files:
        if not file_path.is_file():
            raise FileNotFoundError(
                f"_copy_build_project: required file not found: {file_path}"
            )

    project_dir = tmp_path / "project"
    shutil.copytree(ROOT_DIR / "src", project_dir / "src")
    shutil.copytree(
        ROOT_DIR / "data" / "languages",
        project_dir / "data" / "languages",
    )
    shutil.copytree(ROOT_DIR / "scripts", project_dir / "scripts")
    # Create minimal .git directory so build hooks detect a git repository
    (project_dir / ".git").mkdir()
    shutil.copy2(ROOT_DIR / ".gitignore", project_dir / ".gitignore")
    shutil.copy2(ROOT_DIR / "DATA_LICENSE.md", project_dir / "DATA_LICENSE.md")
    shutil.copy2(ROOT_DIR / "LICENSE", project_dir / "LICENSE")
    shutil.copy2(ROOT_DIR / "NOTICE", project_dir / "NOTICE")
    shutil.copy2(ROOT_DIR / "pyproject.toml", project_dir / "pyproject.toml")
    shutil.copy2(ROOT_DIR / "README.md", project_dir / "README.md")
    shutil.copy2(ROOT_DIR / "hatch_build.py", project_dir / "hatch_build.py")
    assert not (project_dir / "data" / "external").exists()
    return project_dir


def _load_ci_workflow() -> dict[str, object]:
    """Return the parsed CI workflow document."""
    workflow_path = ROOT_DIR / ".github" / "workflows" / "ci.yml"
    return yaml.safe_load(workflow_path.read_text(encoding="utf-8"))


def _stash_phonology_modules() -> dict[str, ModuleType | None]:
    """Remove and return cached phonology modules for temporary isolation."""
    stashed_modules: dict[str, ModuleType | None] = {}
    for module_name in list(sys.modules):
        if module_name == "phonology" or module_name.startswith("phonology."):
            stashed_modules[module_name] = sys.modules.pop(module_name, None)
    return stashed_modules


def _restore_phonology_modules(
    stashed_modules: dict[str, ModuleType | None],
) -> None:
    """Restore previously stashed phonology modules after a temporary isolation."""
    for module_name in list(sys.modules):
        if module_name == "phonology" or module_name.startswith("phonology."):
            sys.modules.pop(module_name, None)
    sys.modules.update(stashed_modules)


def _import_hatch_build_module() -> ModuleType:
    """Import hatch_build with a lightweight hatchling interface stub."""
    interface_module = ModuleType("hatchling.builders.hooks.plugin.interface")

    class DummyBuildHookInterface:
        """Minimal stand-in used by packaging tests."""

    interface_module.BuildHookInterface = DummyBuildHookInterface

    plugin_module = ModuleType("hatchling.builders.hooks.plugin")
    plugin_module.interface = interface_module
    hooks_module = ModuleType("hatchling.builders.hooks")
    hooks_module.plugin = plugin_module
    builders_module = ModuleType("hatchling.builders")
    builders_module.hooks = hooks_module
    hatchling_module = ModuleType("hatchling")
    hatchling_module.builders = builders_module

    sys.modules["hatchling"] = hatchling_module
    sys.modules["hatchling.builders"] = builders_module
    sys.modules["hatchling.builders.hooks"] = hooks_module
    sys.modules["hatchling.builders.hooks.plugin"] = plugin_module
    sys.modules["hatchling.builders.hooks.plugin.interface"] = interface_module
    sys.modules.pop("hatch_build", None)
    return importlib.import_module("hatch_build")


def _extract_sdist(sdist_path: Path, destination_dir: Path) -> Path:
    """Extract a source distribution and return the single project root."""
    with tarfile.open(sdist_path) as archive:
        if sys.version_info >= (3, 11, 4):
            archive.extractall(destination_dir, filter="data")
        else:
            # Validate all members are within destination_dir (CVE-2007-4559)
            destination_resolved = destination_dir.resolve()
            for member in archive.getmembers():
                member_path = destination_dir / member.name
                if not member_path.resolve().is_relative_to(destination_resolved):
                    raise ValueError(f"Attempted path traversal in tar: {member.name}")
                # Also validate symlink/hardlink targets stay within destination_dir
                if member.issym() or member.islnk():
                    linkname = member.linkname
                    # Absolute linknames resolve as-is; relative linknames resolve
                    # from the member's parent directory. Both must stay within
                    # destination_resolved after resolution.
                    if Path(linkname).is_absolute():
                        target_path = Path(linkname).resolve()
                    else:
                        target_path = (member_path.parent / linkname).resolve()
                    if not target_path.is_relative_to(destination_resolved):
                        raise ValueError(
                            f"Tar member {member.name} links to target outside destination: {linkname}"
                        )
            archive.extractall(destination_dir)
        root_names = {name.split("/", 1)[0] for name in archive.getnames() if name}

    assert len(root_names) == 1, f"expected one sdist root directory, got: {sorted(root_names)}"
    return destination_dir / next(iter(root_names))


def test_wheel_force_include_config_and_packaged_layout_support_runtime_loaders(
    tmp_path: Path,
) -> None:
    """Verify wheel force-include assets and packaged layout both satisfy runtime loaders."""
    pyproject = tomllib.loads((ROOT_DIR / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["license-files"] == [
        "LICENSE",
        "NOTICE",
        "DATA_LICENSE.md",
    ]
    force_include = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    assert force_include == {
        "data/languages/ancient_greek/lexicon/greek_lemmas.json": "phonology/data/languages/ancient_greek/lexicon/greek_lemmas.json",
        "data/languages/ancient_greek/lexicon/greek_lemmas.schema.json": "phonology/data/languages/ancient_greek/lexicon/greek_lemmas.schema.json",
        "data/languages/ancient_greek/lexicon/pos_overrides.yaml": "phonology/data/languages/ancient_greek/lexicon/pos_overrides.yaml",
        "data/languages/ancient_greek/matrices": "phonology/data/languages/ancient_greek/matrices",
        "data/languages/ancient_greek/rules": "phonology/data/languages/ancient_greek/rules",
        "src/web/index.html": "web/index.html",
        "src/web/changelog.html": "web/changelog.html",
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

    lexicon_output = packaged_root / "phonology" / "data" / "languages" / "ancient_greek" / "lexicon" / "greek_lemmas.json"
    if not lexicon_output.is_file():
        lexicon_output.write_text(
            json.dumps(_minimal_lexicon_document(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    for asset_path in EXPECTED_WHEEL_ASSETS:
        assert (packaged_root / asset_path).is_file()

    script = """
from pathlib import Path
import sys

packaged_root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(packaged_root))

import api.main as api_module
import phonology.buck as buck_module
import phonology.distance as distance_module
import phonology.explainer as explainer_module

assert Path(api_module.__file__).resolve().is_relative_to(packaged_root)
assert Path(buck_module.__file__).resolve().is_relative_to(packaged_root)
assert Path(distance_module.__file__).resolve().is_relative_to(packaged_root)
assert Path(explainer_module.__file__).resolve().is_relative_to(packaged_root)

lexicon_entries = api_module.load_lexicon_entries()
buck_data = buck_module.load_buck_data()
matrix = distance_module.load_matrix("attic_doric.json")
rules = explainer_module.load_rules("ancient_greek")
legacy_matrix = distance_module.load_matrix("data/languages/ancient_greek/matrices/attic_doric.json")
legacy_rules = explainer_module.load_rules("data/languages/ancient_greek/rules")

assert lexicon_entries
assert any(entry.get("headword") == "ἄνθρωπος" for entry in lexicon_entries)
assert set(buck_data) == {"grammar_rules", "dialects", "glossary"}
assert any(rule.get("id") == "grc_synt_175" for rule in buck_data["grammar_rules"]["rules"])
assert matrix["a"]["a"] == 0.0
assert legacy_matrix["a"]["a"] == 0.0
assert "CCH-001" in rules
assert "MPH-001" in rules
assert "VSH-001" in rules
assert "CCH-001" in legacy_rules
assert "MPH-001" in legacy_rules
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


def test_sdist_force_include_config_bundles_generated_lexicon_and_metadata() -> None:
    """Verify sdist packaging includes generated lexicon assets and uses the build hook."""
    pyproject = tomllib.loads((ROOT_DIR / "pyproject.toml").read_text(encoding="utf-8"))
    sdist_config = pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]

    assert sdist_config["exclude"] == ["data/external/"]
    assert sdist_config["force-include"] == {
        "data/languages/ancient_greek/lexicon/greek_lemmas.json": "data/languages/ancient_greek/lexicon/greek_lemmas.json",
        "data/languages/ancient_greek/lexicon/greek_lemmas.meta.json": "data/languages/ancient_greek/lexicon/greek_lemmas.meta.json",
    }
    assert sdist_config["hooks"]["custom"]["path"] == "hatch_build.py"


def test_build_hook_prefers_project_src_over_preexisting_phonology_package(
    tmp_path: Path,
) -> None:
    """Verify the build hook imports build helpers from its own project root."""
    project_root = tmp_path / "project"
    project_src = project_root / "src" / "phonology"
    project_src.mkdir(parents=True)
    marker_path = tmp_path / "project-called.json"
    (project_src / "__init__.py").write_text("", encoding="utf-8")
    (project_src / "build_lexicon.py").write_text(
        "import json\n"
        "from pathlib import Path\n\n"
        "def ensure_generated_lexicon(**kwargs):\n"
        f"    Path({str(marker_path)!r}).write_text(json.dumps(kwargs, default=str), encoding='utf-8')\n",
        encoding="utf-8",
    )

    external_root = tmp_path / "external"
    external_pkg = external_root / "phonology"
    external_pkg.mkdir(parents=True)
    (external_pkg / "__init__.py").write_text("", encoding="utf-8")
    (external_pkg / "build_lexicon.py").write_text(
        "def ensure_generated_lexicon(**kwargs):\n"
        "    raise AssertionError('imported external phonology.build_lexicon')\n",
        encoding="utf-8",
    )

    stashed_phonology_modules = _stash_phonology_modules()
    try:
        sys.path.insert(0, str(external_root))
        hatch_build_module = _import_hatch_build_module()
        hook = hatch_build_module.build_hook.__new__(hatch_build_module.build_hook)
        hook.root = str(project_root)

        hook.initialize("standard", {})

        payload = json.loads(marker_path.read_text(encoding="utf-8"))
        assert payload == {
            "project_root": str(project_root),
            "skip_if_present": True,
            "allow_clone": False,
        }
        assert str(project_root / "src") not in sys.path
        imported = importlib.import_module("phonology.build_lexicon")
        assert Path(imported.__file__ or "").resolve() == (
            project_src / "build_lexicon.py"
        ).resolve()
    finally:
        while str(external_root) in sys.path:
            sys.path.remove(str(external_root))
        _restore_phonology_modules(stashed_phonology_modules)
        sys.modules.pop("hatch_build", None)
        for module_name in (
            "hatchling",
            "hatchling.builders",
            "hatchling.builders.hooks",
            "hatchling.builders.hooks.plugin",
            "hatchling.builders.hooks.plugin.interface",
        ):
            sys.modules.pop(module_name, None)


def test_build_hook_skips_lexicon_generation_for_editable_builds(tmp_path: Path) -> None:
    """Verify editable builds bypass the lexicon generation hook."""
    project_root = tmp_path / "project"
    project_src = project_root / "src" / "phonology"
    project_src.mkdir(parents=True)
    marker_path = tmp_path / "editable-called.txt"
    (project_src / "__init__.py").write_text("", encoding="utf-8")
    (project_src / "build_lexicon.py").write_text(
        "from pathlib import Path\n\n"
        "def ensure_generated_lexicon(**kwargs):\n"
        f"    Path({str(marker_path)!r}).write_text('called', encoding='utf-8')\n",
        encoding="utf-8",
    )

    stashed_phonology_modules = _stash_phonology_modules()
    try:
        hatch_build_module = _import_hatch_build_module()
        hook = hatch_build_module.build_hook.__new__(hatch_build_module.build_hook)
        hook.root = str(project_root)
        matrices_source = str((project_root / "data" / "languages" / "ancient_greek" / "matrices").resolve())
        hook.build_config = type(
            "DummyBuildConfig",
            (),
            {
                "get_force_include": staticmethod(
                    lambda: {
                        str((project_root / "data" / "languages" / "ancient_greek" / "lexicon" / "greek_lemmas.json").resolve()):
                        "phonology/data/languages/ancient_greek/lexicon/greek_lemmas.json",
                        matrices_source: "phonology/data/languages/ancient_greek/matrices",
                    }
                )
            },
        )()
        build_data: dict[str, object] = {}

        hook.initialize("editable", build_data)

        assert not marker_path.exists()
        assert str(project_root / "src") not in sys.path
        assert build_data == {
            "force_include_editable": {
                matrices_source: "phonology/data/languages/ancient_greek/matrices",
            }
        }
    finally:
        _restore_phonology_modules(stashed_phonology_modules)
        sys.modules.pop("hatch_build", None)
        for module_name in (
            "hatchling",
            "hatchling.builders",
            "hatchling.builders.hooks",
            "hatchling.builders.hooks.plugin",
            "hatchling.builders.hooks.plugin.interface",
        ):
            sys.modules.pop(module_name, None)


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv CLI not available")
def test_uv_build_generates_missing_lexicon_for_sdist(tmp_path: Path) -> None:
    """Verify sdist builds generate and bundle lexicon payload plus metadata."""
    project_dir = _copy_build_project(tmp_path)
    sdist_dir = tmp_path / "dist-sdist"
    sdist_dir.mkdir()
    lexicon_path = project_dir / "data" / "languages" / "ancient_greek" / "lexicon" / "greek_lemmas.json"
    metadata_path = _metadata_path_for_lexicon(lexicon_path)

    if lexicon_path.exists():
        lexicon_path.unlink()
    if metadata_path.exists():
        metadata_path.unlink()
    _write_fake_lsj_checkout(project_dir)

    result = subprocess.run(
        [_uv_executable(), "build", "--sdist", "--out-dir", str(sdist_dir)],
        cwd=project_dir,
        check=False,
        capture_output=True,
        env=_uv_build_env(tmp_path),
        text=True,
    )

    _assert_uv_build_succeeded_or_skip(result)
    sdist_files = sorted(sdist_dir.glob("*.tar.gz"))
    assert sdist_files, "uv build did not produce an sdist"

    with tarfile.open(sdist_files[0]) as archive:
        names = archive.getnames()
        lexicon_members = [
            name for name in names if name.endswith("data/languages/ancient_greek/lexicon/greek_lemmas.json")
        ]
        metadata_members = [
            name for name in names if name.endswith("data/languages/ancient_greek/lexicon/greek_lemmas.meta.json")
        ]
        assert lexicon_members, "sdist does not contain greek_lemmas.json"
        assert metadata_members, "sdist does not contain greek_lemmas.meta.json"
        lexicon_member = lexicon_members[0]
        metadata_member = metadata_members[0]
        lexicon_file = archive.extractfile(lexicon_member)
        if lexicon_file is None:
            pytest.fail(f"extractfile returned None for {lexicon_member}")
        lexicon_document = json.loads(lexicon_file.read().decode("utf-8"))
        metadata_file = archive.extractfile(metadata_member)
        if metadata_file is None:
            pytest.fail(f"extractfile returned None for {metadata_member}")
        metadata_document = json.loads(metadata_file.read().decode("utf-8"))

        for asset_path in EXPECTED_BUCK_RULE_ASSETS:
            assert any(name.endswith(asset_path.removeprefix("phonology/")) for name in names), (
                f"sdist does not contain {asset_path}"
            )

    assert any(entry.get("headword") == "ἄνθρωπος" for entry in lexicon_document["lemmas"])
    assert metadata_document["schema_version"] == build_lexicon.FINGERPRINT_SCHEMA_VERSION


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv CLI not available")
def test_uv_build_generates_missing_lexicon_for_wheel(tmp_path: Path) -> None:
    """Verify wheel builds succeed from a source tree without a tracked lexicon JSON."""
    project_dir = _copy_build_project(tmp_path)
    wheel_dir = tmp_path / "dist"
    wheel_dir.mkdir()

    lexicon_path = project_dir / "data" / "languages" / "ancient_greek" / "lexicon" / "greek_lemmas.json"
    if lexicon_path.exists():
        lexicon_path.unlink()
    _write_fake_lsj_checkout(project_dir)

    result = subprocess.run(
        [_uv_executable(), "build", "--wheel", "--out-dir", str(wheel_dir)],
        cwd=project_dir,
        check=False,
        capture_output=True,
        env=_uv_build_env(tmp_path),
        text=True,
    )

    _assert_uv_build_succeeded_or_skip(result)
    wheel_files = sorted(wheel_dir.glob("*.whl"))
    assert wheel_files, "uv build did not produce a wheel"

    with zipfile.ZipFile(wheel_files[0]) as wheel_zip:
        asset_names = wheel_zip.namelist()
        for asset_path in EXPECTED_BUCK_RULE_ASSETS:
            assert asset_path in asset_names, f"Missing expected asset in wheel: {asset_path}"
        expected_license_assets = (
            ".dist-info/licenses/DATA_LICENSE.md",
            ".dist-info/licenses/LICENSE",
            ".dist-info/licenses/NOTICE",
        )
        assert all(
            any(name.endswith(license_asset) for name in asset_names)
            for license_asset in expected_license_assets
        )
        assert "phonology/data/languages/ancient_greek/lexicon/greek_lemmas.json" in asset_names
        assert "phonology/data/languages/ancient_greek/lexicon/greek_lemmas.meta.json" not in asset_names
        lexicon_document = json.loads(
            wheel_zip.read("phonology/data/languages/ancient_greek/lexicon/greek_lemmas.json").decode("utf-8")
        )

    assert any(entry.get("headword") == "ἄνθρωπος" for entry in lexicon_document["lemmas"])


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv CLI not available")
def test_uv_build_replaces_stale_existing_lexicon_for_wheel(tmp_path: Path) -> None:
    """Verify stale local lexicon data is regenerated before wheel packaging."""
    project_dir = _copy_build_project(tmp_path)
    wheel_dir = tmp_path / "dist-stale"
    wheel_dir.mkdir()
    lexicon_path = project_dir / "data" / "languages" / "ancient_greek" / "lexicon" / "greek_lemmas.json"
    metadata_path = _metadata_path_for_lexicon(lexicon_path)
    stale_document = _minimal_lexicon_document()
    stale_document["lemmas"] = [
        {
            "id": "LSJ-999999",
            "headword": "παλαιός",
            "transliteration": "palaios",
            "ipa": "palaios",
            "pos": "adjective",
            "gender": "common",
            "gloss": "stale fixture",
            "dialect": "attic",
        }
    ]
    lexicon_path.write_text(
        json.dumps(stale_document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": build_lexicon.FINGERPRINT_SCHEMA_VERSION,
                "fingerprint": "stale",
                "inputs": [],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_fake_lsj_checkout(project_dir)

    result = subprocess.run(
        [_uv_executable(), "build", "--wheel", "--out-dir", str(wheel_dir)],
        cwd=project_dir,
        check=False,
        capture_output=True,
        env=_uv_build_env(tmp_path),
        text=True,
    )

    _assert_uv_build_succeeded_or_skip(result)
    wheel_files = sorted(wheel_dir.glob("*.whl"))
    assert wheel_files, "uv build did not produce a wheel"

    with zipfile.ZipFile(wheel_files[0]) as wheel_zip:
        lexicon_document = json.loads(
            wheel_zip.read("phonology/data/languages/ancient_greek/lexicon/greek_lemmas.json").decode("utf-8")
        )
        wheel_headwords = {entry["headword"] for entry in lexicon_document["lemmas"]}
        assert "phonology/data/languages/ancient_greek/lexicon/greek_lemmas.meta.json" not in wheel_zip.namelist()

    assert "ἄνθρωπος" in wheel_headwords
    assert "παλαιός" not in wheel_headwords


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv CLI not available")
def test_uv_build_reuses_fresh_existing_lexicon_for_wheel(tmp_path: Path) -> None:
    """Verify a fresh local lexicon is reused instead of being regenerated."""
    project_dir = _copy_build_project(tmp_path)
    wheel_dir = tmp_path / "dist-fresh"
    wheel_dir.mkdir()
    lexicon_path = project_dir / "data" / "languages" / "ancient_greek" / "lexicon" / "greek_lemmas.json"
    xml_dir = (
        project_dir / "data" / "external" / "lsj" / "CTS_XML_TEI" / "perseus" / "pdllex" / "grc" / "lsj"
    )
    _write_fake_lsj_checkout(project_dir)
    fresh_document = _minimal_lexicon_document()
    fresh_document["lemmas"] = [
        {
            "id": "LSJ-123456",
            "headword": "ἕτοιμος",
            "transliteration": "hetoimos",
            "ipa": "hétoimos",
            "pos": "adjective",
            "gender": "common",
            "gloss": "fresh fixture",
            "dialect": "attic",
        }
    ]
    lexicon_path.write_text(
        json.dumps(fresh_document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    payload = build_lexicon.build_fingerprint_payload(project_dir, xml_dir)
    _metadata_path_for_lexicon(lexicon_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [_uv_executable(), "build", "--wheel", "--out-dir", str(wheel_dir)],
        cwd=project_dir,
        check=False,
        capture_output=True,
        env=_uv_build_env(tmp_path),
        text=True,
    )

    _assert_uv_build_succeeded_or_skip(result)
    wheel_files = sorted(wheel_dir.glob("*.whl"))
    assert wheel_files, "uv build did not produce a wheel"

    with zipfile.ZipFile(wheel_files[0]) as wheel_zip:
        lexicon_document = json.loads(
            wheel_zip.read("phonology/data/languages/ancient_greek/lexicon/greek_lemmas.json").decode("utf-8")
        )
        wheel_headwords = {entry["headword"] for entry in lexicon_document["lemmas"]}
        assert "phonology/data/languages/ancient_greek/lexicon/greek_lemmas.meta.json" not in wheel_zip.namelist()

    assert "ἕτοιμος" in wheel_headwords
    assert "ἄνθρωπος" not in wheel_headwords


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv CLI not available")
def test_uv_build_reuses_fresh_existing_lexicon_without_git_or_lsj_checkout(
    tmp_path: Path,
) -> None:
    """Verify fresh lexicon reuse works without git or a local LSJ checkout."""
    project_dir = _copy_build_project(tmp_path)
    wheel_dir = tmp_path / "dist-fresh-offline"
    wheel_dir.mkdir()
    lexicon_path = project_dir / "data" / "languages" / "ancient_greek" / "lexicon" / "greek_lemmas.json"
    xml_dir = build_lexicon.default_xml_dir(project_dir)
    fake_bin_dir = tmp_path / "fake-bin"
    fake_bin_dir.mkdir()

    fresh_document = _minimal_lexicon_document()
    fresh_document["lemmas"] = [
        {
            "id": "LSJ-123457",
            "headword": "ἕτοιμος",
            "transliteration": "hetoimos",
            "ipa": "hétoimos",
            "pos": "adjective",
            "gender": "common",
            "gloss": "fresh offline fixture",
            "dialect": "attic",
        }
    ]
    lexicon_path.write_text(
        json.dumps(fresh_document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_fake_lsj_checkout(project_dir)
    payload = build_lexicon.build_fingerprint_payload(project_dir, xml_dir)
    shutil.rmtree(project_dir / "data" / "external")
    _metadata_path_for_lexicon(lexicon_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    env = _uv_build_env(tmp_path)
    # Intentionally isolate PATH to only fake_bin_dir to simulate an environment
    # without git/LSJ. Another test later uses PATH appending to simulate presence.
    env["PATH"] = str(fake_bin_dir)

    result = subprocess.run(
        [_uv_executable(), "build", "--wheel", "--out-dir", str(wheel_dir)],
        cwd=project_dir,
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    _assert_uv_build_succeeded_or_skip(result)
    wheel_files = sorted(wheel_dir.glob("*.whl"))
    assert wheel_files, "uv build did not produce a wheel"

    with zipfile.ZipFile(wheel_files[0]) as wheel_zip:
        lexicon_document = json.loads(
            wheel_zip.read("phonology/data/languages/ancient_greek/lexicon/greek_lemmas.json").decode("utf-8")
        )
        assert "phonology/data/languages/ancient_greek/lexicon/greek_lemmas.meta.json" not in wheel_zip.namelist()
        wheel_headwords = {entry["headword"] for entry in lexicon_document["lemmas"]}

    assert "ἕτοιμος" in wheel_headwords
    assert "ἄνθρωπος" not in wheel_headwords


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv CLI not available")
def test_uv_build_env_override_regenerates_instead_of_reusing_offline_shortcut(
    tmp_path: Path,
) -> None:
    """Verify PROTEUS_LSJ_REPO_DIR disables offline reuse of stale cached lexicon data."""
    project_dir = _copy_build_project(tmp_path)
    wheel_dir = tmp_path / "dist-env-override"
    wheel_dir.mkdir()
    lexicon_path = project_dir / "data" / "languages" / "ancient_greek" / "lexicon" / "greek_lemmas.json"
    default_xml_dir = build_lexicon.default_xml_dir(project_dir)
    override_repo_dir = tmp_path / "override-lsj"

    _write_fake_lsj_checkout(project_dir)
    _write_fake_lsj_checkout_at(
        override_repo_dir,
        key="lo/gos",
        orth="lo/gos",
        gen="o(",
        gloss="word",
    )
    fresh_document = _minimal_lexicon_document()
    fresh_document["lemmas"] = [
        {
            "id": "LSJ-123458",
            "headword": "ἕτοιμος",
            "transliteration": "hetoimos",
            "ipa": "hétoimos",
            "pos": "adjective",
            "gender": "common",
            "gloss": "fresh fixture",
            "dialect": "attic",
        }
    ]
    lexicon_path.write_text(
        json.dumps(fresh_document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    payload = build_lexicon.build_fingerprint_payload(project_dir, default_xml_dir)
    _metadata_path_for_lexicon(lexicon_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    env = _uv_build_env(tmp_path)
    env[build_lexicon.LSJ_REPO_DIR_ENV_VAR] = str(override_repo_dir)
    result = subprocess.run(
        [_uv_executable(), "build", "--wheel", "--out-dir", str(wheel_dir)],
        cwd=project_dir,
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    _assert_uv_build_succeeded_or_skip(result)
    wheel_files = sorted(wheel_dir.glob("*.whl"))
    assert wheel_files, "uv build did not produce a wheel"

    with zipfile.ZipFile(wheel_files[0]) as wheel_zip:
        lexicon_document = json.loads(
            wheel_zip.read("phonology/data/languages/ancient_greek/lexicon/greek_lemmas.json").decode("utf-8")
        )
        wheel_headwords = {entry["headword"] for entry in lexicon_document["lemmas"]}

    assert "λόγος" in wheel_headwords
    assert "ἕτοιμος" not in wheel_headwords


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv CLI not available")
def test_uv_build_wheel_from_extracted_sdist_reuses_packaged_lexicon_offline(
    tmp_path: Path,
) -> None:
    """Verify sdist -> wheel rebuilds stay offline by reusing bundled lexicon metadata."""
    project_dir = _copy_build_project(tmp_path)
    sdist_dir = tmp_path / "dist-sdist-offline"
    wheel_dir = tmp_path / "dist-wheel-from-sdist"
    extracted_dir = tmp_path / "extracted-sdist"
    fake_bin_dir = tmp_path / "fake-bin"
    sdist_dir.mkdir()
    wheel_dir.mkdir()
    extracted_dir.mkdir()
    fake_bin_dir.mkdir()

    lexicon_path = project_dir / "data" / "languages" / "ancient_greek" / "lexicon" / "greek_lemmas.json"
    metadata_path = _metadata_path_for_lexicon(lexicon_path)
    if lexicon_path.exists():
        lexicon_path.unlink()
    if metadata_path.exists():
        metadata_path.unlink()
    _write_fake_lsj_checkout(project_dir)

    sdist_result = subprocess.run(
        [_uv_executable(), "build", "--sdist", "--out-dir", str(sdist_dir)],
        cwd=project_dir,
        check=False,
        capture_output=True,
        env=_uv_build_env(tmp_path),
        text=True,
    )

    _assert_uv_build_succeeded_or_skip(sdist_result)
    sdist_files = sorted(sdist_dir.glob("*.tar.gz"))
    assert sdist_files, "uv build did not produce an sdist"
    extracted_project = _extract_sdist(sdist_files[0], extracted_dir)
    assert not (extracted_project / "data" / "external").exists()

    _create_fake_binary(
        fake_bin_dir / "git",
        "#!/usr/bin/env bash\n"
        "echo git-should-not-run >&2\n"
        "exit 99\n",
    )
    env = _uv_build_env(tmp_path)
    env["PATH"] = f"{fake_bin_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}"

    wheel_result = subprocess.run(
        [_uv_executable(), "build", "--wheel", "--out-dir", str(wheel_dir)],
        cwd=extracted_project,
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    _assert_uv_build_succeeded_or_skip(wheel_result)
    assert "git-should-not-run" not in wheel_result.stderr
    wheel_files = sorted(wheel_dir.glob("*.whl"))
    assert wheel_files, "uv build did not produce a wheel from the extracted sdist"

    with zipfile.ZipFile(wheel_files[0]) as wheel_zip:
        lexicon_document = json.loads(
            wheel_zip.read("phonology/data/languages/ancient_greek/lexicon/greek_lemmas.json").decode("utf-8")
        )
        assert "phonology/data/languages/ancient_greek/lexicon/greek_lemmas.meta.json" not in wheel_zip.namelist()

    assert any(entry.get("headword") == "ἄνθρωπος" for entry in lexicon_document["lemmas"])


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv CLI not available")
def test_uv_build_regenerates_invalid_existing_lexicon_when_local_checkout_exists(
    tmp_path: Path,
) -> None:
    """Verify invalid cached lexicon output is regenerated before wheel packaging."""
    project_dir = _copy_build_project(tmp_path)
    wheel_dir = tmp_path / "dist-invalid-regenerated"
    wheel_dir.mkdir()
    lexicon_path = project_dir / "data" / "languages" / "ancient_greek" / "lexicon" / "greek_lemmas.json"
    xml_dir = (
        project_dir / "data" / "external" / "lsj" / "CTS_XML_TEI" / "perseus" / "pdllex" / "grc" / "lsj"
    )

    _write_fake_lsj_checkout(project_dir)
    lexicon_path.write_text("{invalid json\n", encoding="utf-8")
    payload = build_lexicon.build_fingerprint_payload(project_dir, xml_dir)
    _metadata_path_for_lexicon(lexicon_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [_uv_executable(), "build", "--wheel", "--out-dir", str(wheel_dir)],
        cwd=project_dir,
        check=False,
        capture_output=True,
        env=_uv_build_env(tmp_path),
        text=True,
    )

    _assert_uv_build_succeeded_or_skip(result)
    wheel_files = sorted(wheel_dir.glob("*.whl"))
    assert wheel_files, "uv build did not produce a wheel"

    with zipfile.ZipFile(wheel_files[0]) as wheel_zip:
        lexicon_document = json.loads(
            wheel_zip.read("phonology/data/languages/ancient_greek/lexicon/greek_lemmas.json").decode("utf-8")
        )
        wheel_headwords = {entry["headword"] for entry in lexicon_document["lemmas"]}

    assert "ἄνθρωπος" in wheel_headwords


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv CLI not available")
def test_uv_build_fails_when_existing_lexicon_is_invalid_and_no_checkout_exists(
    tmp_path: Path,
) -> None:
    """Verify invalid cached lexicon output is not reused without local inputs."""
    project_dir = _copy_build_project(tmp_path)
    wheel_dir = tmp_path / "dist-invalid-no-checkout"
    wheel_dir.mkdir()
    lexicon_path = project_dir / "data" / "languages" / "ancient_greek" / "lexicon" / "greek_lemmas.json"
    xml_dir = build_lexicon.default_xml_dir(project_dir)

    _write_fake_lsj_checkout(project_dir)
    lexicon_path.write_text("{invalid json\n", encoding="utf-8")
    payload = build_lexicon.build_fingerprint_payload(project_dir, xml_dir)
    shutil.rmtree(project_dir / "data" / "external")
    _metadata_path_for_lexicon(lexicon_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [_uv_executable(), "build", "--wheel", "--out-dir", str(wheel_dir)],
        cwd=project_dir,
        check=False,
        capture_output=True,
        env=_uv_build_env(tmp_path),
        text=True,
    )

    _assert_uv_build_failed_with_message_or_skip(result, "build-time cloning is disabled")
    assert "data/languages/ancient_greek/lexicon/greek_lemmas.json" in result.stderr


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv CLI not available")
def test_uv_build_fails_without_lexicon_or_local_lsj_checkout(tmp_path: Path) -> None:
    """Verify wheel builds fail fast instead of cloning LSJ data during the build."""
    project_dir = _copy_build_project(tmp_path)
    wheel_dir = tmp_path / "dist-missing-inputs"
    wheel_dir.mkdir()
    lexicon_path = project_dir / "data" / "languages" / "ancient_greek" / "lexicon" / "greek_lemmas.json"
    metadata_path = _metadata_path_for_lexicon(lexicon_path)

    if lexicon_path.exists():
        lexicon_path.unlink()
    if metadata_path.exists():
        metadata_path.unlink()

    result = subprocess.run(
        [_uv_executable(), "build", "--wheel", "--out-dir", str(wheel_dir)],
        cwd=project_dir,
        check=False,
        capture_output=True,
        env=_uv_build_env(tmp_path),
        text=True,
    )

    _assert_uv_build_failed_with_message_or_skip(result, "build-time cloning is disabled")
    assert "data/languages/ancient_greek/lexicon/greek_lemmas.json" in result.stderr
    assert "--xml-dir" in result.stderr
    assert "--lsj-repo-dir" in result.stderr
    assert build_lexicon.LSJ_REPO_DIR_ENV_VAR in result.stderr


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv CLI not available")
def test_uv_sync_succeeds_without_lexicon_or_lsj_checkout_for_editable_builds(
    tmp_path: Path,
) -> None:
    """Verify editable installs succeed before lexicon generation, but search stays unready."""
    project_dir = _copy_build_project(tmp_path)
    lexicon_path = project_dir / "data" / "languages" / "ancient_greek" / "lexicon" / "greek_lemmas.json"
    metadata_path = _metadata_path_for_lexicon(lexicon_path)

    if lexicon_path.exists():
        lexicon_path.unlink()
    if metadata_path.exists():
        metadata_path.unlink()
    shutil.rmtree(project_dir / "data" / "external", ignore_errors=True)

    result = subprocess.run(
        [_uv_executable(), "sync", "--all-extras", "--dev"],
        cwd=project_dir,
        check=False,
        capture_output=True,
        env=_uv_build_env(tmp_path),
        text=True,
    )

    _assert_uv_build_succeeded_or_skip(result)

    # Verify the editable install exposes packaged matrices while reporting
    # lexicon-backed search dependencies as not ready.
    verify_script = """
import phonology
from fastapi.testclient import TestClient
import api.main as api_main

from phonology.distance import load_matrix

matrix = load_matrix("attic_doric.json")
assert "a" in matrix
assert "a" in matrix["a"]

try:
    api_main.load_lexicon_entries()
except ValueError as exc:
    assert "Lexicon file not found" in str(exc)
else:
    raise AssertionError("Expected load_lexicon_entries() to fail without generated lexicon")

api_main.app.state.disable_startup_warmup = True
with TestClient(api_main.app) as client:
    ready_response = client.get("/ready")
    assert ready_response.status_code == 503
    assert ready_response.json() == {
        "detail": api_main._SEARCH_DEPENDENCIES_LEXICON_NOT_READY_DETAIL
    }

    search_response = client.post("/search", json={"query_form": "λόγος"})
    assert search_response.status_code == 503
    assert search_response.json() == {
        "detail": api_main._SEARCH_DEPENDENCIES_LEXICON_NOT_READY_DETAIL
    }

print("editable install verified: matrix ok, lexicon pending")
"""

    if os.name == "nt":
        venv_dir = "Scripts"
        venv_exe = "python.exe"
    else:
        venv_dir = "bin"
        venv_exe = "python"

    venv_python_path = project_dir / ".venv" / venv_dir / venv_exe
    assert venv_python_path.exists(), (
        f"Expected venv interpreter at {venv_python_path} after successful uv sync"
    )
    venv_python = str(venv_python_path)
    verify_result = subprocess.run(
        [venv_python, "-c", verify_script],
        cwd=project_dir,
        check=False,
        capture_output=True,
        text=True,
    )

    assert verify_result.returncode == 0, (
        "Editable install verification failed using interpreter "
        f"{venv_python!r} (expected venv interpreter at {venv_python_path!s}).\n"
        f"stderr:\n{verify_result.stderr}\nstdout:\n{verify_result.stdout}"
    )
    assert "editable install verified: matrix ok, lexicon pending" in verify_result.stdout


def test_ci_workflow_caches_lexicon_json_and_metadata_together() -> None:
    """Verify the CI cache stores both the lexicon payload and freshness metadata."""
    workflow = _load_ci_workflow()
    jobs = workflow["jobs"]
    test_job = jobs["test"]
    steps = test_job["steps"]
    cache_step = next(
        (step for step in steps if step.get("name") == "Cache generated lexicon"), None
    )
    assert cache_step is not None, "expected a 'Cache generated lexicon' step in CI workflow"
    cache_path = cache_step["with"]["path"]

    assert "data/languages/ancient_greek/lexicon/greek_lemmas.json" in cache_path
    assert "data/languages/ancient_greek/lexicon/greek_lemmas.meta.json" in cache_path


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
