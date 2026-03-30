"""Tests for build-time lexicon generation helpers."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from functools import partial
from pathlib import Path

import pytest

from phonology import build_lexicon
import phonology.lsj_extractor as lsj_extractor_module

pytestmark = pytest.mark.usefixtures("reset_pos_overrides_cache")

ROOT_DIR = Path(__file__).resolve().parents[1]


def _raise_assertion(message: str, *_args: object, **_kwargs: object) -> None:
    raise AssertionError(message)


def _write_output_and_succeed(output_path: Path) -> int:
    output_path.write_text('{"fresh": true}\n', encoding="utf-8")
    return 0


def _freshness_payload(fingerprint: str = "fresh") -> dict[str, object]:
    return {
        "schema_version": build_lexicon.FINGERPRINT_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "inputs": [],
    }


def _minimal_lexicon_document() -> dict[str, object]:
    return {
        "schema_version": "2.0.0",
        "_meta": {
            "source": "LSJ (test fixture)",
            "encoding": "Unicode NFC",
            "ipa_system": "scholarly Ancient Greek IPA",
            "dialect": "attic",
            "version": "test",
            "last_updated": "2026-03-29T00:00:00Z",
            "license": "CC-BY-SA 4.0",
            "contributors": ["Proteus maintainers"],
            "data_schema_ref": "data/lexicon/greek_lemmas.schema.json",
            "description": "Build helper test fixture.",
            "note": "Generated during unit tests.",
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


def _write_schema(project_root: Path) -> None:
    schema_dir = project_root / "data" / "lexicon"
    schema_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        ROOT_DIR / "data" / "lexicon" / "greek_lemmas.schema.json",
        schema_dir / "greek_lemmas.schema.json",
    )


def _write_valid_lexicon_output(project_root: Path, output_path: Path) -> None:
    _write_schema(project_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_minimal_lexicon_document(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_default_fingerprint_inputs(project_root: Path) -> tuple[Path, Path]:
    """Write a minimal tracked source file and default LSJ XML input."""
    tracked_source = project_root / "src" / "phonology" / "ipa_converter.py"
    tracked_source.parent.mkdir(parents=True, exist_ok=True)
    tracked_source.write_text("# source\n", encoding="utf-8")

    xml_dir = build_lexicon.default_xml_dir(project_root)
    xml_dir.mkdir(parents=True, exist_ok=True)
    (xml_dir / "grc.lsj.perseus-eng1.xml").write_text("<TEI/>\n", encoding="utf-8")
    return tracked_source, xml_dir


def test_run_extractor_raises_when_pos_overrides_fail_strict_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    xml_dir = tmp_path / "xml"
    output_path = tmp_path / "data" / "lexicon" / "greek_lemmas.json"
    xml_dir.mkdir(parents=True)

    def fail_load(*, cli_mode: bool = False) -> dict[str, frozenset[str]]:
        assert cli_mode is True
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    monkeypatch.setattr(lsj_extractor_module, "_load_pos_overrides", fail_load)
    monkeypatch.setattr(
        lsj_extractor_module,
        "main",
        partial(_raise_assertion, "extractor main should not run after strict preload failure"),
    )

    with pytest.raises(UnicodeDecodeError):
        build_lexicon._run_extractor(xml_dir=xml_dir, output_path=output_path)


def test_ensure_generated_lexicon_reuses_fresh_output_for_missing_custom_lsj_xml_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "data" / "lexicon" / "greek_lemmas.json"
    _write_valid_lexicon_output(tmp_path, output_path)

    tracked_source = tmp_path / "src" / "phonology" / "ipa_converter.py"
    tracked_source.parent.mkdir(parents=True, exist_ok=True)
    tracked_source.write_text("# source\n", encoding="utf-8")
    tracked_source_record = build_lexicon._current_record_from_path(
        tracked_source,
        "src/phonology/ipa_converter.py",
    )

    custom_xml_dir = tmp_path.parent / "shared-lsj" / "xml"
    custom_xml_dir.mkdir(parents=True, exist_ok=True)
    custom_xml_path = custom_xml_dir / "grc.lsj.perseus-eng7.xml"
    custom_xml_path.write_text("<TEI/>\n", encoding="utf-8")
    custom_xml_record = build_lexicon._current_record_from_path(
        custom_xml_path,
        str(custom_xml_path),
    )
    shutil.rmtree(custom_xml_dir.parent)

    records = [tracked_source_record, custom_xml_record]
    build_lexicon._write_metadata(
        build_lexicon._metadata_path_for_output(output_path),
        {
            "schema_version": build_lexicon.FINGERPRINT_SCHEMA_VERSION,
            "fingerprint": build_lexicon._fingerprint_digest_for_records(records),
            "inputs": records,
        },
    )
    monkeypatch.setattr(
        build_lexicon,
        "ensure_lsj_checkout",
        partial(_raise_assertion, "should not resolve checkout"),
    )

    did_generate = build_lexicon.ensure_generated_lexicon(
        project_root=tmp_path,
        output_path=output_path,
        skip_if_present=True,
        allow_clone=False,
    )

    assert did_generate is False


def test_ensure_generated_lexicon_reuses_fresh_output_when_env_checkout_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "data" / "lexicon" / "greek_lemmas.json"
    _write_valid_lexicon_output(tmp_path, output_path)

    tracked_source = tmp_path / "src" / "phonology" / "ipa_converter.py"
    tracked_source.parent.mkdir(parents=True, exist_ok=True)
    tracked_source.write_text("# source\n", encoding="utf-8")
    tracked_source_record = build_lexicon._current_record_from_path(
        tracked_source,
        "src/phonology/ipa_converter.py",
    )

    records = [tracked_source_record]
    build_lexicon._write_metadata(
        build_lexicon._metadata_path_for_output(output_path),
        {
            "schema_version": build_lexicon.FINGERPRINT_SCHEMA_VERSION,
            "fingerprint": build_lexicon._fingerprint_digest_for_records(records),
            "inputs": records,
        },
    )
    monkeypatch.setenv(build_lexicon.LSJ_REPO_DIR_ENV_VAR, str(tmp_path / "missing-lsj"))
    monkeypatch.setattr(
        build_lexicon,
        "ensure_lsj_checkout",
        partial(_raise_assertion, "should not resolve checkout"),
    )

    did_generate = build_lexicon.ensure_generated_lexicon(
        project_root=tmp_path,
        output_path=output_path,
        skip_if_present=True,
        allow_clone=False,
    )

    assert did_generate is False


def test_ensure_generated_lexicon_does_not_shortcut_offline_reuse_when_env_checkout_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "data" / "lexicon" / "greek_lemmas.json"
    _write_valid_lexicon_output(tmp_path, output_path)

    tracked_source = tmp_path / "src" / "phonology" / "ipa_converter.py"
    tracked_source.parent.mkdir(parents=True, exist_ok=True)
    tracked_source.write_text("# source\n", encoding="utf-8")
    tracked_source_record = build_lexicon._current_record_from_path(
        tracked_source,
        "src/phonology/ipa_converter.py",
    )
    build_lexicon._write_metadata(
        build_lexicon._metadata_path_for_output(output_path),
        {
            "schema_version": build_lexicon.FINGERPRINT_SCHEMA_VERSION,
            "fingerprint": build_lexicon._fingerprint_digest_for_records([tracked_source_record]),
            "inputs": [tracked_source_record],
        },
    )

    repo_dir = tmp_path / "override-lsj"
    xml_dir = build_lexicon._lsj_xml_dir_for_repo(repo_dir)
    xml_dir.mkdir(parents=True, exist_ok=True)
    (xml_dir / "grc.lsj.perseus-eng1.xml").write_text("<TEI/>\n", encoding="utf-8")
    monkeypatch.setenv(build_lexicon.LSJ_REPO_DIR_ENV_VAR, str(repo_dir))

    freshness_payload = _freshness_payload("env-override")
    captured: dict[str, object] = {}

    def fake_checkout(
        project_root: Path,
        *,
        xml_dir: Path | None = None,
        lsj_repo_dir: Path | None = None,
        resolved_lsj_repo_dir: Path | None = None,
        allow_clone: bool = True,
    ) -> Path:
        captured["project_root"] = project_root
        captured["xml_dir_arg"] = xml_dir
        captured["lsj_repo_dir_arg"] = lsj_repo_dir
        captured["resolved_lsj_repo_dir_arg"] = resolved_lsj_repo_dir
        captured["allow_clone"] = allow_clone
        if xml_dir is not None:
            return xml_dir
        if resolved_lsj_repo_dir is None:
            raise ValueError("resolved_lsj_repo_dir must not be None when xml_dir is None")
        return build_lexicon._lsj_xml_dir_for_repo(resolved_lsj_repo_dir)

    def fake_run_extractor(
        *,
        xml_dir: Path,
        output_path: Path,
        limit: int | None = None,
        dry_run: bool = False,
    ) -> int:
        captured["xml_dir"] = xml_dir
        captured["output_path"] = output_path
        captured["limit"] = limit
        captured["dry_run"] = dry_run
        output_path.write_text('{"fresh": true}\n', encoding="utf-8")
        return 0

    monkeypatch.setattr(build_lexicon, "ensure_lsj_checkout", fake_checkout)
    monkeypatch.setattr(
        build_lexicon,
        "build_fingerprint_payload",
        lambda project_root, xml_dir: freshness_payload,
    )
    monkeypatch.setattr(build_lexicon, "_run_extractor", fake_run_extractor)

    did_generate = build_lexicon.ensure_generated_lexicon(
        project_root=tmp_path,
        output_path=output_path,
        skip_if_present=True,
        allow_clone=False,
    )

    assert did_generate is True
    assert captured == {
        "project_root": tmp_path,
        "xml_dir_arg": None,
        "lsj_repo_dir_arg": None,
        "resolved_lsj_repo_dir_arg": repo_dir,
        "allow_clone": False,
        "xml_dir": xml_dir,
        "output_path": output_path,
        "limit": None,
        "dry_run": False,
    }
    assert json.loads(
        build_lexicon._metadata_path_for_output(output_path).read_text(encoding="utf-8")
    ) == freshness_payload


def test_ensure_generated_lexicon_rejects_missing_non_lsj_input_for_offline_reuse(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "data" / "lexicon" / "greek_lemmas.json"
    _write_valid_lexicon_output(tmp_path, output_path)

    tracked_source = tmp_path / "src" / "phonology" / "ipa_converter.py"
    tracked_source.parent.mkdir(parents=True, exist_ok=True)
    tracked_source.write_text("# source\n", encoding="utf-8")
    tracked_source_record = build_lexicon._current_record_from_path(
        tracked_source,
        "src/phonology/ipa_converter.py",
    )

    missing_input = tmp_path.parent / "shared-inputs" / "not-lsj.txt"
    missing_input.parent.mkdir(parents=True, exist_ok=True)
    missing_input.write_text("stale\n", encoding="utf-8")
    missing_input_record = build_lexicon._current_record_from_path(
        missing_input,
        str(missing_input),
    )
    shutil.rmtree(missing_input.parent)

    records = [tracked_source_record, missing_input_record]
    build_lexicon._write_metadata(
        build_lexicon._metadata_path_for_output(output_path),
        {
            "schema_version": build_lexicon.FINGERPRINT_SCHEMA_VERSION,
            "fingerprint": build_lexicon._fingerprint_digest_for_records(records),
            "inputs": records,
        },
    )

    with pytest.raises(FileNotFoundError, match="build-time cloning is disabled"):
        build_lexicon.ensure_generated_lexicon(
            project_root=tmp_path,
            output_path=output_path,
            skip_if_present=True,
            allow_clone=False,
        )


def test_ensure_generated_lexicon_skips_existing_fresh_output_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    xml_dir = tmp_path / "fake-xml"
    xml_dir.mkdir()
    output_path = tmp_path / "data" / "lexicon" / "greek_lemmas.json"
    _write_valid_lexicon_output(tmp_path, output_path)
    freshness_payload = _freshness_payload()
    build_lexicon._write_metadata(
        build_lexicon._metadata_path_for_output(output_path),
        freshness_payload,
    )

    monkeypatch.setattr(
        build_lexicon,
        "ensure_lsj_checkout",
        lambda project_root, *, xml_dir=None, lsj_repo_dir=None, resolved_lsj_repo_dir=None, allow_clone=True: xml_dir,
    )
    monkeypatch.setattr(
        build_lexicon,
        "build_fingerprint_payload",
        lambda project_root, xml_dir: freshness_payload,
    )
    monkeypatch.setattr(
        build_lexicon,
        "_run_extractor",
        partial(_raise_assertion, "should not extract"),
    )

    did_generate = build_lexicon.ensure_generated_lexicon(
        project_root=tmp_path,
        output_path=output_path,
        xml_dir=xml_dir,
        skip_if_present=True,
    )

    assert did_generate is False


def test_ensure_generated_lexicon_reruns_when_output_exists_and_skip_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    xml_dir = tmp_path / "fake-xml"
    xml_dir.mkdir()
    output_path = tmp_path / "data" / "lexicon" / "greek_lemmas.json"
    output_path.parent.mkdir(parents=True)
    output_path.write_text('{"stale": true}\n', encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_checkout(
        project_root: Path,
        *,
        xml_dir: Path | None = None,
        lsj_repo_dir: Path | None = None,
        resolved_lsj_repo_dir: Path | None = None,
        allow_clone: bool = True,
    ) -> Path:
        captured["project_root"] = project_root
        captured["xml_dir_arg"] = xml_dir
        captured["lsj_repo_dir_arg"] = lsj_repo_dir
        captured["resolved_lsj_repo_dir_arg"] = resolved_lsj_repo_dir
        captured["allow_clone"] = allow_clone
        return xml_dir or Path("missing")

    def fake_run_extractor(
        *,
        xml_dir: Path,
        output_path: Path,
        limit: int | None = None,
        dry_run: bool = False,
    ) -> int:
        captured["xml_dir"] = xml_dir
        captured["output_path"] = output_path
        captured["limit"] = limit
        captured["dry_run"] = dry_run
        output_path.write_text('{"fresh": true}\n', encoding="utf-8")
        return 0

    monkeypatch.setattr(build_lexicon, "ensure_lsj_checkout", fake_checkout)
    monkeypatch.setattr(
        build_lexicon,
        "build_fingerprint_payload",
        lambda project_root, xml_dir: _freshness_payload("skip-disabled"),
    )
    monkeypatch.setattr(build_lexicon, "_run_extractor", fake_run_extractor)

    did_generate = build_lexicon.ensure_generated_lexicon(
        project_root=tmp_path,
        output_path=output_path,
        xml_dir=xml_dir,
        skip_if_present=False,
    )

    assert did_generate is True
    assert output_path.read_text(encoding="utf-8") == '{"fresh": true}\n'
    assert captured == {
        "project_root": tmp_path,
        "xml_dir_arg": xml_dir,
        "lsj_repo_dir_arg": None,
        "resolved_lsj_repo_dir_arg": None,
        "allow_clone": True,
        "xml_dir": xml_dir,
        "output_path": output_path,
        "limit": None,
        "dry_run": False,
    }


def test_ensure_generated_lexicon_reruns_when_output_exists_but_metadata_is_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    xml_dir = tmp_path / "fake-xml"
    xml_dir.mkdir()
    output_path = tmp_path / "data" / "lexicon" / "greek_lemmas.json"
    output_path.parent.mkdir(parents=True)
    output_path.write_text('{"stale": true}\n', encoding="utf-8")
    metadata_path = build_lexicon._metadata_path_for_output(output_path)
    build_lexicon._write_metadata(
        metadata_path,
        {
            "schema_version": build_lexicon.FINGERPRINT_SCHEMA_VERSION,
            "fingerprint": "old",
            "inputs": [],
        },
    )
    freshness_payload = _freshness_payload("new")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        build_lexicon,
        "ensure_lsj_checkout",
        lambda project_root, *, xml_dir=None, lsj_repo_dir=None, resolved_lsj_repo_dir=None, allow_clone=True: xml_dir,
    )
    monkeypatch.setattr(
        build_lexicon,
        "build_fingerprint_payload",
        lambda project_root, xml_dir: freshness_payload,
    )

    def fake_run_extractor(
        *,
        xml_dir: Path,
        output_path: Path,
        limit: int | None = None,
        dry_run: bool = False,
    ) -> int:
        captured["xml_dir"] = xml_dir
        captured["output_path"] = output_path
        captured["limit"] = limit
        captured["dry_run"] = dry_run
        output_path.write_text('{"fresh": true}\n', encoding="utf-8")
        return 0

    monkeypatch.setattr(build_lexicon, "_run_extractor", fake_run_extractor)

    did_generate = build_lexicon.ensure_generated_lexicon(
        project_root=tmp_path,
        output_path=output_path,
        xml_dir=xml_dir,
        skip_if_present=True,
    )

    assert did_generate is True
    assert output_path.read_text(encoding="utf-8") == '{"fresh": true}\n'
    assert json.loads(metadata_path.read_text(encoding="utf-8")) == freshness_payload
    assert captured == {
        "xml_dir": xml_dir,
        "output_path": output_path,
        "limit": None,
        "dry_run": False,
    }


def test_ensure_generated_lexicon_reruns_when_metadata_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    xml_dir = tmp_path / "fake-xml"
    xml_dir.mkdir()
    output_path = tmp_path / "data" / "lexicon" / "greek_lemmas.json"
    output_path.parent.mkdir(parents=True)
    output_path.write_text('{"stale": true}\n', encoding="utf-8")
    freshness_payload = _freshness_payload("new")

    monkeypatch.setattr(
        build_lexicon,
        "ensure_lsj_checkout",
        lambda project_root, *, xml_dir=None, lsj_repo_dir=None, resolved_lsj_repo_dir=None, allow_clone=True: xml_dir,
    )
    monkeypatch.setattr(
        build_lexicon,
        "build_fingerprint_payload",
        lambda project_root, xml_dir: freshness_payload,
    )
    monkeypatch.setattr(
        build_lexicon,
        "_run_extractor",
        lambda **kwargs: _write_output_and_succeed(kwargs["output_path"]),
    )

    did_generate = build_lexicon.ensure_generated_lexicon(
        project_root=tmp_path,
        output_path=output_path,
        xml_dir=xml_dir,
        skip_if_present=True,
    )

    assert did_generate is True
    assert json.loads(
        build_lexicon._metadata_path_for_output(output_path).read_text(encoding="utf-8")
    ) == freshness_payload


def test_ensure_generated_lexicon_reruns_when_metadata_is_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    xml_dir = tmp_path / "fake-xml"
    xml_dir.mkdir()
    output_path = tmp_path / "data" / "lexicon" / "greek_lemmas.json"
    output_path.parent.mkdir(parents=True)
    output_path.write_text('{"stale": true}\n', encoding="utf-8")
    metadata_path = build_lexicon._metadata_path_for_output(output_path)
    metadata_path.write_text("{invalid json\n", encoding="utf-8")
    freshness_payload = _freshness_payload("new")

    monkeypatch.setattr(
        build_lexicon,
        "ensure_lsj_checkout",
        lambda project_root, *, xml_dir=None, lsj_repo_dir=None, resolved_lsj_repo_dir=None, allow_clone=True: xml_dir,
    )
    monkeypatch.setattr(
        build_lexicon,
        "build_fingerprint_payload",
        lambda project_root, xml_dir: freshness_payload,
    )
    monkeypatch.setattr(
        build_lexicon,
        "_run_extractor",
        lambda **kwargs: _write_output_and_succeed(kwargs["output_path"]),
    )

    did_generate = build_lexicon.ensure_generated_lexicon(
        project_root=tmp_path,
        output_path=output_path,
        xml_dir=xml_dir,
        skip_if_present=True,
    )

    assert did_generate is True
    assert json.loads(metadata_path.read_text(encoding="utf-8")) == freshness_payload


def test_ensure_generated_lexicon_reuses_fresh_output_without_default_lsj_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracked_source, xml_dir = _write_default_fingerprint_inputs(tmp_path)
    output_path = tmp_path / "data" / "lexicon" / "greek_lemmas.json"
    _write_valid_lexicon_output(tmp_path, output_path)

    monkeypatch.setattr(
        build_lexicon,
        "_FINGERPRINT_INPUTS",
        (Path("src/phonology/ipa_converter.py"),),
    )
    payload = build_lexicon.build_fingerprint_payload(tmp_path, xml_dir)
    build_lexicon._write_metadata(build_lexicon._metadata_path_for_output(output_path), payload)
    shutil.rmtree(build_lexicon._default_lsj_repo_dir(tmp_path))

    monkeypatch.setattr(
        build_lexicon,
        "ensure_lsj_checkout",
        partial(_raise_assertion, "should not require checkout"),
    )
    monkeypatch.setattr(
        build_lexicon,
        "_run_extractor",
        partial(_raise_assertion, "should not extract"),
    )

    did_generate = build_lexicon.ensure_generated_lexicon(
        project_root=tmp_path,
        output_path=output_path,
        skip_if_present=True,
    )

    assert tracked_source.read_text(encoding="utf-8") == "# source\n"
    assert did_generate is False


def test_ensure_generated_lexicon_reruns_when_existing_output_is_invalid_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    xml_dir = tmp_path / "fake-xml"
    xml_dir.mkdir()
    output_path = tmp_path / "data" / "lexicon" / "greek_lemmas.json"
    _write_schema(tmp_path)
    output_path.write_text("{invalid json\n", encoding="utf-8")
    freshness_payload = _freshness_payload()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        build_lexicon,
        "ensure_lsj_checkout",
        lambda project_root, *, xml_dir=None, lsj_repo_dir=None, resolved_lsj_repo_dir=None, allow_clone=True: xml_dir,
    )
    monkeypatch.setattr(
        build_lexicon,
        "build_fingerprint_payload",
        lambda project_root, xml_dir: freshness_payload,
    )

    def fake_run_extractor(
        *,
        xml_dir: Path,
        output_path: Path,
        limit: int | None = None,
        dry_run: bool = False,
    ) -> int:
        captured["xml_dir"] = xml_dir
        captured["output_path"] = output_path
        captured["limit"] = limit
        captured["dry_run"] = dry_run
        _write_valid_lexicon_output(tmp_path, output_path)
        return 0

    build_lexicon._write_metadata(
        build_lexicon._metadata_path_for_output(output_path),
        freshness_payload,
    )
    monkeypatch.setattr(build_lexicon, "_run_extractor", fake_run_extractor)

    did_generate = build_lexicon.ensure_generated_lexicon(
        project_root=tmp_path,
        output_path=output_path,
        xml_dir=xml_dir,
        skip_if_present=True,
    )

    assert did_generate is True
    assert captured == {
        "xml_dir": xml_dir,
        "output_path": output_path,
        "limit": None,
        "dry_run": False,
    }


def test_ensure_generated_lexicon_reruns_when_existing_output_is_schema_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    xml_dir = tmp_path / "fake-xml"
    xml_dir.mkdir()
    output_path = tmp_path / "data" / "lexicon" / "greek_lemmas.json"
    _write_schema(tmp_path)
    output_path.write_text(
        json.dumps({"schema_version": "2.0.0", "_meta": {}, "lemmas": []}, indent=2) + "\n",
        encoding="utf-8",
    )
    freshness_payload = _freshness_payload()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        build_lexicon,
        "ensure_lsj_checkout",
        lambda project_root, *, xml_dir=None, lsj_repo_dir=None, resolved_lsj_repo_dir=None, allow_clone=True: xml_dir,
    )
    monkeypatch.setattr(
        build_lexicon,
        "build_fingerprint_payload",
        lambda project_root, xml_dir: freshness_payload,
    )

    def fake_run_extractor(
        *,
        xml_dir: Path,
        output_path: Path,
        limit: int | None = None,
        dry_run: bool = False,
    ) -> int:
        captured["xml_dir"] = xml_dir
        captured["output_path"] = output_path
        captured["limit"] = limit
        captured["dry_run"] = dry_run
        _write_valid_lexicon_output(tmp_path, output_path)
        return 0

    build_lexicon._write_metadata(
        build_lexicon._metadata_path_for_output(output_path),
        freshness_payload,
    )
    monkeypatch.setattr(build_lexicon, "_run_extractor", fake_run_extractor)

    did_generate = build_lexicon.ensure_generated_lexicon(
        project_root=tmp_path,
        output_path=output_path,
        xml_dir=xml_dir,
        skip_if_present=True,
    )

    assert did_generate is True
    assert captured == {
        "xml_dir": xml_dir,
        "output_path": output_path,
        "limit": None,
        "dry_run": False,
    }


def test_ensure_generated_lexicon_does_not_reuse_invalid_output_without_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracked_source, xml_dir = _write_default_fingerprint_inputs(tmp_path)
    output_path = tmp_path / "data" / "lexicon" / "greek_lemmas.json"
    _write_schema(tmp_path)
    output_path.write_text("{invalid json\n", encoding="utf-8")

    monkeypatch.setattr(
        build_lexicon,
        "_FINGERPRINT_INPUTS",
        (Path("src/phonology/ipa_converter.py"),),
    )
    payload = build_lexicon.build_fingerprint_payload(tmp_path, xml_dir)
    build_lexicon._write_metadata(build_lexicon._metadata_path_for_output(output_path), payload)
    shutil.rmtree(build_lexicon._default_lsj_repo_dir(tmp_path))

    monkeypatch.setattr(
        build_lexicon,
        "ensure_lsj_checkout",
        partial(_raise_assertion, "should require checkout"),
    )
    monkeypatch.setattr(
        build_lexicon,
        "_run_extractor",
        partial(_raise_assertion, "should not extract"),
    )

    with pytest.raises(AssertionError, match="should require checkout"):
        build_lexicon.ensure_generated_lexicon(
            project_root=tmp_path,
            output_path=output_path,
            skip_if_present=True,
        )

    assert tracked_source.read_text(encoding="utf-8") == "# source\n"


def test_ensure_generated_lexicon_reruns_when_source_changes_and_default_lsj_checkout_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracked_source, xml_dir = _write_default_fingerprint_inputs(tmp_path)
    output_path = tmp_path / "data" / "lexicon" / "greek_lemmas.json"
    output_path.parent.mkdir(parents=True)
    output_path.write_text('{"stale": true}\n', encoding="utf-8")

    monkeypatch.setattr(
        build_lexicon,
        "_FINGERPRINT_INPUTS",
        (Path("src/phonology/ipa_converter.py"),),
    )
    payload = build_lexicon.build_fingerprint_payload(tmp_path, xml_dir)
    build_lexicon._write_metadata(build_lexicon._metadata_path_for_output(output_path), payload)
    tracked_source.write_text("# changed source\n", encoding="utf-8")
    shutil.rmtree(build_lexicon._default_lsj_repo_dir(tmp_path))

    captured: dict[str, object] = {}
    regenerated_payload = _freshness_payload("after-change")

    def fake_checkout(
        project_root: Path,
        *,
        xml_dir: Path | None = None,
        lsj_repo_dir: Path | None = None,
        resolved_lsj_repo_dir: Path | None = None,
        allow_clone: bool = True,
    ) -> Path:
        captured["project_root"] = project_root
        captured["xml_dir_arg"] = xml_dir
        captured["lsj_repo_dir_arg"] = lsj_repo_dir
        captured["resolved_lsj_repo_dir_arg"] = resolved_lsj_repo_dir
        captured["allow_clone"] = allow_clone
        regenerated_xml_dir = project_root / "replacement-xml"
        regenerated_xml_dir.mkdir(exist_ok=True)
        return regenerated_xml_dir

    def fake_run_extractor(
        *,
        xml_dir: Path,
        output_path: Path,
        limit: int | None = None,
        dry_run: bool = False,
    ) -> int:
        captured["xml_dir"] = xml_dir
        captured["output_path"] = output_path
        captured["limit"] = limit
        captured["dry_run"] = dry_run
        output_path.write_text('{"fresh": true}\n', encoding="utf-8")
        return 0

    monkeypatch.setattr(build_lexicon, "ensure_lsj_checkout", fake_checkout)
    monkeypatch.setattr(
        build_lexicon,
        "build_fingerprint_payload",
        lambda project_root, xml_dir: regenerated_payload,
    )
    monkeypatch.setattr(build_lexicon, "_run_extractor", fake_run_extractor)

    did_generate = build_lexicon.ensure_generated_lexicon(
        project_root=tmp_path,
        output_path=output_path,
        skip_if_present=True,
    )

    assert did_generate is True
    assert captured == {
        "project_root": tmp_path,
        "xml_dir_arg": None,
        "lsj_repo_dir_arg": None,
        "resolved_lsj_repo_dir_arg": None,
        "allow_clone": True,
        "xml_dir": tmp_path / "replacement-xml",
        "output_path": output_path,
        "limit": None,
        "dry_run": False,
    }
    assert json.loads(
        build_lexicon._metadata_path_for_output(output_path).read_text(encoding="utf-8")
    ) == regenerated_payload


def test_ensure_generated_lexicon_runs_checkout_and_extractor_when_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    xml_dir = tmp_path / "fake-xml"
    xml_dir.mkdir()
    output_path = tmp_path / "data" / "lexicon" / "greek_lemmas.json"
    captured: dict[str, object] = {}

    def fake_checkout(
        project_root: Path,
        *,
        xml_dir: Path | None = None,
        lsj_repo_dir: Path | None = None,
        resolved_lsj_repo_dir: Path | None = None,
        allow_clone: bool = True,
    ) -> Path:
        captured["project_root"] = project_root
        captured["xml_dir_arg"] = xml_dir
        captured["lsj_repo_dir_arg"] = lsj_repo_dir
        captured["resolved_lsj_repo_dir_arg"] = resolved_lsj_repo_dir
        captured["allow_clone"] = allow_clone
        return xml_dir or Path("missing")

    def fake_run_extractor(
        *,
        xml_dir: Path,
        output_path: Path,
        limit: int | None = None,
        dry_run: bool = False,
    ) -> int:
        captured["xml_dir"] = xml_dir
        captured["output_path"] = output_path
        captured["limit"] = limit
        captured["dry_run"] = dry_run
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text('{"lemmas": []}\n', encoding="utf-8")
        return 0

    monkeypatch.setattr(build_lexicon, "ensure_lsj_checkout", fake_checkout)
    monkeypatch.setattr(
        build_lexicon,
        "build_fingerprint_payload",
        lambda project_root, xml_dir: _freshness_payload("missing-output"),
    )
    monkeypatch.setattr(build_lexicon, "_run_extractor", fake_run_extractor)

    did_generate = build_lexicon.ensure_generated_lexicon(
        project_root=tmp_path,
        output_path=output_path,
        xml_dir=xml_dir,
        limit=5,
    )

    assert did_generate is True
    assert captured == {
        "project_root": tmp_path,
        "xml_dir_arg": xml_dir,
        "lsj_repo_dir_arg": None,
        "resolved_lsj_repo_dir_arg": None,
        "allow_clone": True,
        "xml_dir": xml_dir,
        "output_path": output_path,
        "limit": 5,
        "dry_run": False,
    }
    assert output_path.is_file()


def test_ensure_generated_lexicon_raises_when_extractor_returns_nonzero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    xml_dir = tmp_path / "fake-xml"
    xml_dir.mkdir()
    output_path = tmp_path / "data" / "lexicon" / "greek_lemmas.json"
    resolved_xml_dir = xml_dir

    monkeypatch.setattr(
        build_lexicon,
        "ensure_lsj_checkout",
        lambda project_root, *, xml_dir=None, lsj_repo_dir=None, resolved_lsj_repo_dir=None, allow_clone=True: resolved_xml_dir,
    )
    monkeypatch.setattr(
        build_lexicon,
        "build_fingerprint_payload",
        lambda project_root, xml_dir: _freshness_payload("failure-case"),
    )
    monkeypatch.setattr(build_lexicon, "_run_extractor", lambda **kwargs: 7)

    with pytest.raises(RuntimeError, match="LSJ extraction failed with exit code 7"):
        build_lexicon.ensure_generated_lexicon(
            project_root=tmp_path,
            output_path=output_path,
            xml_dir=xml_dir,
        )


def test_ensure_generated_lexicon_fails_fast_without_checkout_when_clone_disabled(
    tmp_path: Path,
) -> None:
    """Clone disabled without existing checkout must raise with actionable hints."""
    output_path = tmp_path / "data" / "lexicon" / "greek_lemmas.json"

    with pytest.raises(FileNotFoundError, match="build-time cloning is disabled") as exc_info:
        build_lexicon.ensure_generated_lexicon(
            project_root=tmp_path,
            output_path=output_path,
            skip_if_present=True,
            allow_clone=False,
        )

    message = str(exc_info.value)
    assert "data/lexicon/greek_lemmas.json" in message
    assert "--xml-dir" in message
    assert "--lsj-repo-dir" in message
    assert build_lexicon.LSJ_REPO_DIR_ENV_VAR in message


def test_build_fingerprint_payload_tracks_xml_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path
    tracked_source = project_root / "src" / "phonology" / "ipa_converter.py"
    tracked_source.parent.mkdir(parents=True, exist_ok=True)
    tracked_source.write_text("# source\n", encoding="utf-8")
    schema_path = project_root / "data" / "lexicon" / "greek_lemmas.schema.json"
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text("{}\n", encoding="utf-8")
    xml_dir = project_root / "xml"
    xml_dir.mkdir()
    first_xml = xml_dir / "grc.lsj.perseus-eng1.xml"
    second_xml = xml_dir / "grc.lsj.perseus-eng2.xml"
    first_xml.write_text("<TEI/>\n", encoding="utf-8")
    second_xml.write_text("<TEI/>\n", encoding="utf-8")

    monkeypatch.setattr(
        build_lexicon,
        "_FINGERPRINT_INPUTS",
        (
            Path("src/phonology/ipa_converter.py"),
            Path("data/lexicon/greek_lemmas.schema.json"),
        ),
    )

    payload = build_lexicon.build_fingerprint_payload(project_root, xml_dir)

    assert payload["schema_version"] == build_lexicon.FINGERPRINT_SCHEMA_VERSION
    assert isinstance(payload["fingerprint"], str) and payload["fingerprint"]
    assert [record["path"] for record in payload["inputs"]] == [
        "src/phonology/ipa_converter.py",
        "data/lexicon/greek_lemmas.schema.json",
        "xml/grc.lsj.perseus-eng1.xml",
        "xml/grc.lsj.perseus-eng2.xml",
    ]


def test_ensure_lsj_checkout_uses_python_timeout_and_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path
    xml_dir = build_lexicon.default_xml_dir(project_root)
    repo_dir = project_root / "data" / "external" / "lsj"
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    repo_dir.mkdir()

    calls: list[tuple[list[str], str | None, int | None]] = []
    clone_attempts = 0

    def fake_run_subprocess(
        command: list[str],
        *,
        cwd: Path | None = None,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        nonlocal clone_attempts
        calls.append((command, str(cwd) if cwd is not None else None, timeout))
        if command[1] == "clone":
            clone_attempts += 1
            if clone_attempts == 1:
                raise subprocess.TimeoutExpired(command, timeout=timeout or 0)
            repo_dir.mkdir(parents=True, exist_ok=True)
            xml_dir.mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(build_lexicon.shutil, "which", lambda name: "/usr/bin/git")
    monkeypatch.setattr(build_lexicon, "_run_subprocess", fake_run_subprocess)
    monkeypatch.setattr(build_lexicon.time, "sleep", lambda _: None)

    resolved_xml_dir = build_lexicon.ensure_lsj_checkout(project_root)

    assert resolved_xml_dir == xml_dir
    clone_calls = [call for call in calls if call[0][1] == "clone"]
    assert len(clone_calls) == 2
    assert all(call[0][0] == "/usr/bin/git" for call in clone_calls)
    assert all(call[2] == build_lexicon._CLONE_TIMEOUT_SECONDS for call in clone_calls)
    assert "timeout" not in clone_calls[0][0]


def test_ensure_lsj_checkout_fails_after_clone_retries_are_exhausted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path
    clone_calls: list[tuple[list[str], int | None]] = []

    def fake_run_subprocess(
        command: list[str],
        *,
        cwd: Path | None = None,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del cwd
        clone_calls.append((command, timeout))
        raise subprocess.TimeoutExpired(command, timeout=timeout or 0)

    monkeypatch.setattr(build_lexicon.shutil, "which", lambda name: "/usr/bin/git")
    monkeypatch.setattr(build_lexicon, "_run_subprocess", fake_run_subprocess)
    monkeypatch.setattr(build_lexicon.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="Failed to clone Perseus LSJ") as exc_info:
        build_lexicon.ensure_lsj_checkout(project_root)

    assert isinstance(exc_info.value.__cause__, subprocess.TimeoutExpired)
    assert len(clone_calls) == build_lexicon._CLONE_MAX_ATTEMPTS
    assert all(command[1] == "clone" for command, _timeout in clone_calls)
    assert all(
        timeout == build_lexicon._CLONE_TIMEOUT_SECONDS for _command, timeout in clone_calls
    )


def test_infer_lsj_repo_dir_normalizes_equivalent_default_xml_path(tmp_path: Path) -> None:
    project_root = tmp_path
    equivalent_xml_dir = (
        build_lexicon._default_lsj_repo_dir(project_root)
        / "CTS_XML_TEI"
        / "perseus"
        / "pdllex"
        / "grc"
        / "segment"
        / ".."
        / "lsj"
    )

    inferred_repo_dir = build_lexicon._infer_lsj_repo_dir(project_root, equivalent_xml_dir)

    assert inferred_repo_dir == build_lexicon._default_lsj_repo_dir(project_root)


def test_ensure_lsj_checkout_skips_clone_for_custom_xml_dir_outside_default_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    custom_xml_dir = tmp_path / "custom" / "xml"
    caplog.set_level(logging.INFO, logger=build_lexicon.logger.name)
    monkeypatch.setattr(
        build_lexicon,
        "_clone_lsj_repo",
        partial(_raise_assertion, "should not clone"),
    )

    with pytest.raises(FileNotFoundError, match="LSJ XML directory not found"):
        build_lexicon.ensure_lsj_checkout(tmp_path, xml_dir=custom_xml_dir)

    assert (
        f"Skipping LSJ clone because XML directory {custom_xml_dir} is outside the default "
        f"checkout path {build_lexicon.default_xml_dir(tmp_path)}"
        in caplog.text
    )


def test_ensure_lsj_checkout_fails_fast_when_clone_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        build_lexicon,
        "_clone_lsj_repo",
        partial(_raise_assertion, "should not clone"),
    )

    with pytest.raises(FileNotFoundError, match="build-time cloning is disabled") as exc_info:
        build_lexicon.ensure_lsj_checkout(tmp_path, allow_clone=False)

    message = str(exc_info.value)
    assert str(build_lexicon.default_xml_dir(tmp_path)) in message
    assert "data/lexicon/greek_lemmas.json" in message


def test_ensure_lsj_checkout_uses_env_override_for_default_xml_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_dir = tmp_path / "custom-lsj"
    expected_xml_dir = build_lexicon._lsj_xml_dir_for_repo(repo_dir)
    captured: list[Path] = []

    def fake_clone_lsj_repo(lsj_repo_dir: Path, project_root: Path) -> None:
        captured.append(lsj_repo_dir)
        expected_xml_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv(build_lexicon.LSJ_REPO_DIR_ENV_VAR, str(repo_dir))
    monkeypatch.setattr(build_lexicon, "_clone_lsj_repo", fake_clone_lsj_repo)

    resolved_xml_dir = build_lexicon.ensure_lsj_checkout(tmp_path)

    assert resolved_xml_dir == expected_xml_dir
    assert captured == [repo_dir]


def test_ensure_lsj_checkout_prefers_pre_resolved_repo_dir_over_env_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_dir = tmp_path / "pre-resolved-lsj"
    env_repo_dir = tmp_path / "env-lsj"
    expected_xml_dir = build_lexicon._lsj_xml_dir_for_repo(repo_dir)
    captured: list[Path] = []

    def fake_clone_lsj_repo(lsj_repo_dir: Path, project_root: Path) -> None:
        del project_root
        captured.append(lsj_repo_dir)
        expected_xml_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv(build_lexicon.LSJ_REPO_DIR_ENV_VAR, str(env_repo_dir))
    monkeypatch.setattr(build_lexicon, "_clone_lsj_repo", fake_clone_lsj_repo)

    resolved_xml_dir = build_lexicon.ensure_lsj_checkout(
        tmp_path,
        resolved_lsj_repo_dir=repo_dir,
    )

    assert resolved_xml_dir == expected_xml_dir
    assert captured == [repo_dir]


def test_ensure_lsj_checkout_rejects_both_repo_dir_arguments(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_lexicon.ensure_lsj_checkout(
            tmp_path,
            lsj_repo_dir=tmp_path / "explicit-lsj",
            resolved_lsj_repo_dir=tmp_path / "resolved-lsj",
        )


def test_clone_lsj_repo_rejects_repo_dir_outside_project_root_with_shared_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure is_relative_to rejects paths that share a prefix but are not children."""
    project_root = tmp_path / "myproject"
    project_root.mkdir()

    # "myproject-evil" shares the "myproject" prefix but is a sibling directory
    evil_repo_dir = tmp_path / "myproject-evil" / "data" / "external" / "lsj"
    evil_repo_dir.mkdir(parents=True)

    monkeypatch.setattr(build_lexicon.shutil, "which", lambda name: "/usr/bin/git")

    with pytest.raises(RuntimeError, match="Safety check failed"):
        build_lexicon._clone_lsj_repo(evil_repo_dir, project_root)


def test_clone_lsj_repo_rejects_symlink_repo_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Symlink lsj_repo_dir must not be passed to shutil.rmtree."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    real_dir = project_root / "data" / "external" / "real-lsj"
    real_dir.mkdir(parents=True)
    symlink_dir = project_root / "data" / "external" / "lsj"
    symlink_dir.symlink_to(real_dir)

    monkeypatch.setattr(build_lexicon.shutil, "which", lambda name: "/usr/bin/git")

    with pytest.raises(RuntimeError, match="Safety check failed.*symbolic link"):
        build_lexicon._clone_lsj_repo(symlink_dir, project_root)

    # The symlink itself must still exist, proving shutil.rmtree was not called
    # (rmtree on a symlink would remove the symlink itself, leaving it dangling).
    assert symlink_dir.is_symlink()
    assert real_dir.is_dir()


@pytest.mark.parametrize(
    ("no_clone_flag", "expected_allow_clone"),
    [
        ([], True),
        (["--no-clone"], False),
    ],
)
def test_run_cli_passes_allow_clone_based_on_no_clone_flag(
    monkeypatch: pytest.MonkeyPatch,
    no_clone_flag: list[str],
    expected_allow_clone: bool,
) -> None:
    """--no-clone should forward allow_clone=False; omitting it should keep the default True."""
    captured: dict[str, object] = {}

    def fake_ensure(**kwargs: object) -> bool:
        captured.update(kwargs)
        return False

    monkeypatch.setattr(build_lexicon, "ensure_generated_lexicon", fake_ensure)
    monkeypatch.setattr(sys, "argv", ["build_lexicon.py", *no_clone_flag])

    assert build_lexicon.run_cli() == 0
    assert captured.get("allow_clone") is expected_allow_clone


@pytest.mark.parametrize(
    ("argv_suffix", "expect_traceback"),
    [
        ([], False),
        (["--verbose"], True),
    ],
)
def test_run_cli_logs_expected_failure_detail(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    argv_suffix: list[str],
    expect_traceback: bool,
) -> None:
    def fail(**kwargs: object) -> bool:
        del kwargs
        raise RuntimeError("boom")

    monkeypatch.setattr(build_lexicon, "ensure_generated_lexicon", fail)
    monkeypatch.setattr(sys, "argv", ["build_lexicon.py", *argv_suffix])
    caplog.set_level(logging.ERROR, logger=build_lexicon.logger.name)

    assert build_lexicon.run_cli() == 1

    error_records = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert error_records
    assert error_records[-1].getMessage().startswith("Lexicon generation failed")
    assert (error_records[-1].exc_info is not None) is expect_traceback
