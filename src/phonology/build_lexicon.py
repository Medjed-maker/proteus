"""Helpers for generating the packaged LSJ lexicon on demand."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CLONE_TIMEOUT_SECONDS = 600
_CLONE_MAX_ATTEMPTS = 3
LSJ_REPO_DIR_ENV_VAR = "PROTEUS_LSJ_REPO_DIR"
_LSJ_XML_RELATIVE_PATH = Path("CTS_XML_TEI") / "perseus" / "pdllex" / "grc" / "lsj"
_LSJ_XML_FILENAME_RE = re.compile(r"^grc\.lsj\.perseus-eng\d*\.xml$")
FINGERPRINT_SCHEMA_VERSION = 1
_FINGERPRINT_INPUTS = (
    Path("src/phonology/build_lexicon.py"),
    Path("src/phonology/lsj_extractor.py"),
    Path("src/phonology/ipa_converter.py"),
    Path("src/phonology/transliterate.py"),
    Path("src/phonology/betacode.py"),
    Path("data/lexicon/pos_overrides.yaml"),
    Path("data/lexicon/greek_lemmas.schema.json"),
)


def _default_project_root() -> Path:
    """Return the repository root for this source checkout."""
    return Path(__file__).resolve().parents[2]


def _default_output_path(project_root: Path) -> Path:
    """Return the default generated lexicon output path."""
    return project_root / "data" / "lexicon" / "greek_lemmas.json"


def _metadata_path_for_output(output_path: Path) -> Path:
    """Return the sidecar metadata path for a generated lexicon output."""
    return output_path.with_name(f"{output_path.stem}.meta.json")


def _default_lsj_repo_dir(project_root: Path) -> Path:
    """Return the default LSJ repository checkout path."""
    return project_root / "data" / "external" / "lsj"


def default_xml_dir(project_root: Path) -> Path:
    """Return the default LSJ XML checkout path."""
    return _default_lsj_repo_dir(project_root) / _LSJ_XML_RELATIVE_PATH


def _lsj_xml_dir_for_repo(lsj_repo_dir: Path) -> Path:
    """Return the LSJ XML directory for a repository checkout root."""
    return lsj_repo_dir / _LSJ_XML_RELATIVE_PATH


def _resolve_lsj_repo_dir_override(lsj_repo_dir: Path | None) -> Path | None:
    """Return an explicit LSJ repo directory override from args or environment."""
    if lsj_repo_dir is not None:
        return lsj_repo_dir.expanduser()

    override = os.environ.get(LSJ_REPO_DIR_ENV_VAR)
    if not override:
        return None
    return Path(override).expanduser()


def _infer_lsj_repo_dir(project_root: Path, resolved_xml_dir: Path) -> Path | None:
    """Infer the LSJ repo root only for the default checkout layout."""
    default_repo_dir = _default_lsj_repo_dir(project_root)
    default_xml_dir = _lsj_xml_dir_for_repo(default_repo_dir).resolve(strict=False)
    normalized_xml_dir = resolved_xml_dir.resolve(strict=False)
    if normalized_xml_dir == default_xml_dir:
        return default_repo_dir

    logger.info(
        "Skipping LSJ clone because XML directory %s is outside the default checkout path %s; "
        "set %s or pass lsj_repo_dir to clone into a custom location.",
        resolved_xml_dir,
        default_xml_dir,
        LSJ_REPO_DIR_ENV_VAR,
    )
    return None


def _run_subprocess(
    command: list[str],
    *,
    cwd: Path | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess command with text-mode output enabled."""
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        check=True,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def _validate_lsj_repo_dir_target(lsj_repo_dir: Path, project_root: Path) -> None:
    """Validate that the LSJ checkout target is safe to create or replace."""
    if lsj_repo_dir.is_symlink():
        logger.error(
            "Refusing to replace lsj_repo_dir %s: it is a symbolic link",
            lsj_repo_dir,
        )
        raise RuntimeError(
            f"Safety check failed: lsj_repo_dir {lsj_repo_dir} is a symbolic link; "
            "remove the symlink manually before retrying."
        )

    if not lsj_repo_dir.exists():
        return

    resolved_project_root = project_root.resolve()
    resolved_path = lsj_repo_dir.resolve(strict=False)
    if not resolved_path.is_relative_to(resolved_project_root):
        logger.error(
            "Refusing to replace existing lsj_repo_dir %s: resolved path %s is not under project root %s",
            lsj_repo_dir,
            resolved_path,
            resolved_project_root,
        )
        raise RuntimeError(
            "Safety check failed: existing lsj_repo_dir "
            f"{resolved_path} is not under project root {resolved_project_root}"
        )


def _cleanup_directory(path: Path) -> None:
    """Best-effort cleanup for temporary clone directories."""
    if not path.exists() and not path.is_symlink():
        return
    shutil.rmtree(path, ignore_errors=True)


def _replace_lsj_checkout(prepared_repo_dir: Path, target_repo_dir: Path) -> None:
    """Replace the target LSJ checkout with a prepared sibling directory."""
    backup_repo_dir = target_repo_dir.with_name(f"{target_repo_dir.name}.bak")
    _cleanup_directory(backup_repo_dir)
    try:
        if target_repo_dir.exists():
            target_repo_dir.replace(backup_repo_dir)
        prepared_repo_dir.replace(target_repo_dir)
    except Exception:
        if not target_repo_dir.exists() and backup_repo_dir.exists():
            try:
                backup_repo_dir.replace(target_repo_dir)
            except Exception as rollback_exc:
                logger.exception(
                    "Failed to rollback backup from %s to %s: %s",
                    backup_repo_dir,
                    target_repo_dir,
                    rollback_exc,
                )
        raise
    else:
        _cleanup_directory(backup_repo_dir)


def _clone_lsj_repo(lsj_repo_dir: Path, project_root: Path) -> None:
    """Clone the Perseus LSJ repository with retry and a Python timeout."""
    git_executable = shutil.which("git")
    if git_executable is None:
        raise RuntimeError("git is required to clone Perseus LSJ data")

    lsj_repo_dir.parent.mkdir(parents=True, exist_ok=True)
    _validate_lsj_repo_dir_target(lsj_repo_dir, project_root)

    last_error: subprocess.CalledProcessError | subprocess.TimeoutExpired | None = None
    temp_repo_dir = lsj_repo_dir.with_name(f"{lsj_repo_dir.name}.tmp")
    clone_succeeded = False
    for attempt in range(1, _CLONE_MAX_ATTEMPTS + 1):
        _cleanup_directory(temp_repo_dir)
        clone_command = [
            git_executable,
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
            "https://github.com/PerseusDL/lexica.git",
            str(temp_repo_dir),
        ]
        try:
            _run_subprocess(clone_command, timeout=_CLONE_TIMEOUT_SECONDS)
            _run_subprocess(
                [git_executable, "sparse-checkout", "set", "CTS_XML_TEI/perseus/pdllex/grc/lsj"],
                cwd=temp_repo_dir,
                timeout=_CLONE_TIMEOUT_SECONDS,
            )
            xml_dir = _lsj_xml_dir_for_repo(temp_repo_dir)
            if not xml_dir.is_dir():
                raise FileNotFoundError(
                    f"LSJ XML directory not found after clone: {xml_dir}"
                )
            _replace_lsj_checkout(temp_repo_dir, lsj_repo_dir)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as err:
            last_error = err
            if attempt == _CLONE_MAX_ATTEMPTS:
                break
            logger.warning(
                "LSJ clone failed (attempt %d/%d): %s",
                attempt,
                _CLONE_MAX_ATTEMPTS,
                err,
            )
            _cleanup_directory(temp_repo_dir)
            time.sleep(2 ** attempt + random.uniform(0, 1))
            continue
        except Exception:
            _cleanup_directory(temp_repo_dir)
            raise
        clone_succeeded = True
        break

    if not clone_succeeded:
        _cleanup_directory(temp_repo_dir)
        raise RuntimeError(
            f"Failed to clone Perseus LSJ after {_CLONE_MAX_ATTEMPTS} attempts"
        ) from last_error


def ensure_lsj_checkout(
    project_root: Path,
    *,
    xml_dir: Path | None = None,
    lsj_repo_dir: Path | None = None,
    resolved_lsj_repo_dir: Path | None = None,
    allow_clone: bool = True,
) -> Path:
    """Ensure the LSJ checkout exists and return the XML directory.

    Args:
        project_root: Path to the repository root used to resolve default LSJ paths.
        xml_dir: Optional path to the LSJ XML directory. When provided, use this
            directory directly instead of the default checkout layout.
        lsj_repo_dir: Optional checkout root for the LSJ repository. When provided
            directly, derive ``xml_dir`` from this checkout root if ``xml_dir``
            was omitted.
        resolved_lsj_repo_dir: Optional pre-resolved LSJ checkout root. When
            provided, use this path as-is instead of re-reading the environment
            variable specified by LSJ_REPO_DIR_ENV_VAR or expanding ``lsj_repo_dir``.
        allow_clone: When True (default), clone the default LSJ checkout if the
            XML directory is missing. When False, fail fast instead of cloning.
    """
    if lsj_repo_dir is not None and resolved_lsj_repo_dir is not None:
        raise ValueError("lsj_repo_dir and resolved_lsj_repo_dir are mutually exclusive")

    selected_lsj_repo_dir: Path | None = None
    if xml_dir is not None:
        resolved_xml_dir = xml_dir.expanduser()
    else:
        if resolved_lsj_repo_dir is not None:
            selected_lsj_repo_dir = resolved_lsj_repo_dir
        elif lsj_repo_dir is not None:
            selected_lsj_repo_dir = lsj_repo_dir.expanduser()
        else:
            selected_lsj_repo_dir = _resolve_lsj_repo_dir_override(None)

        if selected_lsj_repo_dir is not None:
            resolved_xml_dir = _lsj_xml_dir_for_repo(selected_lsj_repo_dir)
        else:
            resolved_xml_dir = default_xml_dir(project_root)

    if resolved_xml_dir.is_dir():
        return resolved_xml_dir

    clone_repo_dir = selected_lsj_repo_dir or _infer_lsj_repo_dir(project_root, resolved_xml_dir)
    if clone_repo_dir is None:
        raise FileNotFoundError(
            f"LSJ XML directory not found and no checkout root was provided: {resolved_xml_dir}"
        )

    if not allow_clone:
        raise FileNotFoundError(
            "LSJ XML directory not found and build-time cloning is disabled: "
            f"{resolved_xml_dir}. Generate data/lexicon/greek_lemmas.json ahead of time, "
            f"or provide a local LSJ checkout via --xml-dir, --lsj-repo-dir, or {LSJ_REPO_DIR_ENV_VAR}."
        )

    logger.info("Cloning Perseus LSJ repository into %s", clone_repo_dir)
    _clone_lsj_repo(clone_repo_dir, project_root)

    if not resolved_xml_dir.is_dir():
        raise FileNotFoundError(
            f"LSJ XML directory not found after clone: {resolved_xml_dir}"
        )
    return resolved_xml_dir


def _run_extractor(
    *,
    xml_dir: Path,
    output_path: Path,
    limit: int | None = None,
    dry_run: bool = False,
) -> int:
    """Invoke the LSJ extractor using the in-tree source module."""
    from phonology.lsj_extractor import _load_pos_overrides, main as extract_lsj

    _load_pos_overrides(cli_mode=True)

    return extract_lsj(
        xml_dir=xml_dir,
        output_path=output_path,
        limit=limit,
        dry_run=dry_run,
    )


def _fingerprint_path_label(path: Path, project_root: Path) -> str:
    """Return a stable path label for fingerprint input records."""
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return resolved_path.as_posix()


def _tracked_input_records(project_root: Path, xml_dir: Path) -> list[dict[str, Any]]:
    """Return fingerprint records for source inputs and LSJ XML files."""
    from phonology.lsj_extractor import find_xml_files

    tracked_paths = [project_root / relative_path for relative_path in _FINGERPRINT_INPUTS]
    tracked_paths.extend(find_xml_files(xml_dir))

    records: list[dict[str, Any]] = []
    for path in tracked_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Fingerprint input not found: {path}. "
                "Ensure all tracked source files exist."
            )
        stat = path.stat()
        content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(
            {
                "path": _fingerprint_path_label(path, project_root),
                "size": stat.st_size,
                # Persist mtime_ns next to the _fingerprint_path_label-derived path
                # for debugging/audit; build_fingerprint_payload excludes mtime_ns
                # from the digest to avoid timestamp-only fingerprint churn.
                "mtime_ns": stat.st_mtime_ns,
                "content_hash": content_hash,
            }
        )
    return records


def _fingerprint_digest_for_records(records: list[dict[str, Any]]) -> str:
    """Return the fingerprint digest for a sequence of tracked input records."""
    digest = hashlib.sha256()
    for record in records:
        digest.update(record["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["size"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(record["content_hash"].encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _is_missing_lsj_xml_input(path_label: str, project_root: Path) -> bool:
    """Return True when ``path_label`` looks like a missing LSJ XML input file."""
    candidate = Path(path_label)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return bool(_LSJ_XML_FILENAME_RE.fullmatch(candidate.name))


def _current_record_from_path(path: Path, path_label: str) -> dict[str, Any]:
    """Return a fresh fingerprint record for an existing path."""
    stat = path.stat()
    return {
        "path": path_label,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "content_hash": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _metadata_input_records_for_reuse(
    metadata_payload: dict[str, Any],
    project_root: Path,
) -> list[dict[str, Any]] | None:
    """Rebuild fingerprint records from metadata for reuse without an LSJ checkout."""
    raw_inputs = metadata_payload.get("inputs")
    if not isinstance(raw_inputs, list):
        logger.info("Freshness metadata is missing a valid inputs list")
        return None

    records: list[dict[str, Any]] = []
    for raw_record in raw_inputs:
        if not isinstance(raw_record, dict):
            logger.info("Freshness metadata contains a non-object input record")
            return None

        path_label = raw_record.get("path")
        if not isinstance(path_label, str) or not path_label:
            logger.info("Freshness metadata contains an input record without a valid path")
            return None

        candidate_path = Path(path_label)
        if not candidate_path.is_absolute():
            candidate_path = project_root / candidate_path

        if candidate_path.exists():
            records.append(_current_record_from_path(candidate_path, path_label))
            continue

        if not _is_missing_lsj_xml_input(path_label, project_root):
            logger.info("Freshness input is missing and cannot be reused: %s", path_label)
            return None

        size = raw_record.get("size")
        content_hash = raw_record.get("content_hash")
        if not isinstance(size, int):
            logger.info(
                "Freshness metadata for missing LSJ XML input has a non-integer size: %s",
                path_label,
            )
            return None
        if size < 0:
            logger.info(
                "Freshness metadata for missing LSJ XML input has a negative size: %s",
                path_label,
            )
            return None
        if not isinstance(content_hash, str):
            logger.info(
                "Freshness metadata for missing LSJ XML input has a non-string content_hash: %s",
                path_label,
            )
            return None
        if not content_hash:
            logger.info(
                "Freshness metadata for missing LSJ XML input has an empty content_hash: %s",
                path_label,
            )
            return None

        mtime_ns = raw_record.get("mtime_ns")
        if mtime_ns is not None and not isinstance(mtime_ns, int):
            logger.info(
                "Freshness metadata for missing LSJ XML input has a non-integer mtime_ns: %s",
                path_label,
            )
            return None

        records.append(
            {
                "path": path_label,
                "size": size,
                "mtime_ns": mtime_ns,
                "content_hash": content_hash,
            }
        )
    return records


def _is_output_fresh_without_checkout(
    *,
    project_root: Path,
    output_path: Path,
    metadata_path: Path,
) -> bool:
    """Return True when metadata proves the output is fresh without an LSJ checkout."""
    if not _is_reusable_output_document(project_root=project_root, output_path=output_path):
        return False

    actual_payload = _read_metadata(metadata_path)
    if actual_payload is None:
        return False

    current_records = _metadata_input_records_for_reuse(actual_payload, project_root)
    if current_records is None:
        return False
    return actual_payload["fingerprint"] == _fingerprint_digest_for_records(current_records)


def build_fingerprint_payload(project_root: Path, xml_dir: Path) -> dict[str, Any]:
    """Build freshness metadata for generated lexicon reuse checks.

    Args:
        project_root: Path to the repository root used to resolve tracked source
            and data inputs.
        xml_dir: Path to the LSJ XML input directory whose XML files are included
            in the fingerprint.

    Returns:
        A dictionary with:
        - ``schema_version``: ``int`` schema version for the metadata payload.
        - ``fingerprint``: ``str`` checksum computed from the tracked input
          records.
        - ``inputs``: ``list[dict[str, Any]]`` of tracked input records gathered
          by ``_tracked_input_records`` and digested by
          ``_fingerprint_digest_for_records``.
    """
    inputs = _tracked_input_records(project_root, xml_dir)
    return {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "fingerprint": _fingerprint_digest_for_records(inputs),
        "inputs": inputs,
    }


def _read_metadata(metadata_path: Path) -> dict[str, Any] | None:
    """Read sidecar freshness metadata, returning ``None`` when invalid."""
    try:
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.info("Freshness metadata missing at %s", metadata_path)
        return None
    except (OSError, UnicodeError, json.JSONDecodeError) as err:
        logger.warning("Freshness metadata at %s is unreadable: %s", metadata_path, err)
        return None

    if not isinstance(raw, dict):
        logger.warning("Freshness metadata at %s must be a JSON object", metadata_path)
        return None
    if raw.get("schema_version") != FINGERPRINT_SCHEMA_VERSION:
        logger.info(
            "Freshness metadata has an unsupported schema version: found=%s expected=%s at %s",
            raw.get("schema_version"),
            FINGERPRINT_SCHEMA_VERSION,
            metadata_path,
        )
        return None
    if not isinstance(raw.get("fingerprint"), str) or not raw["fingerprint"]:
        logger.warning("Freshness metadata at %s is missing a valid fingerprint", metadata_path)
        return None
    return raw


def _is_output_fresh(
    *,
    project_root: Path,
    output_path: Path,
    metadata_path: Path,
    expected_payload: dict[str, Any],
) -> bool:
    """Return True when the existing output and sidecar metadata are current."""
    if not _is_reusable_output_document(project_root=project_root, output_path=output_path):
        return False

    actual_payload = _read_metadata(metadata_path)
    if actual_payload is None:
        return False
    return actual_payload.get("fingerprint") == expected_payload["fingerprint"]


def _is_reusable_output_document(*, project_root: Path, output_path: Path) -> bool:
    """Check whether an existing lexicon JSON file is valid and safe to reuse.

    Args:
        project_root: Repository root containing the schema file at
            ``data/lexicon/greek_lemmas.schema.json``.
        output_path: Path to the existing lexicon JSON file to validate.

    Returns:
        True if the output file exists, contains valid JSON, is a JSON object,
        and passes schema validation via ``phonology.lsj_extractor.validate_document``.
        False if the file is missing, unreadable, malformed, or fails validation.

    Raises:
        This function does not raise exceptions. All expected errors
        (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError, ValueError)
        are caught and logged, returning False to signal that regeneration is needed.

    Side Effects:
        Logs warnings at the ``logger.warning`` level when the output is invalid
        or unreadable, including the reason for rejection.
    """
    if not output_path.is_file():
        return False

    schema_path = project_root / "data" / "lexicon" / "greek_lemmas.schema.json"
    try:
        document = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            logger.warning(
                "Existing lexicon output at %s is not a JSON object; forcing regeneration",
                output_path,
            )
            return False

        from phonology.lsj_extractor import validate_document

        validate_document(document, schema_path=schema_path)
    except FileNotFoundError as err:
        logger.warning(
            "Existing lexicon output at %s cannot be reused because a required file is missing: %s",
            output_path,
            err,
        )
        return False
    except (OSError, UnicodeError, json.JSONDecodeError) as err:
        logger.warning(
            "Existing lexicon output at %s is unreadable or invalid JSON; forcing regeneration: %s",
            output_path,
            err,
        )
        return False
    except ValueError as err:
        logger.warning(
            "Existing lexicon output at %s failed validation; forcing regeneration: %s",
            output_path,
            err,
        )
        return False

    return True


def _write_metadata(metadata_path: Path, payload: dict[str, Any]) -> None:
    """Persist sidecar freshness metadata next to the generated lexicon."""
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def ensure_generated_lexicon(
    *,
    project_root: Path | None = None,
    output_path: Path | None = None,
    xml_dir: Path | None = None,
    lsj_repo_dir: Path | None = None,
    allow_clone: bool = True,
    limit: int | None = None,
    dry_run: bool = False,
    skip_if_present: bool = False,
) -> bool:
    """Ensure the generated LSJ lexicon exists.

    Args:
        project_root: Path to the repository root.
        output_path: Path to write the generated lexicon.
        xml_dir: Optional path to the LSJ XML directory.
        lsj_repo_dir: Optional checkout root for the LSJ repository.
        allow_clone: When True (default), clone the LSJ checkout if missing.
            When False, fail fast instead of cloning.
        limit: Process only the first N entries (for development).
        dry_run: Parse and validate without writing output.
        skip_if_present: Skip extraction when output exists and is fresh.

    Returns ``True`` when extraction ran and ``False`` when an existing output
    was reused because ``skip_if_present=True`` and the output is still fresh.
    """
    resolved_project_root = project_root or _default_project_root()
    resolved_output_path = output_path or _default_output_path(resolved_project_root)
    metadata_path = _metadata_path_for_output(resolved_output_path)
    resolved_lsj_repo_dir = _resolve_lsj_repo_dir_override(lsj_repo_dir)
    has_existing_env_lsj_checkout = (
        lsj_repo_dir is None
        and resolved_lsj_repo_dir is not None
        and _lsj_xml_dir_for_repo(resolved_lsj_repo_dir).is_dir()
    )
    if (
        skip_if_present
        and xml_dir is None
        and lsj_repo_dir is None
        and not has_existing_env_lsj_checkout
        and _is_output_fresh_without_checkout(
            project_root=resolved_project_root,
            output_path=resolved_output_path,
            metadata_path=metadata_path,
        )
    ):
        logger.info(
            "Lexicon already exists at %s and matches current inputs; skipping generation",
            resolved_output_path,
        )
        return False

    resolved_xml_dir = ensure_lsj_checkout(
        resolved_project_root,
        xml_dir=xml_dir,
        resolved_lsj_repo_dir=resolved_lsj_repo_dir,
        allow_clone=allow_clone,
    )
    expected_payload = build_fingerprint_payload(resolved_project_root, resolved_xml_dir)

    if skip_if_present and _is_output_fresh(
        project_root=resolved_project_root,
        output_path=resolved_output_path,
        metadata_path=metadata_path,
        expected_payload=expected_payload,
    ):
        logger.info(
            "Lexicon already exists at %s and matches current inputs; skipping generation",
            resolved_output_path,
        )
        return False

    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    exit_code = _run_extractor(
        xml_dir=resolved_xml_dir,
        output_path=resolved_output_path,
        limit=limit,
        dry_run=dry_run,
    )
    if exit_code != 0:
        raise RuntimeError(f"LSJ extraction failed with exit code {exit_code}")
    if not dry_run:
        _write_metadata(metadata_path, expected_payload)
    return True


def run_cli() -> int:
    """Parse arguments and generate the lexicon."""
    parser = argparse.ArgumentParser(
        description="Ensure the generated Greek lexicon exists.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=_default_project_root(),
        help="Repository root containing src/, data/, and scripts/.",
    )
    parser.add_argument(
        "--xml-dir",
        type=Path,
        default=None,
        help="Directory containing grc.lsj.perseus-eng*.xml files.",
    )
    parser.add_argument(
        "--lsj-repo-dir",
        type=Path,
        default=None,
        help=(
            "Checkout root for the Perseus LSJ repository "
            f"(default env override: {LSJ_REPO_DIR_ENV_VAR})."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: data/lexicon/greek_lemmas.json).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N entries (for development).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate without writing output.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--if-missing",
        action="store_true",
        help="Skip extraction only when the output JSON exists and is fresh.",
    )
    parser.add_argument(
        "--no-clone",
        action="store_true",
        help="Fail fast if the LSJ XML directory is missing instead of cloning.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        ensure_generated_lexicon(
            project_root=args.project_root,
            output_path=args.output,
            xml_dir=args.xml_dir,
            lsj_repo_dir=args.lsj_repo_dir,
            allow_clone=not args.no_clone,
            limit=args.limit,
            dry_run=args.dry_run,
            skip_if_present=args.if_missing,
        )
    except KeyboardInterrupt:
        logger.info("Lexicon generation interrupted by user")
        return 130
    except Exception as exc:
        if args.verbose:
            logger.exception("Lexicon generation failed")
        else:
            logger.error("Lexicon generation failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
