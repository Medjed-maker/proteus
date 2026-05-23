"""Corpus adapter interfaces and static metadata adapter implementation."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import yaml
from pydantic import ValidationError

from ._models import _FORBIDDEN_TEXT_FIELDS as _FORBIDDEN_SOURCE_TEXT_FIELDS
from ._models import SourceReference

# Promotion gate: the static loader only accepts metadata documents whose
# ``_meta.status`` matches one of these values. Promoting a corpus source file
# to a more authoritative status (e.g. ``"reviewed"``) must be paired with
# updating this constant.
_ALLOWED_META_STATUSES = frozenset({"proof_of_concept"})


class CorpusSourceDataError(ValueError):
    """Raised when corpus source metadata is missing or malformed."""


class CorpusAdapter(Protocol):
    """Lookup source metadata for a ranked search candidate."""

    def lookup(
        self,
        *,
        entry_id: str,
        headword: str,
        language: str,
    ) -> tuple[SourceReference, ...]:
        """Return public source metadata for one search result."""
        ...


class EmptyCorpusAdapter:
    """Corpus adapter that returns no source metadata."""

    def lookup(
        self,
        *,
        entry_id: str,
        headword: str,
        language: str,
    ) -> tuple[SourceReference, ...]:
        return ()


class CompositeCorpusAdapter:
    """Combine multiple corpus adapters while preserving adapter order."""

    def __init__(self, adapters: Iterable[CorpusAdapter]) -> None:
        self._adapters = tuple(adapters)

    def lookup(
        self,
        *,
        entry_id: str,
        headword: str,
        language: str,
    ) -> tuple[SourceReference, ...]:
        references: list[SourceReference] = []
        for adapter in self._adapters:
            references.extend(
                adapter.lookup(
                    entry_id=entry_id,
                    headword=headword,
                    language=language,
                )
            )
        return tuple(references)


class StaticCorpusAdapter:
    """Corpus adapter backed by a checked-in YAML metadata map."""

    def __init__(
        self,
        references_by_entry_id: Mapping[str, tuple[SourceReference, ...]],
    ) -> None:
        self._references_by_entry_id = dict(references_by_entry_id)

    def lookup(
        self,
        *,
        entry_id: str,
        headword: str,
        language: str,
    ) -> tuple[SourceReference, ...]:
        return self._references_by_entry_id.get(entry_id, ())


EMPTY_CORPUS_ADAPTER = EmptyCorpusAdapter()


def _reject_forbidden_fields(value: Any, *, path: str) -> None:
    if isinstance(value, dict):
        forbidden = sorted(set(value) & _FORBIDDEN_SOURCE_TEXT_FIELDS)
        if forbidden:
            raise CorpusSourceDataError(
                f"{path} contains forbidden source-text fields: "
                + ", ".join(forbidden)
            )
        for key, nested in value.items():
            _reject_forbidden_fields(nested, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_forbidden_fields(nested, path=f"{path}[{index}]")


def _validate_meta_status(payload: Mapping[str, Any]) -> None:
    meta = payload.get("_meta")
    if not isinstance(meta, Mapping):
        raise CorpusSourceDataError("Corpus source YAML must define a _meta mapping")
    status = meta.get("status")
    if not isinstance(status, str) or not status.strip():
        raise CorpusSourceDataError(
            "Corpus source YAML _meta.status must be a non-empty string"
        )
    if status not in _ALLOWED_META_STATUSES:
        allowed = ", ".join(sorted(_ALLOWED_META_STATUSES))
        raise CorpusSourceDataError(
            f"Unsupported corpus source _meta.status {status!r}; "
            f"expected one of: {allowed}"
        )


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CorpusSourceDataError(f"Could not read corpus source data: {path}") from exc
    except yaml.YAMLError as exc:
        raise CorpusSourceDataError(f"Could not parse corpus source data: {path}") from exc

    if not isinstance(payload, dict):
        raise CorpusSourceDataError("Corpus source YAML must be a mapping")
    _reject_forbidden_fields(payload, path="$")
    _validate_meta_status(payload)
    return payload


def _parse_reference(raw_reference: Any, *, entry_id: str, index: int) -> SourceReference:
    if not isinstance(raw_reference, dict):
        raise CorpusSourceDataError(
            f"Corpus source entry {entry_id!r} item {index} must be a mapping"
        )
    try:
        return SourceReference.model_validate(raw_reference)
    except ValidationError as exc:
        raise CorpusSourceDataError(
            f"Corpus source entry {entry_id!r} item {index} is invalid: {exc}"
        ) from exc


def _parse_entries(raw_entries: Any) -> dict[str, tuple[SourceReference, ...]]:
    if not isinstance(raw_entries, dict):
        raise CorpusSourceDataError("Corpus source YAML must define an entries mapping")

    parsed: dict[str, tuple[SourceReference, ...]] = {}
    for entry_id, raw_references in raw_entries.items():
        if not isinstance(entry_id, str) or not entry_id.strip():
            raise CorpusSourceDataError("Corpus source entry ids must be non-empty strings")
        normalized_entry_id = entry_id.strip()
        if normalized_entry_id in parsed:
            raise CorpusSourceDataError(
                "Duplicate corpus source entry id "
                f"{entry_id!r} normalizes to {normalized_entry_id!r}"
            )
        if not isinstance(raw_references, list):
            raise CorpusSourceDataError(
                f"Corpus source entry {entry_id!r} must contain a list of references"
            )
        parsed[normalized_entry_id] = tuple(
            _parse_reference(
                raw_reference,
                entry_id=normalized_entry_id,
                index=index,
            )
            for index, raw_reference in enumerate(raw_references)
        )
    return parsed


@lru_cache(maxsize=None)
def _cached_load_static_corpus_adapter(resolved_path: Path) -> StaticCorpusAdapter:
    """Memoized loader keyed by a resolved absolute ``Path``.

    The cache uses ``maxsize=None`` because corpus source files are checked-in
    and the set of distinct paths is small and bounded by the number of
    language profiles. Within one interpreter process, edits to the underlying
    YAML are not picked up; restart the process to refresh data.
    """
    payload = _load_yaml_mapping(resolved_path)
    return StaticCorpusAdapter(_parse_entries(payload.get("entries")))


def load_static_corpus_adapter(path: str | Path) -> StaticCorpusAdapter:
    """Load and validate a static YAML corpus adapter.

    Paths are normalised via ``Path.resolve()`` before lookup so that the same
    file referenced via different spellings (relative vs absolute, ``str`` vs
    ``Path``) shares one cached adapter instance. The cache lives for the
    lifetime of the interpreter; restart the process to pick up data changes.
    """
    return _cached_load_static_corpus_adapter(Path(path).resolve())


def safe_lookup(
    adapter: CorpusAdapter,
    *,
    entry_id: str | None,
    headword: str,
    language: str,
    logger: logging.Logger,
) -> tuple[SourceReference, ...]:
    """Run ``adapter.lookup`` while degrading failures to an empty tuple.

    Returns ``()`` when ``entry_id`` is ``None``. When the adapter raises an
    ``Exception`` (Protocol implementations may raise anything), the failure is
    logged at warning level and an empty tuple is returned so the surrounding
    search request can still complete. ``BaseException`` subclasses
    (e.g. ``KeyboardInterrupt``, ``SystemExit``) propagate unchanged.
    """
    if entry_id is None:
        return ()
    try:
        return adapter.lookup(
            entry_id=entry_id,
            headword=headword,
            language=language,
        )
    except Exception as err:
        logger.warning(
            "Corpus adapter lookup failed for language=%s entry_id=%r headword=%r: %s",
            language,
            entry_id,
            headword,
            err,
            exc_info=True,
        )
        return ()


__all__ = [
    "CompositeCorpusAdapter",
    "CorpusAdapter",
    "CorpusSourceDataError",
    "EMPTY_CORPUS_ADAPTER",
    "EmptyCorpusAdapter",
    "StaticCorpusAdapter",
    "load_static_corpus_adapter",
    "safe_lookup",
]
