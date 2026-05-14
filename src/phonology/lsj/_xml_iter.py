"""LSJ extractor helper module.

Logger uses the literal ``phonology.lsj_extractor`` name so existing
``caplog.set_level(logger="phonology.lsj_extractor")`` blocks in
``tests/test_lsj_extractor.py`` keep capturing diagnostics from this module
after the split.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("phonology.lsj_extractor")

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ._extract import extract_entry


def iter_xml_entries(xml_path: Path) -> Iterator[dict[str, Any]]:
    """Yield lemma dicts from a single LSJ XML file using streaming parse."""
    from lxml import etree  # type: ignore[import-untyped]

    # Harden iterparse against XXE and billion-laughs style attacks. LSJ
    # sources are normally trusted offline inputs, but disabling entity
    # resolution, DTD loading, and network access removes the attack surface.
    context = etree.iterparse(
        str(xml_path),
        events=("end",),
        tag="entryFree",
        recover=True,
        resolve_entities=False,
        load_dtd=False,
        no_network=True,
    )
    for _event, element in context:
        entry = extract_entry(element)
        if entry is not None:
            yield entry
        # Free memory: remove all previously processed siblings.
        # The current element stays in the tree until the next iteration.
        element.clear()
        parent = element.getparent()
        while parent is not None and element.getprevious() is not None:
            del parent[0]


def find_xml_files(xml_dir: Path) -> list[Path]:
    """Find and sort all LSJ XML files in the given directory."""
    files = sorted(
        xml_dir.glob("grc.lsj.perseus-eng*.xml"),
        key=lambda p: int("".join(filter(str.isdigit, p.stem.split("eng")[-1])) or "0"),
    )
    if not files:
        raise FileNotFoundError(
            f"No LSJ XML files found in {xml_dir} (expected grc.lsj.perseus-eng*.xml)"
        )
    return files


def extract_all(xml_dir: Path, *, limit: int | None = None) -> Iterator[dict[str, Any]]:
    """Yield lemma dicts from all LSJ XML files in order."""
    files = find_xml_files(xml_dir)
    count = 0
    seen_ids: set[str] = set()

    for xml_file in files:
        logger.info("Processing %s", xml_file.name)
        for entry in iter_xml_entries(xml_file):
            # Deduplicate by ID
            if entry["id"] in seen_ids:
                continue
            seen_ids.add(entry["id"])

            yield entry
            count += 1
            if limit is not None and count >= limit:
                return


# ---------------------------------------------------------------------------
# Document building
# ---------------------------------------------------------------------------


