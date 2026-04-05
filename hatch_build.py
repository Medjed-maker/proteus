"""Custom Hatch build hook for ensuring the generated lexicon exists."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

logger = logging.getLogger(__name__)


class build_hook(BuildHookInterface):
    """Ensure wheel builds package a fresh generated lexicon asset."""

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        """Prepare the build environment and ensure lexicon assets exist.

        Args:
            version: Hatch build version string (``"editable"`` for editable
                installs, ``"standard"`` for wheel/sdist builds).
            build_data: Mutable Hatch build metadata.  For editable installs,
                ``force_include_editable`` is populated with force-include
                entries excluding the generated lexicon.

        Side Effects:
            For standard builds, ensures the project's ``src`` directory is on
            ``sys.path`` and calls
            ``ensure_generated_lexicon(project_root=..., skip_if_present=True)``.
            Importing ``phonology.build_lexicon`` may occur as part of this setup.
            Stale lexicon outputs are regenerated instead of being silently reused.

            For editable builds, lexicon generation is skipped entirely and the
            lexicon asset is excluded from the editable force-include set.
        """
        if version == "editable":
            lexicon_source = (Path(self.root) / "data" / "lexicon" / "greek_lemmas.json").resolve(
                strict=False
            )
            editable_force_include = {
                source: destination
                for source, destination in self.build_config.get_force_include().items()
                if Path(source).resolve(strict=False) != lexicon_source
            }
            build_data["force_include_editable"] = editable_force_include
            logger.info(
                "Skipping lexicon build hook for editable builds; explicit extraction remains the caller's responsibility."
            )
            return

        src_dir = str(Path(self.root, "src"))
        added_to_path = src_dir not in sys.path
        if added_to_path:
            sys.path.insert(0, src_dir)

        try:
            from phonology.build_lexicon import ensure_generated_lexicon

            ensure_generated_lexicon(
                project_root=Path(self.root),
                skip_if_present=True,
                allow_clone=False,
            )
        except Exception as exc:
            logger.exception(
                "Build hook failed: ensure_generated_lexicon(project_root=%s, skip_if_present=True, allow_clone=False) "
                "raised %s: %s. Wheel builds reuse fresh lexicon outputs or require a local LSJ checkout; "
                "the build hook does not clone external LSJ data.",
                self.root,
                type(exc).__name__,
                exc,
            )
            raise
        finally:
            if added_to_path:
                try:
                    sys.path.remove(src_dir)
                except ValueError:
                    pass
