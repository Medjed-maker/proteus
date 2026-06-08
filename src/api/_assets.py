"""Packaged HTML asset loading for the REST application."""

from __future__ import annotations

import html
import logging
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import HTMLResponse

logger = logging.getLogger(__name__)

_FRONTEND_PATH = Path(__file__).resolve().parents[1] / "web" / "index.html"
_CHANGELOG_PATH = Path(__file__).resolve().parents[1] / "web" / "changelog.html"
_STATIC_DIR = Path(__file__).resolve().parents[1] / "web" / "static"


def _load_html_asset(path: Path, *, label: str) -> str | None:
    """Load and cache a packaged HTML document."""
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        logger.exception("Failed to read %s HTML from %s", label, path)
        return None


def _load_frontend_html() -> str | None:
    """Load the packaged frontend HTML document."""
    return _load_html_asset(_FRONTEND_PATH, label="frontend")


def _load_changelog_html() -> str | None:
    """Load the packaged changelog HTML document."""
    return _load_html_asset(_CHANGELOG_PATH, label="changelog")


def _build_deprecation_link_header(base_url: str, docs_path: str) -> str:
    """Return an absolute URI Link header value per RFC 8288."""
    base = base_url.rstrip("/")
    path = docs_path.lstrip("/")
    return f'<{base}/{path}>; rel="deprecation"'


def render_html_asset(
    html_document: str | None,
    *,
    app_version: str,
    missing_detail: str,
) -> HTMLResponse:
    """Render a cached HTML document with the current application version."""
    if html_document is None:
        raise HTTPException(status_code=404, detail=missing_detail)
    escaped_version = html.escape(app_version)
    return HTMLResponse(html_document.replace("{{APP_VERSION}}", escaped_version))


__all__ = [
    "_CHANGELOG_PATH",
    "_FRONTEND_PATH",
    "_STATIC_DIR",
    "_build_deprecation_link_header",
    "_load_changelog_html",
    "_load_frontend_html",
    "render_html_asset",
]
