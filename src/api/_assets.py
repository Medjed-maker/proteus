"""Packaged HTML asset loading for the REST application."""

from __future__ import annotations

import html
import json
import logging
import os
from pathlib import Path
from urllib.parse import quote

from fastapi import HTTPException
from fastapi.responses import HTMLResponse

logger = logging.getLogger(__name__)

_FRONTEND_PATH = Path(__file__).resolve().parents[1] / "web" / "index.html"
_CHANGELOG_PATH = Path(__file__).resolve().parents[1] / "web" / "changelog.html"
_STATIC_DIR = Path(__file__).resolve().parents[1] / "web" / "static"
_GOOGLE_ANALYTICS_ENV_VAR = "PROTEUS_GOOGLE_ANALYTICS_ID"
_GOOGLE_ANALYTICS_TAG_PLACEHOLDER = "{{GOOGLE_ANALYTICS_TAG}}"


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


def _javascript_string_literal(value: str) -> str:
    """Return a JavaScript string literal safe to embed in an HTML ``<script>``.

    ``json.dumps`` produces a valid JS string literal; the extra ``<``/``>``/``&``
    escaping then prevents the value from breaking out of the script element
    (e.g. via a literal ``</script>``) when the document is parsed as HTML.
    """
    return (
        json.dumps(value)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _build_google_analytics_tag() -> str:
    """Return the Google Analytics tag when deployment config enables it."""
    measurement_id = os.environ.get(_GOOGLE_ANALYTICS_ENV_VAR, "").strip()
    if not measurement_id:
        return ""

    escaped_id = html.escape(quote(measurement_id, safe=""), quote=True)
    script_id = _javascript_string_literal(measurement_id)
    return (
        "  <!-- Google tag (gtag.js) -->\n"
        f'  <script async src="https://www.googletagmanager.com/gtag/js?id={escaped_id}"></script>\n'
        "  <script>\n"
        "    window.dataLayer = window.dataLayer || [];\n"
        "    function gtag(){dataLayer.push(arguments);}\n"
        "    gtag('js', new Date());\n"
        "\n"
        f"    gtag('config', {script_id});\n"
        "  </script>\n"
    )


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
    rendered = html_document.replace("{{APP_VERSION}}", escaped_version)
    # Replace the placeholder together with its trailing newline so the line is
    # removed entirely when analytics is disabled (no stray blank line), and the
    # tag's own trailing newline keeps the surrounding markup tidy when enabled.
    rendered = rendered.replace(
        _GOOGLE_ANALYTICS_TAG_PLACEHOLDER + "\n",
        _build_google_analytics_tag(),
    )
    return HTMLResponse(rendered)


__all__ = [
    "_CHANGELOG_PATH",
    "_FRONTEND_PATH",
    "_GOOGLE_ANALYTICS_ENV_VAR",
    "_STATIC_DIR",
    "_build_deprecation_link_header",
    "_load_changelog_html",
    "_load_frontend_html",
    "render_html_asset",
]
