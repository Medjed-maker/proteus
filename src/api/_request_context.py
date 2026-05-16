"""Request metadata helpers shared by REST and future MCP surfaces."""

from __future__ import annotations

import os
import re
from urllib.parse import urlencode, urlparse
import uuid

from fastapi import Request

from ._models import RequestEcho, SearchRequest

PUBLIC_BASE_URL_ENV_VAR = "PROTEUS_PUBLIC_BASE_URL"
_MIN_REQUEST_ID_LENGTH = 8
_MAX_REQUEST_ID_LENGTH = 64
_HEX_REQUEST_ID_RE = re.compile(r"[0-9a-fA-F]+")


def generate_request_id() -> str:
    """Return a fresh UUID4 hex request identifier."""
    return uuid.uuid4().hex


def is_valid_request_id(value: str) -> bool:
    """Return True when ``value`` is a bounded hex request identifier.

    Accepts only plain hexadecimal characters (no ``0x`` prefix, no whitespace,
    no sign markers). ``int(value, 16)`` would silently accept ``"0xff..."``
    within the length bounds, which would round-trip into request traces as a
    non-canonical id; the explicit regex closes that gap.
    """
    if not (_MIN_REQUEST_ID_LENGTH <= len(value) <= _MAX_REQUEST_ID_LENGTH):
        return False
    return _HEX_REQUEST_ID_RE.fullmatch(value) is not None


def validate_public_base_url(value: str) -> None:
    """Raise ``ValueError`` if ``value`` is not a usable public base URL.

    Centralizes the rule set so startup validation and per-request construction
    in :func:`build_verification_url` agree. Empty strings are rejected here;
    callers that want to treat "unset" as a fallback should branch *before*
    calling this validator (see :func:`resolve_public_base_url`).
    """
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError("base must be a non-empty URL")
    parsed = urlparse(value)
    if not (parsed.scheme and parsed.netloc):
        raise ValueError("base must be an absolute URL with scheme and host")
    if parsed.query or parsed.fragment:
        raise ValueError("base must not include query or fragment")


def resolve_public_base_url(fastapi_request: Request) -> str:
    """Resolve the public base URL for reproducibility links."""
    env_base_url = os.environ.get(PUBLIC_BASE_URL_ENV_VAR, "").strip()
    if env_base_url:
        return env_base_url
    return str(fastapi_request.base_url)


def build_verification_url(base: str, request: SearchRequest) -> str:
    """Build a deterministic URL that can reproduce a search request."""
    validate_public_base_url(base)

    query = urlencode(
        [
            ("q", request.query_form),
            ("language", request.language),
            ("dialect", request.dialect_hint),
            ("max_candidates", str(request.max_candidates)),
            ("response_language", request.response_language),
        ]
    )
    return f"{base.rstrip('/')}/?{query}"


def build_request_echo(request: SearchRequest) -> RequestEcho:
    """Return sanitized validated request parameters for response metadata."""
    return RequestEcho(
        query_form=request.query_form,
        language=request.language,
        dialect_hint=request.dialect_hint,
        max_candidates=request.max_candidates,
        response_language=request.response_language,
    )
