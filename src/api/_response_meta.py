"""Response metadata construction helpers."""

from __future__ import annotations

from datetime import datetime, timezone

from ._constants import API_VERSION, SCHEMA_VERSION
from ._models import DataVersions, ResponseMeta, SearchRequest
from ._request_context import build_request_echo, build_verification_url


def build_response_meta(
    *,
    request_id: str,
    request: SearchRequest | None,
    base_url: str | None,
    engine_version: str,
    data_versions: DataVersions,
    ruleset_versions: dict[str, str],
) -> ResponseMeta:
    """Build shared API response metadata without importing the FastAPI app.

    ``verification_url`` and ``request_echo`` are populated only when both
    ``request`` and ``base_url`` are not ``None``. Otherwise, ``verification_url``
    is empty and ``request_echo`` is ``None``. When populated,
    ``verification_url`` contains the deterministic URL from
    ``build_verification_url`` and ``request_echo`` contains the sanitized
    request parameters from ``build_request_echo``.
    """
    verification_url = ""
    request_echo = None
    if request is not None and base_url is not None:
        verification_url = build_verification_url(base_url, request)
        request_echo = build_request_echo(request)

    return ResponseMeta(
        api_version=API_VERSION,
        schema_version=SCHEMA_VERSION,
        engine_version=engine_version,
        data_versions=data_versions,
        ruleset_versions=ruleset_versions,
        request_id=request_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        verification_url=verification_url,
        request_echo=request_echo,
    )
