"""Shared API version constants."""

API_VERSION: str = "1.0"
"""Public Proteus REST API version."""

SCHEMA_VERSION: str = "1.0.0"
"""Response schema version.

Bump the minor version for backward-compatible response additions. Bump the
major version when response fields or semantics change incompatibly.
"""
