"""Shared pytest fixtures for the test suite."""

from collections.abc import Generator
import sys
from types import ModuleType

import pytest

from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def _clear_pos_overrides(lsx: ModuleType | None) -> None:
    """Clear the module's ``_pos_overrides`` cache attribute.

    Args:
        lsx: The ``phonology.lsj_extractor`` module, or ``None`` if not yet imported.
    """
    if lsx is not None:
        lsx._pos_overrides = None


@pytest.fixture
def reset_pos_overrides_cache() -> Generator[None, None, None]:
    """Reset the lsj_extractor _pos_overrides cache before and after each test.

    Applied via ``pytestmark`` in test modules that exercise POS extraction
    (test_lsj_extractor, test_build_lexicon).
    """
    _clear_pos_overrides(sys.modules.get("phonology.lsj_extractor"))
    yield
    _clear_pos_overrides(sys.modules.get("phonology.lsj_extractor"))
