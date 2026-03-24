"""Shared pytest fixtures for the test suite."""

import pytest

from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client
