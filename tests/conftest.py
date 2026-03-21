"""Shared pytest fixtures for the test suite."""

from fastapi.testclient import TestClient
import pytest

from proteus.api.main import app


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client
