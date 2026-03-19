"""Placeholder tests — replace with real tests as modules are implemented."""

import pytest


def test_project_imports() -> None:
    """Smoke test: verify the package structure is importable."""
    import importlib

    for module in [
        "proteus.phonology.ipa_converter",
        "proteus.phonology.distance",
        "proteus.phonology.search",
        "proteus.phonology.explainer",
        "proteus.api.main",
    ]:
        importlib.import_module(module)


def test_api_health(client) -> None:
    """Health endpoint returns 200 ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from proteus.api.main import app

    return TestClient(app)
