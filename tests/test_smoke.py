"""Cross-cutting smoke tests for the Proteus package."""


def test_project_imports() -> None:
    """Smoke test: verify the package structure is importable."""
    import importlib

    for module in [
        "phonology.ipa_converter",
        "phonology.distance",
        "phonology.search",
        "phonology.explainer",
        "api.main",
    ]:
        importlib.import_module(module)
