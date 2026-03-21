"""Cross-cutting smoke tests for the Proteus package."""


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
