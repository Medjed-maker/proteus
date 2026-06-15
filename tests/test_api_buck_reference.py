"""Tests for Buck reference REST endpoints."""

from __future__ import annotations

import unicodedata

from fastapi.testclient import TestClient


def test_buck_rules_endpoint_filters_by_numeric_section(
    client: TestClient,
) -> None:
    """Verify numeric section filters match canonical Buck section strings."""
    response = client.get(
        "/languages/ancient_greek/buck/rules",
        params={"section": 41.4},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] >= 1
    assert payload["metadata"]["status"] == "provisional"
    assert payload["metadata"]["review_status"] == "not_expert_reviewed"
    assert payload["metadata"]["citation_ready"] is False
    assert "not expert-reviewed" in payload["metadata"]["review_note"]
    rule = next(
        (rule for rule in payload["rules"] if rule["id"] == "grc_phon_41_4"),
        None,
    )
    assert rule is not None, "Expected Buck rule grc_phon_41_4 in section results"
    assert rule["buck_section"] == "41.4"
    assert rule["review_status"] == "not_expert_reviewed"
    assert rule["citation_ready"] is False


def test_buck_rule_endpoint_returns_typed_transformation_alias(
    client: TestClient,
) -> None:
    """Verify single-rule responses serialize transformation.from by alias."""
    response = client.get("/languages/ancient_greek/buck/rules/grc_phon_8")

    assert response.status_code == 200
    payload = response.json()
    assert payload["rule"]["id"] == "grc_phon_8"
    assert payload["rule"]["transformation"]["from"] == "ā"
    assert "from_" not in payload["rule"]["transformation"]


def test_buck_rules_endpoint_reports_total_before_pagination(
    client: TestClient,
) -> None:
    """Verify rules total reports all matches before the page limit is applied."""
    response = client.get(
        "/languages/ancient_greek/buck/rules",
        params={"limit": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert len(payload["rules"]) == 1
    assert payload["total"] > payload["count"]


def test_buck_dialects_endpoint_reports_total_before_pagination(
    client: TestClient,
) -> None:
    """Verify dialects total reports all matches before pagination."""
    response = client.get(
        "/languages/ancient_greek/buck/dialects",
        params={"limit": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["total"] >= payload["count"]
    assert payload["total"] > 1


def test_buck_glossary_endpoint_reports_total_before_pagination(
    client: TestClient,
) -> None:
    """Verify glossary total reports matching entries before pagination."""
    response = client.get(
        "/languages/ancient_greek/buck/glossary",
        params={"limit": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] <= 1
    assert payload["total"] >= payload["count"]


def test_buck_rules_endpoint_rejects_blank_section(client: TestClient) -> None:
    """Verify blank section filters are rejected with a clear client error."""
    response = client.get(
        "/languages/ancient_greek/buck/rules",
        params={"section": ""},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Buck section must not be empty"


def test_buck_dialects_endpoint_filters_by_kind_and_group(
    client: TestClient,
) -> None:
    """Verify dialect list filters by exact kind and group values."""
    response = client.get(
        "/languages/ancient_greek/buck/dialects",
        params={"kind": "dialect", "group": "Attic-Ionic"},
    )

    assert response.status_code == 200
    payload = response.json()
    dialect_ids = {dialect["id"] for dialect in payload["dialects"]}
    assert {"attic", "ionic"}.issubset(dialect_ids)
    assert all(dialect["kind"] == "dialect" for dialect in payload["dialects"])
    assert all(dialect["group"] == "Attic-Ionic" for dialect in payload["dialects"])


def test_buck_dialect_endpoint_can_include_inherited_rules(
    client: TestClient,
) -> None:
    """Verify dialect detail can include rules inherited from its parent chain."""
    response = client.get(
        "/languages/ancient_greek/buck/dialects/attic",
        params={"include_rules": True, "include_inherited": True},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["dialect"]["id"] == "attic"
    assert any(rule["id"] == "grc_phon_41_4" for rule in payload["rules"])
    assert all(rule["review_status"] == "not_expert_reviewed" for rule in payload["rules"])


def test_buck_glossary_endpoint_uses_nfc_exact_matching(
    client: TestClient,
) -> None:
    """Verify glossary lookup normalizes input to NFC before exact matching."""
    response = client.get(
        "/languages/ancient_greek/buck/glossary",
        params={"standard_form": unicodedata.normalize("NFD", "λαός")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert [entry["word"] for entry in payload["entries"]] == ["λεώς"]
    entry = payload["entries"][0]
    assert entry["buck_ref"] == {"section": "41.4", "page": 130}
    assert entry["review_status"] == "not_expert_reviewed"
    assert entry["citation_ready"] is False


def test_buck_glossary_endpoint_preserves_inscription_number_arrays(
    client: TestClient,
) -> None:
    """Verify integer-array inscription numbers remain arrays in REST JSON."""
    response = client.get(
        "/languages/ancient_greek/buck/glossary",
        params={"standard_form": "ἐν"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["entries"][0]["word"] == "ἴν"
    assert payload["entries"][0]["inscription_no"] == [47]


def test_buck_glossary_endpoint_rejects_unimplemented_accent_insensitive_search(
    client: TestClient,
) -> None:
    """Verify accent-insensitive glossary search is rejected until supported."""
    response = client.get(
        "/languages/ancient_greek/buck/glossary",
        params={"standard_form": "λαος", "accent_insensitive": True},
    )

    assert response.status_code == 400
    assert "accent_insensitive glossary search is not supported yet" in response.json()[
        "detail"
    ]


def test_buck_endpoints_reject_unsupported_language(client: TestClient) -> None:
    """Verify Buck endpoints return 404 for languages without Buck data."""
    response = client.get("/languages/old_english/buck/rules")

    assert response.status_code == 404
    assert response.json()["detail"] == (
        "Buck reference data is not available for language 'old_english'"
    )


def test_buck_rule_endpoint_rejects_unknown_rule(client: TestClient) -> None:
    """Verify unknown Buck rule ids return 404 with a specific detail."""
    response = client.get("/languages/ancient_greek/buck/rules/unknown")

    assert response.status_code == 404
    assert response.json()["detail"] == "Unknown Buck rule: unknown"


def test_buck_dialect_endpoint_rejects_unknown_dialect(client: TestClient) -> None:
    """Verify unknown Buck dialect ids return 404 with a specific detail."""
    response = client.get("/languages/ancient_greek/buck/dialects/unknown")

    assert response.status_code == 404
    assert response.json()["detail"] == "Unknown Buck dialect: unknown"
