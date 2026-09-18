import json

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_interpreter
from app.main import app
from tests.conftest import MappingInterpreter


def test_health_exact_response():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_structural_invalid_request_returns_400(valid_request_dict):
    client = TestClient(app)
    valid_request_dict["extra"] = True
    response = client.post("/optimize-energy", json=valid_request_dict)
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid request body."}


def test_malformed_json_returns_400():
    client = TestClient(app)
    response = client.post(
        "/optimize-energy",
        content="{not-json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 400


def test_semantically_invalid_request_returns_422(valid_request_dict, no_op_batch):
    valid_request_dict["hours"][23]["hour"] = 22
    app.dependency_overrides[get_interpreter] = lambda: MappingInterpreter({"TEST-001": no_op_batch})
    try:
        response = TestClient(app).post("/optimize-energy", json=valid_request_dict)
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 422


def test_missing_api_key_is_controlled_500(valid_request_dict, monkeypatch):
    from app.interpreter import client as client_module

    monkeypatch.setattr(client_module.settings, "openai_api_key", None)
    response = TestClient(app).post("/optimize-energy", json=valid_request_dict)
    assert response.status_code == 500
    assert response.json() == {"detail": "Directive interpretation is temporarily unavailable."}


def test_wrong_scalar_type_returns_400(valid_request_dict):
    valid_request_dict["hours"][0]["demand_kwh"] = "100"
    response = TestClient(app).post("/optimize-energy", json=valid_request_dict)
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid request body."}


@pytest.mark.parametrize("location", ["hour", "battery"])
def test_nested_extra_fields_return_400(valid_request_dict, location):
    if location == "hour":
        valid_request_dict["hours"][0]["unexpected"] = 1
    else:
        valid_request_dict["battery"]["unexpected"] = 1
    response = TestClient(app).post("/optimize-energy", json=valid_request_dict)
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid request body."}


@pytest.mark.parametrize(
    "payload,headers",
    [
        (None, None),
        ("", {"content-type": "application/json"}),
        ("{not-valid-json", {"content-type": "application/json"}),
        ("[]", {"content-type": "application/json"}),
        ("null", {"content-type": "application/json"}),
        ('"primitive string"', {"content-type": "application/json"}),
        ("12345", {"content-type": "application/json"}),
    ],
)
def test_body_level_structural_failures_return_400_json(payload, headers):
    client = TestClient(app)
    if payload is None:
        response = client.post("/optimize-energy")
    else:
        response = client.post("/optimize-energy", content=payload, headers=headers)
    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {"detail": "Invalid request body."}


def test_zero_capacity_battery_accepted(valid_request_dict, no_op_batch):
    valid_request_dict["battery"] = {
        "capacity_kwh": 0.0,
        "initial_energy_kwh": 0.0,
        "minimum_energy_kwh": 0.0,
        "max_charge_kwh_per_hour": 0.0,
        "max_discharge_kwh_per_hour": 0.0,
    }
    app.dependency_overrides[get_interpreter] = lambda: MappingInterpreter(
        {"TEST-001": no_op_batch}
    )
    try:
        response = TestClient(app).post("/optimize-energy", json=valid_request_dict)
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    plan = response.json()["hourly_plan"]
    assert all(row["battery_action"] == "idle" for row in plan)
    assert all(row["battery_kwh"] == 0.0 for row in plan)
    assert all(row["battery_energy_after_kwh"] == 0.0 for row in plan)


def test_float_hour_fails_structurally(valid_request_dict):
    valid_request_dict["hours"][0]["hour"] = 0.0
    response = TestClient(app).post("/optimize-energy", json=valid_request_dict)
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid request body."}


def test_boolean_for_float_fails_structurally(valid_request_dict):
    valid_request_dict["hours"][0]["demand_kwh"] = True
    response = TestClient(app).post("/optimize-energy", json=valid_request_dict)
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid request body."}

