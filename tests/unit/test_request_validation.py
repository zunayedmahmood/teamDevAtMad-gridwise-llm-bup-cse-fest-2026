import math

import pytest
from pydantic import ValidationError

from app.errors import SemanticInputError
from app.models.request import EnergyRequest, validate_request_semantics


def parse_and_validate(payload: dict) -> EnergyRequest:
    request = EnergyRequest.model_validate(payload)
    validate_request_semantics(request)
    return request


def test_valid_request_passes(valid_request_dict):
    parse_and_validate(valid_request_dict)


def test_unsorted_complete_hours_passes(valid_request_dict):
    valid_request_dict["hours"] = list(reversed(valid_request_dict["hours"]))
    parse_and_validate(valid_request_dict)


@pytest.mark.parametrize("count", [0, 4])
def test_note_count_fails(valid_request_dict, count):
    valid_request_dict["operator_notes"] = ["valid"] * count
    request = EnergyRequest.model_validate(valid_request_dict)
    with pytest.raises(SemanticInputError):
        validate_request_semantics(request)


def test_blank_note_fails(valid_request_dict):
    valid_request_dict["operator_notes"] = ["   "]
    request = EnergyRequest.model_validate(valid_request_dict)
    with pytest.raises(SemanticInputError):
        validate_request_semantics(request)


@pytest.mark.parametrize("count", [23, 25])
def test_wrong_hour_count_fails(valid_request_dict, count):
    hours = valid_request_dict["hours"]
    if count == 23:
        valid_request_dict["hours"] = hours[:23]
    else:
        valid_request_dict["hours"] = hours + [dict(hours[-1], hour=23)]
    request = EnergyRequest.model_validate(valid_request_dict)
    with pytest.raises(SemanticInputError):
        validate_request_semantics(request)


def test_duplicate_hour_fails(valid_request_dict):
    valid_request_dict["hours"][23]["hour"] = 22
    request = EnergyRequest.model_validate(valid_request_dict)
    with pytest.raises(SemanticInputError):
        validate_request_semantics(request)


def test_missing_hour_fails(valid_request_dict):
    valid_request_dict["hours"] = [row for row in valid_request_dict["hours"] if row["hour"] != 17]
    valid_request_dict["hours"].append({"hour": 16, "demand_kwh": 1.0, "solar_kwh": 0.0, "tariff_bdt_per_kwh": 1.0})
    request = EnergyRequest.model_validate(valid_request_dict)
    with pytest.raises(SemanticInputError):
        validate_request_semantics(request)


@pytest.mark.parametrize(
    "updates",
    [
        {"initial_energy_kwh": 101.0},
        {"minimum_energy_kwh": 101.0},
        {"initial_energy_kwh": 10.0, "minimum_energy_kwh": 20.0},
    ],
)
def test_invalid_battery_relationships_fail(valid_request_dict, updates):
    valid_request_dict["battery"].update(updates)
    request = EnergyRequest.model_validate(valid_request_dict)
    with pytest.raises(SemanticInputError):
        validate_request_semantics(request)


@pytest.mark.parametrize("field", ["demand_kwh", "solar_kwh", "tariff_bdt_per_kwh"])
def test_negative_hour_numbers_fail_structurally(valid_request_dict, field):
    valid_request_dict["hours"][0][field] = -1.0
    with pytest.raises(ValidationError):
        EnergyRequest.model_validate(valid_request_dict)


@pytest.mark.parametrize("field", ["max_charge_kwh_per_hour", "max_discharge_kwh_per_hour"])
def test_negative_battery_rates_fail_structurally(valid_request_dict, field):
    valid_request_dict["battery"][field] = -1.0
    with pytest.raises(ValidationError):
        EnergyRequest.model_validate(valid_request_dict)


def test_numeric_string_fails_structurally(valid_request_dict):
    valid_request_dict["hours"][0]["demand_kwh"] = "12"
    with pytest.raises(ValidationError):
        EnergyRequest.model_validate(valid_request_dict)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_fails_structurally(valid_request_dict, value):
    valid_request_dict["hours"][0]["demand_kwh"] = value
    with pytest.raises(ValidationError):
        EnergyRequest.model_validate(valid_request_dict)


@pytest.mark.parametrize(
    "location,field",
    [("top", "extra"), ("hour", "unexpected"), ("battery", "unexpected")],
)
def test_unknown_fields_fail(valid_request_dict, location, field):
    if location == "top":
        valid_request_dict[field] = 1
    elif location == "hour":
        valid_request_dict["hours"][0][field] = 1
    else:
        valid_request_dict["battery"][field] = 1
    with pytest.raises(ValidationError):
        EnergyRequest.model_validate(valid_request_dict)
