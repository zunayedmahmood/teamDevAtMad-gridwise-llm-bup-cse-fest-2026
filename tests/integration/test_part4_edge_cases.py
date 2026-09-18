from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_interpreter
from app.main import app
from app.models.llm import LLMDirective, LLMDirectiveBatch
from app.models.response import DirectiveType, EnergyResponse
from tests.conftest import MappingInterpreter


def directive(
    note_index: int,
    directive_type: DirectiveType,
    *,
    hours=None,
    factor=None,
    minimum_energy_kwh=None,
    max_grid_kwh=None,
) -> LLMDirective:
    applies = directive_type is not DirectiveType.NO_OP
    return LLMDirective(
        note_index=note_index,
        applies=applies,
        directive_type=directive_type,
        hours=hours,
        factor=factor,
        minimum_energy_kwh=minimum_energy_kwh,
        max_grid_kwh=max_grid_kwh,
        explanation="Deterministic test interpretation.",
    )


def run_case(request_dict: dict, directives: list[LLMDirective]) -> EnergyResponse:
    batch = LLMDirectiveBatch(directives=directives)
    scenario_id = request_dict["scenario_id"]
    app.dependency_overrides[get_interpreter] = lambda: MappingInterpreter(
        {scenario_id: batch}
    )
    try:
        response = TestClient(app).post("/optimize-energy", json=request_dict)
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200, response.text
    return EnergyResponse.model_validate_json(response.text)


def base_case(valid_request_dict: dict, scenario_id: str, notes: list[str]) -> dict:
    request = deepcopy(valid_request_dict)
    request["scenario_id"] = scenario_id
    request["operator_notes"] = notes
    for hour in request["hours"]:
        hour["solar_kwh"] = 20.0
    return request


def test_no_op_with_unsorted_input_hours(valid_request_dict):
    request = base_case(valid_request_dict, "EDGE-UNSORTED", ["Irrelevant note"])
    request["hours"] = list(reversed(request["hours"]))
    response = run_case(request, [directive(0, DirectiveType.NO_OP)])
    assert [row.hour for row in response.hourly_plan] == list(range(24))


def test_all_three_notes_active(valid_request_dict):
    request = base_case(valid_request_dict, "EDGE-THREE", ["a", "b", "c"])
    response = run_case(
        request,
        [
            directive(0, DirectiveType.SOLAR_REDUCTION, hours=[10], factor=0.5),
            directive(1, DirectiveType.NO_CHARGE_WINDOW, hours=[1]),
            directive(2, DirectiveType.MAX_GRID_WINDOW, hours=[5], max_grid_kwh=90.0),
        ],
    )
    assert len(response.directive_interpretation) == 3


def test_overlapping_reserves(valid_request_dict):
    request = base_case(valid_request_dict, "EDGE-RESERVE", ["a", "b"])
    response = run_case(
        request,
        [
            directive(0, DirectiveType.MINIMUM_BATTERY_RESERVE, hours=[5, 6], minimum_energy_kwh=30.0),
            directive(1, DirectiveType.MINIMUM_BATTERY_RESERVE, hours=[5, 6], minimum_energy_kwh=40.0),
        ],
    )
    assert response.hourly_plan[5].battery_energy_after_kwh >= 40.0 - 1e-7


def test_overlapping_grid_caps(valid_request_dict):
    request = base_case(valid_request_dict, "EDGE-GRID-CAPS", ["a", "b"])
    response = run_case(
        request,
        [
            directive(0, DirectiveType.MAX_GRID_WINDOW, hours=[5], max_grid_kwh=90.0),
            directive(1, DirectiveType.MAX_GRID_WINDOW, hours=[5], max_grid_kwh=80.0),
        ],
    )
    assert response.hourly_plan[5].grid_kwh <= 80.0 + 1e-7


def test_charge_and_discharge_both_prohibited(valid_request_dict):
    request = base_case(valid_request_dict, "EDGE-IDLE", ["a", "b"])
    response = run_case(
        request,
        [
            directive(0, DirectiveType.NO_CHARGE_WINDOW, hours=[5]),
            directive(1, DirectiveType.NO_DISCHARGE_WINDOW, hours=[5]),
        ],
    )
    assert response.hourly_plan[5].battery_action == "idle"
    assert response.hourly_plan[5].battery_kwh == 0.0


@pytest.mark.parametrize("factor", [0.0, 1.0])
def test_solar_reduction_factor_boundaries(valid_request_dict, factor):
    request = base_case(valid_request_dict, f"EDGE-SOLAR-{factor}", ["a"])
    response = run_case(
        request,
        [directive(0, DirectiveType.SOLAR_REDUCTION, hours=[10], factor=factor)],
    )
    assert response.hourly_plan[10].solar_used_kwh <= 20.0 * factor + 1e-7


@pytest.mark.parametrize("field", ["max_charge_kwh_per_hour", "max_discharge_kwh_per_hour"])
def test_zero_battery_rate_limits(valid_request_dict, field):
    request = base_case(valid_request_dict, f"EDGE-ZERO-{field}", ["irrelevant"])
    request["battery"][field] = 0.0
    response = run_case(request, [directive(0, DirectiveType.NO_OP)])
    if field == "max_charge_kwh_per_hour":
        assert all(row.battery_action != "charge" for row in response.hourly_plan)
    else:
        assert all(row.battery_action != "discharge" for row in response.hourly_plan)


def test_capacity_equals_baseline_minimum(valid_request_dict):
    request = base_case(valid_request_dict, "EDGE-FIXED-BATTERY", ["irrelevant"])
    request["battery"].update(
        capacity_kwh=50.0,
        initial_energy_kwh=50.0,
        minimum_energy_kwh=50.0,
    )
    response = run_case(request, [directive(0, DirectiveType.NO_OP)])
    assert all(row.battery_energy_after_kwh == pytest.approx(50.0) for row in response.hourly_plan)


@pytest.mark.parametrize("profile", ["flat", "spread"])
def test_tariff_profiles(valid_request_dict, profile):
    request = base_case(valid_request_dict, f"EDGE-TARIFF-{profile}", ["irrelevant"])
    for hour in request["hours"]:
        hour["tariff_bdt_per_kwh"] = 10.0 if profile == "flat" else (1.0 if hour["hour"] < 12 else 100.0)
    response = run_case(request, [directive(0, DirectiveType.NO_OP)])
    assert len(response.hourly_plan) == 24


@pytest.mark.parametrize("profile", ["zero_solar", "zero_demand"])
def test_zero_energy_profiles(valid_request_dict, profile):
    request = base_case(valid_request_dict, f"EDGE-{profile}", ["irrelevant"])
    if profile == "zero_solar":
        for hour in request["hours"]:
            hour["solar_kwh"] = 0.0
    else:
        for hour in request["hours"][:4]:
            hour["demand_kwh"] = 0.0
    response = run_case(request, [directive(0, DirectiveType.NO_OP)])
    assert len(response.hourly_plan) == 24


def test_cross_midnight_hour_membership(valid_request_dict):
    request = base_case(valid_request_dict, "EDGE-CROSS-MIDNIGHT", ["No charging 10 PM to 2 AM"])
    response = run_case(
        request,
        [directive(0, DirectiveType.NO_CHARGE_WINDOW, hours=[0, 1, 22, 23])],
    )
    adjustment = response.directive_interpretation[0].structured_adjustment
    assert adjustment.hours == [0, 1, 22, 23]
    for hour in [0, 1, 22, 23]:
        assert response.hourly_plan[hour].battery_action != "charge"
