from copy import deepcopy

import pytest

from app.errors import InternalPlanValidationError
from app.models.internal import ValidatedDirective, canonicalize_request
from app.models.response import DirectiveType, HourlyPlanEntry
from app.optimizer.compiler import compile_constraints
from app.optimizer.lp import solve_lp
from app.optimizer.numeric import VALIDATION_TOLERANCE
from app.optimizer.replay import build_hourly_plan, calculate_totals, replay_and_validate


def make_valid_setup(valid_request):
    canonical = canonicalize_request(valid_request)
    compiled = compile_constraints(canonical, [])
    solution = solve_lp(canonical, compiled)
    plan = build_hourly_plan(solution)
    return canonical, compiled, solution, plan


def test_mutation_flipped_battery_sign_fails_replay(valid_request):
    canonical, compiled, solution, plan = make_valid_setup(valid_request)
    mutated_plan = [row.model_copy() for row in plan]
    # Invert action on hour 0
    new_action = "discharge" if mutated_plan[0].battery_action == "charge" else "charge"
    mutated_plan[0] = mutated_plan[0].model_copy(
        update={"battery_action": new_action, "battery_kwh": 10.0}
    )
    with pytest.raises(InternalPlanValidationError, match="energy balance failed"):
        replay_and_validate(canonical, compiled, mutated_plan)


def test_mutation_missing_end_of_day_neutrality_fails_replay(valid_request):
    canonical, compiled, solution, plan = make_valid_setup(valid_request)
    mutated_plan = [row.model_copy() for row in plan]
    mutated_plan[23] = mutated_plan[23].model_copy(
        update={"battery_energy_after_kwh": mutated_plan[23].battery_energy_after_kwh + 5.0}
    )
    with pytest.raises(InternalPlanValidationError):
        replay_and_validate(canonical, compiled, mutated_plan)


def test_mutation_solar_exceeding_effective_fails_replay(valid_request):
    canonical, compiled, solution, plan = make_valid_setup(valid_request)
    mutated_plan = [row.model_copy() for row in plan]
    mutated_plan[12] = mutated_plan[12].model_copy(
        update={
            "solar_used_kwh": compiled.effective_solar_kwh[12] + 10.0,
            "grid_kwh": max(0.0, mutated_plan[12].grid_kwh - 10.0),
        }
    )
    with pytest.raises(InternalPlanValidationError, match="solar usage exceeds effective solar"):
        replay_and_validate(canonical, compiled, mutated_plan)


def test_mutation_charging_in_no_charge_hour_fails_replay(valid_request):
    canonical = canonicalize_request(valid_request)
    directive = ValidatedDirective(0, True, DirectiveType.NO_CHARGE_WINDOW, (5,), None, None, None, "test")
    compiled = compile_constraints(canonical, [directive])
    solution = solve_lp(canonical, compiled)
    plan = build_hourly_plan(solution)

    mutated_plan = [row.model_copy() for row in plan]
    mutated_plan[5] = mutated_plan[5].model_copy(
        update={
            "battery_action": "charge",
            "battery_kwh": 10.0,
            "grid_kwh": mutated_plan[5].grid_kwh + 10.0,
        }
    )
    with pytest.raises(InternalPlanValidationError, match="charging is prohibited"):
        replay_and_validate(canonical, compiled, mutated_plan)


def test_mutation_discharging_in_no_discharge_hour_fails_replay(valid_request):
    canonical = canonicalize_request(valid_request)
    directive = ValidatedDirective(0, True, DirectiveType.NO_DISCHARGE_WINDOW, (5,), None, None, None, "test")
    compiled = compile_constraints(canonical, [directive])
    solution = solve_lp(canonical, compiled)
    plan = build_hourly_plan(solution)

    mutated_plan = [row.model_copy() for row in plan]
    mutated_plan[5] = mutated_plan[5].model_copy(
        update={
            "battery_action": "discharge",
            "battery_kwh": 10.0,
            "grid_kwh": max(0.0, mutated_plan[5].grid_kwh - 10.0),
        }
    )
    with pytest.raises(InternalPlanValidationError, match="discharging is prohibited"):
        replay_and_validate(canonical, compiled, mutated_plan)


def test_mutation_exceeding_grid_cap_fails_replay(valid_request):
    canonical = canonicalize_request(valid_request)
    # Give high cap 200 so LP solves feasibly
    directive = ValidatedDirective(0, True, DirectiveType.MAX_GRID_WINDOW, (8,), None, None, 200.0, "cap")
    compiled = compile_constraints(canonical, [directive])
    solution = solve_lp(canonical, compiled)
    plan = build_hourly_plan(solution)

    # Mutate plan grid_kwh in hour 8 to exceed 200
    mutated_plan = [row.model_copy() for row in plan]
    mutated_plan[8] = mutated_plan[8].model_copy(update={"grid_kwh": 205.0})
    with pytest.raises(InternalPlanValidationError, match="grid cap exceeded"):
        replay_and_validate(canonical, compiled, mutated_plan)


def test_mutation_violating_battery_reserve_fails_replay(valid_request):
    canonical = canonicalize_request(valid_request)
    directive = ValidatedDirective(0, True, DirectiveType.MINIMUM_BATTERY_RESERVE, (10,), None, 60.0, None, "reserve")
    compiled = compile_constraints(canonical, [directive])
    solution = solve_lp(canonical, compiled)
    plan = build_hourly_plan(solution)

    mutated_plan = [row.model_copy() for row in plan]
    # Set reported after energy below 60
    mutated_plan[10] = mutated_plan[10].model_copy(update={"battery_energy_after_kwh": 40.0})
    with pytest.raises(InternalPlanValidationError):
        replay_and_validate(canonical, compiled, mutated_plan)


def test_mutation_exceeding_capacity_fails_replay(valid_request):
    canonical, compiled, solution, plan = make_valid_setup(valid_request)
    mutated_plan = [row.model_copy() for row in plan]
    mutated_plan[3] = mutated_plan[3].model_copy(
        update={"battery_energy_after_kwh": canonical.battery.capacity_kwh + 10.0}
    )
    with pytest.raises(InternalPlanValidationError, match="battery capacity exceeded|battery state is inconsistent"):
        replay_and_validate(canonical, compiled, mutated_plan)


def test_mutation_unsorted_hours_fails_replay(valid_request):
    canonical, compiled, solution, plan = make_valid_setup(valid_request)
    mutated_plan = list(reversed(plan))
    with pytest.raises(InternalPlanValidationError, match="hourly plan must be sorted"):
        replay_and_validate(canonical, compiled, mutated_plan)


def test_mutation_wrong_hour_count_fails_replay(valid_request):
    canonical, compiled, solution, plan = make_valid_setup(valid_request)
    with pytest.raises(InternalPlanValidationError, match="exactly 24 rows"):
        replay_and_validate(canonical, compiled, plan[:23])


def test_mutation_rounding_breaks_energy_balance(valid_request):
    canonical, compiled, solution, plan = make_valid_setup(valid_request)
    mutated_plan = [row.model_copy() for row in plan]
    mutated_plan[0] = mutated_plan[0].model_copy(
        update={"grid_kwh": mutated_plan[0].grid_kwh + 0.05}
    )
    with pytest.raises(InternalPlanValidationError, match="energy balance failed"):
        replay_and_validate(canonical, compiled, mutated_plan)


def test_mutation_objective_mismatch_detected(valid_request):
    canonical, compiled, solution, plan = make_valid_setup(valid_request)
    totals = calculate_totals(canonical, plan)
    faked_objective = solution.objective_value + 0.1
    assert abs(totals.total_cost_bdt - faked_objective) > VALIDATION_TOLERANCE
