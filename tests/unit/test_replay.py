from copy import deepcopy
from dataclasses import replace

import pytest

from app.errors import InternalPlanValidationError
from app.models.internal import CompiledConstraints, canonicalize_request
from app.optimizer.lp import solve_lp
from app.optimizer.replay import build_hourly_plan, replay_and_validate


def make_plan(valid_request):
    request = canonicalize_request(valid_request)
    compiled = CompiledConstraints(
        effective_solar_kwh=tuple(request.hours_by_hour[h].solar_kwh for h in range(24)),
        minimum_battery_kwh=tuple(request.battery.minimum_energy_kwh for _ in range(24)),
        can_charge=tuple(True for _ in range(24)),
        can_discharge=tuple(True for _ in range(24)),
        max_grid_kwh=tuple(None for _ in range(24)),
    )
    plan = build_hourly_plan(solve_lp(request, compiled))
    replay_and_validate(request, compiled, plan)
    return request, compiled, plan


def test_valid_plan_replays(valid_request):
    request, compiled, plan = make_plan(valid_request)
    replay_and_validate(request, compiled, plan)


def test_grid_balance_mutation_is_rejected(valid_request):
    request, compiled, plan = make_plan(valid_request)
    plan = deepcopy(plan)
    plan[0].grid_kwh += 1.0
    with pytest.raises(InternalPlanValidationError):
        replay_and_validate(request, compiled, plan)


def test_solar_above_effective_mutation_is_rejected(valid_request):
    request, compiled, plan = make_plan(valid_request)
    plan = deepcopy(plan)
    plan[0].solar_used_kwh = compiled.effective_solar_kwh[0] + 1.0
    with pytest.raises(InternalPlanValidationError):
        replay_and_validate(request, compiled, plan)


def test_battery_action_without_positive_amount_is_rejected(valid_request):
    request, compiled, plan = make_plan(valid_request)
    plan = deepcopy(plan)
    plan[0].battery_action = "charge"
    plan[0].battery_kwh = 0.0
    with pytest.raises(InternalPlanValidationError):
        replay_and_validate(request, compiled, plan)


def test_idle_with_nonzero_battery_is_rejected(valid_request):
    request, compiled, plan = make_plan(valid_request)
    plan = deepcopy(plan)
    plan[0].battery_action = "idle"
    plan[0].battery_kwh = 1.0
    with pytest.raises(InternalPlanValidationError):
        replay_and_validate(request, compiled, plan)


def test_charge_rate_mutation_is_rejected(valid_request):
    request, compiled, plan = make_plan(valid_request)
    plan = deepcopy(plan)
    plan[0].battery_action = "charge"
    plan[0].battery_kwh = request.battery.max_charge_kwh_per_hour + 1.0
    with pytest.raises(InternalPlanValidationError):
        replay_and_validate(request, compiled, plan)


def test_discharge_rate_mutation_is_rejected(valid_request):
    request, compiled, plan = make_plan(valid_request)
    plan = deepcopy(plan)
    plan[0].battery_action = "discharge"
    plan[0].battery_kwh = request.battery.max_discharge_kwh_per_hour + 1.0
    with pytest.raises(InternalPlanValidationError):
        replay_and_validate(request, compiled, plan)


def test_charge_in_no_charge_hour_is_rejected(valid_request):
    request, compiled, plan = make_plan(valid_request)
    plan = deepcopy(plan)
    can_charge = list(compiled.can_charge)
    can_charge[0] = False
    compiled = replace(compiled, can_charge=tuple(can_charge))
    plan[0].battery_action = "charge"
    plan[0].battery_kwh = 1.0
    with pytest.raises(InternalPlanValidationError):
        replay_and_validate(request, compiled, plan)


def test_discharge_in_no_discharge_hour_is_rejected(valid_request):
    request, compiled, plan = make_plan(valid_request)
    plan = deepcopy(plan)
    can_discharge = list(compiled.can_discharge)
    can_discharge[0] = False
    compiled = replace(compiled, can_discharge=tuple(can_discharge))
    plan[0].battery_action = "discharge"
    plan[0].battery_kwh = 1.0
    with pytest.raises(InternalPlanValidationError):
        replay_and_validate(request, compiled, plan)


def test_grid_cap_mutation_is_rejected(valid_request):
    request, compiled, plan = make_plan(valid_request)
    caps = list(compiled.max_grid_kwh)
    caps[0] = max(0.0, plan[0].grid_kwh - 1.0)
    compiled = replace(compiled, max_grid_kwh=tuple(caps))
    with pytest.raises(InternalPlanValidationError):
        replay_and_validate(request, compiled, plan)


def test_reported_battery_state_mismatch_is_rejected(valid_request):
    request, compiled, plan = make_plan(valid_request)
    plan = deepcopy(plan)
    plan[0].battery_energy_after_kwh += 1.0
    with pytest.raises(InternalPlanValidationError):
        replay_and_validate(request, compiled, plan)


def test_negative_reported_battery_state_is_rejected(valid_request):
    request, compiled, plan = make_plan(valid_request)
    plan = deepcopy(plan)
    plan[0].battery_energy_after_kwh = -1.0
    with pytest.raises(InternalPlanValidationError):
        replay_and_validate(request, compiled, plan)


def test_reserve_mutation_is_rejected(valid_request):
    request, compiled, plan = make_plan(valid_request)
    minimums = list(compiled.minimum_battery_kwh)
    minimums[0] = min(request.battery.capacity_kwh, plan[0].battery_energy_after_kwh + 1.0)
    compiled = replace(compiled, minimum_battery_kwh=tuple(minimums))
    with pytest.raises(InternalPlanValidationError):
        replay_and_validate(request, compiled, plan)


def test_capacity_mutation_is_rejected(valid_request):
    request, compiled, plan = make_plan(valid_request)
    request = replace(
        request,
        battery=request.battery.model_copy(update={"max_charge_kwh_per_hour": 100.0}),
    )
    plan = deepcopy(plan)
    charge = request.battery.capacity_kwh - request.battery.initial_energy_kwh + 1.0
    plan[0].battery_action = "charge"
    plan[0].battery_kwh = charge
    plan[0].grid_kwh = request.hours_by_hour[0].demand_kwh + charge - plan[0].solar_used_kwh
    plan[0].battery_energy_after_kwh = request.battery.initial_energy_kwh + charge
    with pytest.raises(InternalPlanValidationError):
        replay_and_validate(request, compiled, plan)


def test_final_battery_neutrality_mutation_is_rejected(valid_request):
    request, compiled, plan = make_plan(valid_request)
    plan = deepcopy(plan)
    row = plan[23]
    # The flat-tariff reference solution charges in the final hour. Reducing that
    # charge by 1 kWh keeps the row internally consistent but violates final neutrality.
    assert row.battery_action == "charge" and row.battery_kwh >= 1.0
    row.battery_kwh -= 1.0
    row.grid_kwh -= 1.0
    row.battery_energy_after_kwh -= 1.0
    with pytest.raises(InternalPlanValidationError, match="neutrality"):
        replay_and_validate(request, compiled, plan)


def test_duplicate_hour_is_rejected(valid_request):
    request, compiled, plan = make_plan(valid_request)
    plan = deepcopy(plan)
    plan[23].hour = 22
    with pytest.raises(InternalPlanValidationError):
        replay_and_validate(request, compiled, plan)


def test_missing_hour_is_rejected(valid_request):
    request, compiled, plan = make_plan(valid_request)
    with pytest.raises(InternalPlanValidationError):
        replay_and_validate(request, compiled, plan[:-1])
