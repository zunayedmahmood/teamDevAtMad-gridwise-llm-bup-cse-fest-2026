from copy import deepcopy

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


@pytest.mark.parametrize("mutation", ["grid", "solar", "state", "action", "final"])
def test_mutations_are_rejected(valid_request, mutation):
    request, compiled, plan = make_plan(valid_request)
    plan = deepcopy(plan)
    if mutation == "grid":
        plan[0].grid_kwh += 1.0
    elif mutation == "solar":
        plan[0].solar_used_kwh = compiled.effective_solar_kwh[0] + 1.0
    elif mutation == "state":
        plan[0].battery_energy_after_kwh += 1.0
    elif mutation == "action":
        plan[0].battery_action = "idle"
        plan[0].battery_kwh = 1.0
    elif mutation == "final":
        plan[23].battery_energy_after_kwh += 1.0
    with pytest.raises(InternalPlanValidationError):
        replay_and_validate(request, compiled, plan)
