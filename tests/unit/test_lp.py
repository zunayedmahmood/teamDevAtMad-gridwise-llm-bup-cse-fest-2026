import pytest

from app.errors import InternalPlanValidationError, OptimizationInfeasibleError
from app.models.internal import CompiledConstraints, canonicalize_request
from app.optimizer.lp import VariableIndex, solve_lp
from app.optimizer.replay import build_hourly_plan, calculate_totals, replay_and_validate


def baseline_compiled(request):
    return CompiledConstraints(
        effective_solar_kwh=tuple(request.hours_by_hour[h].solar_kwh for h in range(24)),
        minimum_battery_kwh=tuple(request.battery.minimum_energy_kwh for _ in range(24)),
        can_charge=tuple(True for _ in range(24)),
        can_discharge=tuple(True for _ in range(24)),
        max_grid_kwh=tuple(None for _ in range(24)),
    )


def test_simple_arbitrage(valid_request):
    for row in valid_request.hours:
        row.tariff_bdt_per_kwh = 10.0
    valid_request.hours[0].tariff_bdt_per_kwh = 1.0
    valid_request.hours[1].tariff_bdt_per_kwh = 100.0
    request = canonicalize_request(valid_request)
    compiled = baseline_compiled(request)
    plan = build_hourly_plan(solve_lp(request, compiled))
    replay_and_validate(request, compiled, plan)
    assert plan[0].battery_action == "charge"
    assert plan[1].battery_action == "discharge"
    assert plan[-1].battery_energy_after_kwh == pytest.approx(request.battery.initial_energy_kwh, abs=1e-6)


def test_no_charge_forces_next_best_hour(valid_request):
    for row in valid_request.hours:
        row.tariff_bdt_per_kwh = 50.0
    valid_request.hours[0].tariff_bdt_per_kwh = 1.0
    valid_request.hours[1].tariff_bdt_per_kwh = 2.0
    valid_request.hours[2].tariff_bdt_per_kwh = 100.0
    request = canonicalize_request(valid_request)
    base = baseline_compiled(request)
    compiled = CompiledConstraints(
        base.effective_solar_kwh,
        base.minimum_battery_kwh,
        tuple(False if h == 0 else True for h in range(24)),
        base.can_discharge,
        base.max_grid_kwh,
    )
    plan = build_hourly_plan(solve_lp(request, compiled))
    replay_and_validate(request, compiled, plan)
    assert plan[0].battery_action != "charge"
    assert plan[1].battery_action == "charge"


def test_grid_cap_is_respected(valid_request):
    request = canonicalize_request(valid_request)
    base = baseline_compiled(request)
    compiled = CompiledConstraints(
        base.effective_solar_kwh,
        base.minimum_battery_kwh,
        base.can_charge,
        base.can_discharge,
        tuple(80.0 if h == 5 else None for h in range(24)),
    )
    plan = build_hourly_plan(solve_lp(request, compiled))
    replay_and_validate(request, compiled, plan)
    assert plan[5].grid_kwh <= 80.0 + 1e-6


def test_infeasible_grid_cap_raises(valid_request):
    valid_request.battery.initial_energy_kwh = valid_request.battery.minimum_energy_kwh
    valid_request.battery.max_charge_kwh_per_hour = 0.0
    valid_request.battery.max_discharge_kwh_per_hour = 0.0
    request = canonicalize_request(valid_request)
    base = baseline_compiled(request)
    compiled = CompiledConstraints(
        base.effective_solar_kwh,
        base.minimum_battery_kwh,
        base.can_charge,
        base.can_discharge,
        tuple(50.0 if h == 0 else None for h in range(24)),
    )
    with pytest.raises(OptimizationInfeasibleError):
        solve_lp(request, compiled)


def test_flat_tariff_has_no_arbitrage_cost_advantage(valid_request):
    for row in valid_request.hours:
        row.tariff_bdt_per_kwh = 10.0
        row.solar_kwh = 0.0
    request = canonicalize_request(valid_request)
    compiled = baseline_compiled(request)
    plan = build_hourly_plan(solve_lp(request, compiled))
    replay_and_validate(request, compiled, plan)
    totals = calculate_totals(request, plan)
    expected_grid = sum(row.demand_kwh for row in valid_request.hours)
    assert totals.total_grid_kwh == pytest.approx(expected_grid, abs=1e-6)
    assert totals.total_cost_bdt == pytest.approx(expected_grid * 10.0, abs=1e-6)


def test_lp_has_exactly_96_variables():
    assert VariableIndex().size == 96


def test_no_discharge_blocks_expensive_hour(valid_request):
    for row in valid_request.hours:
        row.tariff_bdt_per_kwh = 10.0
    valid_request.hours[0].tariff_bdt_per_kwh = 1.0
    valid_request.hours[1].tariff_bdt_per_kwh = 100.0
    request = canonicalize_request(valid_request)
    base = baseline_compiled(request)
    compiled = CompiledConstraints(
        base.effective_solar_kwh,
        base.minimum_battery_kwh,
        base.can_charge,
        tuple(False if h == 1 else True for h in range(24)),
        base.max_grid_kwh,
    )

    solution = solve_lp(request, compiled)
    plan = build_hourly_plan(solution)
    replay_and_validate(request, compiled, plan)

    assert solution.battery_net_kwh[1] >= -1e-9
    assert plan[1].battery_action != "discharge"


def test_reserve_bound_is_respected(valid_request):
    request = canonicalize_request(valid_request)
    base = baseline_compiled(request)
    reserve = 45.0
    compiled = CompiledConstraints(
        base.effective_solar_kwh,
        tuple(reserve if h == 5 else base.minimum_battery_kwh[h] for h in range(24)),
        base.can_charge,
        base.can_discharge,
        base.max_grid_kwh,
    )

    solution = solve_lp(request, compiled)
    plan = build_hourly_plan(solution)
    replay_and_validate(request, compiled, plan)

    assert solution.battery_energy_after_kwh[5] >= reserve - 1e-7


def test_end_of_day_neutrality_is_encoded_in_lp(valid_request):
    valid_request.hours[23].tariff_bdt_per_kwh = 100.0
    request = canonicalize_request(valid_request)
    compiled = baseline_compiled(request)

    solution = solve_lp(request, compiled)

    assert solution.battery_energy_after_kwh[23] == pytest.approx(
        request.battery.initial_energy_kwh,
        abs=1e-7,
    )


def test_objective_matches_grid_cost(valid_request):
    request = canonicalize_request(valid_request)
    compiled = baseline_compiled(request)

    solution = solve_lp(request, compiled)
    cost_from_solution = sum(
        solution.grid_kwh[h] * request.hours_by_hour[h].tariff_bdt_per_kwh
        for h in range(24)
    )

    assert solution.objective_value == pytest.approx(cost_from_solution, abs=1e-7)


def test_both_charge_and_discharge_prohibited_force_idle(valid_request):
    request = canonicalize_request(valid_request)
    base = baseline_compiled(request)
    compiled = CompiledConstraints(
        base.effective_solar_kwh,
        base.minimum_battery_kwh,
        tuple(False if h == 3 else True for h in range(24)),
        tuple(False if h == 3 else True for h in range(24)),
        base.max_grid_kwh,
    )

    solution = solve_lp(request, compiled)

    assert solution.battery_net_kwh[3] == pytest.approx(0.0, abs=1e-9)


def test_pre_solver_rejects_malformed_compiled_vector(valid_request):
    request = canonicalize_request(valid_request)
    base = baseline_compiled(request)
    malformed = CompiledConstraints(
        base.effective_solar_kwh[:-1],
        base.minimum_battery_kwh,
        base.can_charge,
        base.can_discharge,
        base.max_grid_kwh,
    )

    with pytest.raises(InternalPlanValidationError, match="24 hourly values"):
        solve_lp(request, malformed)


def test_pre_solver_rejects_negative_effective_solar(valid_request):
    request = canonicalize_request(valid_request)
    base = baseline_compiled(request)
    malformed = CompiledConstraints(
        tuple(-1.0 if h == 7 else base.effective_solar_kwh[h] for h in range(24)),
        base.minimum_battery_kwh,
        base.can_charge,
        base.can_discharge,
        base.max_grid_kwh,
    )

    with pytest.raises(InternalPlanValidationError, match="effective solar"):
        solve_lp(request, malformed)


def test_unexpected_solver_status_raises_controlled_internal_error(valid_request, monkeypatch):
    from types import SimpleNamespace

    from app.optimizer import lp as lp_module

    request = canonicalize_request(valid_request)
    compiled = baseline_compiled(request)
    monkeypatch.setattr(
        lp_module,
        "linprog",
        lambda **kwargs: SimpleNamespace(success=False, status=4, message="solver failed"),
    )

    with pytest.raises(InternalPlanValidationError, match="status 4"):
        solve_lp(request, compiled)
