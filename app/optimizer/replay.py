import math

from app.errors import InternalPlanValidationError
from app.models.internal import CanonicalRequest, CompiledConstraints, LPSolution, Totals
from app.models.response import HourlyPlanEntry
from app.optimizer.numeric import EPS, REPLAY_TOLERANCE, clean_number


def _fail(message: str) -> None:
    raise InternalPlanValidationError(message)


def _battery_action(net_kwh: float) -> tuple[str, float]:
    if net_kwh > EPS:
        return "charge", clean_number(net_kwh)
    if net_kwh < -EPS:
        return "discharge", clean_number(-net_kwh)
    return "idle", 0.0


def build_hourly_plan(solution: LPSolution) -> list[HourlyPlanEntry]:
    plan: list[HourlyPlanEntry] = []
    for hour in range(24):
        action, amount = _battery_action(solution.battery_net_kwh[hour])
        plan.append(
            HourlyPlanEntry(
                hour=hour,
                grid_kwh=clean_number(solution.grid_kwh[hour]),
                solar_used_kwh=clean_number(solution.solar_used_kwh[hour]),
                battery_action=action,
                battery_kwh=amount,
                battery_energy_after_kwh=clean_number(solution.battery_energy_after_kwh[hour]),
            )
        )
    return plan


def replay_and_validate(
    request: CanonicalRequest,
    compiled: CompiledConstraints,
    plan: list[HourlyPlanEntry],
) -> None:
    tol = REPLAY_TOLERANCE
    if len(plan) != 24:
        _fail("hourly plan must contain exactly 24 rows")
    if [row.hour for row in plan] != list(range(24)):
        _fail("hourly plan must be sorted and cover hours 0 through 23 exactly")

    energy = float(request.battery.initial_energy_kwh)
    for hour, row in enumerate(plan):
        values = [row.grid_kwh, row.solar_used_kwh, row.battery_kwh, row.battery_energy_after_kwh]
        if not all(math.isfinite(value) for value in values):
            _fail(f"hour {hour}: non-finite public plan value")
        if (
            row.grid_kwh < -tol
            or row.solar_used_kwh < -tol
            or row.battery_kwh < -tol
            or row.battery_energy_after_kwh < -tol
        ):
            _fail(f"hour {hour}: negative public energy value")
        if row.solar_used_kwh > compiled.effective_solar_kwh[hour] + tol:
            _fail(f"hour {hour}: solar usage exceeds effective solar")

        if row.battery_action == "charge":
            if row.battery_kwh <= EPS:
                _fail(f"hour {hour}: charge action must have positive magnitude")
            charge = row.battery_kwh
            discharge = 0.0
        elif row.battery_action == "discharge":
            if row.battery_kwh <= EPS:
                _fail(f"hour {hour}: discharge action must have positive magnitude")
            charge = 0.0
            discharge = row.battery_kwh
        elif row.battery_action == "idle":
            if abs(row.battery_kwh) > tol:
                _fail(f"hour {hour}: idle action must have zero magnitude")
            charge = 0.0
            discharge = 0.0
        else:
            _fail(f"hour {hour}: invalid battery action")

        if not compiled.can_charge[hour] and charge > tol:
            _fail(f"hour {hour}: charging is prohibited")
        if not compiled.can_discharge[hour] and discharge > tol:
            _fail(f"hour {hour}: discharging is prohibited")
        if charge > request.battery.max_charge_kwh_per_hour + tol:
            _fail(f"hour {hour}: charge rate exceeded")
        if discharge > request.battery.max_discharge_kwh_per_hour + tol:
            _fail(f"hour {hour}: discharge rate exceeded")

        cap = compiled.max_grid_kwh[hour]
        if cap is not None and row.grid_kwh > cap + tol:
            _fail(f"hour {hour}: grid cap exceeded")

        lhs = row.grid_kwh + row.solar_used_kwh + discharge
        rhs = request.hours_by_hour[hour].demand_kwh + charge
        if abs(lhs - rhs) > tol:
            _fail(f"hour {hour}: energy balance failed")

        energy = energy + charge - discharge
        if energy > request.battery.capacity_kwh + tol:
            _fail(f"hour {hour}: battery capacity exceeded")
        if energy < compiled.minimum_battery_kwh[hour] - tol:
            _fail(f"hour {hour}: battery reserve violated")
        if abs(energy - row.battery_energy_after_kwh) > tol:
            _fail(f"hour {hour}: reported battery state is inconsistent")

    if abs(energy - request.battery.initial_energy_kwh) > tol:
        _fail("end-of-day battery neutrality failed")


def calculate_totals(request: CanonicalRequest, plan: list[HourlyPlanEntry]) -> Totals:
    total_grid = float(sum(row.grid_kwh for row in plan))
    total_cost = float(
        sum(
            row.grid_kwh * request.hours_by_hour[row.hour].tariff_bdt_per_kwh
            for row in plan
        )
    )
    peak_grid = float(max(row.grid_kwh for row in plan))
    return Totals(total_grid, total_cost, peak_grid)
