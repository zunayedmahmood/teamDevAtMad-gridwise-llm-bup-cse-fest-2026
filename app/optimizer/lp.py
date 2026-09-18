from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from app.errors import InternalPlanValidationError, OptimizationInfeasibleError
from app.models.internal import CanonicalRequest, CompiledConstraints, LPSolution


@dataclass(frozen=True)
class VariableIndex:
    n: int = 24

    def grid(self, hour: int) -> int:
        return hour

    def solar(self, hour: int) -> int:
        return self.n + hour

    def battery_net(self, hour: int) -> int:
        return 2 * self.n + hour

    def energy(self, hour: int) -> int:
        return 3 * self.n + hour

    @property
    def size(self) -> int:
        return 4 * self.n


def _validate_compiled_constraints(
    request: CanonicalRequest,
    compiled: CompiledConstraints,
) -> None:
    vectors = (
        compiled.effective_solar_kwh,
        compiled.minimum_battery_kwh,
        compiled.can_charge,
        compiled.can_discharge,
        compiled.max_grid_kwh,
    )
    if any(len(values) != 24 for values in vectors):
        raise InternalPlanValidationError("compiled constraints must contain 24 hourly values")

    capacity = request.battery.capacity_kwh
    for hour in range(24):
        if compiled.effective_solar_kwh[hour] < 0:
            raise InternalPlanValidationError(f"hour {hour}: effective solar cannot be negative")
        if compiled.minimum_battery_kwh[hour] > capacity:
            raise OptimizationInfeasibleError("minimum battery reserve exceeds capacity")
        cap = compiled.max_grid_kwh[hour]
        if cap is not None and cap < 0:
            raise InternalPlanValidationError(f"hour {hour}: grid cap cannot be negative")


def _build_bounds(
    request: CanonicalRequest,
    compiled: CompiledConstraints,
    index: VariableIndex,
) -> list[tuple[float | None, float | None]]:
    bounds: list[tuple[float | None, float | None]] = [(None, None)] * index.size
    battery = request.battery

    for hour in range(24):
        bounds[index.grid(hour)] = (0.0, compiled.max_grid_kwh[hour])
        bounds[index.solar(hour)] = (0.0, compiled.effective_solar_kwh[hour])

        lower = -battery.max_discharge_kwh_per_hour
        upper = battery.max_charge_kwh_per_hour
        if not compiled.can_discharge[hour]:
            lower = 0.0
        if not compiled.can_charge[hour]:
            upper = 0.0
        bounds[index.battery_net(hour)] = (float(lower), float(upper))

        bounds[index.energy(hour)] = (
            compiled.minimum_battery_kwh[hour],
            battery.capacity_kwh,
        )

    return bounds


def solve_lp(request: CanonicalRequest, compiled: CompiledConstraints) -> LPSolution:
    _validate_compiled_constraints(request, compiled)
    index = VariableIndex()
    c = np.zeros(index.size, dtype=float)
    for hour in range(24):
        c[index.grid(hour)] = request.hours_by_hour[hour].tariff_bdt_per_kwh

    equality_rows: list[np.ndarray] = []
    equality_rhs: list[float] = []

    for hour in range(24):
        row = np.zeros(index.size, dtype=float)
        row[index.grid(hour)] = 1.0
        row[index.solar(hour)] = 1.0
        row[index.battery_net(hour)] = -1.0
        equality_rows.append(row)
        equality_rhs.append(request.hours_by_hour[hour].demand_kwh)

    for hour in range(24):
        row = np.zeros(index.size, dtype=float)
        row[index.energy(hour)] = 1.0
        row[index.battery_net(hour)] = -1.0
        if hour == 0:
            equality_rows.append(row)
            equality_rhs.append(request.battery.initial_energy_kwh)
        else:
            row[index.energy(hour - 1)] = -1.0
            equality_rows.append(row)
            equality_rhs.append(0.0)

    final_row = np.zeros(index.size, dtype=float)
    final_row[index.energy(23)] = 1.0
    equality_rows.append(final_row)
    equality_rhs.append(request.battery.initial_energy_kwh)

    result = linprog(
        c=c,
        A_eq=np.asarray(equality_rows),
        b_eq=np.asarray(equality_rhs),
        bounds=_build_bounds(request, compiled, index),
        method="highs",
    )

    if not result.success:
        if result.status == 2:
            raise OptimizationInfeasibleError("The supplied energy constraints are infeasible.")
        raise InternalPlanValidationError(f"LP solver failed with status {result.status}")

    values = result.x
    return LPSolution(
        grid_kwh=tuple(float(values[index.grid(h)]) for h in range(24)),
        solar_used_kwh=tuple(float(values[index.solar(h)]) for h in range(24)),
        battery_net_kwh=tuple(float(values[index.battery_net(h)]) for h in range(24)),
        battery_energy_after_kwh=tuple(float(values[index.energy(h)]) for h in range(24)),
        objective_value=float(result.fun),
    )
