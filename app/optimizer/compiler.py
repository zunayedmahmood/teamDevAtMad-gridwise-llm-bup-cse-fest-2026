from app.errors import OptimizationInfeasibleError
from app.models.internal import CanonicalRequest, CompiledConstraints, ValidatedDirective
from app.models.response import DirectiveType


def compile_constraints(
    request: CanonicalRequest,
    directives: list[ValidatedDirective],
) -> CompiledConstraints:
    effective_factor = [1.0] * 24
    minimum_battery = [request.battery.minimum_energy_kwh] * 24
    can_charge = [True] * 24
    can_discharge = [True] * 24
    max_grid: list[float | None] = [None] * 24

    for directive in directives:
        if not directive.applies:
            continue
        hours = directive.hours or ()

        if directive.directive_type is DirectiveType.SOLAR_REDUCTION:
            factor = float(directive.factor)
            for hour in hours:
                effective_factor[hour] = min(effective_factor[hour], factor)

        elif directive.directive_type is DirectiveType.MINIMUM_BATTERY_RESERVE:
            reserve = float(directive.minimum_energy_kwh)
            for hour in hours:
                minimum_battery[hour] = max(minimum_battery[hour], reserve)

        elif directive.directive_type is DirectiveType.NO_CHARGE_WINDOW:
            for hour in hours:
                can_charge[hour] = False

        elif directive.directive_type is DirectiveType.NO_DISCHARGE_WINDOW:
            for hour in hours:
                can_discharge[hour] = False

        elif directive.directive_type is DirectiveType.MAX_GRID_WINDOW:
            cap = float(directive.max_grid_kwh)
            for hour in hours:
                max_grid[hour] = cap if max_grid[hour] is None else min(max_grid[hour], cap)

    if any(value > request.battery.capacity_kwh for value in minimum_battery):
        raise OptimizationInfeasibleError("minimum battery reserve exceeds capacity")

    effective_solar = tuple(
        float(request.hours_by_hour[hour].solar_kwh * effective_factor[hour])
        for hour in range(24)
    )

    return CompiledConstraints(
        effective_solar_kwh=effective_solar,
        minimum_battery_kwh=tuple(float(value) for value in minimum_battery),
        can_charge=tuple(can_charge),
        can_discharge=tuple(can_discharge),
        max_grid_kwh=tuple(max_grid),
    )
