from dataclasses import dataclass

from app.models.request import BatteryInput, HourInput, EnergyRequest
from app.models.response import DirectiveType


@dataclass(frozen=True)
class CanonicalRequest:
    scenario_id: str
    operator_notes: tuple[str, ...]
    hours_by_hour: dict[int, HourInput]
    battery: BatteryInput


@dataclass(frozen=True)
class ValidatedDirective:
    note_index: int
    applies: bool
    directive_type: DirectiveType
    hours: tuple[int, ...] | None
    factor: float | None
    minimum_energy_kwh: float | None
    max_grid_kwh: float | None
    explanation: str


@dataclass(frozen=True)
class CompiledConstraints:
    effective_solar_kwh: tuple[float, ...]
    minimum_battery_kwh: tuple[float, ...]
    can_charge: tuple[bool, ...]
    can_discharge: tuple[bool, ...]
    max_grid_kwh: tuple[float | None, ...]


@dataclass(frozen=True)
class LPSolution:
    grid_kwh: tuple[float, ...]
    solar_used_kwh: tuple[float, ...]
    battery_net_kwh: tuple[float, ...]
    battery_energy_after_kwh: tuple[float, ...]
    objective_value: float


@dataclass(frozen=True)
class Totals:
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float


def canonicalize_request(req: EnergyRequest) -> CanonicalRequest:
    return CanonicalRequest(
        scenario_id=req.scenario_id,
        operator_notes=tuple(req.operator_notes),
        hours_by_hour={item.hour: item for item in req.hours},
        battery=req.battery,
    )
