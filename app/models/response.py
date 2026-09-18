from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


class StrictResponseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )


class DirectiveType(str, Enum):
    SOLAR_REDUCTION = "solar_reduction"
    MINIMUM_BATTERY_RESERVE = "minimum_battery_reserve"
    NO_CHARGE_WINDOW = "no_charge_window"
    NO_DISCHARGE_WINDOW = "no_discharge_window"
    MAX_GRID_WINDOW = "max_grid_window"
    NO_OP = "no_op"


class SolarReductionAdjustment(StrictResponseModel):
    hours: list[int]
    factor: float


class MinimumBatteryReserveAdjustment(StrictResponseModel):
    hours: list[int]
    minimum_energy_kwh: float


class NoChargeAdjustment(StrictResponseModel):
    hours: list[int]


class NoDischargeAdjustment(StrictResponseModel):
    hours: list[int]


class MaxGridAdjustment(StrictResponseModel):
    hours: list[int]
    max_grid_kwh: float


StructuredAdjustment = (
    SolarReductionAdjustment
    | MinimumBatteryReserveAdjustment
    | NoChargeAdjustment
    | NoDischargeAdjustment
    | MaxGridAdjustment
)


class DirectiveInterpretation(StrictResponseModel):
    note_index: int
    applies: bool
    directive_type: DirectiveType
    structured_adjustment: StructuredAdjustment | None
    explanation: str

    @model_validator(mode="after")
    def validate_adjustment_shape(self):
        if self.directive_type is DirectiveType.NO_OP:
            if self.applies or self.structured_adjustment is not None:
                raise ValueError("no_op must use applies=false and no adjustment")
            return self

        if not self.applies or self.structured_adjustment is None:
            raise ValueError("active directives require applies=true and an adjustment")

        expected_fields = {
            DirectiveType.SOLAR_REDUCTION: {"hours", "factor"},
            DirectiveType.MINIMUM_BATTERY_RESERVE: {"hours", "minimum_energy_kwh"},
            DirectiveType.NO_CHARGE_WINDOW: {"hours"},
            DirectiveType.NO_DISCHARGE_WINDOW: {"hours"},
            DirectiveType.MAX_GRID_WINDOW: {"hours", "max_grid_kwh"},
        }[self.directive_type]
        actual_fields = set(self.structured_adjustment.model_dump())
        if actual_fields != expected_fields:
            raise ValueError("structured_adjustment does not match directive_type")
        return self


class HourlyPlanEntry(StrictResponseModel):
    hour: int
    grid_kwh: float
    solar_used_kwh: float
    battery_action: Literal["charge", "discharge", "idle"]
    battery_kwh: float
    battery_energy_after_kwh: float


class EnergyResponse(StrictResponseModel):
    scenario_id: str
    directive_interpretation: list[DirectiveInterpretation]
    hourly_plan: list[HourlyPlanEntry]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str
