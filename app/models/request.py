import math

from pydantic import BaseModel, ConfigDict, Field

from app.errors import SemanticInputError


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )


class HourInput(StrictModel):
    hour: int = Field(ge=0, le=23)
    demand_kwh: float = Field(ge=0)
    solar_kwh: float = Field(ge=0)
    tariff_bdt_per_kwh: float = Field(ge=0)


class BatteryInput(StrictModel):
    capacity_kwh: float = Field(ge=0)
    initial_energy_kwh: float = Field(ge=0)
    minimum_energy_kwh: float = Field(ge=0)
    max_charge_kwh_per_hour: float = Field(ge=0)
    max_discharge_kwh_per_hour: float = Field(ge=0)


class EnergyRequest(StrictModel):
    scenario_id: str
    operator_notes: list[str]
    hours: list[HourInput]
    battery: BatteryInput


def validate_request_semantics(req: EnergyRequest) -> None:
    if not req.scenario_id.strip():
        raise SemanticInputError("scenario_id must not be empty")

    if not 1 <= len(req.operator_notes) <= 3:
        raise SemanticInputError("operator_notes must contain between 1 and 3 notes")

    for index, note in enumerate(req.operator_notes):
        if not note.strip():
            raise SemanticInputError(f"operator_notes[{index}] must not be empty")

    if len(req.hours) != 24:
        raise SemanticInputError("hours must contain exactly 24 entries")

    hour_ids = [item.hour for item in req.hours]
    if len(set(hour_ids)) != 24:
        raise SemanticInputError("hour identifiers must be unique")
    if set(hour_ids) != set(range(24)):
        raise SemanticInputError("hours must cover exactly 0 through 23")

    numeric_values = []
    for item in req.hours:
        numeric_values.extend(
            [item.demand_kwh, item.solar_kwh, item.tariff_bdt_per_kwh]
        )
    b = req.battery
    numeric_values.extend(
        [
            b.capacity_kwh,
            b.initial_energy_kwh,
            b.minimum_energy_kwh,
            b.max_charge_kwh_per_hour,
            b.max_discharge_kwh_per_hour,
        ]
    )
    if not all(math.isfinite(value) for value in numeric_values):
        raise SemanticInputError("all numeric values must be finite")

    if b.initial_energy_kwh > b.capacity_kwh:
        raise SemanticInputError("initial battery energy exceeds capacity")
    if b.minimum_energy_kwh > b.capacity_kwh:
        raise SemanticInputError("minimum battery energy exceeds capacity")
    if b.initial_energy_kwh < b.minimum_energy_kwh:
        raise SemanticInputError("initial battery energy is below baseline minimum")
