import math

from app.errors import DirectiveValidationError
from app.models.internal import CanonicalRequest, ValidatedDirective
from app.models.llm import LLMDirective, LLMDirectiveBatch
from app.models.response import (
    DirectiveInterpretation,
    DirectiveType,
    MaxGridAdjustment,
    MinimumBatteryReserveAdjustment,
    NoChargeAdjustment,
    NoDischargeAdjustment,
    SolarReductionAdjustment,
)


def _fail(message: str) -> None:
    raise DirectiveValidationError(message)


def _validate_hours(value: list[int] | None, note_index: int) -> tuple[int, ...]:
    if value is None or len(value) == 0:
        _fail(f"note {note_index}: hours are required")
    if any(hour < 0 or hour > 23 for hour in value):
        _fail(f"note {note_index}: hours must be between 0 and 23")
    if len(set(value)) != len(value):
        _fail(f"note {note_index}: hours must be unique")
    if value != sorted(value):
        _fail(f"note {note_index}: hours must be sorted ascending")
    return tuple(value)


def _must_be_none(note_index: int, **values: object) -> None:
    populated = [name for name, value in values.items() if value is not None]
    if populated:
        _fail(f"note {note_index}: unexpected fields: {', '.join(populated)}")


def _validate_one(d: LLMDirective, request: CanonicalRequest) -> ValidatedDirective:
    if not d.explanation.strip():
        _fail(f"note {d.note_index}: explanation must not be empty")

    kind = d.directive_type
    if kind is DirectiveType.NO_OP:
        if d.applies:
            _fail(f"note {d.note_index}: no_op must use applies=false")
        _must_be_none(
            d.note_index,
            hours=d.hours,
            factor=d.factor,
            minimum_energy_kwh=d.minimum_energy_kwh,
            max_grid_kwh=d.max_grid_kwh,
        )
        return ValidatedDirective(
            note_index=d.note_index,
            applies=False,
            directive_type=kind,
            hours=None,
            factor=None,
            minimum_energy_kwh=None,
            max_grid_kwh=None,
            explanation=d.explanation.strip(),
        )

    if not d.applies:
        _fail(f"note {d.note_index}: non-no_op directives must use applies=true")

    hours = _validate_hours(d.hours, d.note_index)

    if kind is DirectiveType.SOLAR_REDUCTION:
        _must_be_none(
            d.note_index,
            minimum_energy_kwh=d.minimum_energy_kwh,
            max_grid_kwh=d.max_grid_kwh,
        )
        if d.factor is None or not math.isfinite(d.factor) or not 0 <= d.factor <= 1:
            _fail(f"note {d.note_index}: solar factor must be between 0 and 1")
        return ValidatedDirective(d.note_index, True, kind, hours, float(d.factor), None, None, d.explanation.strip())

    if kind is DirectiveType.MINIMUM_BATTERY_RESERVE:
        _must_be_none(d.note_index, factor=d.factor, max_grid_kwh=d.max_grid_kwh)
        reserve = d.minimum_energy_kwh
        if reserve is None or not math.isfinite(reserve) or not 0 <= reserve <= request.battery.capacity_kwh:
            _fail(f"note {d.note_index}: reserve must be between 0 and battery capacity")
        return ValidatedDirective(d.note_index, True, kind, hours, None, float(reserve), None, d.explanation.strip())

    if kind is DirectiveType.NO_CHARGE_WINDOW:
        _must_be_none(
            d.note_index,
            factor=d.factor,
            minimum_energy_kwh=d.minimum_energy_kwh,
            max_grid_kwh=d.max_grid_kwh,
        )
        return ValidatedDirective(d.note_index, True, kind, hours, None, None, None, d.explanation.strip())

    if kind is DirectiveType.NO_DISCHARGE_WINDOW:
        _must_be_none(
            d.note_index,
            factor=d.factor,
            minimum_energy_kwh=d.minimum_energy_kwh,
            max_grid_kwh=d.max_grid_kwh,
        )
        return ValidatedDirective(d.note_index, True, kind, hours, None, None, None, d.explanation.strip())

    if kind is DirectiveType.MAX_GRID_WINDOW:
        _must_be_none(d.note_index, factor=d.factor, minimum_energy_kwh=d.minimum_energy_kwh)
        cap = d.max_grid_kwh
        if cap is None or not math.isfinite(cap) or cap < 0:
            _fail(f"note {d.note_index}: grid cap must be finite and non-negative")
        return ValidatedDirective(d.note_index, True, kind, hours, None, None, float(cap), d.explanation.strip())

    _fail(f"note {d.note_index}: unsupported directive type")
    raise AssertionError("unreachable")


def validate_llm_batch(batch: LLMDirectiveBatch, request: CanonicalRequest) -> list[ValidatedDirective]:
    expected_count = len(request.operator_notes)
    if len(batch.directives) != expected_count:
        _fail("wrong directive count")

    seen: set[int] = set()
    validated: list[ValidatedDirective] = []
    for directive in batch.directives:
        if directive.note_index < 0 or directive.note_index >= expected_count:
            _fail("invalid note_index")
        if directive.note_index in seen:
            _fail("duplicate note_index")
        seen.add(directive.note_index)
        validated.append(_validate_one(directive, request))

    if seen != set(range(expected_count)):
        _fail("missing note_index")

    return sorted(validated, key=lambda item: item.note_index)


def to_public_interpretation(d: ValidatedDirective) -> DirectiveInterpretation:
    if d.directive_type is DirectiveType.NO_OP:
        adjustment = None
    elif d.directive_type is DirectiveType.SOLAR_REDUCTION:
        adjustment = SolarReductionAdjustment(hours=list(d.hours or ()), factor=float(d.factor))
    elif d.directive_type is DirectiveType.MINIMUM_BATTERY_RESERVE:
        adjustment = MinimumBatteryReserveAdjustment(
            hours=list(d.hours or ()), minimum_energy_kwh=float(d.minimum_energy_kwh)
        )
    elif d.directive_type is DirectiveType.NO_CHARGE_WINDOW:
        adjustment = NoChargeAdjustment(hours=list(d.hours or ()))
    elif d.directive_type is DirectiveType.NO_DISCHARGE_WINDOW:
        adjustment = NoDischargeAdjustment(hours=list(d.hours or ()))
    elif d.directive_type is DirectiveType.MAX_GRID_WINDOW:
        adjustment = MaxGridAdjustment(hours=list(d.hours or ()), max_grid_kwh=float(d.max_grid_kwh))
    else:
        raise DirectiveValidationError("unsupported validated directive")

    return DirectiveInterpretation(
        note_index=d.note_index,
        applies=d.applies,
        directive_type=d.directive_type,
        structured_adjustment=adjustment,
        explanation=d.explanation,
    )
