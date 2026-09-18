import pytest
from pydantic import ValidationError

from app.models.response import (
    DirectiveInterpretation,
    DirectiveType,
    HourlyPlanEntry,
    NoChargeAdjustment,
    SolarReductionAdjustment,
)


def test_directive_enum_exact_values():
    assert {item.value for item in DirectiveType} == {
        "solar_reduction",
        "minimum_battery_reserve",
        "no_charge_window",
        "no_discharge_window",
        "max_grid_window",
        "no_op",
    }


def test_response_model_forbids_extra_fields():
    with pytest.raises(ValidationError):
        HourlyPlanEntry(
            hour=0,
            grid_kwh=1.0,
            solar_used_kwh=0.0,
            battery_action="idle",
            battery_kwh=0.0,
            battery_energy_after_kwh=10.0,
            extra=1,
        )


def test_no_op_response_semantics_are_enforced():
    with pytest.raises(ValidationError):
        DirectiveInterpretation(
            note_index=0,
            applies=True,
            directive_type=DirectiveType.NO_OP,
            structured_adjustment=None,
            explanation="irrelevant",
        )


def test_adjustment_shape_must_match_directive_type():
    with pytest.raises(ValidationError):
        DirectiveInterpretation(
            note_index=0,
            applies=True,
            directive_type=DirectiveType.SOLAR_REDUCTION,
            structured_adjustment=NoChargeAdjustment(hours=[12]),
            explanation="mismatch",
        )

    valid = DirectiveInterpretation(
        note_index=0,
        applies=True,
        directive_type=DirectiveType.SOLAR_REDUCTION,
        structured_adjustment=SolarReductionAdjustment(hours=[12], factor=0.2),
        explanation="valid",
    )
    assert valid.directive_type is DirectiveType.SOLAR_REDUCTION
