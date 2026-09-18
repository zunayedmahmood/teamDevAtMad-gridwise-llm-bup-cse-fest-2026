from app.interpreter.validator import validate_llm_batch
from app.models.internal import canonicalize_request
from app.models.llm import LLMDirective, LLMDirectiveBatch
from app.models.response import DirectiveType
from app.optimizer.compiler import compile_constraints


def make_directive(index, kind, hours, *, factor=None, reserve=None, cap=None):
    return LLMDirective(
        note_index=index,
        applies=True,
        directive_type=kind,
        hours=hours,
        factor=factor,
        minimum_energy_kwh=reserve,
        max_grid_kwh=cap,
        explanation="test",
    )


def test_compiler_applies_overlaps(valid_request):
    valid_request.operator_notes[:] = ["a", "b", "c"]
    canonical = canonicalize_request(valid_request)
    batch = LLMDirectiveBatch(
        directives=[
            make_directive(0, DirectiveType.SOLAR_REDUCTION, [10, 11], factor=0.5),
            make_directive(1, DirectiveType.MINIMUM_BATTERY_RESERVE, [11, 12], reserve=60.0),
            make_directive(2, DirectiveType.MAX_GRID_WINDOW, [12], cap=80.0),
        ]
    )
    directives = validate_llm_batch(batch, canonical)
    compiled = compile_constraints(canonical, directives)
    assert compiled.effective_solar_kwh[10] == 0.0
    assert compiled.minimum_battery_kwh[11] == 60.0
    assert compiled.max_grid_kwh[12] == 80.0
    assert compiled.minimum_battery_kwh[0] == valid_request.battery.minimum_energy_kwh


def test_restrictive_overlap_policies(valid_request):
    valid_request.operator_notes[:] = ["a", "b", "c"]
    valid_request.hours[10].solar_kwh = 100.0
    canonical = canonicalize_request(valid_request)
    batch = LLMDirectiveBatch(
        directives=[
            make_directive(0, DirectiveType.SOLAR_REDUCTION, [10], factor=0.7),
            make_directive(1, DirectiveType.SOLAR_REDUCTION, [10], factor=0.4),
            make_directive(2, DirectiveType.MINIMUM_BATTERY_RESERVE, [10], reserve=70.0),
        ]
    )
    compiled = compile_constraints(canonical, validate_llm_batch(batch, canonical))
    assert compiled.effective_solar_kwh[10] == 40.0
    assert compiled.minimum_battery_kwh[10] == 70.0


def test_compiler_applies_no_charge_and_no_discharge(valid_request):
    valid_request.operator_notes[:] = ["a", "b"]
    canonical = canonicalize_request(valid_request)
    batch = LLMDirectiveBatch(
        directives=[
            make_directive(0, DirectiveType.NO_CHARGE_WINDOW, [4, 5]),
            make_directive(1, DirectiveType.NO_DISCHARGE_WINDOW, [5, 6]),
        ]
    )

    compiled = compile_constraints(canonical, validate_llm_batch(batch, canonical))

    assert compiled.can_charge[4] is False
    assert compiled.can_charge[5] is False
    assert compiled.can_discharge[5] is False
    assert compiled.can_discharge[6] is False
    assert compiled.can_charge[6] is True
    assert compiled.can_discharge[4] is True


def test_compiler_uses_strictest_overlapping_grid_cap(valid_request):
    valid_request.operator_notes[:] = ["a", "b"]
    canonical = canonicalize_request(valid_request)
    batch = LLMDirectiveBatch(
        directives=[
            make_directive(0, DirectiveType.MAX_GRID_WINDOW, [8], cap=100.0),
            make_directive(1, DirectiveType.MAX_GRID_WINDOW, [8], cap=60.0),
        ]
    )

    compiled = compile_constraints(canonical, validate_llm_batch(batch, canonical))

    assert compiled.max_grid_kwh[8] == 60.0


def test_compiler_uses_highest_overlapping_reserve(valid_request):
    valid_request.operator_notes[:] = ["a", "b"]
    canonical = canonicalize_request(valid_request)
    batch = LLMDirectiveBatch(
        directives=[
            make_directive(0, DirectiveType.MINIMUM_BATTERY_RESERVE, [9], reserve=60.0),
            make_directive(1, DirectiveType.MINIMUM_BATTERY_RESERVE, [9], reserve=80.0),
        ]
    )

    compiled = compile_constraints(canonical, validate_llm_batch(batch, canonical))

    assert compiled.minimum_battery_kwh[9] == 80.0


def test_no_op_leaves_baseline_constraints_unchanged(valid_request):
    canonical = canonicalize_request(valid_request)
    batch = LLMDirectiveBatch(
        directives=[
            LLMDirective(
                note_index=0,
                applies=False,
                directive_type=DirectiveType.NO_OP,
                hours=None,
                factor=None,
                minimum_energy_kwh=None,
                max_grid_kwh=None,
                explanation="irrelevant",
            )
        ]
    )

    compiled = compile_constraints(canonical, validate_llm_batch(batch, canonical))

    assert compiled.effective_solar_kwh == tuple(
        canonical.hours_by_hour[h].solar_kwh for h in range(24)
    )
    assert compiled.minimum_battery_kwh == tuple(
        canonical.battery.minimum_energy_kwh for _ in range(24)
    )
    assert all(compiled.can_charge)
    assert all(compiled.can_discharge)
    assert all(cap is None for cap in compiled.max_grid_kwh)
