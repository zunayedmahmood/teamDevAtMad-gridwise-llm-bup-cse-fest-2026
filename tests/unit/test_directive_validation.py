import pytest

from app.errors import DirectiveValidationError
from app.interpreter.validator import validate_llm_batch
from app.models.internal import canonicalize_request
from app.models.llm import LLMDirective, LLMDirectiveBatch
from app.models.response import DirectiveType


def directive(**overrides) -> LLMDirective:
    data = dict(
        note_index=0,
        applies=True,
        directive_type=DirectiveType.SOLAR_REDUCTION,
        hours=[12, 13],
        factor=0.2,
        minimum_energy_kwh=None,
        max_grid_kwh=None,
        explanation="Solar remains at 20%.",
    )
    data.update(overrides)
    return LLMDirective(**data)


def validate_one(valid_request, item):
    canonical = canonicalize_request(valid_request)
    return validate_llm_batch(LLMDirectiveBatch(directives=[item]), canonical)


def test_valid_solar_reduction(valid_request):
    result = validate_one(valid_request, directive())
    assert result[0].factor == 0.2


@pytest.mark.parametrize("factor", [-0.1, 1.1])
def test_invalid_solar_factor(valid_request, factor):
    with pytest.raises(DirectiveValidationError):
        validate_one(valid_request, directive(factor=factor))


@pytest.mark.parametrize("hours", [[12, 12], [-1], [24], [13, 12], []])
def test_invalid_hours(valid_request, hours):
    with pytest.raises(DirectiveValidationError):
        validate_one(valid_request, directive(hours=hours))


def test_invalid_note_index(valid_request):
    with pytest.raises(DirectiveValidationError):
        validate_one(valid_request, directive(note_index=1))


def test_no_op_requires_false_and_nulls(valid_request):
    bad = directive(
        applies=True,
        directive_type=DirectiveType.NO_OP,
        hours=None,
        factor=None,
    )
    with pytest.raises(DirectiveValidationError):
        validate_one(valid_request, bad)


def test_valid_no_op(valid_request):
    item = directive(
        applies=False,
        directive_type=DirectiveType.NO_OP,
        hours=None,
        factor=None,
        explanation="Irrelevant.",
    )
    assert validate_one(valid_request, item)[0].applies is False


def test_reserve_greater_than_capacity_fails(valid_request):
    item = directive(
        directive_type=DirectiveType.MINIMUM_BATTERY_RESERVE,
        factor=None,
        minimum_energy_kwh=101.0,
    )
    with pytest.raises(DirectiveValidationError):
        validate_one(valid_request, item)


def test_grid_cap_negative_fails(valid_request):
    item = directive(
        directive_type=DirectiveType.MAX_GRID_WINDOW,
        factor=None,
        max_grid_kwh=-1.0,
    )
    with pytest.raises(DirectiveValidationError):
        validate_one(valid_request, item)


def test_wrong_field_for_directive_fails(valid_request):
    item = directive(minimum_energy_kwh=20.0)
    with pytest.raises(DirectiveValidationError):
        validate_one(valid_request, item)


def test_wrong_directive_count_fails(valid_request):
    canonical = canonicalize_request(valid_request)
    with pytest.raises(DirectiveValidationError):
        validate_llm_batch(LLMDirectiveBatch(directives=[]), canonical)


def test_duplicate_note_index_fails(valid_request):
    valid_request.operator_notes[:] = ["a", "b"]
    canonical = canonicalize_request(valid_request)
    batch = LLMDirectiveBatch(directives=[directive(note_index=0), directive(note_index=0)])
    with pytest.raises(DirectiveValidationError):
        validate_llm_batch(batch, canonical)


def test_no_op_with_hours_fails(valid_request):
    item = directive(
        applies=False,
        directive_type=DirectiveType.NO_OP,
        hours=[12],
        factor=None,
    )
    with pytest.raises(DirectiveValidationError):
        validate_one(valid_request, item)


def test_no_charge_with_factor_fails(valid_request):
    item = directive(
        directive_type=DirectiveType.NO_CHARGE_WINDOW,
        factor=0.5,
    )
    with pytest.raises(DirectiveValidationError):
        validate_one(valid_request, item)
