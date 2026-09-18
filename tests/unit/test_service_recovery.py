import pytest

from app.errors import LLMInterpretationError
from app.models.llm import LLMDirective, LLMDirectiveBatch
from app.models.response import DirectiveType
from app.services.optimization_service import optimize_energy


def valid_no_op_batch():
    return LLMDirectiveBatch(
        directives=[
            LLMDirective(
                note_index=0,
                applies=False,
                directive_type=DirectiveType.NO_OP,
                hours=None,
                factor=None,
                minimum_energy_kwh=None,
                max_grid_kwh=None,
                explanation="Irrelevant note.",
            )
        ]
    )


class DomainRecoveryInterpreter:
    def __init__(self):
        self.recovery_calls = 0

    async def interpret(self, request):
        return LLMDirectiveBatch(
            directives=[
                LLMDirective(
                    note_index=0,
                    applies=True,
                    directive_type=DirectiveType.SOLAR_REDUCTION,
                    hours=[10],
                    factor=2.0,
                    minimum_energy_kwh=None,
                    max_grid_kwh=None,
                    explanation="invalid",
                )
            ]
        )

    async def recover(self, request, previous, validation_error):
        self.recovery_calls += 1
        assert previous is not None
        return valid_no_op_batch()


class ParseRecoveryInterpreter:
    def __init__(self):
        self.recovery_calls = 0

    async def interpret(self, request):
        raise LLMInterpretationError("no structured output")

    async def recover(self, request, previous, validation_error):
        self.recovery_calls += 1
        assert previous is None
        return valid_no_op_batch()


class FailedRecoveryInterpreter(DomainRecoveryInterpreter):
    async def recover(self, request, previous, validation_error):
        self.recovery_calls += 1
        return await self.interpret(request)


@pytest.mark.asyncio
async def test_domain_validation_failure_gets_one_recovery(valid_request):
    interpreter = DomainRecoveryInterpreter()
    response = await optimize_energy(valid_request, interpreter)
    assert response.directive_interpretation[0].directive_type is DirectiveType.NO_OP
    assert interpreter.recovery_calls == 1


@pytest.mark.asyncio
async def test_parse_failure_gets_one_recovery(valid_request):
    interpreter = ParseRecoveryInterpreter()
    response = await optimize_energy(valid_request, interpreter)
    assert response.directive_interpretation[0].directive_type is DirectiveType.NO_OP
    assert interpreter.recovery_calls == 1


@pytest.mark.asyncio
async def test_invalid_recovery_fails_controlled(valid_request):
    interpreter = FailedRecoveryInterpreter()
    with pytest.raises(LLMInterpretationError):
        await optimize_energy(valid_request, interpreter)
    assert interpreter.recovery_calls == 1


class ParseThenInvalidRecoveryInterpreter:
    def __init__(self):
        self.recovery_calls = 0

    async def interpret(self, request):
        raise LLMInterpretationError("no structured output")

    async def recover(self, request, previous, validation_error):
        self.recovery_calls += 1
        return LLMDirectiveBatch(
            directives=[
                LLMDirective(
                    note_index=0,
                    applies=True,
                    directive_type=DirectiveType.SOLAR_REDUCTION,
                    hours=[10],
                    factor=2.0,
                    minimum_energy_kwh=None,
                    max_grid_kwh=None,
                    explanation="still invalid",
                )
            ]
        )


@pytest.mark.asyncio
async def test_parse_failure_never_gets_more_than_one_recovery(valid_request):
    interpreter = ParseThenInvalidRecoveryInterpreter()
    with pytest.raises(LLMInterpretationError):
        await optimize_energy(valid_request, interpreter)
    assert interpreter.recovery_calls == 1
