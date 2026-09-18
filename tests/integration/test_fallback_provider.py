import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.dependencies import get_interpreter
from app.errors import LLMConfigurationError, LLMInterpretationError, LLMProviderError
from app.main import app
from app.models.llm import LLMDirective, LLMDirectiveBatch
from app.models.response import DirectiveType
from app.services.optimization_service import optimize_energy


class ProviderFailureThenFallbackSuccessInterpreter:
    def __init__(self, fallback_batch: LLMDirectiveBatch):
        self.fallback_batch = fallback_batch
        self.interpret_calls = 0
        self.recover_calls = 0

    async def interpret(self, request):
        self.interpret_calls += 1
        raise LLMProviderError("Simulated primary provider outage 503")

    async def recover(self, request, previous, validation_error):
        self.recover_calls += 1
        assert previous is None  # Primary failed before producing structured output
        assert isinstance(validation_error, LLMProviderError)
        return self.fallback_batch


class BothFailInterpreter:
    def __init__(self):
        self.interpret_calls = 0
        self.recover_calls = 0

    async def interpret(self, request):
        self.interpret_calls += 1
        raise LLMProviderError("Primary provider error")

    async def recover(self, request, previous, validation_error):
        self.recover_calls += 1
        raise LLMProviderError("Fallback provider error")


class ConfigErrorInterpreter:
    def __init__(self):
        self.interpret_calls = 0
        self.recover_calls = 0

    async def interpret(self, request):
        self.interpret_calls += 1
        raise LLMConfigurationError("OPENAI_API_KEY is not configured")

    async def recover(self, request, previous, validation_error):
        self.recover_calls += 1
        raise AssertionError("Fallback must not be called on permanent configuration error")


class FallbackFailsValidationInterpreter:
    def __init__(self):
        self.interpret_calls = 0
        self.recover_calls = 0

    async def interpret(self, request):
        self.interpret_calls += 1
        raise LLMProviderError("Primary provider error")

    async def recover(self, request, previous, validation_error):
        self.recover_calls += 1
        # Return batch with invalid hours
        return LLMDirectiveBatch(
            directives=[
                LLMDirective(
                    note_index=0,
                    applies=True,
                    directive_type=DirectiveType.NO_CHARGE_WINDOW,
                    hours=[25],  # Out of range hour
                    factor=None,
                    minimum_energy_kwh=None,
                    max_grid_kwh=None,
                    explanation="invalid",
                )
            ]
        )


@pytest.mark.asyncio
async def test_primary_provider_failure_fallback_succeeds(valid_request, no_op_batch):
    interpreter = ProviderFailureThenFallbackSuccessInterpreter(no_op_batch)
    response = await optimize_energy(valid_request, interpreter)

    assert response.scenario_id == valid_request.scenario_id
    assert response.directive_interpretation[0].directive_type is DirectiveType.NO_OP
    assert interpreter.interpret_calls == 1
    assert interpreter.recover_calls == 1


def test_primary_provider_failure_fallback_endpoint_integration(valid_request_dict, no_op_batch):
    interpreter = ProviderFailureThenFallbackSuccessInterpreter(no_op_batch)
    app.dependency_overrides[get_interpreter] = lambda: interpreter
    try:
        response = TestClient(app).post("/optimize-energy", json=valid_request_dict)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["scenario_id"] == valid_request_dict["scenario_id"]
    assert interpreter.interpret_calls == 1
    assert interpreter.recover_calls == 1


@pytest.mark.asyncio
async def test_primary_interpretation_failure_fallback_succeeds(valid_request, no_op_batch):
    class ParseFailInterpreter:
        def __init__(self):
            self.interpret_calls = 0
            self.recover_calls = 0

        async def interpret(self, request):
            self.interpret_calls += 1
            raise LLMInterpretationError("Unparseable output")

        async def recover(self, request, previous, validation_error):
            self.recover_calls += 1
            return no_op_batch

    interpreter = ParseFailInterpreter()
    response = await optimize_energy(valid_request, interpreter)
    assert response.scenario_id == valid_request.scenario_id
    assert interpreter.interpret_calls == 1
    assert interpreter.recover_calls == 1


def test_both_primary_and_fallback_fail_returns_sanitized_500(valid_request_dict):
    interpreter = BothFailInterpreter()
    app.dependency_overrides[get_interpreter] = lambda: interpreter
    try:
        response = TestClient(app).post("/optimize-energy", json=valid_request_dict)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 500
    assert response.json() == {"detail": "Directive interpretation is temporarily unavailable."}
    assert "Traceback" not in response.text
    assert interpreter.interpret_calls == 1
    assert interpreter.recover_calls == 1


@pytest.mark.asyncio
async def test_configuration_error_fails_immediately_without_fallback(valid_request):
    interpreter = ConfigErrorInterpreter()
    with pytest.raises(LLMConfigurationError):
        await optimize_energy(valid_request, interpreter)

    assert interpreter.interpret_calls == 1
    assert interpreter.recover_calls == 0


@pytest.mark.asyncio
async def test_insufficient_deadline_skips_provider_fallback(valid_request, no_op_batch, monkeypatch):
    interpreter = ProviderFailureThenFallbackSuccessInterpreter(no_op_batch)
    monkeypatch.setattr(settings, "optimize_deadline_seconds", 1.0)

    with pytest.raises(LLMProviderError, match="Directive recovery budget exhausted"):
        await optimize_energy(valid_request, interpreter)

    assert interpreter.interpret_calls == 1
    assert interpreter.recover_calls == 0


@pytest.mark.asyncio
async def test_fallback_output_fails_validation_is_rejected(valid_request):
    interpreter = FallbackFailsValidationInterpreter()

    with pytest.raises(LLMInterpretationError, match="deterministic validation"):
        await optimize_energy(valid_request, interpreter)

    assert interpreter.interpret_calls == 1
    assert interpreter.recover_calls == 1


@pytest.mark.asyncio
async def test_exactly_one_fallback_attempt_enforced(valid_request, no_op_batch):
    interpreter = ProviderFailureThenFallbackSuccessInterpreter(no_op_batch)
    await optimize_energy(valid_request, interpreter)

    assert interpreter.interpret_calls == 1
    assert interpreter.recover_calls <= 1
