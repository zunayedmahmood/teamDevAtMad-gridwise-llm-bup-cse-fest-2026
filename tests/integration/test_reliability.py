import asyncio
import logging

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.dependencies import get_interpreter
from app.errors import LLMProviderError
from app.main import app
from app.services.optimization_service import optimize_energy
from tests.conftest import MappingInterpreter


class ProviderFailureInterpreter:
    async def interpret(self, request):
        raise LLMProviderError("provider unavailable")

    async def recover(self, request, previous, validation_error):
        raise AssertionError("provider errors should not enter semantic recovery")


class SlowInterpreter:
    async def interpret(self, request):
        await asyncio.sleep(0.05)
        raise AssertionError("deadline should fire first")

    async def recover(self, request, previous, validation_error):
        raise AssertionError("recovery should not run")


def test_provider_failure_returns_generic_500(valid_request_dict):
    app.dependency_overrides[get_interpreter] = lambda: ProviderFailureInterpreter()
    try:
        response = TestClient(app).post("/optimize-energy", json=valid_request_dict)
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 500
    assert response.json() == {
        "detail": "Directive interpretation is temporarily unavailable."
    }


@pytest.mark.asyncio
async def test_overall_deadline_is_enforced(valid_request, monkeypatch):
    monkeypatch.setattr(settings, "optimize_deadline_seconds", 0.001)
    with pytest.raises(LLMProviderError, match="deadline"):
        await optimize_energy(valid_request, SlowInterpreter())


def test_repeated_valid_requests_are_stable_and_request_ids_are_unique(
    valid_request_dict, no_op_batch
):
    app.dependency_overrides[get_interpreter] = lambda: MappingInterpreter(
        {"TEST-001": no_op_batch}
    )
    try:
        client = TestClient(app)
        responses = [client.post("/optimize-energy", json=valid_request_dict) for _ in range(3)]
    finally:
        app.dependency_overrides.clear()

    assert all(response.status_code == 200 for response in responses)
    assert responses[0].json() == responses[1].json() == responses[2].json()
    request_ids = [response.headers.get("X-Request-ID") for response in responses]
    assert all(request_ids)
    assert len(set(request_ids)) == 3


def test_optimization_log_contains_stage_metrics(valid_request_dict, no_op_batch, caplog):
    app.dependency_overrides[get_interpreter] = lambda: MappingInterpreter(
        {"TEST-001": no_op_batch}
    )
    try:
        with caplog.at_level(logging.INFO, logger="gridwise.optimization"):
            response = TestClient(app).post("/optimize-energy", json=valid_request_dict)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    message = next(
        record.getMessage()
        for record in caplog.records
        if record.name == "gridwise.optimization"
    )
    for field in (
        "request_id=",
        "scenario_id=TEST-001",
        "total_latency_ms=",
        "llm_latency_ms=",
        "solver_latency_ms=",
        "replay_latency_ms=",
        "llm_attempts=1",
        "model=",
        "directive_count=1",
        "solver_status=success",
    ):
        assert field in message
