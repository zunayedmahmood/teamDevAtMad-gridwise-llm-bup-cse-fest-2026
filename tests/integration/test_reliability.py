import asyncio
import logging
from dataclasses import replace

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
        "compile_latency_ms=",
        "solver_latency_ms=",
        "replay_latency_ms=",
        "serialize_latency_ms=",
        "llm_attempts=1",
        "model=",
        "directive_count=1",
        "solver_status=success",
    ):
        assert field in message


def test_objective_mismatch_returns_internal_schedule_validation_500(
    valid_request_dict, no_op_batch, monkeypatch
):
    from app.services import optimization_service as service_module

    original_solve = service_module.solve_lp

    def mismatched_solve(request, compiled):
        solution = original_solve(request, compiled)
        return replace(solution, objective_value=solution.objective_value + 1.0)

    monkeypatch.setattr(service_module, "solve_lp", mismatched_solve)
    app.dependency_overrides[get_interpreter] = lambda: MappingInterpreter(
        {"TEST-001": no_op_batch}
    )
    try:
        response = TestClient(app).post("/optimize-energy", json=valid_request_dict)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal schedule validation failed."}


def test_hourly_values_are_not_mutated_after_replay(
    valid_request_dict, no_op_batch, monkeypatch
):
    from app.services import optimization_service as service_module

    captured = {}
    original_replay = service_module.replay_and_validate

    def capturing_replay(request, compiled, plan):
        captured["plan"] = [row.model_dump() for row in plan]
        return original_replay(request, compiled, plan)

    monkeypatch.setattr(service_module, "replay_and_validate", capturing_replay)
    app.dependency_overrides[get_interpreter] = lambda: MappingInterpreter(
        {"TEST-001": no_op_batch}
    )
    try:
        response = TestClient(app).post("/optimize-energy", json=valid_request_dict)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["hourly_plan"] == captured["plan"]


def test_plan_summary_is_deterministic(valid_request_dict, no_op_batch):
    app.dependency_overrides[get_interpreter] = lambda: MappingInterpreter(
        {"TEST-001": no_op_batch}
    )
    try:
        first = TestClient(app).post("/optimize-energy", json=valid_request_dict)
        second = TestClient(app).post("/optimize-energy", json=valid_request_dict)
    finally:
        app.dependency_overrides.clear()

    assert first.status_code == second.status_code == 200
    assert first.json()["plan_summary"] == second.json()["plan_summary"]
    assert "no additional operator constraints" in first.json()["plan_summary"]


class CountingNoOpInterpreter:
    def __init__(self, batch):
        self.batch = batch
        self.calls = 0

    async def interpret(self, request):
        self.calls += 1
        return self.batch

    async def recover(self, request, previous, validation_error):
        raise AssertionError("valid no-op interpretation should not require recovery")


def test_one_hundred_requests_are_stateless_and_all_reach_interpreter(
    valid_request_dict, no_op_batch
):
    interpreter = CountingNoOpInterpreter(no_op_batch)
    app.dependency_overrides[get_interpreter] = lambda: interpreter
    try:
        client = TestClient(app)
        for index in range(100):
            payload = dict(valid_request_dict)
            payload["scenario_id"] = f"RELIABILITY-{index:03d}"
            response = client.post("/optimize-energy", json=payload)
            assert response.status_code == 200
            assert response.json()["scenario_id"] == payload["scenario_id"]
    finally:
        app.dependency_overrides.clear()

    assert interpreter.calls == 100


@pytest.mark.asyncio
async def test_ten_concurrent_requests_do_not_cross_contaminate(
    valid_request_dict, no_op_batch
):
    import copy

    import httpx

    interpreter = CountingNoOpInterpreter(no_op_batch)
    app.dependency_overrides[get_interpreter] = lambda: interpreter
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            payloads = []
            for index in range(10):
                payload = copy.deepcopy(valid_request_dict)
                payload["scenario_id"] = f"CONCURRENT-{index:02d}"
                payloads.append(payload)
            responses = await asyncio.gather(
                *(client.post("/optimize-energy", json=payload) for payload in payloads)
            )
    finally:
        app.dependency_overrides.clear()

    assert interpreter.calls == 10
    assert all(response.status_code == 200 for response in responses)
    assert [response.json()["scenario_id"] for response in responses] == [
        payload["scenario_id"] for payload in payloads
    ]


def test_health_survives_missing_openai_key(valid_request_dict, monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", None)
    get_interpreter.cache_clear()
    try:
        client = TestClient(app)
        health = client.get("/health")
        optimize = client.post("/optimize-energy", json=valid_request_dict)
    finally:
        get_interpreter.cache_clear()

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert optimize.status_code == 500
    assert optimize.json() == {
        "detail": "Directive interpretation is temporarily unavailable."
    }


def test_provider_failure_log_has_category_and_internal_detail(
    valid_request_dict, caplog
):
    app.dependency_overrides[get_interpreter] = lambda: ProviderFailureInterpreter()
    try:
        with caplog.at_level(logging.WARNING, logger="app.api.error_handlers"):
            response = TestClient(app).post("/optimize-energy", json=valid_request_dict)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 500
    message = next(
        record.getMessage()
        for record in caplog.records
        if record.name == "app.api.error_handlers"
    )
    assert "exception_category=LLMProviderError" in message
    assert "detail=provider unavailable" in message
    assert "request_id=" in message
