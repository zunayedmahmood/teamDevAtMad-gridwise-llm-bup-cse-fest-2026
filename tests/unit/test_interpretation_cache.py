import asyncio
from copy import deepcopy
import pytest

from app.config import settings
from app.errors import DirectiveValidationError, LLMInterpretationError, LLMProviderError
from app.interpreter.cache import (
    InterpretationCache,
    SingleFlight,
    compute_cache_key,
    interpretation_cache,
    single_flight,
)
from app.models.internal import CanonicalRequest, ValidatedDirective, canonicalize_request
from app.models.llm import LLMDirective, LLMDirectiveBatch
from app.models.request import EnergyRequest
from app.models.response import DirectiveType
from app.services.optimization_service import optimize_energy


class CountingInterpreter:
    def __init__(self, batch: LLMDirectiveBatch, delay: float = 0.0):
        self.batch = batch
        self.delay = delay
        self.calls = 0

    async def interpret(self, request):
        self.calls += 1
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        return self.batch

    async def recover(self, request, previous, validation_error):
        raise AssertionError("recovery should not be called in this test")


@pytest.mark.asyncio
async def test_a_cache_miss_calls_interpreter_once(valid_request, no_op_batch):
    interpreter = CountingInterpreter(no_op_batch)
    response = await optimize_energy(valid_request, interpreter)

    assert response.scenario_id == valid_request.scenario_id
    assert interpreter.calls == 1
    assert len(interpretation_cache) == 1


@pytest.mark.asyncio
async def test_b_exact_repeat_cache_hit_skips_llm_and_runs_optimizer(valid_request, no_op_batch):
    interpreter = CountingInterpreter(no_op_batch)
    # First call - cache miss
    resp1 = await optimize_energy(valid_request, interpreter)
    assert interpreter.calls == 1

    # Second call with same request - cache hit!
    resp2 = await optimize_energy(valid_request, interpreter)
    assert interpreter.calls == 1  # Interpreter was NOT called again!

    assert resp1.directive_interpretation == resp2.directive_interpretation
    assert resp1.total_cost_bdt == resp2.total_cost_bdt
    assert len(resp2.hourly_plan) == 24


def test_c_different_notes_do_not_collide(valid_request_dict):
    req1 = canonicalize_request(EnergyRequest.model_validate(valid_request_dict))
    req2_dict = deepcopy(valid_request_dict)
    req2_dict["operator_notes"] = ["Do not charge from 2 PM to 4 PM."]
    req2 = canonicalize_request(EnergyRequest.model_validate(req2_dict))

    key1 = compute_cache_key(req1)
    key2 = compute_cache_key(req2)
    assert key1 != key2


def test_d_interpretation_context_affects_key(valid_request_dict):
    req1 = canonicalize_request(EnergyRequest.model_validate(valid_request_dict))
    req2_dict = deepcopy(valid_request_dict)
    req2_dict["battery"]["capacity_kwh"] = 200.0  # Different capacity context
    req2 = canonicalize_request(EnergyRequest.model_validate(req2_dict))

    key1 = compute_cache_key(req1)
    key2 = compute_cache_key(req2)
    assert key1 != key2


def test_e_prompt_version_invalidation(valid_request_dict):
    req = canonicalize_request(EnergyRequest.model_validate(valid_request_dict))
    key1 = compute_cache_key(req, prompt_version="gridwise-directive-v1")
    key2 = compute_cache_key(req, prompt_version="gridwise-directive-v2")
    assert key1 != key2


def test_f_model_identifier_invalidation(valid_request_dict):
    req = canonicalize_request(EnergyRequest.model_validate(valid_request_dict))
    key1 = compute_cache_key(req, model="gpt-5.6-sol")
    key2 = compute_cache_key(req, model="gpt-6-astra")
    assert key1 != key2


@pytest.mark.asyncio
async def test_g_failed_interpretation_is_not_cached(valid_request, no_op_batch):
    class FailOnceInterpreter:
        def __init__(self):
            self.calls = 0

        async def interpret(self, request):
            self.calls += 1
            if self.calls == 1:
                raise LLMProviderError("Transient 503")
            return no_op_batch

        async def recover(self, request, previous, validation_error):
            # Simulate fallback also failing on first call
            raise LLMProviderError("Fallback also failed")

    interpreter = FailOnceInterpreter()
    with pytest.raises(LLMProviderError):
        await optimize_energy(valid_request, interpreter)

    assert len(interpretation_cache) == 0

    # Second call with another interpreter must attempt LLM again
    second_interpreter = CountingInterpreter(no_op_batch)
    response = await optimize_energy(valid_request, second_interpreter)
    assert response.scenario_id == valid_request.scenario_id
    assert second_interpreter.calls == 1
    assert len(interpretation_cache) == 1


@pytest.mark.asyncio
async def test_h_validation_failure_is_not_cached(valid_request):
    bad_batch = LLMDirectiveBatch(
        directives=[
            LLMDirective(
                note_index=0,
                applies=True,
                directive_type=DirectiveType.NO_CHARGE_WINDOW,
                hours=[25],  # Invalid hour
                factor=None,
                minimum_energy_kwh=None,
                max_grid_kwh=None,
                explanation="bad",
            )
        ]
    )

    class ValidationFailInterpreter:
        async def interpret(self, request):
            return bad_batch

        async def recover(self, request, previous, validation_error):
            return bad_batch

    with pytest.raises(LLMInterpretationError):
        await optimize_energy(valid_request, ValidationFailInterpreter())

    assert len(interpretation_cache) == 0


@pytest.mark.asyncio
async def test_i_ttl_expiration(valid_request, no_op_batch, monkeypatch):
    cache = InterpretationCache(ttl_seconds=0.01)
    monkeypatch.setattr("app.services.optimization_service.interpretation_cache", cache)

    interpreter = CountingInterpreter(no_op_batch)
    await optimize_energy(valid_request, interpreter)
    assert interpreter.calls == 1

    # Wait for TTL to expire
    await asyncio.sleep(0.02)

    # Next call must miss cache
    await optimize_energy(valid_request, interpreter)
    assert interpreter.calls == 2


@pytest.mark.asyncio
async def test_j_bounded_eviction():
    cache = InterpretationCache(max_entries=3, ttl_seconds=100.0)
    directive = ValidatedDirective(
        note_index=0,
        applies=False,
        directive_type=DirectiveType.NO_OP,
        hours=None,
        factor=None,
        minimum_energy_kwh=None,
        max_grid_kwh=None,
        explanation="ok",
    )

    for i in range(5):
        await cache.set(f"key-{i}", [directive])

    assert len(cache) <= 3
    # First two keys should have been evicted
    assert await cache.get("key-0") is None
    assert await cache.get("key-1") is None
    assert await cache.get("key-4") is not None


@pytest.mark.asyncio
async def test_k_concurrent_identical_requests_single_flight(valid_request, no_op_batch):
    # Delayed interpreter so requests definitely overlap in flight
    interpreter = CountingInterpreter(no_op_batch, delay=0.05)

    # Run 5 identical requests concurrently
    tasks = [optimize_energy(valid_request, interpreter) for _ in range(5)]
    responses = await asyncio.gather(*tasks)

    # Only 1 LLM call must have occurred!
    assert interpreter.calls == 1
    assert len(responses) == 5
    for r in responses:
        assert r.scenario_id == valid_request.scenario_id


@pytest.mark.asyncio
async def test_l_unrelated_requests_are_not_serialized(valid_request_dict, no_op_batch):
    req1 = EnergyRequest.model_validate(valid_request_dict)
    req2_dict = deepcopy(valid_request_dict)
    req2_dict["scenario_id"] = "TEST-002"
    req2_dict["operator_notes"] = ["Completely different note."]
    req2 = EnergyRequest.model_validate(req2_dict)

    interpreter = CountingInterpreter(no_op_batch, delay=0.02)

    tasks = [
        optimize_energy(req1, interpreter),
        optimize_energy(req2, interpreter),
    ]
    responses = await asyncio.gather(*tasks)

    # Both requests must execute because keys are different
    assert interpreter.calls == 2
    assert responses[0].scenario_id == "TEST-001"
    assert responses[1].scenario_id == "TEST-002"


@pytest.mark.asyncio
async def test_m_leader_failure_cleans_inflight_registry(valid_request, no_op_batch):
    class FailThenSucceedInterpreter:
        def __init__(self):
            self.calls = 0

        async def interpret(self, request):
            self.calls += 1
            if self.calls == 1:
                raise LLMProviderError("First call fails")
            return no_op_batch

        async def recover(self, request, previous, validation_error):
            raise LLMProviderError("Recovery fails")

    interpreter = FailThenSucceedInterpreter()

    # First call fails
    with pytest.raises(LLMProviderError):
        await optimize_energy(valid_request, interpreter)

    # Registry must be clean
    assert len(single_flight._inflight) == 0

    # Second call should not hang or fail due to stale in-flight state
    resp = await optimize_energy(valid_request, interpreter)
    assert resp.scenario_id == valid_request.scenario_id
    assert interpreter.calls == 2


@pytest.mark.asyncio
async def test_n_one_waiter_cancellation_does_not_cancel_shared_task(valid_request, no_op_batch):
    interpreter = CountingInterpreter(no_op_batch, delay=0.08)

    task1 = asyncio.create_task(optimize_energy(valid_request, interpreter))
    # Give task1 a tiny slice to register in flight
    await asyncio.sleep(0.01)
    task2 = asyncio.create_task(optimize_energy(valid_request, interpreter))

    # Cancel task1 (the leader)
    await asyncio.sleep(0.01)
    task1.cancel()

    # task1 raises CancelledError
    with pytest.raises(asyncio.CancelledError):
        await task1

    # task2 must still complete successfully because the shared task is shielded
    resp2 = await task2
    assert resp2.scenario_id == valid_request.scenario_id
    assert interpreter.calls == 1


@pytest.mark.asyncio
async def test_o_timed_out_waiter_does_not_poison_cache(valid_request, no_op_batch, monkeypatch):
    interpreter = CountingInterpreter(no_op_batch, delay=0.06)

    # Start a request with a tiny deadline that will time out
    monkeypatch.setattr(settings, "optimize_deadline_seconds", 0.02)
    with pytest.raises(LLMProviderError, match="exceeded its deadline"):
        await optimize_energy(valid_request, interpreter)

    # Allow the underlying interpretation task to complete in background
    await asyncio.sleep(0.08)

    # Reset deadline to normal
    monkeypatch.setattr(settings, "optimize_deadline_seconds", 25.0)

    # Next request succeeds cleanly
    resp = await optimize_energy(valid_request, interpreter)
    assert resp.scenario_id == valid_request.scenario_id


@pytest.mark.asyncio
async def test_p_returned_cache_objects_cannot_mutate_stored_state(valid_request, no_op_batch):
    interpreter = CountingInterpreter(no_op_batch)
    resp1 = await optimize_energy(valid_request, interpreter)

    # Directives retrieved from cache
    cache_key = compute_cache_key(canonicalize_request(valid_request))
    cached_first = await interpretation_cache.get(cache_key)
    assert cached_first is not None

    # Mutating an attribute (if it were mutable) or re-fetching should yield pristine object
    cached_second = await interpretation_cache.get(cache_key)
    assert cached_first == cached_second
    # Objects are distinct instances
    assert cached_first[0] is not cached_second[0]
