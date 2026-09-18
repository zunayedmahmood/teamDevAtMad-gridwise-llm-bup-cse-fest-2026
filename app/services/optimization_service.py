import asyncio
import logging
import time

from app.config import settings
from app.errors import (
    DirectiveValidationError,
    InternalPlanValidationError,
    LLMInterpretationError,
    LLMProviderError,
)
from app.interpreter.client import DirectiveInterpreter
from app.interpreter.validator import to_public_interpretation, validate_llm_batch
from app.models.internal import CanonicalRequest, Totals, ValidatedDirective, canonicalize_request
from app.models.request import EnergyRequest, validate_request_semantics
from app.models.response import EnergyResponse
from app.optimizer.compiler import compile_constraints
from app.optimizer.numeric import VALIDATION_TOLERANCE, clean_number
from app.optimizer.lp import solve_lp
from app.optimizer.replay import build_hourly_plan, calculate_totals, replay_and_validate

logger = logging.getLogger("gridwise.optimization")


def build_summary(
    directives: list[ValidatedDirective],
    totals: Totals,
) -> str:
    active = [directive.directive_type.value for directive in directives if directive.applies]
    constraints = ", ".join(active) if active else "no additional operator constraints"
    return (
        f"Optimized a 24-hour schedule under {constraints}. "
        f"Total grid import is {totals.total_grid_kwh:.2f} kWh, "
        f"total cost is {totals.total_cost_bdt:.2f} BDT, "
        f"and peak grid import is {totals.peak_grid_kwh:.2f} kWh."
    )


async def _optimize_within_deadline(
    request: EnergyRequest,
    interpreter: DirectiveInterpreter,
    deadline_at: float,
    request_id: str,
) -> EnergyResponse:
    started = time.perf_counter()
    validate_request_semantics(request)
    canonical = canonicalize_request(request)

    worst_case_recovery_seconds = (
        settings.openai_timeout_seconds * (settings.openai_max_retries + 1) + 0.5
    )

    llm_attempts = 1
    recovered = False
    llm_started = time.perf_counter()
    try:
        raw_batch = await interpreter.interpret(canonical)
    except LLMInterpretationError as first_error:
        remaining = deadline_at - time.monotonic()
        if remaining <= worst_case_recovery_seconds:
            raise LLMInterpretationError(
                "Directive recovery budget exhausted"
            ) from first_error
        llm_attempts = 2
        recovered = True
        raw_batch = await interpreter.recover(canonical, None, first_error)

    try:
        directives = validate_llm_batch(raw_batch, canonical)
    except DirectiveValidationError as first_error:
        if recovered:
            raise LLMInterpretationError(
                "Directive interpretation failed deterministic validation"
            ) from first_error
        remaining = deadline_at - time.monotonic()
        if remaining <= worst_case_recovery_seconds:
            raise LLMInterpretationError(
                "Directive recovery budget exhausted"
            ) from first_error
        llm_attempts = 2
        repaired = await interpreter.recover(canonical, raw_batch, first_error)
        try:
            directives = validate_llm_batch(repaired, canonical)
        except DirectiveValidationError as second_error:
            raise LLMInterpretationError(
                "Directive interpretation failed deterministic validation"
            ) from second_error
    llm_latency_ms = (time.perf_counter() - llm_started) * 1000

    compile_started = time.perf_counter()
    compiled = compile_constraints(canonical, directives)
    compile_latency_ms = (time.perf_counter() - compile_started) * 1000

    solver_started = time.perf_counter()
    solution = solve_lp(canonical, compiled)
    solver_latency_ms = (time.perf_counter() - solver_started) * 1000

    hourly_plan = sorted(build_hourly_plan(solution), key=lambda row: row.hour)

    replay_started = time.perf_counter()
    replay_and_validate(canonical, compiled, hourly_plan)
    replay_latency_ms = (time.perf_counter() - replay_started) * 1000
    totals = calculate_totals(canonical, hourly_plan)
    if abs(totals.total_cost_bdt - solution.objective_value) > VALIDATION_TOLERANCE:
        raise InternalPlanValidationError("recomputed cost does not match solver objective")

    public_directives = sorted(
        [to_public_interpretation(directive) for directive in directives],
        key=lambda item: item.note_index,
    )

    response = EnergyResponse(
        scenario_id=request.scenario_id,
        directive_interpretation=public_directives,
        hourly_plan=hourly_plan,
        total_grid_kwh=clean_number(totals.total_grid_kwh),
        total_cost_bdt=clean_number(totals.total_cost_bdt),
        peak_grid_kwh=clean_number(totals.peak_grid_kwh),
        plan_summary=build_summary(directives, totals),
    )

    total_latency_ms = (time.perf_counter() - started) * 1000
    model = settings.openai_fallback_model if llm_attempts == 2 else settings.openai_model
    logger.info(
        "request_id=%s scenario_id=%s status=200 total_latency_ms=%.2f "
        "llm_latency_ms=%.2f compile_latency_ms=%.2f solver_latency_ms=%.2f replay_latency_ms=%.2f "
        "llm_attempts=%d model=%s directive_count=%d solver_status=success",
        request_id,
        canonical.scenario_id,
        total_latency_ms,
        llm_latency_ms,
        compile_latency_ms,
        solver_latency_ms,
        replay_latency_ms,
        llm_attempts,
        model,
        len(directives),
    )
    return response


async def optimize_energy(
    request: EnergyRequest,
    interpreter: DirectiveInterpreter,
    request_id: str = "-",
) -> EnergyResponse:
    deadline_seconds = settings.optimize_deadline_seconds
    deadline_at = time.monotonic() + deadline_seconds
    try:
        async with asyncio.timeout(deadline_seconds):
            return await _optimize_within_deadline(
                request,
                interpreter,
                deadline_at,
                request_id,
            )
    except TimeoutError as exc:
        raise LLMProviderError(
            "Optimization request exceeded its deadline"
        ) from exc
