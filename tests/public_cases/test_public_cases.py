import pytest

from app.interpreter.validator import validate_llm_batch
from app.models.internal import canonicalize_request
from app.models.request import EnergyRequest, validate_request_semantics
from app.optimizer.compiler import compile_constraints
from app.optimizer.lp import solve_lp
from app.optimizer.replay import build_hourly_plan, calculate_totals, replay_and_validate
from tests.conftest import expected_to_llm_batch


@pytest.mark.parametrize("case_index", range(10))
def test_all_public_cases_optimizer_matches_reference_cost(public_cases, case_index):
    case = public_cases[case_index]
    request = EnergyRequest.model_validate(case["input"])
    validate_request_semantics(request)
    canonical = canonicalize_request(request)
    directives = validate_llm_batch(expected_to_llm_batch(case), canonical)
    compiled = compile_constraints(canonical, directives)
    plan = build_hourly_plan(solve_lp(canonical, compiled))
    replay_and_validate(canonical, compiled, plan)
    totals = calculate_totals(canonical, plan)

    expected = case["expected_output"]
    assert totals.total_cost_bdt == pytest.approx(expected["total_cost_bdt"], abs=0.01)
    assert totals.total_grid_kwh == pytest.approx(expected["total_grid_kwh"], abs=0.01)
