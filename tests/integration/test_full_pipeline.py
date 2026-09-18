import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_interpreter
from app.main import app
from app.models.response import EnergyResponse
from tests.conftest import MappingInterpreter, expected_to_llm_batch


@pytest.mark.parametrize("case_index", range(10))
def test_public_case_full_pipeline_with_ground_truth_interpreter(public_cases, case_index):
    case = public_cases[case_index]
    batch = expected_to_llm_batch(case)
    app.dependency_overrides[get_interpreter] = lambda: MappingInterpreter({case["id"]: batch})
    try:
        response = TestClient(app).post("/optimize-energy", json=case["input"])
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200, response.text
    parsed = EnergyResponse.model_validate_json(response.text)
    assert parsed.scenario_id == case["id"]
    actual_interpretations = [
        {
            "note_index": item.note_index,
            "applies": item.applies,
            "directive_type": item.directive_type.value,
            "structured_adjustment": (None if item.structured_adjustment is None else item.structured_adjustment.model_dump()),
        }
        for item in parsed.directive_interpretation
    ]
    expected_interpretations = [
        {
            "note_index": item["note_index"],
            "applies": item["applies"],
            "directive_type": item["directive_type"],
            "structured_adjustment": item["structured_adjustment"],
        }
        for item in case["expected_output"]["directive_interpretation"]
    ]
    assert actual_interpretations == expected_interpretations
    assert parsed.total_cost_bdt == pytest.approx(case["expected_output"]["total_cost_bdt"], abs=0.01)
    assert parsed.total_grid_kwh == pytest.approx(case["expected_output"]["total_grid_kwh"], abs=0.01)
