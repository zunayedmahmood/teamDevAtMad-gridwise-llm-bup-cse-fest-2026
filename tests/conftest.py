import json
from pathlib import Path

import pytest

from app.interpreter.cache import interpretation_cache, single_flight
from app.models.llm import LLMDirective, LLMDirectiveBatch
from app.models.request import EnergyRequest
from app.models.response import DirectiveType


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "public_sample_cases.json"


@pytest.fixture(autouse=True)
def _reset_cache_and_single_flight():
    interpretation_cache.clear()
    single_flight.clear()
    yield
    interpretation_cache.clear()
    single_flight.clear()


@pytest.fixture(scope="session")
def public_cases() -> list[dict]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["cases"]


@pytest.fixture
def valid_request_dict() -> dict:
    return {
        "scenario_id": "TEST-001",
        "operator_notes": ["The cafeteria menu changes tomorrow."],
        "hours": [
            {
                "hour": hour,
                "demand_kwh": 100.0,
                "solar_kwh": 0.0,
                "tariff_bdt_per_kwh": 10.0,
            }
            for hour in range(24)
        ],
        "battery": {
            "capacity_kwh": 100.0,
            "initial_energy_kwh": 50.0,
            "minimum_energy_kwh": 20.0,
            "max_charge_kwh_per_hour": 25.0,
            "max_discharge_kwh_per_hour": 25.0,
        },
    }


def expected_to_llm_batch(case: dict) -> LLMDirectiveBatch:
    directives: list[LLMDirective] = []
    for expected in case["expected_output"]["directive_interpretation"]:
        adjustment = expected["structured_adjustment"] or {}
        directives.append(
            LLMDirective(
                note_index=expected["note_index"],
                applies=expected["applies"],
                directive_type=DirectiveType(expected["directive_type"]),
                hours=adjustment.get("hours"),
                factor=adjustment.get("factor"),
                minimum_energy_kwh=adjustment.get("minimum_energy_kwh"),
                max_grid_kwh=adjustment.get("max_grid_kwh"),
                explanation=expected["explanation"],
            )
        )
    return LLMDirectiveBatch(directives=directives)


class MappingInterpreter:
    def __init__(self, batches: dict[str, LLMDirectiveBatch]):
        self.batches = batches
        self.recovery_calls = 0

    async def interpret(self, request):
        return self.batches[request.scenario_id]

    async def recover(self, request, previous, validation_error):
        self.recovery_calls += 1
        return self.batches[request.scenario_id]


@pytest.fixture
def no_op_batch() -> LLMDirectiveBatch:
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
                explanation="This note does not affect the energy schedule.",
            )
        ]
    )


@pytest.fixture
def valid_request(valid_request_dict) -> EnergyRequest:
    return EnergyRequest.model_validate(valid_request_dict)
