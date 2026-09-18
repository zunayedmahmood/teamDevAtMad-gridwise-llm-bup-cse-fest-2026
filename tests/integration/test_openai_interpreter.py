import json
import os
from pathlib import Path

import pytest

from app.interpreter.client import DirectiveInterpreter
from app.interpreter.validator import validate_llm_batch
from app.models.internal import canonicalize_request
from app.models.request import EnergyRequest


pytestmark = pytest.mark.live_openai
RUN_OPENAI = os.getenv("RUN_OPENAI_TESTS") == "1" and bool(os.getenv("OPENAI_API_KEY"))
PARAPHRASE_PATH = Path(__file__).parents[1] / "fixtures" / "paraphrase_cases.json"


def request_for_notes(notes: list[str], battery: dict, scenario_id: str) -> EnergyRequest:
    return EnergyRequest.model_validate(
        {
            "scenario_id": scenario_id,
            "operator_notes": notes,
            "hours": [
                {
                    "hour": h,
                    "demand_kwh": 100.0,
                    "solar_kwh": 20.0,
                    "tariff_bdt_per_kwh": 10.0,
                }
                for h in range(24)
            ],
            "battery": battery,
        }
    )


def actual_semantics(directives) -> list[dict]:
    rows = []
    for directive in directives:
        adjustment = None
        if directive.hours is not None:
            adjustment = {"hours": list(directive.hours)}
            if directive.factor is not None:
                adjustment["factor"] = directive.factor
            if directive.minimum_energy_kwh is not None:
                adjustment["minimum_energy_kwh"] = directive.minimum_energy_kwh
            if directive.max_grid_kwh is not None:
                adjustment["max_grid_kwh"] = directive.max_grid_kwh
        rows.append(
            {
                "note_index": directive.note_index,
                "applies": directive.applies,
                "directive_type": directive.directive_type.value,
                "structured_adjustment": adjustment,
            }
        )
    return rows


def expected_semantics(items: list[dict]) -> list[dict]:
    return [
        {
            "note_index": item["note_index"],
            "applies": item["applies"],
            "directive_type": item["directive_type"],
            "structured_adjustment": item["structured_adjustment"],
        }
        for item in items
    ]


PARAPHRASE_CASES = json.loads(PARAPHRASE_PATH.read_text(encoding="utf-8"))["cases"]


@pytest.mark.skipif(not RUN_OPENAI, reason="set RUN_OPENAI_TESTS=1 and OPENAI_API_KEY to enable")
@pytest.mark.parametrize("case_index", range(len(PARAPHRASE_CASES)))
async def test_live_interpreter_paraphrase_corpus(case_index):
    payload = json.loads(PARAPHRASE_PATH.read_text(encoding="utf-8"))
    case = payload["cases"][case_index]
    request = request_for_notes(case["notes"], payload["battery"], case["id"])
    canonical = canonicalize_request(request)

    batch = await DirectiveInterpreter().interpret(canonical)
    directives = validate_llm_batch(batch, canonical)

    assert actual_semantics(directives) == expected_semantics(case["expected"])


@pytest.mark.skipif(not RUN_OPENAI, reason="set RUN_OPENAI_TESTS=1 and OPENAI_API_KEY to enable")
@pytest.mark.parametrize("case_index", range(10))
async def test_live_interpreter_official_public_cases(public_cases, case_index):
    case = public_cases[case_index]
    request = EnergyRequest.model_validate(case["input"])
    canonical = canonicalize_request(request)

    batch = await DirectiveInterpreter().interpret(canonical)
    directives = validate_llm_batch(batch, canonical)

    assert actual_semantics(directives) == expected_semantics(
        case["expected_output"]["directive_interpretation"]
    )
