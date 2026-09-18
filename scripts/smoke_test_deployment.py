#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.interpreter.validator import validate_llm_batch
from app.models.internal import canonicalize_request
from app.models.request import EnergyRequest, validate_request_semantics
from app.models.response import EnergyResponse
from app.optimizer.compiler import compile_constraints
from app.optimizer.replay import calculate_totals, replay_and_validate
from scripts.run_public_cases import NUMERIC_TOLERANCE, expected_batch


def validate_sample(case: dict, response: httpx.Response) -> None:
    response.raise_for_status()
    parsed = EnergyResponse.model_validate_json(response.text)
    if parsed.scenario_id != case["id"]:
        raise RuntimeError("scenario_id was not echoed correctly")

    request = EnergyRequest.model_validate(case["input"])
    validate_request_semantics(request)
    canonical = canonicalize_request(request)
    directives = validate_llm_batch(expected_batch(case), canonical)
    compiled = compile_constraints(canonical, directives)
    replay_and_validate(canonical, compiled, parsed.hourly_plan)
    totals = calculate_totals(canonical, parsed.hourly_plan)

    if abs(parsed.total_grid_kwh - totals.total_grid_kwh) > NUMERIC_TOLERANCE:
        raise RuntimeError("reported total_grid_kwh does not match returned plan")
    if abs(parsed.total_cost_bdt - totals.total_cost_bdt) > NUMERIC_TOLERANCE:
        raise RuntimeError("reported total_cost_bdt does not match returned plan")
    if abs(parsed.peak_grid_kwh - totals.peak_grid_kwh) > NUMERIC_TOLERANCE:
        raise RuntimeError("reported peak_grid_kwh does not match returned plan")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test a deployed GridWise API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--health-only", action="store_true")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--cases",
        default=str(ROOT / "tests" / "fixtures" / "public_sample_cases.json"),
    )
    args = parser.parse_args()

    with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=args.timeout) as client:
        health = client.get("/health")
        health.raise_for_status()
        if health.json() != {"status": "ok"}:
            raise RuntimeError(f"Unexpected health response: {health.text}")
        print("health: PASS")

        if not args.health_only:
            cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))["cases"]
            case = cases[0]
            response = client.post("/optimize-energy", json=case["input"])
            validate_sample(case, response)
            print("sample schema + replay: PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
