#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.interpreter.validator import validate_llm_batch
from app.models.internal import canonicalize_request
from app.models.llm import LLMDirective, LLMDirectiveBatch
from app.models.request import EnergyRequest, validate_request_semantics
from app.models.response import DirectiveType, EnergyResponse
from app.optimizer.compiler import compile_constraints
from app.optimizer.replay import calculate_totals, replay_and_validate


def expected_batch(case: dict) -> LLMDirectiveBatch:
    directives = []
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


def normalized_interpretations(items) -> list[dict]:
    return [
        {
            "note_index": item.note_index,
            "applies": item.applies,
            "directive_type": item.directive_type.value,
            "structured_adjustment": (
                None if item.structured_adjustment is None else item.structured_adjustment.model_dump()
            ),
        }
        for item in items
    ]


def expected_interpretations(case: dict) -> list[dict]:
    return [
        {
            "note_index": item["note_index"],
            "applies": item["applies"],
            "directive_type": item["directive_type"],
            "structured_adjustment": item["structured_adjustment"],
        }
        for item in case["expected_output"]["directive_interpretation"]
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run official public sample cases against a deployed GridWise API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--cases",
        default=str(ROOT / "tests" / "fixtures" / "public_sample_cases.json"),
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))["cases"]
    rows = []
    all_passed = True

    with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=args.timeout) as client:
        for case in cases:
            started = time.perf_counter()
            try:
                response = client.post("/optimize-energy", json=case["input"])
                latency = time.perf_counter() - started
                if response.status_code != 200:
                    raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")

                parsed = EnergyResponse.model_validate_json(response.text)
                interpretation_ok = normalized_interpretations(parsed.directive_interpretation) == expected_interpretations(case)

                request = EnergyRequest.model_validate(case["input"])
                validate_request_semantics(request)
                canonical = canonicalize_request(request)
                ground_truth = validate_llm_batch(expected_batch(case), canonical)
                compiled = compile_constraints(canonical, ground_truth)
                replay_and_validate(canonical, compiled, parsed.hourly_plan)
                totals = calculate_totals(canonical, parsed.hourly_plan)

                totals_ok = (
                    abs(parsed.total_grid_kwh - totals.total_grid_kwh) <= 0.01
                    and abs(parsed.total_cost_bdt - totals.total_cost_bdt) <= 0.01
                    and abs(parsed.peak_grid_kwh - totals.peak_grid_kwh) <= 0.01
                )
                cost_ok = abs(totals.total_cost_bdt - case["expected_output"]["total_cost_bdt"]) <= 0.01
                plan_ok = totals_ok
                passed = interpretation_ok and plan_ok and cost_ok
                detail = ""
            except Exception as exc:
                latency = time.perf_counter() - started
                interpretation_ok = plan_ok = cost_ok = passed = False
                detail = str(exc).replace("\n", " ")[:100]

            all_passed &= passed
            rows.append((case["id"], interpretation_ok, plan_ok, cost_ok, latency, detail))

    print(f"{'CASE':<12} {'INTERP':<8} {'PLAN':<8} {'COST':<8} {'LATENCY':>9}  DETAIL")
    for case_id, interp, plan, cost, latency, detail in rows:
        print(
            f"{case_id:<12} {'PASS' if interp else 'FAIL':<8} {'PASS' if plan else 'FAIL':<8} "
            f"{'PASS' if cost else 'FAIL':<8} {latency:>8.3f}s  {detail}"
        )
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
