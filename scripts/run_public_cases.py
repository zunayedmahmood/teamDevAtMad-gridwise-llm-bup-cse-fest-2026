#!/usr/bin/env python3
import argparse
import json
import math
import statistics
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

NUMERIC_TOLERANCE = 0.01


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


def adjustments_equal(actual: dict | None, expected: dict | None) -> bool:
    if actual is None or expected is None:
        return actual is expected
    if set(actual) != set(expected):
        return False
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, (int, float)) and not isinstance(expected_value, bool):
            if not math.isclose(float(actual_value), float(expected_value), abs_tol=NUMERIC_TOLERANCE):
                return False
        elif actual_value != expected_value:
            return False
    return True


def interpretations_equal(actual: list[dict], expected: list[dict]) -> bool:
    if len(actual) != len(expected):
        return False
    for actual_item, expected_item in zip(actual, expected):
        if actual_item["note_index"] != expected_item["note_index"]:
            return False
        if actual_item["applies"] != expected_item["applies"]:
            return False
        if actual_item["directive_type"] != expected_item["directive_type"]:
            return False
        if not adjustments_equal(
            actual_item["structured_adjustment"],
            expected_item["structured_adjustment"],
        ):
            return False
    return True


def percentile95(values: list[float]) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


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
            http_ok = False
            status_code = None
            try:
                response = client.post("/optimize-energy", json=case["input"])
                latency = time.perf_counter() - started
                status_code = response.status_code
                http_ok = response.status_code == 200
                if not http_ok:
                    raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")

                parsed = EnergyResponse.model_validate_json(response.text)
                interpretation_ok = interpretations_equal(
                    normalized_interpretations(parsed.directive_interpretation),
                    expected_interpretations(case),
                )

                request = EnergyRequest.model_validate(case["input"])
                validate_request_semantics(request)
                canonical = canonicalize_request(request)
                ground_truth = validate_llm_batch(expected_batch(case), canonical)
                compiled = compile_constraints(canonical, ground_truth)
                replay_and_validate(canonical, compiled, parsed.hourly_plan)
                totals = calculate_totals(canonical, parsed.hourly_plan)

                totals_ok = (
                    abs(parsed.total_grid_kwh - totals.total_grid_kwh) <= NUMERIC_TOLERANCE
                    and abs(parsed.total_cost_bdt - totals.total_cost_bdt) <= NUMERIC_TOLERANCE
                    and abs(parsed.peak_grid_kwh - totals.peak_grid_kwh) <= NUMERIC_TOLERANCE
                )
                cost_ok = (
                    abs(totals.total_cost_bdt - case["expected_output"]["total_cost_bdt"])
                    <= NUMERIC_TOLERANCE
                )
                plan_ok = totals_ok
                passed = http_ok and interpretation_ok and plan_ok and cost_ok
                detail = ""
            except Exception as exc:
                latency = time.perf_counter() - started
                interpretation_ok = plan_ok = cost_ok = passed = False
                detail = str(exc).replace("\n", " ")[:100]

            all_passed &= passed
            rows.append(
                (case["id"], status_code, http_ok, interpretation_ok, plan_ok, cost_ok, latency, detail)
            )

    print(
        f"{'CASE':<12} {'HTTP':<8} {'INTERP':<8} {'PLAN':<8} {'COST':<8} "
        f"{'LATENCY':>9}  DETAIL"
    )
    for case_id, status, http_ok, interp, plan, cost, latency, detail in rows:
        http_text = "PASS" if http_ok else (str(status) if status is not None else "FAIL")
        print(
            f"{case_id:<12} {http_text:<8} {'PASS' if interp else 'FAIL':<8} "
            f"{'PASS' if plan else 'FAIL':<8} {'PASS' if cost else 'FAIL':<8} "
            f"{latency:>8.3f}s  {detail}"
        )

    latencies = [row[6] for row in rows]
    passed_count = sum(1 for row in rows if row[2] and row[3] and row[4] and row[5])
    print()
    print(f"{passed_count}/{len(rows)} passed")
    if latencies:
        print(f"p50 {statistics.median(latencies):.3f}s")
        print(f"p95 {percentile95(latencies):.3f}s")

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
