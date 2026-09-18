#!/usr/bin/env python3
import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import settings
from app.interpreter.client import DirectiveInterpreter
from app.interpreter.validator import validate_llm_batch
from app.models.internal import canonicalize_request
from app.models.request import EnergyRequest


@dataclass
class Metrics:
    notes: int = 0
    type_ok: int = 0
    hours_ok: int = 0
    numeric_ok: int = 0
    full_ok: int = 0
    failed_calls: int = 0


def public_expected(case: dict) -> list[dict]:
    return case["expected_output"]["directive_interpretation"]


def paraphrase_request(case: dict, battery: dict) -> EnergyRequest:
    return EnergyRequest.model_validate(
        {
            "scenario_id": case["id"],
            "operator_notes": case["notes"],
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


def expected_numeric(adjustment: dict | None) -> tuple[str | None, float | None]:
    adjustment = adjustment or {}
    for name in ("factor", "minimum_energy_kwh", "max_grid_kwh"):
        if name in adjustment:
            return name, float(adjustment[name])
    return None, None


def score(expected: list[dict], actual, metrics: Metrics) -> None:
    by_index = {directive.note_index: directive for directive in actual}
    for item in expected:
        metrics.notes += 1
        directive = by_index.get(item["note_index"])
        if directive is None:
            continue

        type_ok = directive.directive_type.value == item["directive_type"]
        expected_adjustment = item["structured_adjustment"]
        expected_hours = None if expected_adjustment is None else expected_adjustment.get("hours")
        actual_hours = None if directive.hours is None else list(directive.hours)
        hours_ok = actual_hours == expected_hours

        numeric_field, numeric_value = expected_numeric(expected_adjustment)
        actual_numeric = None if numeric_field is None else getattr(directive, numeric_field)
        numeric_ok = (
            actual_numeric is None
            if numeric_value is None
            else actual_numeric is not None and abs(float(actual_numeric) - numeric_value) <= 0.01
        )
        applies_ok = directive.applies == item["applies"]

        metrics.type_ok += int(type_ok)
        metrics.hours_ok += int(hours_ok)
        metrics.numeric_ok += int(numeric_ok)
        metrics.full_ok += int(type_ok and hours_ok and numeric_ok and applies_ok)


async def evaluate_request(
    interpreter: DirectiveInterpreter,
    request: EnergyRequest,
    expected: list[dict],
    metrics: Metrics,
    latencies: list[float],
) -> None:
    canonical = canonicalize_request(request)
    started = time.perf_counter()
    try:
        batch = await interpreter.interpret(canonical)
        directives = validate_llm_batch(batch, canonical)
        latencies.append(time.perf_counter() - started)
        score(expected, directives, metrics)
    except Exception:
        latencies.append(time.perf_counter() - started)
        metrics.failed_calls += 1
        metrics.notes += len(expected)


async def run_model(model: str, public_cases: list[dict], paraphrases: dict) -> tuple[Metrics, list[float]]:
    old_model = settings.openai_model
    settings.openai_model = model
    interpreter = DirectiveInterpreter()
    metrics = Metrics()
    latencies: list[float] = []
    try:
        for case in public_cases:
            await evaluate_request(
                interpreter,
                EnergyRequest.model_validate(case["input"]),
                public_expected(case),
                metrics,
                latencies,
            )
        for case in paraphrases["cases"]:
            await evaluate_request(
                interpreter,
                paraphrase_request(case, paraphrases["battery"]),
                case["expected"],
                metrics,
                latencies,
            )
    finally:
        settings.openai_model = old_model
    return metrics, latencies


def pct(value: int, total: int) -> str:
    return f"{100.0 * value / total:5.1f}%" if total else "  n/a"


async def async_main(args) -> int:
    public_cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))["cases"]
    paraphrases = json.loads(Path(args.paraphrases).read_text(encoding="utf-8"))

    print(
        f"{'MODEL':<20} {'TYPE':>7} {'HOURS':>7} {'NUM':>7} {'FULL':>7} "
        f"{'FAIL':>6} {'AVG_S':>8} {'P95_S':>8}"
    )
    all_ok = True
    for model in args.models:
        metrics, latencies = await run_model(model, public_cases, paraphrases)
        values = np.asarray(latencies, dtype=float)
        avg = float(values.mean()) if len(values) else float("nan")
        p95 = float(np.percentile(values, 95)) if len(values) else float("nan")
        print(
            f"{model:<20} {pct(metrics.type_ok, metrics.notes):>7} "
            f"{pct(metrics.hours_ok, metrics.notes):>7} "
            f"{pct(metrics.numeric_ok, metrics.notes):>7} "
            f"{pct(metrics.full_ok, metrics.notes):>7} "
            f"{metrics.failed_calls:>6} {avg:>8.3f} {p95:>8.3f}"
        )
        all_ok &= metrics.full_ok == metrics.notes and metrics.failed_calls == 0
    return 0 if all_ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark OpenAI interpreter models on official and paraphrase cases."
    )
    parser.add_argument("models", nargs="+", help="OpenAI model IDs to test")
    parser.add_argument(
        "--cases",
        default=str(ROOT / "tests" / "fixtures" / "public_sample_cases.json"),
    )
    parser.add_argument(
        "--paraphrases",
        default=str(ROOT / "tests" / "fixtures" / "paraphrase_cases.json"),
    )
    args = parser.parse_args()
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
