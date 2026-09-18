#!/usr/bin/env python3
import argparse
import asyncio
import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Result:
    latency_seconds: float
    category: str


def unique_payload(case: dict, index: int) -> dict:
    payload = copy.deepcopy(case["input"])
    payload["scenario_id"] = f"BENCH-{index:05d}-{payload['scenario_id']}"
    payload["operator_notes"][0] = (
        f"{payload['operator_notes'][0]} Benchmark request reference {index:05d}; "
        "the reference is not an energy instruction."
    )
    return payload


def categorize_failure(response: httpx.Response) -> str:
    if response.status_code >= 500:
        try:
            detail = str(response.json().get("detail", "")).lower()
        except Exception:
            return "http_5xx"
        if "directive interpretation" in detail or "interpret operator" in detail:
            return "openai_failure"
        if "schedule validation" in detail or "optimization validation" in detail:
            return "replay_failure"
        return "http_5xx"
    return "other_http"


async def run_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    case: dict,
    index: int,
) -> Result:
    from app.models.response import EnergyResponse

    payload = unique_payload(case, index)
    async with semaphore:
        started = time.perf_counter()
        try:
            response = await client.post("/optimize-energy", json=payload)
        except httpx.TimeoutException:
            return Result(time.perf_counter() - started, "timeout")
        except httpx.HTTPError:
            return Result(time.perf_counter() - started, "transport_failure")

    latency = time.perf_counter() - started
    if response.status_code != 200:
        return Result(latency, categorize_failure(response))

    try:
        data = response.json()
    except json.JSONDecodeError:
        return Result(latency, "invalid_json")
    try:
        EnergyResponse.model_validate_json(response.text)
    except ValidationError:
        return Result(latency, "schema_failure")

    return Result(latency, "success")


async def async_main(args) -> int:
    import sys

    sys.path.insert(0, str(ROOT))
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))["cases"]
    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = httpx.Timeout(args.timeout)

    async with httpx.AsyncClient(base_url=args.base_url.rstrip("/"), timeout=timeout) as client:
        tasks = [
            run_one(client, semaphore, cases[index % len(cases)], index)
            for index in range(args.requests)
        ]
        results = await asyncio.gather(*tasks)

    latencies = np.asarray([item.latency_seconds for item in results], dtype=float)
    counts: dict[str, int] = {}
    for item in results:
        counts[item.category] = counts.get(item.category, 0) + 1

    successes = counts.get("success", 0)
    failures = args.requests - successes
    print(f"requests: {args.requests}")
    print(f"concurrency: {args.concurrency}")
    print(f"success_count: {successes}")
    print(f"failure_count: {failures}")
    for category in (
        "http_5xx",
        "timeout",
        "invalid_json",
        "schema_failure",
        "replay_failure",
        "openai_failure",
        "transport_failure",
        "other_http",
    ):
        print(f"{category}: {counts.get(category, 0)}")

    if len(latencies):
        for percentile in (50, 90, 95, 99):
            print(f"p{percentile}: {np.percentile(latencies, percentile):.3f}s")
        print(f"max: {latencies.max():.3f}s")

    return 0 if failures == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure concurrent end-to-end /optimize-energy latency and reliability."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--cases",
        default=str(ROOT / "tests" / "fixtures" / "public_sample_cases.json"),
    )
    args = parser.parse_args()
    if args.requests < 1:
        parser.error("--requests must be >= 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1")
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
