#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure end-to-end /optimize-energy latency.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--cases",
        default=str(ROOT / "tests" / "fixtures" / "public_sample_cases.json"),
    )
    args = parser.parse_args()

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))["cases"]
    latencies: list[float] = []
    successes = 0

    with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=args.timeout) as client:
        for index in range(args.requests):
            payload = cases[index % len(cases)]["input"]
            started = time.perf_counter()
            try:
                response = client.post("/optimize-energy", json=payload)
                elapsed = time.perf_counter() - started
                latencies.append(elapsed)
                if response.status_code == 200:
                    successes += 1
            except httpx.HTTPError:
                latencies.append(time.perf_counter() - started)

    values = np.asarray(latencies, dtype=float)
    print(f"requests: {args.requests}")
    print(f"success_rate: {successes / args.requests:.2%}")
    for percentile in (50, 90, 95, 99):
        print(f"p{percentile}: {np.percentile(values, percentile):.3f}s")
    print(f"max: {values.max():.3f}s")
    return 0 if successes == args.requests else 1


if __name__ == "__main__":
    raise SystemExit(main())
