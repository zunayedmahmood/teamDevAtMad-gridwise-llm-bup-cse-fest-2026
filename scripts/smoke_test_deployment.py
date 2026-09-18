#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test a deployed GridWise API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--with-sample", action="store_true")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=args.timeout) as client:
        health = client.get("/health")
        health.raise_for_status()
        if health.json() != {"status": "ok"}:
            raise RuntimeError(f"Unexpected health response: {health.text}")
        print("health: PASS")

        if args.with_sample:
            cases = json.loads(
                (ROOT / "tests" / "fixtures" / "public_sample_cases.json").read_text(encoding="utf-8")
            )["cases"]
            response = client.post("/optimize-energy", json=cases[0]["input"])
            response.raise_for_status()
            body = response.json()
            if body.get("scenario_id") != cases[0]["id"]:
                raise RuntimeError("scenario_id was not echoed correctly")
            print("sample optimize: PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
