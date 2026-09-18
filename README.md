# GridWise LLM

Submission-oriented implementation for the BUP CSE Fest 2026 preliminary **Smart Campus Energy Optimization Challenge - LLM-Assisted Operator Directive Interpretation**.

The service accepts a 24-hour demand/solar/tariff scenario plus 1-3 natural-language operator notes. A language model converts every note into one supported structured directive (or `no_op`); deterministic code validates those directives, compiles them into hard constraints, solves a 24-hour linear program, builds the exact public schedule, independently replays that schedule, recomputes totals, and returns the required JSON contract.

## Architecture

```text
POST /optimize-energy
  -> strict Pydantic request parsing
  -> semantic request validation
  -> OpenAI Responses API structured interpretation (1 batched call)
  -> deterministic directive validation
     -> at most 1 bounded recovery call if model semantics are invalid
  -> 24-hour constraint compiler
  -> SciPy/HiGHS linear program
  -> exact public hourly plan
  -> independent deterministic replay of that exact plan
  -> totals recomputed from the replayed plan
  -> deterministic summary
  -> strict response model
```

The LLM **does not optimize**, compute totals, or validate its own schedule. Natural-language interpretation is the only model-driven stage.

## Supported directives

`solar_reduction`, `minimum_battery_reserve`, `no_charge_window`, `no_discharge_window`, `max_grid_window`, and `no_op` are the only directive types accepted by the implementation.

## Technology

- Python 3.12
- FastAPI + Pydantic v2
- OpenAI Responses API + Pydantic Structured Outputs
- `gpt-5.6-sol` primary model
- `gpt-6-astra` optional recovery model
- NumPy + SciPy `linprog(method="highs")`
- Pytest
- Docker

## Configuration

Copy the example file and set your real API key locally:

```bash
cp .env.example .env
```

Required for `/optimize-energy`:

```dotenv
OPENAI_API_KEY=your_key_here
```

Defaults:

```dotenv
OPENAI_MODEL=gpt-5.6-sol
OPENAI_FALLBACK_MODEL=gpt-6-astra
OPENAI_TIMEOUT_SECONDS=5
OPENAI_MAX_RETRIES=1
APP_HOST=0.0.0.0
APP_PORT=8000
LOG_LEVEL=INFO
OPTIMIZE_DEADLINE_SECONDS=25
```

`/health` never requires an API key and never calls OpenAI or the optimizer.

## Local quickstart

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and set OPENAI_API_KEY
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl -i http://127.0.0.1:8000/health
```

Expected body:

```json
{"status":"ok"}
```

To run the first official public sample without manually copying 24 hourly rows:

```bash
python - <<'PY'
import json
from pathlib import Path
cases=json.loads(Path('tests/fixtures/public_sample_cases.json').read_text())['cases']
Path('/tmp/gridwise_sample.json').write_text(json.dumps(cases[0]['input']))
PY
curl -sS -X POST http://127.0.0.1:8000/optimize-energy \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/gridwise_sample.json
```

The complete official sample request **and expected response** for all 10 organizer cases are retained verbatim in `tests/fixtures/public_sample_cases.json`; equivalent optimal schedules are accepted by the challenge even when their hourly action sequence differs from the reference.

## Tests

Run the deterministic suite:

```bash
pytest -q
```

The suite covers strict request handling, semantic validation, all directive guardrails, recovery/deadline behavior, compiler behavior, LP behavior, replay mutation rejection, API error/status behavior, all 10 public sample cases, and the structure of a balanced 36-note paraphrase/prompt-injection corpus. Public-case optimizer tests use the organizer-provided expected interpretations so the mathematical pipeline can be verified without consuming API credits.

Live OpenAI tests are intentionally skipped unless both the explicit flag and API key are present:

```bash
RUN_OPENAI_TESTS=1 OPENAI_API_KEY=... pytest -q -m live_openai
```

Run the public cases against a running local/deployed endpoint (this exercises the real LLM path):

```bash
python scripts/run_public_cases.py --base-url http://127.0.0.1:8000
```

Benchmark interpreter models on the official public cases plus the local 36-note paraphrase corpus:

```bash
OPENAI_API_KEY=... python scripts/benchmark_models.py gpt-5.6-sol gpt-5.6-terra gpt-6-astra
```

The benchmark reports directive-type, hour-set, numeric-parameter, and full-note accuracy plus failure rate, average latency, and p95 latency.

Benchmark end-to-end latency (this performs real model calls and therefore incurs API usage):

```bash
python scripts/benchmark_latency.py --base-url http://127.0.0.1:8000 --requests 50
```

Deployment smoke test:

```bash
python scripts/smoke_test_deployment.py --base-url https://YOUR_HOST
python scripts/smoke_test_deployment.py --base-url https://YOUR_HOST --with-sample
```

## HTTP behavior

- `GET /health` -> `200` and exactly `{"status":"ok"}`.
- malformed JSON, wrong types, missing required fields, extra fields, and other structural schema failures -> `400` with a generic body.
- semantically invalid but structurally valid input -> `422`.
- a mathematically infeasible well-formed scenario -> `422`.
- provider/interpretation failures after the bounded recovery path -> controlled `500` without provider internals or secrets.
- internal solver/replay contradictions -> controlled `500`; invalid plans are never returned.

## Optimization model

For each hour `h`, the LP has grid import `g_h`, used solar `s_h`, signed battery-net energy `x_h` (positive = charge, negative = discharge), and battery state `E_h`.

It minimizes:

```text
sum(tariff[h] * g_h)
```

subject to energy balance, solar bounds, battery capacity/minimum reserve, charge/discharge rate limits, operator directive constraints, non-negative grid import, and final battery state equal to initial battery state. The challenge provides no battery-efficiency parameter, so transitions are lossless.

## Deterministic assumptions where the official statement is silent

- Time windows are `[start, end)` as specified. Cross-midnight windows are supported and returned as unique sorted hours, e.g. 10 PM-2 AM -> `[0,1,22,23]`.
- Overlapping minimum reserves use the maximum active reserve.
- Overlapping grid caps use the minimum active cap.
- Charge/discharge prohibitions combine by union.
- Overlapping solar reductions use the **minimum remaining-solar factor**. This is the main competition-specific assumption that should be changed if organizers clarify different overlap semantics.
- No interpretation cache is used in the baseline.
- No regex/phrase-matching fallback replaces the required LLM interpretation path.

## Numerical policy

Solver values are not rounded before validation. Only values with absolute magnitude below `1e-9` are normalized to zero. The exact public hourly values are then replayed with a strict internal tolerance before return. Totals are calculated **after** successful replay from the same public plan.

## Docker

Build locally:

```bash
docker build -t gridwise-llm:latest .
```

Run:

```bash
docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  gridwise-llm:latest
```

Then verify:

```bash
curl http://127.0.0.1:8000/health
```

For final competition submission, publish the tested image to the required registry and replace your submission placeholder with the exact pullable tag/digest. No credentials are baked into this Dockerfile.

## Security and failure handling

Operator notes are passed as untrusted data under a developer instruction that explicitly forbids following instructions embedded in notes. No tools, web access, filesystem access, or function-calling loop is exposed to the model. API keys are read only from environment configuration; `.env` is ignored by Git; request logs never include Authorization headers or API-key values. Public `500` responses are generic.

## Known operational dependency

`/optimize-energy` depends on a hosted OpenAI model. The service deliberately does not invent a deterministic NLP fallback because the challenge requires a language-capable generative model in the interpretation path. The deployment owner must provide valid credentials, quota, and network access. `/health` remains independent of that dependency.

## Repository map

```text
app/api/              HTTP routes and exception mapping
app/models/           strict public, LLM, and internal models
app/interpreter/      OpenAI client, prompt, deterministic guardrail validator
app/optimizer/        compiler, LP, numerical policy, replay
app/services/         end-to-end orchestration and deadline/recovery logic
app/observability/    request IDs and request logging
tests/                unit, integration, and all public-sample tests
scripts/              public-case runner, model benchmark, latency benchmark, deployment smoke test
```

## External dependencies / credits

This project uses FastAPI, Pydantic, Uvicorn, the OpenAI Python SDK, NumPy, SciPy/HiGHS, HTTPX, Pytest, and pytest-asyncio. The optimization and challenge-specific validation/orchestration logic in this repository is project code.
