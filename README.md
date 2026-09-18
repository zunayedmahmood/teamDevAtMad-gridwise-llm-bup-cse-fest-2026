# GridWise LLM

## Overview

GridWise LLM is the submission-oriented backend for the BUP CSE Fest 2026 preliminary **Smart Campus Energy Optimization Challenge - LLM-Assisted Operator Directive Interpretation**.

The API accepts one 24-hour demand/solar/tariff scenario, battery parameters, and 1-3 natural-language operator notes. It returns:

1. one structured interpretation per operator note, and
2. a minimum-cost 24-hour energy schedule that satisfies the physical battery model and all interpreted operator constraints.

The implementation intentionally keeps uncertainty limited to language understanding. Everything after note interpretation is deterministic.

## Architecture

```text
POST /optimize-energy
  -> strict Pydantic request parsing
  -> semantic request validation
  -> OpenAI Responses API structured interpretation (one batched call)
  -> deterministic directive-domain validation
     -> at most one bounded semantic recovery call
  -> 24-hour constraint compiler
  -> SciPy/HiGHS linear program
  -> exact public hourly plan + near-zero cleanup
  -> independent deterministic replay of those exact public values
  -> totals recomputed from the replayed plan
  -> solver-objective consistency check
  -> deterministic summary
  -> strict EnergyResponse validation
```

Public endpoints:

```text
GET  /health
POST /optimize-energy
```

## Why the LLM is used

The LLM is used only for the part that actually requires natural-language understanding: converting each operator note into exactly one supported directive.

Supported directive types are:

```text
solar_reduction
minimum_battery_reserve
no_charge_window
no_discharge_window
max_grid_window
no_op
```

The model does **not** solve the optimization problem, compute totals, generate the hourly plan, or validate its own schedule.

## Why deterministic components are used

Normal code is used whenever the task can be defined exactly:

- strict request and response validation,
- directive-domain validation,
- 24-hour constraint compilation,
- minimum-cost optimization,
- battery and energy-balance enforcement,
- plan replay,
- total and peak calculations,
- deterministic output ordering and summary construction.

This makes the mathematical result reproducible and independently checkable even though the incoming operator notes are natural language.

## Requirements

- Python 3.12
- a valid OpenAI API key for `POST /optimize-energy`
- network access to the OpenAI API
- Docker only if using the container deployment path

`GET /health` does not require an API key and does not call OpenAI or the optimizer.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | empty | Required for optimization requests |
| `OPENAI_MODEL` | `gpt-5.6-sol` | Primary directive interpreter |
| `OPENAI_FALLBACK_MODEL` | `gpt-6-astra` | Exceptional semantic-recovery model |
| `OPENAI_TIMEOUT_SECONDS` | `5` | Per-model-call timeout |
| `OPENAI_MAX_RETRIES` | `1` | Bounded SDK/network retry count |
| `APP_HOST` | `0.0.0.0` | Service bind host |
| `APP_PORT` | `8000` | Service bind port |
| `LOG_LEVEL` | `INFO` | Python log level |
| `OPTIMIZE_DEADLINE_SECONDS` | `25` | Overall `/optimize-energy` deadline |

Copy the checked-in template and add the real key locally:

```bash
cp .env.example .env
```

Never commit `.env` or a real API key.

## Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
# edit .env and set OPENAI_API_KEY
```

All direct project dependencies are pinned in `requirements.txt` to prevent SDK and solver drift.

## Run locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

The production baseline deliberately starts with one worker. The service is async and the LP is tiny; increase workers only after measured concurrency testing justifies it.

## Health check

```bash
curl -i http://127.0.0.1:8000/health
```

Expected body:

```json
{"status":"ok"}
```

Startup performs no OpenAI request, solver call, model download, or remote health check.

## Optimize example

A complete tested request and response generated from the working schema are included in:

```text
examples/sample_request.json
examples/sample_response.json
```

Run the exact request against a local service:

```bash
curl -sS -X POST http://127.0.0.1:8000/optimize-energy \
  -H 'Content-Type: application/json' \
  --data-binary @examples/sample_request.json
```

The corresponding checked response begins with:

```json
{
  "scenario_id": "SAMPLE-01",
  "directive_interpretation": [
    {
      "note_index": 0,
      "applies": true,
      "directive_type": "solar_reduction",
      "structured_adjustment": {
        "hours": [12, 13],
        "factor": 0.25
      }
    }
  ],
  "hourly_plan": ["24 typed hourly entries"],
  "total_grid_kwh": 2692.5,
  "total_cost_bdt": 38365.0,
  "peak_grid_kwh": 187.5,
  "plan_summary": "..."
}
```

The full file contains the actual 24-row response. Equivalent optimal schedules may differ in hourly actions while still being valid when replayed and cost-equivalent.

## OpenAI model configuration

Default primary model:

```text
gpt-5.6-sol
reasoning.effort = none
```

Exceptional recovery model:

```text
gpt-6-astra
reasoning.effort = low
```

Both model IDs are configurable through environment variables. The baseline uses no interpretation cache; every normal optimization request reaches the OpenAI interpreter.

Before final submission, benchmark the three planned candidates with real credentials:

```bash
OPENAI_API_KEY="$OPENAI_API_KEY" \
python scripts/benchmark_models.py \
  gpt-5.6-sol gpt-5.6-terra gpt-6-astra
```

The script reports directive-type, hour-set, numeric-parameter, and full-note accuracy, provider failure rate, p50 latency, and p95 latency over the official public notes plus the local 36-note paraphrase/prompt-injection corpus.

## Optimizer

The deterministic core uses `scipy.optimize.linprog(method="highs")`.

For each hour `h`, the LP contains:

```text
g_h = grid import
s_h = solar used
x_h = signed battery net energy (positive charge, negative discharge)
E_h = battery energy after the hour
```

The objective is:

```text
minimize sum(tariff[h] * g_h)
```

Constraints enforce hourly energy balance, solar availability, battery charge/discharge rates, capacity, minimum reserve, operator constraints, non-negative grid import, and `E_23 = initial_energy`.

The problem statement provides no battery-efficiency parameter, so battery transitions are lossless.

## Deterministic guardrails

The public contract uses strict Pydantic v2 models with `extra="forbid"` and no string-to-number coercion.

HTTP behavior:

- malformed JSON, wrong types, missing fields, extra fields, or other structural schema failures -> `400`;
- structurally valid but semantically invalid input -> `422`;
- mathematically infeasible valid constraints -> `422`;
- exhausted provider/interpretation failures -> controlled `500`;
- internal solver/replay contradictions -> controlled `500`.

The LLM's structured output is still treated as untrusted. Deterministic validation checks note coverage, directive/field compatibility, hours, bounds, battery reserve values, and grid caps before optimization.

`directive_interpretation` is sorted by `note_index`; `hourly_plan` is sorted by `hour`.

The exact judge-facing hourly values are near-zero-normalized, then independently replayed using business rules rather than LP matrices. If replay fails, the plan is never returned. No hourly numeric value is mutated after replay. Totals are recomputed from that replayed plan and the total cost is cross-checked against the solver objective.

## Tests

Run the deterministic suite:

```bash
pytest -q
```

Live OpenAI tests are intentionally gated:

```bash
RUN_OPENAI_TESTS=1 OPENAI_API_KEY="$OPENAI_API_KEY" \
pytest -q -m live_openai
```

The deterministic suite covers strict request handling, semantic validation, directive validation, bounded recovery/deadline behavior, compiler rules, the 96-variable LP, replay mutation rejection, failure mappings, repeated/concurrent request isolation, and all ten organizer public cases with known interpretations.

## Public sample runner

Against a running local or deployed service:

```bash
python scripts/run_public_cases.py \
  --base-url http://127.0.0.1:8000
```

For every organizer case it checks HTTP status, semantic directive fields, strict response schema, independent local replay under organizer-ground-truth constraints, returned totals, reference optimal cost, and latency. Explanation prose is not compared exactly, and equivalent optimal schedules are accepted.

## Deployment smoke test

```bash
python scripts/smoke_test_deployment.py \
  --base-url https://YOUR_PUBLIC_HOST
```

The default smoke test performs:

```text
GET /health
POST one organizer sample
strict EnergyResponse validation
independent local replay
returned-total consistency checks
```

Use `--health-only` only for a readiness check that intentionally avoids an OpenAI call.

The command exits non-zero on failure.

## Latency and reliability benchmark

Run mostly unique note payloads so the result reflects actual uncached interpretation latency:

```bash
python scripts/benchmark_latency.py \
  --base-url https://YOUR_PUBLIC_HOST \
  --requests 100 \
  --concurrency 5
```

The script reports:

```text
success_count
failure_count
http_5xx
timeout
invalid_json
schema_failure
replay_failure
openai_failure
transport_failure
other_http
p50
p90
p95
p99
max
```

A moderate `--concurrency 5` to `10` is enough for the submission hardening check; this project is not intended to be a large-load benchmark.

## Docker

Build:

```bash
docker build -t gridwise-llm:local .
```

Run with the key supplied only at runtime:

```bash
docker run --rm \
  -p 8000:8000 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e APP_HOST=0.0.0.0 \
  -e APP_PORT=8000 \
  gridwise-llm:local
```

Check container health:

```bash
curl http://127.0.0.1:8000/health
```

Then run the public-case runner against the container.

The Dockerfile uses `python:3.12-slim`, installs only the pinned dependencies and application code, starts one Uvicorn worker, and contains a `/health` healthcheck. `.env`, Git metadata, virtual environments, caches, tests, and bytecode are excluded from the build context.

## Deployment

The target platform must provide:

- public HTTPS without authentication,
- runtime environment variables,
- Python/container deployment,
- enough memory for SciPy,
- continuous availability during judging.

Recommended startup command outside Docker:

```bash
uvicorn app.main:app \
  --host "${APP_HOST:-0.0.0.0}" \
  --port "${APP_PORT:-8000}" \
  --workers 1
```

Set all configuration as runtime environment variables; do not bake credentials into an image.

After deployment, record and verify separately:

```text
public HTTPS URL
container registry image:tag
immutable image digest (if the registry provides one)
chosen deployment region
```

Then run:

```bash
python scripts/smoke_test_deployment.py --base-url https://YOUR_PUBLIC_HOST
python scripts/run_public_cases.py --base-url https://YOUR_PUBLIC_HOST
python scripts/benchmark_latency.py --base-url https://YOUR_PUBLIC_HOST --requests 100 --concurrency 5
```

Do not treat a successful local image as proof that a registry image or public deployment is reachable; pull/test the published image from a clean environment and test the public URL from outside the developer network.

## Known assumptions

Where the official challenge material is silent, this implementation uses these documented rules:

- time intervals are start-inclusive and end-exclusive;
- cross-midnight windows are normalized to unique sorted hours, e.g. 10 PM-2 AM -> `[0,1,22,23]`;
- overlapping minimum-battery reserves use the highest active reserve;
- overlapping maximum-grid constraints use the lowest active cap;
- charge/discharge prohibitions combine by union;
- overlapping `solar_reduction` directives use the **minimum remaining-solar factor**;
- battery transitions are lossless because no efficiency parameter exists in the challenge input;
- no interpretation cache or regex fallback is used.

If organizers clarify different solar-overlap semantics, only the compiler policy and its tests should change.

## Numerical policy

Solver precision is preserved. Only values with absolute magnitude below `1e-9` are cleaned to `0.0`. Replay uses the internal validation tolerance before the response is built. Public numeric values are JSON numbers, never formatted numeric strings.

## Observability

Successful optimization logs include:

```text
timestamp
request_id
scenario_id
status
model
llm_attempts
directive_count
total_latency_ms
llm_latency_ms
compile_latency_ms
solver_latency_ms
replay_latency_ms
```

Every HTTP response receives an `X-Request-ID` header. Failure logs contain the request ID, exception category, and internal technical detail. Public error responses remain generic and do not expose stack traces, provider responses, secrets, solver internals, or OpenAI metadata.

## Security

- `.env` is ignored by Git and Docker.
- API keys are read only from runtime configuration.
- The Dockerfile contains no API key.
- Authorization headers and API-key values are never logged.
- Operator notes are sent as untrusted user data under a developer instruction that explicitly forbids following instructions embedded in the notes.
- The model receives no tools, web access, filesystem access, agent loop, or conversation memory.
- Production responses contain only the official response schema.

Secret-hygiene checks before submission:

```bash
git grep -n 'sk-'
git grep -n 'OPENAI_API_KEY='
```

Documentation placeholders such as `OPENAI_API_KEY="$OPENAI_API_KEY"` are expected; real secret-looking values are not.

## Repository hygiene

Before creating the submission archive or publishing the repository:

```bash
git status
git log --oneline --decorate -n 10
```

Confirm there is no `.env`, local virtual environment, `__pycache__`, `.pytest_cache`, test-output dump, log file, or accidental large binary in the committed source.

Repository visibility must follow the hackathon rules; do not make the repository public earlier than permitted.

## Repository map

```text
app/api/              HTTP routes and centralized exception mapping
app/models/           strict public, LLM, and internal models
app/interpreter/      OpenAI client, prompt, and directive validator
app/optimizer/        constraint compiler, LP, numeric policy, replay
app/services/         end-to-end orchestration and request deadline
app/observability/    request IDs and structured request logging
examples/             exact tested sample request and response
tests/                unit, integration, reliability, and public-case tests
scripts/              public runner, model/latency benchmarks, deployment smoke test
```

## Final submission checks

Before submission, run:

```bash
pytest -q
python scripts/smoke_test_deployment.py --base-url https://YOUR_PUBLIC_HOST
python scripts/run_public_cases.py --base-url https://YOUR_PUBLIC_HOST
python scripts/benchmark_latency.py --base-url https://YOUR_PUBLIC_HOST --requests 100 --concurrency 5
```

With credentials, also run:

```bash
RUN_OPENAI_TESTS=1 OPENAI_API_KEY="$OPENAI_API_KEY" pytest -q -m live_openai
OPENAI_API_KEY="$OPENAI_API_KEY" python scripts/benchmark_models.py gpt-5.6-sol gpt-5.6-terra gpt-6-astra
```

Part 5 completes the build/hardening phase. The separate Part 6 recheck document should still be run as an independent final submission audit.
