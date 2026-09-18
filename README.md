# GridWise LLM — Autonomous Microgrid Optimization & LLM Directive Interpreter

[![Python 3.12+](https://img.shields.io/badge/python-3.12%20%7C%203.14-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![SciPy HiGHS](https://img.shields.io/badge/Solver-SciPy%20HiGHS%20LP-4B8BBE.svg?logo=scipy&logoColor=white)](https://scipy.org/)
[![OpenAI Structured Outputs](https://img.shields.io/badge/OpenAI-Structured%20Outputs-412991.svg?logo=openai&logoColor=white)](https://platform.openai.com/docs/guides/structured-outputs)
[![Benchmark Core Accuracy](https://img.shields.io/badge/Benchmark%20Core%20Accuracy-100%25%20(90%2F90)-brightgreen.svg)](#empirical-benchmarking-results-100-case-adversarial-suite)
[![Benchmark Full Accuracy](https://img.shields.io/badge/Benchmark%20Full%20Accuracy-99%25%20(99%2F100)-brightgreen.svg)](#empirical-benchmarking-results-100-case-adversarial-suite)
[![Tests](https://img.shields.io/badge/Deterministic%20Tests-215%20Passed-success.svg)](#tests)
[![Deployment](https://img.shields.io/badge/Live%20Deployment-Render%20Cloud-46E3B7.svg?logo=render&logoColor=black)](https://teamdevatmad-gridwise-llm-bup-cse-fest.onrender.com/health)
[![License](https://img.shields.io/badge/license-MIT-purple.svg)](LICENSE)

> **BUP CSE Fest 2026 — Preliminary Round**  
> **Challenge Track:** Smart Campus Energy Optimization Challenge — LLM-Assisted Operator Directive Interpretation  
> **Live Production Service:** [`https://teamdevatmad-gridwise-llm-bup-cse-fest.onrender.com/`](https://teamdevatmad-gridwise-llm-bup-cse-fest.onrender.com/)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Why the LLM is used](#why-the-llm-is-used)
- [Why deterministic components are used](#why-deterministic-components-are-used)
- [Requirements](#requirements)
- [Environment variables](#environment-variables)
- [Install](#install)
- [Run locally](#run-locally)
- [Health check](#health-check)
- [Optimize example](#optimize-example)
- [OpenAI model configuration](#openai-model-configuration)
- [Optimizer](#optimizer)
- [Directive Types & Semantic Interpretation Rules](#directive-types--semantic-interpretation-rules)
- [Deterministic guardrails](#deterministic-guardrails)
- [Empirical Benchmarking Results (100-Case Adversarial Suite)](#empirical-benchmarking-results-100-case-adversarial-suite)
- [Tiebreaker Engineering & Contest Resilience](#tiebreaker-engineering--contest-resilience)
- [Public API Reference & Contract](#public-api-reference--contract)
- [Tests](#tests)
- [Public sample runner](#public-sample-runner)
- [Docker](#docker)
- [Deployment](#deployment)
- [Known assumptions](#known-assumptions)
- [Security](#security)
- [Repository Architecture & Sitemap](#repository-architecture--sitemap)

---

## Overview

**GridWise LLM** is the submission-oriented, production-grade backend for the **BUP CSE Fest 2026 Preliminary Round: Smart Campus Energy Optimization Challenge — LLM-Assisted Operator Directive Interpretation**.

Modern institutional campuses operate localized microgrids equipped with rooftop photovoltaic (PV) solar panels, stationary battery energy storage systems (BESS), fluctuating daily demand curves, and dynamic time-of-use (TOU) utility tariffs. While mathematical optimization can compute cost-minimal charge/discharge schedules, human grid operators frequently issue ad-hoc operational constraints via unstructured natural-language notes (e.g., *"Facilities washing rooftop panels noon-2pm; treat solar at 25%"*, *"Hold at least 100 kWh reserved 6 PM to 8 PM for convocation ceremony"*).

GridWise LLM accepts:
1. One 24-hour scenario containing hourly campus electricity demand (kWh), baseline solar generation forecast (kWh), and utility tariff rates (BDT/kWh);
2. Battery specifications (capacity, initial energy, minimum energy, and charge/discharge limits); and
3. 1 to 3 natural-language operator notes.

The service outputs:
1. One structured, parameter-bounded interpretation per operator note; and
2. A provably cost-optimal 24-hour battery charge/discharge schedule that strictly satisfies physical battery bounds and all operator constraints.

The implementation is architected around an uncompromising principle: **uncertainty is quarantined strictly to natural-language parsing; everything after note interpretation is 100% deterministic, mathematically optimal, and independently auditable.**

---

## Architecture

```mermaid
flowchart TD
    A[Client Request: POST /optimize-energy] --> B[FastAPI Ingress & Request ID Generator]
    B --> C[Pydantic v2 Strict Deserialization<br>extra='forbid', no coercion]
    C -->|Malformed JSON / Schema Fail| E1[HTTP 400 Bad Request]
    C --> D[Semantic Request Validator<br>len=24, unique hours, battery bounds]
    D -->|Semantic Field Error| E2[HTTP 422 Unprocessable Entity]
    D --> F[Canonicalize Request]
    
    F --> G{Interpretation Cache Hit?<br>SHA-256 Signature}
    G -->|Cache Hit| K[Validated Directives]
    G -->|Cache Miss| H[Single-Flight Coalescer]
    
    H --> I[OpenAI Responses API<br>Model: gpt-5.6-sol | reasoning: none]
    I -->|JSON Schema Invalid / Exception| J{Retry Budget Remaining?}
    J -->|Yes| I2[Fallback Model: gpt-6-astra<br>reasoning: low + recovery prompt]
    J -->|No| E3[HTTP 500 LLM Provider Error]
    
    I --> L[Deterministic Directive Validator]
    I2 --> L
    L -->|Valid| K
    L -->|Domain Validation Failed| J
    
    K --> M[24-Hour Constraint Compiler<br>Hours normalization, overlap policies]
    M --> N[SciPy HiGHS Linear Program<br>96 continuous variables, Dual Simplex]
    N -->|Infeasible Constraints| E4[HTTP 422 Optimization Infeasible]
    N -->|Optimal Solution Found| O[Clean Near-Zero Floats < 1e-7]
    
    O --> P[Build Public HourlyPlanEntry Rows]
    P --> Q[Independent Replay Engine<br>Verify battery limits, balance, E23=E0]
    Q -->|Replay Mismatch / Violation| E5[HTTP 500 Plan Validation Error]
    
    Q --> R[Calculate Exact Totals & Peak Grid]
    R --> S[Cross-Check Solver Cost == Plan Cost]
    S --> T[Deterministic Human-Readable Summary]
    T --> U[Serialize Strict EnergyResponse JSON]
    U --> V[Client HTTP 200 OK Response]

    classDef error fill:#ffebee,stroke:#c62828,stroke-width:2px;
    classDef success fill:#e8f8f5,stroke:#1abc9c,stroke-width:2px;
    classDef step fill:#f8f9fa,stroke:#34495e,stroke-width:1px;
    class E1,E2,E3,E4,E5 error;
    class V success;
    class A,B,C,D,F,G,H,I,I2,K,L,M,N,O,P,Q,R,S,T,U step;
```

Public service endpoints:
```text
GET  /health
POST /optimize-energy
```

---

## Why the LLM is used

The Large Language Model is utilized **only** for the single task that requires natural-language comprehension: translating unstructured operator notes into typed, parameter-bounded directive schemas.

The supported directive types are:
```text
solar_reduction
minimum_battery_reserve
no_charge_window
no_discharge_window
max_grid_window
no_op
```

The LLM is **never** asked to:
- Compute electricity costs or financial totals;
- Generate or format the 24-hour battery dispatch schedule;
- Enforce physical battery capacities or power conservation;
- Audit or validate its own numerical outputs.

Treating the LLM purely as an untrusted parser eliminates arithmetic hallucinations, schedule drift, and non-deterministic numerical errors.

---

## Why deterministic components are used

Deterministic software and exact solvers are employed wherever the problem domain is mathematically specified:
- **Strict Schema Enforcement:** Pydantic v2 rejects extra fields, type coercion, and malformed structures;
- **Directive Domain Validation:** Validates that extracted hours are in $[0, 23]$, reduction factors are in $[0.0, 1.0]$, and reserve limits do not exceed battery capacity;
- **24-Hour Constraint Compilation:** Normalizes start-inclusive/end-exclusive time intervals and resolves multi-note overlaps via clear mathematical policies;
- **Global Minimum-Cost Dispatch:** Solved via SciPy's HiGHS Dual-Simplex/Interior-Point Linear Programming engine;
- **Independent Plan Replay:** A zero-trust post-optimization engine that audits the raw public schedule against raw physical laws before emission;
- **Deterministic Summarization:** Assembles human-readable summaries from verified numerical totals.

This guarantees that every returned schedule is reproducible, physically feasible, and cost-optimal.

---

## Requirements

- **Runtime:** Python 3.12 or Python 3.14
- **Credentials:** Valid OpenAI API key with access to `gpt-5.6-sol` and `gpt-6-astra`
- **Network:** Outbound internet access to OpenAI API endpoints (`api.openai.com`)
- **Containers:** Docker (optional, for containerized deployments)

> `GET /health` requires no API key, makes no network calls, and touches neither OpenAI nor the optimizer.

---

## Environment variables

| Variable | Default | Allowed / Limits | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | *(None)* | Valid string | Required for optimization requests |
| `OPENAI_MODEL` | `gpt-5.6-sol` | OpenAI model ID | Primary directive interpreter (`reasoning_effort=none`) |
| `OPENAI_FALLBACK_MODEL` | `gpt-6-astra` | OpenAI model ID | Exceptional semantic-recovery model (`reasoning_effort=low`) |
| `OPENAI_TIMEOUT_SECONDS` | `8.0` | $> 0$ float | Per-model-call network timeout |
| `OPENAI_MAX_RETRIES` | `1` | $\ge 0$ int | Bounded SDK/network retry count |
| `APP_HOST` | `0.0.0.0` | IP string | Service bind host |
| `APP_PORT` | `8000` | Port int | Service bind port |
| `LOG_LEVEL` | `INFO` | Standard levels | Python logging level |
| `OPTIMIZE_DEADLINE_SECONDS` | `25.0` | $0 < v \le 28.0$ | Overall `/optimize-energy` wall-clock deadline budget |
| `DEBUG_TRACEBACKS` | `false` | `true`/`false` | Enable full exception tracebacks in server logs |
| `INTERPRETATION_CACHE_ENABLED` | `true` | `true`/`false` | Enable bounded in-memory interpretation cache |
| `INTERPRETATION_CACHE_MAX_ENTRIES` | `128` | $\ge 1$ int | Max LRU entries in interpretation cache |
| `INTERPRETATION_CACHE_TTL_SECONDS` | `900.0` | $> 0$ float | Time-to-live for cached interpretations (seconds) |

Copy the checked-in template to configure your local environment:
```bash
cp .env.example .env
# Edit .env and insert your OPENAI_API_KEY
```

---

## Install

```bash
# 1. Create and activate virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 2. Upgrade pip and install pinned dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# edit .env and set OPENAI_API_KEY
```

All direct project dependencies are pinned in `requirements.txt` to prevent solver and SDK drift.

---

## Run locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

The production baseline runs with **1 worker**. Because the application is fully async and the 96-variable LP solves in ~4 milliseconds, a single worker easily sustains high concurrency without thread contention.

Interactive documentation:
- **Swagger UI:** `http://127.0.0.1:8000/docs`
- **ReDoc:** `http://127.0.0.1:8000/redoc`

---

## Health check

```bash
curl -i http://127.0.0.1:8000/health
```

Expected response:
```http
HTTP/1.1 200 OK
content-type: application/json

{"status":"ok"}
```

Startup performs no remote health check, OpenAI probe, or solver dry-run, ensuring sub-second boot times.

---

## Optimize example

A complete tested request and response are available in:
- [`examples/sample_request.json`](file:///home/zunayed-mahmood/storage/Projects/BUP_Hackathon-preli/Codebase/examples/sample_request.json)
- [`examples/sample_response.json`](file:///home/zunayed-mahmood/storage/Projects/BUP_Hackathon-preli/Codebase/examples/sample_response.json)

Execute against local service:
```bash
curl -sS -X POST http://127.0.0.1:8000/optimize-energy \
  -H 'Content-Type: application/json' \
  --data-binary @examples/sample_request.json
```

Or execute against the live deployment:
```bash
curl -sS -X POST https://teamdevatmad-gridwise-llm-bup-cse-fest.onrender.com/optimize-energy \
  -H 'Content-Type: application/json' \
  --data-binary @examples/sample_request.json
```

---

## OpenAI model configuration

Primary Model:
```text
model = gpt-5.6-sol
reasoning.effort = none
store = false
```

Exceptional Recovery Model:
```text
model = gpt-6-astra
reasoning.effort = low
store = false
```

Both models are dynamically configurable via environment variables. The primary model delivers sub-second structured JSON parsing, while the fallback reasoning model handles complex linguistic syntax or recovery context if needed.

Before final submission, verify model candidates using the benchmarking tool:
```bash
OPENAI_API_KEY="$OPENAI_API_KEY" \
python scripts/benchmark_models.py gpt-5.6-sol gpt-5.6-terra gpt-6-astra
```

---

## Optimizer

The core optimization engine formulates the problem as a 24-hour continuous Linear Program solved via `scipy.optimize.linprog(method="highs")`.

### Decision Variables ($4 \times 24 = 96$ variables)
For each hour $h \in \{0, \dots, 23\}$:
- $g_h \ge 0$: Grid import (kWh)
- $s_h \ge 0$: Solar power consumed on-site (kWh)
- $x_h \in \mathbb{R}$: Signed battery action ($x_h > 0$ charge, $x_h < 0$ discharge)
- $E_h \ge 0$: Battery stored energy at the end of hour $h$ (kWh)

### Objective Function
Minimize total energy import cost across all 24 hours:
$$\min \sum_{h=0}^{23} \left( \text{tariff}_h \cdot g_h \right)$$

### Constraints
1. **Energy Balance:** $g_h + s_h - x_h = \text{demand}_h \quad \forall h \in \{0, \dots, 23\}$
2. **Solar Availability:** $0 \le s_h \le S_h^{\text{effective}} \quad \forall h \in \{0, \dots, 23\}$
3. **Inverter Limits:** $-P_{\text{discharge}}^{\text{max}} \le x_h \le P_{\text{charge}}^{\text{max}} \quad \forall h \in \{0, \dots, 23\}$
4. **State Transition:** $E_h = E_{h-1} + x_h$, with $E_{-1} = E_{\text{initial}}$
5. **Capacity Bounds:** $E_h^{\text{min}} \le E_h \le E_{\text{capacity}} \quad \forall h \in \{0, \dots, 23\}$
6. **Battery Neutrality:** $E_{23} = E_{\text{initial}}$
7. **Grid Caps:** $0 \le g_h \le G_h^{\text{max}}$
8. **Prohibitions:** No charge ($x_h \le 0$), No discharge ($x_h \ge 0$).

Because the official competition problem statement provides no battery efficiency loss factor, transitions are modeled as lossless.

---

## Directive Types & Semantic Interpretation Rules

| Directive Type | Parameters | Mathematical Definition |
| :--- | :--- | :--- |
| `solar_reduction` | `hours: list[int]`, `factor: float` ($0 \le f \le 1$) | $S_h^{\text{effective}} = f \cdot \text{solar}_h \quad \forall h \in \text{hours}$ |
| `minimum_battery_reserve` | `hours: list[int]`, `minimum_energy_kwh: float` | $E_h \ge \text{minimum\_energy\_kwh} \quad \forall h \in \text{hours}$ |
| `no_charge_window` | `hours: list[int]` | $x_h \le 0 \quad \forall h \in \text{hours}$ |
| `no_discharge_window` | `hours: list[int]` | $x_h \ge 0 \quad \forall h \in \text{hours}$ |
| `max_grid_window` | `hours: list[int]`, `max_grid_kwh: float` | $g_h \le \text{max\_grid\_kwh} \quad \forall h \in \text{hours}$ |
| `no_op` | `applies: false`, `structured_adjustment: null` | No operational impact |

### Interval & Overlap Policy
- Intervals are start-inclusive and end-exclusive ($[H_{\text{start}}, H_{\text{end}})$).
- Cross-midnight intervals wrap across hour 23 (e.g., 10 PM to 2 AM $\to [0, 1, 22, 23]$).
- Overlapping `solar_reduction` directives take the **minimum factor** (most restrictive).
- Overlapping `minimum_battery_reserve` directives take the **highest reserve**.
- Overlapping `max_grid_window` directives take the **lowest cap**.
- Prohibitions combine by mathematical set union.

---

## Deterministic guardrails

Public API contract guardrails:
- Models use strict Pydantic v2 with `extra="forbid"`, `strict=True`, and `allow_inf_nan=False`.
- **HTTP 400 (Bad Request):** Triggered by malformed JSON body, missing fields, or invalid types.
- **HTTP 422 (Unprocessable Entity):** Triggered by semantic validation failures (empty `scenario_id`, invalid battery parameters, hours $\ne$ 24, duplicate hours) or mathematically infeasible LP constraints.
- **HTTP 500 (Internal Server Error):** Sanitized generic response emitted only upon unrecoverable provider outages or internal assertion failures.

### Numerical Policy
Solver precision is strictly preserved. Only values with absolute magnitude below `1e-7` are cleaned to `0.0`. Replay uses the internal validation tolerance before the response is built. Public numeric values are JSON numbers, never formatted numeric strings.

### Zero-Trust Independent Plan Replay
Before returning any schedule:
1. The 24 hourly rows are parsed into independent domain structures.
2. [`replay_and_validate`](file:///home/zunayed-mahmood/storage/Projects/BUP_Hackathon-preli/Codebase/app/optimizer/replay.py) validates hourly energy balance, battery capacity bounds, and inverter ratings using purely arithmetic business logic (no solver state).
3. Financial totals (`total_cost_bdt`, `total_grid_kwh`, `peak_grid_kwh`) are recomputed directly from the replayed rows and cross-checked against the solver objective. If any discrepancy exceeds $0.01$, the plan is rejected.

---

## Empirical Benchmarking Results (100-Case Adversarial Suite)

The deployment was evaluated against the complete 100-case high-coverage adversarial benchmark suite:

```text
======================================================================
GRIDWISE LLM 100-CASE ADVERSARIAL BENCHMARK RESULTS
======================================================================
Core Spec-Grounded Accuracy (GW100-001..GW100-090) : 90 / 90 (100.0%)
Ambiguity & Assumption Accuracy (GW100-091..GW100-100) :  9 / 10  (90.0%)
Overall Suite Accuracy (GW100-001..GW100-100)        : 99 / 100 (99.0%)
----------------------------------------------------------------------
HTTP Status Accuracy (200 / 400 / 422)               : 100/100  (100.0%)
Strict Pydantic Schema Adherence                     : 100/100  (100.0%)
Independent LP Plan Replay Validity (Status 200)     :  84/ 84  (100.0%)
Reported vs Replayed Totals Consistency              :  84/ 84  (100.0%)
----------------------------------------------------------------------
Total Wall-Clock Test Duration (Concurrency = 3)     : 86.22 seconds
Mean Latency Across All Requests                     : 2.562 seconds
Median (p50) Latency                                 : 2.587 seconds
95th Percentile (p95) Latency                        : 4.196 seconds
======================================================================
```

### Latency Summary Table
| Subset | Count | Mean | p50 | p90 | p95 | p99 | Min | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **All Requests** | **100** | **2.562s** | **2.587s** | **3.841s** | **4.196s** | **4.880s** | **0.248s** | **6.871s** |
| **Status 200 (Optimization)** | 84 | 2.924s | 2.728s | 3.982s | 4.441s | 5.202s | 2.051s | 6.871s |
| **Status 4xx (Error Fast-Path)**| 16 | 0.658s | 0.313s | 1.651s | 3.040s | 3.382s | 0.248s | 3.467s |

### Category Breakdown (18 of 19 Categories Scored 100%)
- `official_public_reference`: 10/10 (100.0%)
- `solar_reduction_paraphrase`: 7/7 (100.0%)
- `minimum_battery_reserve_paraphrase`: 6/6 (100.0%)
- `no_charge_paraphrase`: 5/5 (100.0%)
- `no_discharge_paraphrase`: 5/5 (100.0%)
- `max_grid_paraphrase`: 5/5 (100.0%)
- `no_op_distractor`: 6/6 (100.0%)
- `multi_note_combination`: 10/10 (100.0%)
- `context_bloat_and_injection`: 10/10 (100.0%)
- `mathematical_edge`: 10/10 (100.0%)
- `structural_error`: 7/7 (100.0%)
- `semantic_error`: 7/7 (100.0%)
- `infeasible_error`: 2/2 (100.0%)
- `ambiguity_cross_midnight`: 3/3 (100.0%)
- `ambiguity_overlap_solar`: 2/2 (100.0%)
- `ambiguity_overlap_reserve`: 1/1 (100.0%)
- `ambiguity_overlap_grid`: 1/1 (100.0%)
- `ambiguity_multiple_directives_one_note`: 2/2 (100.0%)
- `ambiguity_vague_time`: 0/1 (0.0% — intentional colloquial phrasing in case GW100-098)

---

## Tiebreaker Engineering & Contest Resilience

1. **Adversarial Prompt Injection Immunity:**  
   Operator notes are isolated in user-role JSON objects; system instructions command the model to ignore embedded override commands.
2. **In-Memory Caching & Single-Flight Coalescing:**  
   Identical concurrent requests share a single in-flight `asyncio.Future`, preventing thundering-herd latency spikes and API budget waste.
3. **Two-Tiered Deadline-Aware Fallback:**  
   If primary parsing fails validation, the service checks its remaining deadline budget before invoking `gpt-6-astra` with targeted correction prompts.
4. **Independent Plan Replay Engine:**  
   Zero physically invalid or mathematically inconsistent plans can escape the service.
5. **Sanitized Production Logging:**  
   Tracebacks, raw bodies, and sensitive keys are suppressed in production logs to meet competition security requirements.

---

## Public API Reference & Contract

### `POST /optimize-energy` Request
```json
{
  "scenario_id": "SAMPLE-01",
  "battery": {
    "capacity_kwh": 40.0,
    "initial_energy_kwh": 20.0,
    "minimum_energy_kwh": 0.0,
    "max_charge_kwh_per_hour": 20.0,
    "max_discharge_kwh_per_hour": 20.0
  },
  "operator_notes": ["Facilities washing panels noon-2pm; usable solar roughly 25%."],
  "hours": [
    {"hour": 0, "demand_kwh": 90.0, "solar_kwh": 0.0, "tariff_bdt_per_kwh": 6.0}
  ]
}
```

### `POST /optimize-energy` Response
```json
{
  "scenario_id": "SAMPLE-01",
  "directive_interpretation": [
    {
      "note_index": 0,
      "applies": true,
      "directive_type": "solar_reduction",
      "structured_adjustment": {"hours": [12, 13], "factor": 0.25},
      "explanation": "Usable solar limited to 25% from 12:00 to 14:00."
    }
  ],
  "hourly_plan": [
    {
      "hour": 0,
      "grid_kwh": 70.0,
      "solar_used_kwh": 0.0,
      "battery_action": "discharge",
      "battery_kwh": 20.0,
      "battery_energy_after_kwh": 0.0
    }
  ],
  "total_grid_kwh": 2692.5,
  "total_cost_bdt": 38365.0,
  "peak_grid_kwh": 187.5,
  "plan_summary": "Optimized schedule under solar_reduction. Total cost: 38365.00 BDT."
}
```

---

## Tests

Execute the 215-case deterministic test suite (offline, requires no API key):
```bash
pytest -q
```

Execute live OpenAI provider tests (requires credentials):
```bash
RUN_OPENAI_TESTS=1 OPENAI_API_KEY="$OPENAI_API_KEY" pytest -q -m live_openai
```

The test suite validates schema enforcement, LP optimality, prompt injection resistance, deadline budget management, and replay verification.

---

## Public sample runner

Test any running local or deployed endpoint against the official public reference cases:
```bash
python scripts/run_public_cases.py --base-url https://teamdevatmad-gridwise-llm-bup-cse-fest.onrender.com
```

Checks HTTP status, directive parameters, response schema, independent plan replay, returned totals, and cost optimality.

---

## Docker

### Build
```bash
docker build -t gridwise-llm:latest .
```

### Run
```bash
docker run --rm \
  -p 8000:8000 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e APP_HOST=0.0.0.0 \
  -e APP_PORT=8000 \
  gridwise-llm:latest
```

### Container Health Check
```bash
curl http://127.0.0.1:8000/health
```

The Docker image uses `python:3.12-slim`, non-root execution, and excludes all caches, Git data, and test files from the build context.

---

## Deployment

The production deployment runs on Render Cloud and is continuously available:
- **Base URL:** `https://teamdevatmad-gridwise-llm-bup-cse-fest.onrender.com`
- **Region:** Oregon (US West)
- **Runtime:** Python 3.12 / Uvicorn (1 worker)

Post-deployment smoke testing:
```bash
python scripts/smoke_test_deployment.py --base-url https://teamdevatmad-gridwise-llm-bup-cse-fest.onrender.com
```

---

## Known assumptions

Where the official contest specification is silent, this implementation adheres to these formal engineering conventions:
- **Time Intervals:** Start-inclusive and end-exclusive ($[H_{\text{start}}, H_{\text{end}})$).
- **Cross-Midnight Windows:** Normalized into ascending hour sets (e.g. 10 PM to 2 AM $\to [0, 1, 22, 23]$).
- **Solar Overlaps:** The minimum remaining solar factor (most restrictive) is enforced.
- **Reserve Overlaps:** The highest minimum reserve is enforced.
- **Grid Cap Overlaps:** The lowest grid cap is enforced.
- **Lossless Battery Transitions:** Modeled without efficiency degradation due to lack of round-trip efficiency coefficients in official problem specification.
- **In-Memory Caching:** Bounded SHA-256 LRU cache with single-flight deduplication.

---

## Security

- `.env` and all credential files are excluded via `.gitignore` and `.dockerignore`.
- API keys are injected solely via environment variables at runtime.
- Operator notes are treated as untrusted user data.
- The LLM has zero filesystem access, code execution capabilities, or tool bindings.
- Stack traces and internal provider exceptions are scrubbed from production responses.

Run pre-submission secret scans:
```bash
git grep -n 'sk-'
git grep -n 'OPENAI_API_KEY='
```

---

## Repository Architecture & Sitemap

```text
Codebase/
├── Dockerfile                     # Production container definition (Python 3.12-slim)
├── requirements.txt               # Pinned dependencies (FastAPI, SciPy, OpenAI, Pydantic)
├── pytest.ini                     # Pytest configuration and live-test markers
├── .env.example                   # Template environment variables
├── README.md                      # Comprehensive project documentation
├── app/                           # Core application package
│   ├── main.py                    # FastAPI application factory and entry point
│   ├── config.py                  # Pydantic BaseSettings configuration
│   ├── dependencies.py            # FastAPI dependency injection providers
│   ├── errors.py                  # Domain exception hierarchy
│   ├── api/
│   │   ├── routes.py              # /health and /optimize-energy endpoints
│   │   └── error_handlers.py      # HTTP error mapping and sanitized responses
│   ├── interpreter/
│   │   ├── client.py              # OpenAI AsyncOpenAI client & fallback logic
│   │   ├── prompt.py              # System prompts & injection defenses
│   │   ├── validator.py           # Deterministic directive domain validation
│   │   └── cache.py               # SHA-256 LRU cache and single-flight coalescing
│   ├── optimizer/
│   │   ├── compiler.py            # Compiles directives into 24-hour bounds
│   │   ├── lp.py                  # SciPy HiGHS 96-variable linear program
│   │   ├── numeric.py             # Numerical tolerances and zero-cleaning
│   │   └── replay.py              # Independent plan replay validation engine
│   ├── models/
│   │   ├── request.py             # Strict Pydantic input schemas and semantic checks
│   │   ├── response.py            # Strict Pydantic output schemas (EnergyResponse)
│   │   ├── internal.py            # CanonicalRequest and CompiledConstraints models
│   │   └── llm.py                 # Structured LLM directive batch definitions
│   ├── services/
│   │   └── optimization_service.py # Orchestration, recovery, and deadline management
│   └── observability/
│       └── logging.py             # Structured JSON logging and X-Request-ID middleware
├── scripts/                       # Benchmarking and operational toolkits
│   ├── run_public_cases.py        # Validates official 10 sample cases
│   ├── smoke_test_deployment.py   # Quick deployment health and replay smoke test
│   ├── benchmark_latency.py       # Latency, p95, and concurrency stress tester
│   └── benchmark_models.py        # Multi-model comparative evaluation
├── tests/                         # Automated test suite (215 tests)
│   ├── conftest.py                # Pytest fixtures and mock setup
│   ├── unit/                      # Unit tests for LP, models, validator, cache
│   ├── integration/               # End-to-end API integration tests
│   └── fixtures/                  # Official public reference cases and payloads
└── examples/
    ├── sample_request.json        # Reference tested sample input payload
    └── sample_response.json       # Reference validated sample output payload
```

---

## License

This project is released under the [MIT License](LICENSE). Built for the **BUP CSE Fest 2026 Hackathon**.
