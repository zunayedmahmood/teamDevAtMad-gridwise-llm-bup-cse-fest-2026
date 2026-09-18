import json
import re
from pathlib import Path

from app.models.request import EnergyRequest
from app.models.response import EnergyResponse

ROOT = Path(__file__).resolve().parents[2]


def test_requirements_are_exactly_pinned():
    lines = [
        line.strip()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert lines
    assert all("==" in line for line in lines)


def test_env_example_matches_settings_surface():
    env_keys = {
        line.split("=", 1)[0]
        for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    }
    assert env_keys == {
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_FALLBACK_MODEL",
        "OPENAI_TIMEOUT_SECONDS",
        "OPENAI_MAX_RETRIES",
        "APP_HOST",
        "APP_PORT",
        "LOG_LEVEL",
        "OPTIMIZE_DEADLINE_SECONDS",
    }


def test_dockerfile_has_runtime_only_secret_policy_and_healthcheck():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.12-slim" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert 'ENV APP_HOST=0.0.0.0' in dockerfile
    assert 'ENV APP_PORT=8000' in dockerfile
    assert '--workers 1' in dockerfile
    assert not re.search(r"ENV\s+OPENAI_API_KEY\s*=", dockerfile)


def test_checked_examples_match_public_models_and_numeric_types():
    request_data = json.loads((ROOT / "examples" / "sample_request.json").read_text())
    response_text = (ROOT / "examples" / "sample_response.json").read_text()
    response_data = json.loads(response_text)
    EnergyRequest.model_validate(request_data)
    EnergyResponse.model_validate_json(response_text)

    for key in ("total_grid_kwh", "total_cost_bdt", "peak_grid_kwh"):
        assert isinstance(response_data[key], (int, float))
        assert not isinstance(response_data[key], str)


def test_readme_contains_part5_required_sections():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    required = {
        "## Overview",
        "## Architecture",
        "## Why the LLM is used",
        "## Why deterministic components are used",
        "## Requirements",
        "## Environment variables",
        "## Install",
        "## Run locally",
        "## Health check",
        "## Optimize example",
        "## OpenAI model configuration",
        "## Optimizer",
        "## Deterministic guardrails",
        "## Tests",
        "## Public sample runner",
        "## Docker",
        "## Deployment",
        "## Known assumptions",
        "## Security",
    }
    assert required <= set(re.findall(r"^## .+$", readme, flags=re.MULTILINE))
