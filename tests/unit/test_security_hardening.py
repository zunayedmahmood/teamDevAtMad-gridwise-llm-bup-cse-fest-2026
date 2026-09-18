import logging
from pathlib import Path
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.error_handlers import register_exception_handlers
from app.config import settings
from app.optimizer.numeric import EPS


def test_unexpected_error_no_traceback_in_production_logs(caplog):
    test_app = FastAPI()
    register_exception_handlers(test_app)

    @test_app.get("/trigger-crash")
    def trigger_crash():
        raise RuntimeError("super_secret_internal_crash_details")

    client = TestClient(test_app, raise_server_exceptions=False)
    with caplog.at_level(logging.ERROR):
        response = client.get("/trigger-crash")

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal service error."}

    # Verify no traceback is emitted in captured logs when debug_tracebacks is False
    log_text = caplog.text
    assert "Traceback (most recent call last)" not in log_text
    assert "super_secret_internal_crash_details" not in log_text
    assert "exception_category=RuntimeError" in log_text


def test_sanitized_api_error_response():
    test_app = FastAPI()
    register_exception_handlers(test_app)

    @test_app.get("/trigger-error")
    def trigger_error():
        raise ValueError("raw internal database exception")

    client = TestClient(test_app, raise_server_exceptions=False)
    response = client.get("/trigger-error")

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal service error."}
    # Ensure sensitive details are not leaked in body
    assert "ValueError" not in response.text
    assert "database" not in response.text


def test_dockerignore_exists_and_contains_sensitive_patterns():
    repo_root = Path(__file__).resolve().parents[2]
    dockerignore_path = repo_root / ".dockerignore"

    assert dockerignore_path.exists(), ".dockerignore must exist in repository root"
    content = dockerignore_path.read_text(encoding="utf-8")
    patterns = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")]

    required_patterns = [".env", ".git", "__pycache__", ".pytest_cache"]
    for req in required_patterns:
        assert any(req in p for p in patterns), f"Missing required pattern: {req}"


def test_documentation_config_consistency():
    repo_root = Path(__file__).resolve().parents[2]
    readme_text = (repo_root / "README.md").read_text(encoding="utf-8")
    env_example_text = (repo_root / ".env.example").read_text(encoding="utf-8")

    # Reconcile timeout
    assert f"`{settings.openai_timeout_seconds}`" in readme_text or f"`{settings.openai_timeout_seconds:.1f}`" in readme_text
    assert f"OPENAI_TIMEOUT_SECONDS={settings.openai_timeout_seconds}" in env_example_text or f"OPENAI_TIMEOUT_SECONDS={settings.openai_timeout_seconds:.1f}" in env_example_text

    # Reconcile numerical threshold
    assert "`1e-7`" in readme_text or f"`{EPS}`" in readme_text

    # Reconcile deadline
    assert f"`{settings.optimize_deadline_seconds}`" in readme_text or f"`{settings.optimize_deadline_seconds:.1f}`" in readme_text
    assert f"OPTIMIZE_DEADLINE_SECONDS={settings.optimize_deadline_seconds}" in env_example_text or f"OPTIMIZE_DEADLINE_SECONDS={settings.optimize_deadline_seconds:.1f}" in env_example_text
