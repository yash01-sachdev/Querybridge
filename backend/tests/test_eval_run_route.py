"""Tests for the built-in GenAI eval runner route."""

from __future__ import annotations

from models.request import EvalRunRequest
from routes.query import run_built_in_evals


def test_sqlite_eval_run_returns_passing_suite() -> None:
    """The built-in SQLite suite should pass with the deterministic Ollama mock."""

    response = run_built_in_evals(EvalRunRequest(backend="sqlite"))

    assert response.backend == "sqlite"
    assert response.workflow == "langgraph"
    assert response.model == "mock-ollama"
    assert response.total_cases > 0
    assert response.failed_cases == 0
    assert response.passed_cases == response.total_cases
    assert response.cases[0].trace.node_count >= 1


def test_mongodb_eval_run_returns_passing_suite() -> None:
    """The built-in MongoDB suite should pass with the deterministic Ollama mock."""

    response = run_built_in_evals(EvalRunRequest(backend="mongodb"))

    assert response.backend == "mongodb"
    assert response.workflow == "langgraph"
    assert response.model == "mock-ollama"
    assert response.total_cases > 0
    assert response.failed_cases == 0
    assert response.passed_cases == response.total_cases
    assert response.cases[0].trace.node_count >= 1


def test_postgresql_eval_run_returns_passing_suite() -> None:
    """The built-in PostgreSQL-path suite should also pass with the deterministic mock."""

    response = run_built_in_evals(EvalRunRequest(backend="postgresql"))

    assert response.backend == "postgresql"
    assert response.workflow == "langgraph"
    assert response.model == "mock-ollama"
    assert response.total_cases > 0
    assert response.failed_cases == 0
    assert response.passed_cases == response.total_cases
    assert response.cases[0].trace.node_count >= 1
