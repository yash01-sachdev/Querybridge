"""Route-level MongoDB prompt matrix tests for the LangGraph workflow."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException

from models.request import QueryRequest
from routes.query import create_query
from tests.fake_mongo import FakeMongoDatabase
from tests.mongodb_prompt_matrix import MONGODB_PROMPT_CASES

SUCCESS_CASES = [case for case in MONGODB_PROMPT_CASES if case["expected"] == "success"]
ERROR_CASES = [case for case in MONGODB_PROMPT_CASES if case["expected"] == "error"]


@pytest.fixture()
def fake_mongo_database() -> FakeMongoDatabase:
    """Return a small in-memory MongoDB-like database for route tests."""

    return FakeMongoDatabase(
        {
            "users": [
                {"name": "Alice", "email": "alice@example.com", "status": "active"},
                {"name": "Bob", "email": "bob@example.com", "status": "active"},
                {"name": "Charlie", "email": "charlie@example.com", "status": "inactive"},
            ],
            "orders": [
                {"user_name": "Alice", "amount": 120.5, "status": "completed"},
                {"user_name": "Bob", "amount": 60.0, "status": "pending"},
                {"user_name": "Alice", "amount": 300.0, "status": "completed"},
                {"user_name": "Charlie", "amount": 90.0, "status": "cancelled"},
            ],
        }
    )


@pytest.fixture(autouse=True)
def use_fake_mongo_connection(
    fake_mongo_database: FakeMongoDatabase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Swap the live Mongo connection layer for the in-memory test database."""

    def fake_open_backend_connection(backend: str, connection_overrides: object = None):
        return fake_mongo_database, (lambda: None)

    monkeypatch.setattr("routes.query.open_backend_connection", fake_open_backend_connection)


@pytest.mark.parametrize("case", SUCCESS_CASES, ids=[case["name"] for case in SUCCESS_CASES])
def test_mongodb_prompt_matrix_success_cases(case: dict[str, Any]) -> None:
    """Supported MongoDB prompts should complete through the graph route."""

    response = create_query(QueryRequest(question=case["question"], backend="mongodb"))

    assert response.workflow == "langgraph"
    assert response.model == "mock-ollama"
    assert response.trace.node_count >= 6
    assert response.trace.steps[0].name == "select_schema"
    assert response.safety_check.passed is True
    assert response.validation_check.passed is True
    assert response.result.rows == case["rows"]
    assert response.result.row_count == case["row_count"]

    for query_fragment in case["query_fragments"]:
        assert query_fragment in response.compiled_query


@pytest.mark.parametrize("case", ERROR_CASES, ids=[case["name"] for case in ERROR_CASES])
def test_mongodb_prompt_matrix_error_cases(case: dict[str, Any]) -> None:
    """Unsupported or destructive MongoDB prompts should fail clearly."""

    with pytest.raises(HTTPException) as exc_info:
        create_query(QueryRequest(question=case["question"], backend="mongodb"))

    assert exc_info.value.status_code == 400
    assert str(case["plan"]["error"]) in str(exc_info.value.detail)
