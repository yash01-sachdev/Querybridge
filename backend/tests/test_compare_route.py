"""Tests for the model-backed compare route."""

from __future__ import annotations

import pytest

from core.connection_registry import register_connection, remove_connection
from models.request import CompareConnectionIds, QueryCompareRequest
from routes.query import compare_backends
from tests.fake_mongo import FakeMongoDatabase


def test_compare_route_returns_three_model_backed_previews() -> None:
    """The compare route should build previews for SQLite, PostgreSQL, and MongoDB."""

    response = compare_backends(QueryCompareRequest(question="show user emails", backend="sqlite"))

    assert response.question == "show user emails"
    assert len(response.comparisons) == 3

    by_backend = {item.backend: item for item in response.comparisons}

    assert by_backend["sqlite"].success is True
    assert by_backend["sqlite"].schema_source == "built-in learning schema"
    assert by_backend["sqlite"].workflow == "langgraph"
    assert by_backend["sqlite"].model == "mock-ollama"
    assert "SELECT users.email FROM users" in by_backend["sqlite"].compiled_query
    assert by_backend["sqlite"].trace.node_count >= 4

    assert by_backend["postgresql"].success is True
    assert by_backend["postgresql"].schema_source == "built-in learning schema"
    assert by_backend["postgresql"].workflow == "langgraph"
    assert by_backend["postgresql"].model == "mock-ollama"
    assert "SELECT users.email FROM users" in by_backend["postgresql"].compiled_query
    assert by_backend["postgresql"].trace.node_count >= 4

    assert by_backend["mongodb"].success is True
    assert by_backend["mongodb"].schema_source == "built-in learning schema"
    assert by_backend["mongodb"].workflow == "langgraph"
    assert by_backend["mongodb"].model == "mock-ollama"
    assert '"$project"' in by_backend["mongodb"].compiled_query
    assert by_backend["mongodb"].trace.node_count >= 4


def test_compare_route_keeps_backend_level_failures_visible() -> None:
    """Unsupported prompts should return one failed preview per backend instead of crashing the route."""

    response = compare_backends(
        QueryCompareRequest(question="truncate the users table", backend="sqlite")
    )

    assert len(response.comparisons) == 3
    assert all(item.success is False for item in response.comparisons)
    assert all("Only read-only queries are allowed." in item.message for item in response.comparisons)
    assert all(item.trace.node_count >= 1 for item in response.comparisons)


def test_compare_route_uses_live_linked_databases_when_available(
    sqlite_backend: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare mode should switch each backend to its linked live schema when a session id exists."""

    fake_mongo_database = FakeMongoDatabase(
        {
            "users": [
                {"name": "Alice", "email": "alice@example.com"},
                {"name": "Bob", "email": "bob@example.com"},
            ]
        }
    )

    engine = sqlite_backend["engine"]
    sqlite_connection_id = register_connection("sqlite", {"sqlite_path": str(sqlite_backend["path"])})
    postgresql_connection_id = register_connection(
        "postgresql",
        {"postgres_url": "postgresql://demo:demo@localhost:5432/demo"},
    )
    mongodb_connection_id = register_connection(
        "mongodb",
        {"mongo_url": "mongodb://localhost:27017/demo"},
    )

    def fake_open_backend_connection(backend: str, connection_overrides: object = None):
        if backend in {"sqlite", "postgresql"}:
            connection = engine.connect()

            def close_connection() -> None:
                connection.close()

            return connection, close_connection

        return fake_mongo_database, (lambda: None)

    monkeypatch.setattr("routes.query.open_backend_connection", fake_open_backend_connection)

    try:
        response = compare_backends(
            QueryCompareRequest(
                question="show user emails",
                backend="sqlite",
                connection_ids=CompareConnectionIds(
                    sqlite=sqlite_connection_id,
                    postgresql=postgresql_connection_id,
                    mongodb=mongodb_connection_id,
                ),
            )
        )
    finally:
        remove_connection(sqlite_connection_id)
        remove_connection(postgresql_connection_id)
        remove_connection(mongodb_connection_id)

    by_backend = {item.backend: item for item in response.comparisons}

    assert response.schema_source == "linked live database"
    assert by_backend["sqlite"].schema_source == "linked live database"
    assert by_backend["postgresql"].schema_source == "linked live database"
    assert by_backend["mongodb"].schema_source == "linked live database"
    assert all(item.success is True for item in response.comparisons)
