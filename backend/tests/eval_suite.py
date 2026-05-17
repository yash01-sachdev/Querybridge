"""Backend pipeline tests for SQLite, PostgreSQL-style SQL, and Mongo-style flow."""

from __future__ import annotations

import sys
from pathlib import Path
import re

import pytest
from fastapi import HTTPException

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.compiler import compile_query
from core.executor import execute_query
from core.safety import check_safety
from core.validator import validate_query
from models.request import QueryRequest
from routes.query import create_query


def test_sqlite_route_runs_end_to_end(sqlite_backend: dict[str, object]) -> None:
    """The main route should return rows, checks, and explanation for SQLite."""

    response = create_query(
        QueryRequest(
            question="show user emails",
            backend="sqlite",
            connection={"sqlite_path": str(sqlite_backend["path"])},
        )
    )

    assert response.compiled_query == "SELECT users.email FROM users LIMIT 10"
    assert response.safety_check.passed is True
    assert response.validation_check.passed is True
    assert response.result.row_count == 3
    assert response.result.rows[0]["email"] == "alice@example.com"
    assert response.repaired is False
    assert response.execution_time_ms >= 0
    assert response.explanation


def test_sqlite_count_route_returns_count(sqlite_backend: dict[str, object]) -> None:
    """Count questions should compile and execute as one-row summaries."""

    response = create_query(
        QueryRequest(
            question="how many users",
            backend="sqlite",
            connection={"sqlite_path": str(sqlite_backend["path"])},
        )
    )

    assert response.compiled_query == "SELECT COUNT(*) AS count FROM users"
    assert response.result.rows == [{"count": 3}]
    assert response.result.row_count == 1


def test_sqlite_contains_filter_returns_matching_rows(sqlite_backend: dict[str, object]) -> None:
    """Contains-style prompts should compile into a real WHERE clause."""

    response = create_query(
        QueryRequest(
            question="user with name containing any one letter b,z",
            backend="sqlite",
            connection={"sqlite_path": str(sqlite_backend["path"])},
        )
    )

    assert "WHERE" in response.compiled_query
    assert "LOWER(users.name)" in response.compiled_query
    assert response.result.rows == [{"name": "Bob"}]
    assert response.result.row_count == 1


def test_sqlite_ends_with_filter_returns_matching_rows(sqlite_backend: dict[str, object]) -> None:
    """Ends-with prompts should compile into a suffix filter."""

    response = create_query(
        QueryRequest(
            question="user with name ending with b",
            backend="sqlite",
            connection={"sqlite_path": str(sqlite_backend["path"])},
        )
    )

    assert "WHERE" in response.compiled_query
    assert "LOWER(users.name) LIKE '%b'" in response.compiled_query
    assert response.result.rows == [{"name": "Bob"}]
    assert response.result.row_count == 1


def test_sqlite_cross_field_or_contains_query_returns_both_matches(
    sqlite_backend: dict[str, object],
) -> None:
    """Separate field clauses joined by OR should stay separate and return both matches."""

    response = create_query(
        QueryRequest(
            question="show users where name contains alice or email contains bob",
            backend="sqlite",
            connection={"sqlite_path": str(sqlite_backend["path"])},
        )
    )

    assert "LOWER(users.name) LIKE '%alice%'" in response.compiled_query
    assert "LOWER(users.email) LIKE '%bob%'" in response.compiled_query
    assert " OR " in response.compiled_query
    assert response.result.rows == [
        {"name": "Alice", "email": "alice@example.com"},
        {"name": "Bob", "email": "bob@example.com"},
    ]
    assert response.result.row_count == 2


def test_sqlite_join_query_returns_related_rows(sqlite_backend: dict[str, object]) -> None:
    """Questions that span related tables should produce a join and real rows."""

    response = create_query(
        QueryRequest(
            question="show order amounts with user names",
            backend="sqlite",
            connection={"sqlite_path": str(sqlite_backend["path"])},
        )
    )

    assert "JOIN users ON orders.user_id = users.id" in response.compiled_query
    assert response.result.row_count == 4
    assert response.result.rows[0] == {"amount": 120.5, "name": "Alice"}


def test_sqlite_join_count_query_uses_related_filter(sqlite_backend: dict[str, object]) -> None:
    """Count questions should still join when the filter lives on a related table."""

    response = create_query(
        QueryRequest(
            question="how many orders where name is Alice",
            backend="sqlite",
            connection={"sqlite_path": str(sqlite_backend["path"])},
        )
    )

    assert "JOIN users ON orders.user_id = users.id" in response.compiled_query
    assert "users.name = 'alice'" in response.compiled_query.lower()
    assert response.result.rows == [{"count": 2}]


def test_sqlite_route_blocks_destructive_request(sqlite_backend: dict[str, object]) -> None:
    """Destructive prompts should be rejected instead of quietly downgraded to SELECT *."""

    with pytest.raises(HTTPException) as exc_info:
        create_query(
            QueryRequest(
                question="truncate the users table",
                backend="sqlite",
                connection={"sqlite_path": str(sqlite_backend["path"])},
            )
        )

    assert exc_info.value.status_code == 400
    assert "read-only" in str(exc_info.value.detail).lower()


def test_safety_rejects_destructive_sql() -> None:
    """Unsafe SQL should be blocked before execution."""

    is_safe, reason = check_safety("DELETE FROM users", "sqlite", {"tables": {"users": {}}})

    assert is_safe is False
    assert "Only SELECT" in reason or "forbidden" in reason


def test_validator_rejects_unknown_sql_column(sqlite_schema: dict[str, object]) -> None:
    """Validator should reject SQL columns that do not exist."""

    is_valid, reason = validate_query(
        "SELECT users.username FROM users LIMIT 10",
        "sqlite",
        sqlite_schema,
    )

    assert is_valid is False
    assert "Column not found" in reason


def test_validator_rejects_unqualified_column_outside_query_scope(
    sqlite_schema: dict[str, object],
) -> None:
    """Validator should reject unqualified columns that only exist on other tables."""

    is_valid, reason = validate_query(
        "SELECT AVG(amount) AS average_amount FROM users GROUP BY user_id LIMIT 10",
        "sqlite",
        sqlite_schema,
    )

    assert is_valid is False
    assert "referenced tables" in reason


def test_validator_rejects_qualified_column_outside_query_scope(
    sqlite_schema: dict[str, object],
) -> None:
    """Validator should reject qualified columns from tables that are not joined in the SQL."""

    is_valid, reason = validate_query(
        "SELECT AVG(orders.amount) AS average_amount FROM users GROUP BY users.id LIMIT 10",
        "sqlite",
        sqlite_schema,
    )

    assert is_valid is False
    assert "not joined" in reason


def test_validator_accepts_sql_table_aliases(sqlite_schema: dict[str, object]) -> None:
    """Validator should resolve table aliases back to the underlying schema table."""

    is_valid, reason = validate_query(
        "SELECT u.email FROM users AS u LIMIT 10",
        "sqlite",
        sqlite_schema,
    )

    assert is_valid is True
    assert reason == ""


def test_postgresql_style_route_runs_with_sqlite_backing_store(
    sqlite_backend: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """We can smoke-test the PostgreSQL path locally by swapping in a SQLite connection."""

    engine = sqlite_backend["engine"]

    def fake_open_backend_connection(backend: str, connection_overrides: object = None):
        connection = engine.connect()

        def close_connection() -> None:
            connection.close()

        return connection, close_connection

    monkeypatch.setattr("routes.query.open_backend_connection", fake_open_backend_connection)

    response = create_query(QueryRequest(question="show user emails", backend="postgresql"))

    assert response.backend == "postgresql"
    assert "SELECT users.email FROM users LIMIT 10" in response.compiled_query
    assert response.safety_check.passed is True
    assert response.validation_check.passed is True
    assert response.result.row_count == 3


def test_mongodb_pipeline_runs_end_to_end() -> None:
    """Mongo-style planning, compilation, checks, execution, and explanation should work locally."""

    fake_database = FakeMongoDatabase(
        {
            "users": [
                {"name": "Alice", "email": "alice@example.com", "status": "active"},
                {"name": "Bob", "email": "bob@example.com", "status": "active"},
                {"name": "Charlie", "email": "charlie@example.com", "status": "inactive"},
            ]
        }
    )
    schema = {
        "collections": {
            "users": {
                "fields": [
                    {"name": "name", "type": "str"},
                    {"name": "email", "type": "str"},
                    {"name": "status", "type": "str"},
                ]
            }
        }
    }

    query_plan = {
        "operation": "find",
        "collection": "users",
        "match": {},
        "project": {"email": 1},
        "sort": {},
        "limit": 10,
    }
    compiled_query = compile_query(query_plan, "mongodb")
    safety_check = check_safety(compiled_query, "mongodb", schema)
    validation_check = validate_query(compiled_query, "mongodb", schema)
    result = execute_query(compiled_query, "mongodb", fake_database, collection="users")

    assert safety_check[0] is True
    assert validation_check[0] is True
    assert result["row_count"] == 3
    assert result["rows"][0]["email"] == "alice@example.com"


def test_mongodb_count_pipeline_returns_count() -> None:
    """Mongo count questions should compile to a count stage and execute correctly."""

    fake_database = FakeMongoDatabase(
        {
            "users": [
                {"name": "Alice"},
                {"name": "Bob"},
                {"name": "Charlie"},
            ]
        }
    )
    schema = {
        "collections": {
            "users": {
                "fields": [
                    {"name": "name", "type": "str"},
                ]
            }
        }
    }

    query_plan = {
        "operation": "aggregate",
        "collection": "users",
        "intent": "count",
        "match": {},
        "project": {},
        "sort": {},
        "group_by": [],
        "aggregations": [],
        "limit": 10,
    }
    compiled_query = compile_query(query_plan, "mongodb")
    result = execute_query(compiled_query, "mongodb", fake_database, collection="users")

    assert '"$count": "count"' in compiled_query
    assert result["rows"] == [{"count": 3}]


def test_mongodb_ends_with_pipeline_returns_matching_rows() -> None:
    """Mongo planning should support suffix-style text filters."""

    fake_database = FakeMongoDatabase(
        {
            "users": [
                {"name": "Alice"},
                {"name": "Bob"},
                {"name": "Charlie"},
            ]
        }
    )
    schema = {
        "collections": {
            "users": {
                "fields": [
                    {"name": "name", "type": "str"},
                ]
            }
        }
    }

    query_plan = {
        "operation": "find",
        "collection": "users",
        "match": {"name": {"$regex": "b$", "$options": "i"}},
        "project": {"name": 1},
        "sort": {},
        "limit": 10,
    }
    compiled_query = compile_query(query_plan, "mongodb")
    result = execute_query(compiled_query, "mongodb", fake_database, collection="users")

    assert 'b$' in compiled_query
    assert result["rows"] == [{"name": "Bob"}]


def test_mongodb_cross_field_or_contains_pipeline_returns_both_matches() -> None:
    """Mongo planning should preserve OR between separate field clauses."""

    fake_database = FakeMongoDatabase(
        {
            "users": [
                {"name": "Alice", "email": "alice@example.com"},
                {"name": "Bob", "email": "bob@example.com"},
                {"name": "Charlie", "email": "charlie@example.com"},
            ]
        }
    )
    schema = {
        "collections": {
            "users": {
                "fields": [
                    {"name": "name", "type": "str"},
                    {"name": "email", "type": "str"},
                ]
            }
        }
    }

    query_plan = {
        "operation": "find",
        "collection": "users",
        "match": {
            "$or": [
                {"name": {"$regex": "alice", "$options": "i"}},
                {"email": {"$regex": "bob", "$options": "i"}},
            ]
        },
        "project": {"name": 1, "email": 1},
        "sort": {},
        "limit": 10,
    }
    compiled_query = compile_query(query_plan, "mongodb")
    result = execute_query(compiled_query, "mongodb", fake_database, collection="users")

    assert '"$or"' in compiled_query
    assert result["rows"] == [
        {"name": "Alice", "email": "alice@example.com"},
        {"name": "Bob", "email": "bob@example.com"},
    ]


class FakeMongoDatabase:
    """Very small in-memory MongoDB stand-in for local tests."""

    def __init__(self, collections: dict[str, list[dict[str, object]]]) -> None:
        self._collections = {
            name: FakeMongoCollection(documents)
            for name, documents in collections.items()
        }

    def __getitem__(self, collection_name: str) -> "FakeMongoCollection":
        return self._collections[collection_name]


class FakeMongoCollection:
    """Enough aggregate support to exercise the current compiler and executor."""

    def __init__(self, documents: list[dict[str, object]]) -> None:
        self._documents = [dict(document) for document in documents]

    def aggregate(self, pipeline: list[dict[str, object]]) -> list[dict[str, object]]:
        documents = [dict(document) for document in self._documents]

        for stage in pipeline:
            if "$match" in stage:
                documents = _apply_match(documents, stage["$match"])
                continue

            if "$project" in stage:
                documents = _apply_project(documents, stage["$project"])
                continue

            if "$sort" in stage:
                documents = _apply_sort(documents, stage["$sort"])
                continue

            if "$limit" in stage:
                documents = documents[: int(stage["$limit"])]
                continue

            if "$count" in stage:
                return [{str(stage["$count"]): len(documents)}]

        return documents


def _apply_match(documents: list[dict[str, object]], match: dict[str, object]) -> list[dict[str, object]]:
    """Apply a tiny subset of MongoDB matching for local tests."""

    matched_documents: list[dict[str, object]] = []

    for document in documents:
        if _document_matches(document, match):
            matched_documents.append(document)

    return matched_documents


def _document_matches(document: dict[str, object], match: dict[str, object]) -> bool:
    """Check whether one in-memory document passes the fake match clause."""

    for key, expected_value in match.items():
        if key == "$and" and isinstance(expected_value, list):
            return all(
                _document_matches(document, nested_clause)
                for nested_clause in expected_value
                if isinstance(nested_clause, dict)
            )

        if key == "$or" and isinstance(expected_value, list):
            return any(
                _document_matches(document, nested_clause)
                for nested_clause in expected_value
                if isinstance(nested_clause, dict)
            )

        actual_value = document.get(key)

        if isinstance(expected_value, dict) and "$regex" in expected_value:
            regex_pattern = str(expected_value["$regex"])
            regex_flags = re.IGNORECASE if expected_value.get("$options") == "i" else 0
            if actual_value is None or not re.search(regex_pattern, str(actual_value), regex_flags):
                return False
            continue

        if actual_value != expected_value:
            return False

    return True


def _apply_project(documents: list[dict[str, object]], project: dict[str, object]) -> list[dict[str, object]]:
    """Keep only projected fields for each in-memory document."""

    projected_documents: list[dict[str, object]] = []
    included_fields = [field_name for field_name, include in project.items() if include]

    for document in documents:
        projected_documents.append(
            {
                field_name: document.get(field_name)
                for field_name in included_fields
                if field_name in document
            }
        )

    return projected_documents


def _apply_sort(documents: list[dict[str, object]], sort: dict[str, object]) -> list[dict[str, object]]:
    """Sort in-memory documents by the first requested field."""

    if not sort:
        return documents

    field_name, direction = next(iter(sort.items()))
    reverse = int(direction) < 0

    return sorted(documents, key=lambda document: document.get(field_name), reverse=reverse)
