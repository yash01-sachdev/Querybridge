"""Route-level SQL prompt matrix tests for the LangGraph + Ollama workflow."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException

from core.validator import validate_query as real_validate_query
from models.request import QueryRequest
from routes.query import create_query
from tests.sql_prompt_matrix import SQL_PROMPT_CASES

SUCCESS_CASES = [case for case in SQL_PROMPT_CASES if case["expected"] == "success"]
ERROR_CASES = [case for case in SQL_PROMPT_CASES if case["expected"] == "error"]


@pytest.fixture()
def use_sqlite_for_postgresql(
    sqlite_backend: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the local SQLite test database to exercise the PostgreSQL route path."""

    engine = sqlite_backend["engine"]

    def fake_open_backend_connection(backend: str, connection_overrides: object = None):
        connection = engine.connect()

        def close_connection() -> None:
            connection.close()

        return connection, close_connection

    monkeypatch.setattr("routes.query.open_backend_connection", fake_open_backend_connection)


@pytest.mark.parametrize("case", SUCCESS_CASES, ids=[case["name"] for case in SUCCESS_CASES])
@pytest.mark.parametrize("backend", ["sqlite", "postgresql"])
def test_sql_prompt_matrix_success_cases(
    case: dict[str, Any],
    backend: str,
    sqlite_backend: dict[str, object],
    use_sqlite_for_postgresql: None,
) -> None:
    """Supported SQL prompts should plan, compile, validate, execute, and explain."""

    if backend not in case["backends"]:
        pytest.skip(f"{case['name']} does not apply to {backend}")

    request = _build_request(
        question=case["question"],
        backend=backend,
        sqlite_path=str(sqlite_backend["path"]),
    )
    response = create_query(request)

    assert response.workflow == "langgraph"
    assert response.model == "mock-ollama"
    assert response.trace.node_count >= 6
    assert response.trace.steps[0].name == "select_schema"
    assert response.safety_check.passed is True
    assert response.validation_check.passed is True
    assert response.result.rows == case["rows"]
    assert response.result.row_count == case["row_count"]

    for sql_fragment in case["sql_fragments"]:
        assert sql_fragment in response.compiled_query


@pytest.mark.parametrize("case", ERROR_CASES, ids=[case["name"] for case in ERROR_CASES])
@pytest.mark.parametrize("backend", ["sqlite", "postgresql"])
def test_sql_prompt_matrix_error_cases(
    case: dict[str, Any],
    backend: str,
    sqlite_backend: dict[str, object],
    use_sqlite_for_postgresql: None,
) -> None:
    """Unsupported or destructive SQL prompts should fail clearly at the API boundary."""

    if backend not in case["backends"]:
        pytest.skip(f"{case['name']} does not apply to {backend}")

    request = _build_request(
        question=case["question"],
        backend=backend,
        sqlite_path=str(sqlite_backend["path"]),
    )

    with pytest.raises(HTTPException) as exc_info:
        create_query(request)

    assert exc_info.value.status_code == 400
    assert str(case["plan"]["error"]) in str(exc_info.value.detail)


def test_langgraph_route_semantic_check_repairs_invalid_plan_once(
    sqlite_backend: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The graph should let Ollama semantically repair one invalid plan."""

    call_count = {"value": 0}

    def fake_generate_json(prompt: str) -> dict[str, Any]:
        if "Return this schema selection shape" in prompt:
            return {"tables": ["users"]}

        if "Semantic checklist:" in prompt or "Original question" in prompt:
            return {
                "passed": False,
                "reason": "users.username is not in the selected schema.",
                "repaired_plan": {
                    "tables": ["users"],
                    "fields": ["users.email"],
                    "filters": [],
                    "joins": [],
                    "aggregations": [],
                    "group_by": [],
                    "order_by": [],
                    "limit": 10,
                },
            } if "Semantic checklist:" in prompt else {
                "tables": ["users"],
                "fields": ["users.email"],
                "filters": [],
                "joins": [],
                "aggregations": [],
                "group_by": [],
                "order_by": [],
                "limit": 10,
            }

        call_count["value"] += 1
        if call_count["value"] == 1:
            return {
                "tables": ["users"],
                "fields": ["users.username"],
                "filters": [],
                "joins": [],
                "aggregations": [],
                "group_by": [],
                "order_by": [],
                "limit": 10,
            }

        raise AssertionError("Planner should not be called more than once for this repair test.")

    monkeypatch.setattr("core.query_graph.generate_json", fake_generate_json)
    monkeypatch.setattr("core.query_graph.generate_text", lambda prompt: "Mock repaired explanation.")
    monkeypatch.setattr("core.query_graph.get_active_model_name", lambda: "mock-ollama")

    response = create_query(
        QueryRequest(
            question="show user emails",
            backend="sqlite",
            connection={"sqlite_path": str(sqlite_backend["path"])},
        )
    )

    assert response.repaired is True
    assert response.repair_attempts == 1
    assert response.compiled_query == "SELECT users.email FROM users LIMIT 10"
    assert response.result.rows[0]["email"] == "alice@example.com"
    assert response.trace.steps[2].name == "semantic_check"
    assert response.trace.steps[2].status == "repaired"


def test_langgraph_route_semantic_check_repairs_contradictory_pass_response(
    sqlite_backend: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A semantic response that says 'passed' but describes a wrong plan should trigger repair."""

    def fake_generate_json(prompt: str, candidate_models: list[str] | None = None) -> dict[str, Any]:
        if "Return this schema selection shape" in prompt:
            return {"tables": ["users"]}

        if "Semantic checklist:" in prompt:
            return {
                "passed": True,
                "reason": "The plan does not filter users whose name does not contain 'ali'. It should include a filter condition to exclude such users.",
                "repaired_plan": None,
            }

        if "Original question" in prompt:
            return {
                "tables": ["users"],
                "fields": ["users.name"],
                "filters": [{"field": "users.name", "operator": "NOT_CONTAINS", "value": "ali"}],
                "joins": [],
                "aggregations": [],
                "group_by": [],
                "order_by": [{"field": "users.name", "direction": "ASC"}],
                "limit": 10,
            }

        return {
            "tables": ["users"],
            "fields": [],
            "filters": [],
            "joins": [],
            "aggregations": [],
            "group_by": [],
            "order_by": [],
            "limit": 10,
        }

    monkeypatch.setattr("core.query_graph.generate_json", fake_generate_json)
    monkeypatch.setattr("core.query_graph.generate_text", lambda prompt: "Mock contradiction explanation.")
    monkeypatch.setattr("core.query_graph.get_active_model_name", lambda: "mock-ollama")

    response = create_query(
        QueryRequest(
            question="show users where name does not contain ali",
            backend="sqlite",
            connection={"sqlite_path": str(sqlite_backend["path"])},
        )
    )

    assert response.repaired is True
    assert response.repair_attempts == 1
    assert "NOT LOWER(users.name) LIKE '%ali%'" in response.compiled_query
    assert response.result.rows == [{"name": "Bob"}, {"name": "Charlie"}]
    assert response.trace.steps[2].name == "semantic_check"
    assert response.trace.steps[2].status == "repaired"


def test_langgraph_route_stays_model_first_by_default_for_real_model_names(
    sqlite_backend: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real Ollama model name alone should not automatically switch planning to the Python fast path."""

    def fake_generate_json(prompt: str, candidate_models: list[str] | None = None) -> dict[str, Any]:
        if "Return this schema selection shape" in prompt:
            return {"tables": ["users"]}

        if "Semantic checklist:" in prompt:
            return {
                "passed": True,
                "reason": "The plan matches the question and schema.",
                "repaired_plan": None,
            }

        return {
            "tables": ["users"],
            "fields": ["users.email"],
            "filters": [],
            "joins": [],
            "aggregations": [],
            "group_by": [],
            "order_by": [],
            "limit": 10,
        }

    monkeypatch.setattr("core.query_graph.generate_json", fake_generate_json)
    monkeypatch.setattr("core.query_graph.generate_text", lambda prompt: "Mock model-first explanation.")
    monkeypatch.setattr("core.query_graph.get_active_model_name", lambda: "qwen2.5:3b")

    response = create_query(
        QueryRequest(
            question="show user emails",
            backend="sqlite",
            connection={"sqlite_path": str(sqlite_backend["path"])},
        )
    )

    assert response.compiled_query == "SELECT users.email FROM users LIMIT 10"
    assert response.trace.steps[0].details["selection_mode"] == "python"
    assert response.trace.steps[1].details["planning_mode"] == "model"


@pytest.mark.parametrize("placeholder_message", ["short clear reason", "actual specific reason"])
def test_langgraph_route_retries_when_model_returns_placeholder_error(
    sqlite_backend: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    placeholder_message: str,
) -> None:
    """Placeholder model errors should get one stricter model retry, not local fallback."""

    call_count = {"value": 0}

    def fake_generate_json(prompt: str) -> dict[str, Any]:
        if "Return this schema selection shape" in prompt:
            return {"tables": ["users"]}

        if "Semantic checklist:" in prompt:
            return {
                "passed": True,
                "reason": "The plan matches the question and schema.",
                "repaired_plan": None,
            }

        call_count["value"] += 1
        if call_count["value"] == 1:
            return {"error": placeholder_message}

        return {
            "tables": ["users"],
            "fields": ["users.name", "users.email"],
            "filters": [
                {"field": "users.name", "operator": "CONTAINS", "value": "alice"},
                {
                    "field": "users.email",
                    "operator": "CONTAINS",
                    "value": "bob",
                    "combine_with_previous": "OR",
                },
            ],
            "joins": [],
            "aggregations": [],
            "group_by": [],
            "order_by": [],
            "limit": 10,
        }

    monkeypatch.setattr("core.query_graph.generate_json", fake_generate_json)
    monkeypatch.setattr("core.query_graph.generate_text", lambda prompt: "Mock retry explanation.")
    monkeypatch.setattr("core.query_graph.get_active_model_name", lambda: "mock-ollama")

    response = create_query(
        QueryRequest(
            question="show users where name contains alice or email contains bob",
            backend="sqlite",
            connection={"sqlite_path": str(sqlite_backend["path"])},
        )
    )

    assert response.compiled_query
    assert "LOWER(users.name)" in response.compiled_query
    assert " OR " in response.compiled_query
    assert response.result.rows == [
        {"name": "Alice", "email": "alice@example.com"},
        {"name": "Bob", "email": "bob@example.com"},
    ]
    assert response.trace.steps[1].name == "plan_query"
    assert response.trace.steps[1].details["planning_attempts"] == 2


def test_langgraph_route_replaces_placeholder_failure_with_concrete_message(
    sqlite_backend: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated placeholder planning errors should surface a real backend message."""

    call_count = {"value": 0}

    def fake_generate_json(prompt: str) -> dict[str, Any]:
        if "Return this schema selection shape" in prompt:
            return {"tables": ["users"]}

        call_count["value"] += 1
        if call_count["value"] <= 2:
            return {"error": "actual specific reason"}

        raise AssertionError("Semantic review should not run when planning never succeeds.")

    monkeypatch.setattr("core.query_graph.generate_json", fake_generate_json)
    monkeypatch.setattr("core.query_graph.generate_text", lambda prompt: "Mock failure explanation.")
    monkeypatch.setattr("core.query_graph.get_active_model_name", lambda: "mock-ollama")

    with pytest.raises(HTTPException) as exc_info:
        create_query(
            QueryRequest(
                question="how many users",
                backend="sqlite",
                connection={"sqlite_path": str(sqlite_backend["path"])},
            )
        )

    assert exc_info.value.status_code == 400
    assert isinstance(exc_info.value.detail, dict)
    assert exc_info.value.detail["message"] == (
        "Ollama did not return a usable query plan after retrying. "
        "It kept returning placeholder text instead of a real reason."
    )
    assert exc_info.value.detail["trace"]["steps"][1]["details"]["reason"] == (
        "Ollama did not return a usable query plan after retrying. "
        "It kept returning placeholder text instead of a real reason."
    )
    assert exc_info.value.detail["trace"]["steps"][1]["details"]["planning_attempts"] == 2


def test_langgraph_route_canonicalizes_plain_count_plan_before_compile(
    sqlite_backend: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plain count questions should not leak row fields into the compiled SQL or result table."""

    def fake_generate_json(prompt: str) -> dict[str, Any]:
        if "Return this schema selection shape" in prompt:
            return {"tables": ["users", "orders"]}

        if "Semantic checklist:" in prompt:
            return {
                "passed": True,
                "reason": "The plan counts users but still includes row fields.",
                "repaired_plan": None,
            }

        return {
            "tables": ["users"],
            "fields": ["*"],
            "filters": [],
            "joins": [],
            "aggregations": [
                {"function": "COUNT", "field": "users.id", "alias": "count"},
            ],
            "group_by": [],
            "order_by": [],
            "limit": 10,
        }

    monkeypatch.setattr("core.query_graph.generate_json", fake_generate_json)
    monkeypatch.setattr("core.query_graph.generate_text", lambda prompt: "Mock count explanation.")
    monkeypatch.setattr("core.query_graph.get_active_model_name", lambda: "mock-ollama")

    response = create_query(
        QueryRequest(
            question="how many users",
            backend="sqlite",
            connection={"sqlite_path": str(sqlite_backend["path"])},
        )
    )

    assert response.query_plan["fields"] == []
    assert response.query_plan["aggregations"] == [{"function": "COUNT", "field": "*", "alias": "count"}]
    assert response.compiled_query == "SELECT COUNT(*) AS count FROM users"
    assert response.result.columns == ["count"]
    assert response.result.rows == [{"count": 3}]


def test_langgraph_route_repairs_grouped_average_plan_after_validation_scope_error(
    sqlite_backend: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Schema-aware canonicalization should fix grouped averages before a bad retry loop starts."""

    def fake_generate_json(prompt: str) -> dict[str, Any]:
        if "Return this schema selection shape" in prompt:
            return {"tables": ["users", "orders"]}

        if "Semantic checklist:" in prompt:
            return {
                "passed": True,
                "reason": "The plan matches the question and selected schema.",
                "repaired_plan": None,
            }

        return {
            "tables": ["users"],
            "fields": [],
            "filters": [],
            "joins": [],
            "aggregations": [
                {"function": "AVG", "field": "amount", "alias": "average_amount"},
            ],
            "group_by": ["user_id"],
            "order_by": [],
            "limit": 10,
        }

    monkeypatch.setattr("core.query_graph.generate_json", fake_generate_json)
    monkeypatch.setattr("core.query_graph.generate_text", lambda prompt: "Mock grouped average explanation.")
    monkeypatch.setattr("core.query_graph.get_active_model_name", lambda: "mock-ollama")

    response = create_query(
        QueryRequest(
            question="average order amount by user",
            backend="sqlite",
            connection={"sqlite_path": str(sqlite_backend["path"])},
        )
    )

    assert response.repaired is False
    assert response.repair_attempts == 0
    assert "AVG(orders.amount) AS average_amount" in response.compiled_query
    assert "JOIN orders ON orders.user_id = users.id" in response.compiled_query
    assert "GROUP BY users.name" in response.compiled_query
    assert response.result.rows == [
        {"name": "Alice", "average_amount": 210.25},
        {"name": "Bob", "average_amount": 60.0},
        {"name": "Charlie", "average_amount": 90.0},
    ]


def test_langgraph_route_repairs_execution_scope_error_with_python_before_model_retry(
    sqlite_backend: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canonicalization should prevent the old execution-time scope error from ever reaching model repair."""

    def fake_generate_json(prompt: str) -> dict[str, Any]:
        if "Return this schema selection shape" in prompt:
            return {"tables": ["users", "orders"]}

        if "Semantic checklist:" in prompt:
            return {
                "passed": True,
                "reason": "The plan matches the question and selected schema.",
                "repaired_plan": None,
            }

        if "Original question" in prompt:
            raise AssertionError("Model repair should not run when canonicalization already fixed the plan.")

        return {
            "tables": ["users"],
            "fields": [],
            "filters": [],
            "joins": [],
            "aggregations": [
                {"function": "AVG", "field": "orders.amount", "alias": "average_amount"},
            ],
            "group_by": ["users.id"],
            "order_by": [],
            "limit": 10,
        }

    def fake_validate_query(query: str, backend: str, schema: dict[str, Any]) -> tuple[bool, str]:
        if "SELECT AVG(orders.amount) AS average_amount FROM users GROUP BY users.id LIMIT 10" in query:
            return True, ""

        return real_validate_query(query, backend, schema)

    monkeypatch.setattr("core.query_graph.generate_json", fake_generate_json)
    monkeypatch.setattr("core.query_graph.generate_text", lambda prompt: "Mock grouped average explanation.")
    monkeypatch.setattr("core.query_graph.get_active_model_name", lambda: "mock-ollama")
    monkeypatch.setattr("core.query_graph.validate_query", fake_validate_query)

    response = create_query(
        QueryRequest(
            question="average order amount by user",
            backend="sqlite",
            connection={"sqlite_path": str(sqlite_backend["path"])},
        )
    )

    assert response.repaired is False
    assert response.repair_attempts == 0
    assert "AVG(orders.amount) AS average_amount" in response.compiled_query
    assert "JOIN orders ON orders.user_id = users.id" in response.compiled_query
    assert "GROUP BY users.name" in response.compiled_query
    assert response.result.rows == [
        {"name": "Alice", "average_amount": 210.25},
        {"name": "Bob", "average_amount": 60.0},
        {"name": "Charlie", "average_amount": 90.0},
    ]


def test_langgraph_route_repairs_contradictory_semantic_echoed_plan(
    sqlite_backend: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A contradictory semantic response must not pass just because the model echoed the current plan."""

    def fake_generate_json(prompt: str, candidate_models: list[str] | None = None) -> dict[str, Any]:
        if "Return this schema selection shape" in prompt:
            return {"tables": ["users"]}

        if "Semantic checklist:" in prompt:
            return {
                "passed": True,
                "reason": "The plan does not include a filter to exclude users with 'ali' in their name.",
                "plan": {
                    "tables": ["users"],
                    "fields": ["users.name"],
                    "filters": [],
                    "joins": [],
                    "aggregations": [],
                    "group_by": [],
                    "order_by": [],
                    "limit": 10,
                },
            }

        if "Original question" in prompt:
            return {
                "tables": ["users"],
                "fields": ["users.name"],
                "filters": [{"field": "users.name", "operator": "NOT_CONTAINS", "value": "ali"}],
                "joins": [],
                "aggregations": [],
                "group_by": [],
                "order_by": [{"field": "users.name", "direction": "ASC"}],
                "limit": 10,
            }

        return {
            "tables": ["users"],
            "fields": ["users.name"],
            "filters": [],
            "joins": [],
            "aggregations": [],
            "group_by": [],
            "order_by": [],
            "limit": 10,
        }

    monkeypatch.setattr("core.query_graph.generate_json", fake_generate_json)
    monkeypatch.setattr("core.query_graph.generate_text", lambda prompt: "Mock echoed-plan explanation.")
    monkeypatch.setattr("core.query_graph.get_active_model_name", lambda: "mock-ollama")

    response = create_query(
        QueryRequest(
            question="show users where name does not contain ali",
            backend="sqlite",
            connection={"sqlite_path": str(sqlite_backend["path"])},
        )
    )

    assert response.repaired is True
    assert response.repair_attempts == 1
    assert "NOT LOWER(users.name) LIKE '%ali%'" in response.compiled_query
    assert response.result.rows == [{"name": "Bob"}, {"name": "Charlie"}]
    assert response.trace.steps[2].status == "repaired"


def test_langgraph_route_canonicalizes_model_alias_references_before_validation(
    sqlite_backend: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Alias-like model references such as o.amount and u.name should map back to real schema columns."""

    def fake_generate_json(prompt: str, candidate_models: list[str] | None = None) -> dict[str, Any]:
        if "Semantic checklist:" in prompt:
            return {
                "passed": True,
                "reason": "The plan matches the question and schema.",
                "repaired_plan": None,
            }

        return {
            "tables": ["orders", "users"],
            "fields": ["o.amount", "u.name"],
            "filters": [],
            "joins": [{"left": "o.user_id", "right": "u.id"}],
            "aggregations": [],
            "group_by": [],
            "order_by": [{"field": "o.id", "direction": "ASC"}],
            "limit": 10,
        }

    monkeypatch.setattr("core.query_graph.generate_json", fake_generate_json)
    monkeypatch.setattr("core.query_graph.generate_text", lambda prompt: "Mock alias explanation.")
    monkeypatch.setattr("core.query_graph.get_active_model_name", lambda: "mock-ollama")

    response = create_query(
        QueryRequest(
            question="show order amounts with user names",
            backend="sqlite",
            connection={"sqlite_path": str(sqlite_backend["path"])},
        )
    )

    assert "orders.amount" in response.compiled_query
    assert "users.name" in response.compiled_query
    assert "JOIN users ON orders.user_id = users.id" in response.compiled_query
    assert response.validation_check.passed is True
    assert response.result.row_count >= 1


def test_langgraph_route_skips_model_semantic_review_for_simple_projection(
    sqlite_backend: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simple single-table projections should not spend an extra Ollama round-trip on semantic review."""

    def fake_generate_json(prompt: str, candidate_models: list[str] | None = None) -> dict[str, Any]:
        if "Semantic checklist:" in prompt:
            raise AssertionError("Simple projections should skip model semantic review.")

        return {
            "tables": ["users"],
            "fields": ["users.email"],
            "filters": [],
            "joins": [],
            "aggregations": [],
            "group_by": [],
            "order_by": [],
            "limit": 10,
        }

    monkeypatch.setattr("core.query_graph.generate_json", fake_generate_json)
    monkeypatch.setattr("core.query_graph.generate_text", lambda prompt: "Mock projection explanation.")
    monkeypatch.setattr("core.query_graph.get_active_model_name", lambda: "qwen2.5:3b")

    response = create_query(
        QueryRequest(
            question="show user emails",
            backend="sqlite",
            connection={"sqlite_path": str(sqlite_backend["path"])},
        )
    )

    assert response.repaired is False
    assert response.compiled_query == "SELECT users.email FROM users LIMIT 10"
    assert response.trace.steps[2].details["semantic_mode"] == "python-guardrails"


def test_langgraph_route_skips_model_semantic_review_for_clean_grouped_average(
    sqlite_backend: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A structurally clean one-join grouped average should skip redundant semantic review."""

    def fake_generate_json(prompt: str, candidate_models: list[str] | None = None) -> dict[str, Any]:
        if "Semantic checklist:" in prompt:
            raise AssertionError("Clean grouped averages should skip model semantic review.")

        return {
            "tables": ["orders", "users"],
            "fields": ["users.name"],
            "filters": [],
            "joins": [{"left": "orders.user_id", "right": "users.id"}],
            "aggregations": [{"function": "AVG", "field": "orders.amount", "alias": "average_amount"}],
            "group_by": ["users.name"],
            "order_by": [],
            "limit": 10,
        }

    monkeypatch.setattr("core.query_graph.generate_json", fake_generate_json)
    monkeypatch.setattr("core.query_graph.generate_text", lambda prompt: "Mock grouped average explanation.")
    monkeypatch.setattr("core.query_graph.get_active_model_name", lambda: "qwen2.5:3b")

    response = create_query(
        QueryRequest(
            question="average order amount by user",
            backend="sqlite",
            connection={"sqlite_path": str(sqlite_backend["path"])},
        )
    )

    assert response.repaired is False
    assert response.compiled_query == (
        "SELECT users.name, AVG(orders.amount) AS average_amount "
        "FROM orders JOIN users ON orders.user_id = users.id "
        "GROUP BY users.name LIMIT 10"
    )
    assert response.trace.steps[2].details["semantic_mode"] == "python-guardrails"


def test_langgraph_route_drops_literal_join_filters_from_model_repairs(
    sqlite_backend: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model output should not keep a join condition as a bogus string-literal WHERE filter."""

    call_count = {"value": 0}

    def fake_generate_json(prompt: str, candidate_models: list[str] | None = None) -> dict[str, Any]:
        if "Semantic checklist:" in prompt:
            return {
                "passed": True,
                "reason": "The plan matches the question and schema.",
                "repaired_plan": None,
            }

        call_count["value"] += 1
        if call_count["value"] == 1:
            return {
                "tables": ["orders", "users"],
                "fields": ["users.name"],
                "filters": [],
                "joins": [{"left": "orders.user_id", "right": "users.id"}],
                "aggregations": [{"function": "AVG", "field": "orders.amount", "alias": "average_amount"}],
                "group_by": ["users.name"],
                "order_by": [],
                "limit": 10,
            }

        return {
            "tables": ["orders", "users"],
            "fields": ["users.name"],
            "filters": [{"field": "orders.user_id", "operator": "=", "value": "users.id"}],
            "joins": [{"left": "orders.user_id", "right": "users.id"}],
            "aggregations": [{"function": "AVG", "field": "orders.amount", "alias": "average_amount"}],
            "group_by": ["users.name"],
            "order_by": [],
            "limit": 10,
        }

    monkeypatch.setattr("core.query_graph.generate_json", fake_generate_json)
    monkeypatch.setattr("core.query_graph.generate_text", lambda prompt: "Mock repaired average explanation.")
    monkeypatch.setattr("core.query_graph.get_active_model_name", lambda: "mock-ollama")

    response = create_query(
        QueryRequest(
            question="average order amount by user",
            backend="sqlite",
            connection={"sqlite_path": str(sqlite_backend["path"])},
        )
    )

    assert response.compiled_query == (
        "SELECT users.name, AVG(orders.amount) AS average_amount "
        "FROM orders JOIN users ON orders.user_id = users.id "
        "GROUP BY users.name LIMIT 10"
    )
    assert response.result.rows == [
        {"name": "Alice", "average_amount": 210.25},
        {"name": "Bob", "average_amount": 60.0},
        {"name": "Charlie", "average_amount": 90.0},
    ]


def test_langgraph_route_uses_python_fast_path_for_grouped_average(
    sqlite_backend: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The optional Python fast path should still work when explicitly enabled."""

    def fail_generate_json(prompt: str) -> dict[str, Any]:
        raise AssertionError("Python fast path should skip structured model planning for this prompt.")

    monkeypatch.setattr("core.query_graph.generate_json", fail_generate_json)
    monkeypatch.setattr(
        "core.query_graph.generate_text",
        lambda prompt: (_ for _ in ()).throw(
            AssertionError("Python fast path should also skip model explanations for this prompt.")
        ),
    )
    monkeypatch.setattr("core.query_graph.get_active_model_name", lambda: "qwen2.5:3b")
    monkeypatch.setattr("core.query_graph._should_use_python_fast_path", lambda: True)

    response = create_query(
        QueryRequest(
            question="average order amount by user",
            backend="sqlite",
            connection={"sqlite_path": str(sqlite_backend["path"])},
        )
    )

    assert response.repaired is False
    assert "SELECT users.name, AVG(orders.amount) AS avg_amount" in response.compiled_query
    assert "FROM orders JOIN users ON orders.user_id = users.id" in response.compiled_query
    assert "GROUP BY users.name" in response.compiled_query
    assert "ORDER BY users.name ASC" in response.compiled_query
    assert response.compiled_query.endswith("LIMIT 10")
    assert response.result.rows == [
        {"name": "Alice", "avg_amount": 210.25},
        {"name": "Bob", "avg_amount": 60.0},
        {"name": "Charlie", "avg_amount": 90.0},
    ]
    assert response.trace.steps[0].details["selection_mode"] == "python"
    assert response.trace.steps[1].details["planning_mode"] == "python-fast-path"
    assert response.trace.steps[2].details["semantic_mode"] == "python-fast-path"


def test_langgraph_route_uses_python_fast_path_for_not_contains_filter(
    sqlite_backend: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The optional Python fast path should still work when explicitly enabled."""

    def fail_generate_json(prompt: str, candidate_models: list[str] | None = None) -> dict[str, Any]:
        raise AssertionError("Python fast path should skip structured model planning for this prompt.")

    monkeypatch.setattr("core.query_graph.generate_json", fail_generate_json)
    monkeypatch.setattr(
        "core.query_graph.generate_text",
        lambda prompt: (_ for _ in ()).throw(
            AssertionError("Python fast path should also skip model explanations for this prompt.")
        ),
    )
    monkeypatch.setattr("core.query_graph.get_active_model_name", lambda: "qwen2.5:3b")
    monkeypatch.setattr("core.query_graph._should_use_python_fast_path", lambda: True)

    response = create_query(
        QueryRequest(
            question="show users where name does not contain ali",
            backend="sqlite",
            connection={"sqlite_path": str(sqlite_backend["path"])},
        )
    )

    assert response.repaired is False
    assert "NOT LOWER(users.name) LIKE '%ali%'" in response.compiled_query
    assert response.result.rows == [{"name": "Bob"}, {"name": "Charlie"}]
    assert response.trace.steps[0].details["selection_mode"] == "python"
    assert response.trace.steps[1].details["planning_mode"] == "python-fast-path"
    assert response.trace.steps[2].details["semantic_mode"] == "python-fast-path"


def _build_request(question: str, backend: str, sqlite_path: str) -> QueryRequest:
    """Create one request object for either SQLite or PostgreSQL test paths."""

    if backend == "sqlite":
        return QueryRequest(
            question=question,
            backend=backend,
            connection={"sqlite_path": sqlite_path},
        )

    return QueryRequest(question=question, backend=backend)
