"""Built-in GenAI evaluation runner for SQLite, PostgreSQL, and MongoDB."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

from core.eval_fixtures import (
    cleanup_demo_sqlite_database,
    create_demo_mongo_database,
    create_demo_sqlite_database,
)
from core.ollama_client import get_active_model_name
from core.query_graph import QueryGraphError, run_query_graph
from core.schema_extractor import extract_schema
from tests.mongodb_prompt_matrix import MONGODB_PROMPT_CASES
from tests.sql_prompt_matrix import SQL_PROMPT_CASES

WORKFLOW_NAME = "langgraph"


def run_eval_suite(backend: str, base_dir: Path) -> dict[str, Any]:
    """Run the built-in prompt suite for one backend."""

    if backend in {"sqlite", "postgresql"}:
        return _run_sql_eval_suite(backend, base_dir)

    if backend == "mongodb":
        return _run_mongodb_eval_suite()

    raise ValueError(f"Unsupported backend: {backend}")


def _run_sql_eval_suite(backend: str, base_dir: Path) -> dict[str, Any]:
    """Run the shared SQL prompt matrix against a relational demo database."""

    engine, database_path = create_demo_sqlite_database(base_dir)

    try:
        with engine.connect() as connection:
            database_schema = extract_schema("sqlite", connection)
            case_results = [
                _run_case(
                    case=case,
                    backend=backend,
                    database_schema=database_schema,
                    connection=connection,
                )
                for case in SQL_PROMPT_CASES
                if backend in case["backends"]
            ]
    finally:
        cleanup_demo_sqlite_database(engine, database_path)

    return _build_eval_suite_response(
        backend=backend,
        dataset_source="built-in relational demo fixture",
        case_results=case_results,
    )


def _run_mongodb_eval_suite() -> dict[str, Any]:
    """Run the shared MongoDB prompt matrix against the in-memory demo database."""

    connection = create_demo_mongo_database()
    database_schema = extract_schema("mongodb", connection)
    case_results = [
        _run_case(
            case=case,
            backend="mongodb",
            database_schema=database_schema,
            connection=connection,
        )
        for case in MONGODB_PROMPT_CASES
    ]

    return _build_eval_suite_response(
        backend="mongodb",
        dataset_source="built-in MongoDB-style demo fixture",
        case_results=case_results,
    )


def _run_case(
    case: dict[str, Any],
    backend: str,
    database_schema: dict[str, Any],
    connection: Any,
) -> dict[str, Any]:
    """Run and grade one prompt case through the real query graph."""

    started_at = perf_counter()

    try:
        graph_result = run_query_graph(
            question=str(case["question"]),
            backend=backend,
            database_schema=database_schema,
            connection=connection,
        )
    except QueryGraphError as exc:
        latency_ms = round((perf_counter() - started_at) * 1000, 2)
        return _grade_error_case(case, str(exc), exc.trace, latency_ms)
    except ValueError as exc:
        latency_ms = round((perf_counter() - started_at) * 1000, 2)
        return _grade_error_case(case, str(exc), {"steps": [], "node_count": 0}, latency_ms)

    latency_ms = round((perf_counter() - started_at) * 1000, 2)
    return _grade_success_case(case, graph_result, latency_ms)


def _grade_success_case(case: dict[str, Any], graph_result: dict[str, Any], latency_ms: float) -> dict[str, Any]:
    """Check whether one successful graph run matched the expected output."""

    compiled_query = str(graph_result.get("compiled_query", ""))
    result = graph_result.get("result", {})
    actual_rows = result.get("rows", [])
    actual_row_count = int(result.get("row_count", 0))
    trace = graph_result.get("trace", {"steps": [], "node_count": 0})

    if case["expected"] != "success":
        return {
            "name": case["name"],
            "question": case["question"],
            "expected": case["expected"],
            "status": "Fail",
            "passed": False,
            "message": "The prompt should have been rejected, but the graph produced a query.",
            "compiled_query": compiled_query,
            "row_count": actual_row_count,
            "latency_ms": latency_ms,
            "trace": trace,
        }

    expected_fragments = case.get("sql_fragments", case.get("query_fragments", []))
    missing_fragments = [fragment for fragment in expected_fragments if fragment not in compiled_query]
    rows_match = actual_rows == case.get("rows", [])
    row_count_matches = actual_row_count == int(case.get("row_count", 0))
    passed = not missing_fragments and rows_match and row_count_matches

    if passed:
        message = "Matched the expected compiled query fragments and result rows."
    elif missing_fragments:
        message = f"Missing expected query fragment(s): {', '.join(missing_fragments)}"
    elif not rows_match:
        message = "Returned rows did not match the expected demo result."
    else:
        message = "Returned row count did not match the expected demo result."

    return {
        "name": case["name"],
        "question": case["question"],
        "expected": case["expected"],
        "status": "Pass" if passed else "Fail",
        "passed": passed,
        "message": message,
        "compiled_query": compiled_query,
        "row_count": actual_row_count,
        "latency_ms": latency_ms,
        "trace": trace,
    }


def _grade_error_case(
    case: dict[str, Any],
    error_message: str,
    trace: dict[str, Any],
    latency_ms: float,
) -> dict[str, Any]:
    """Check whether one rejected graph run failed for the expected reason."""

    expected_error = str(case.get("plan", {}).get("error", ""))
    passed = case["expected"] == "error" and expected_error in error_message

    if passed:
        message = error_message
    elif case["expected"] == "error":
        message = f"Expected a different error. Actual: {error_message}"
    else:
        message = f"The prompt should have succeeded, but failed with: {error_message}"

    return {
        "name": case["name"],
        "question": case["question"],
        "expected": case["expected"],
        "status": "Pass" if passed else "Fail",
        "passed": passed,
        "message": message,
        "compiled_query": "",
        "row_count": 0,
        "latency_ms": latency_ms,
        "trace": trace,
    }


def _build_eval_suite_response(
    backend: str,
    dataset_source: str,
    case_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize one full eval run."""

    total_cases = len(case_results)
    passed_cases = sum(1 for case in case_results if case["passed"])
    failed_cases = total_cases - passed_cases
    pass_rate = round((passed_cases / total_cases) * 100, 2) if total_cases else 0.0
    avg_latency_ms = round(
        sum(float(case["latency_ms"]) for case in case_results) / total_cases,
        2,
    ) if total_cases else 0.0

    return {
        "backend": backend,
        "workflow": WORKFLOW_NAME,
        "model": get_active_model_name(),
        "dataset_source": dataset_source,
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "pass_rate": pass_rate,
        "avg_latency_ms": avg_latency_ms,
        "cases": case_results,
    }
