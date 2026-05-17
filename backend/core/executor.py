"""Helpers for running compiled read-only queries against supported backends."""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import text

MAX_RETURNED_ROWS = 100


def execute_query(
    query: str,
    backend: str,
    connection: Any,
    collection: str | None = None,
) -> dict[str, Any]:
    """Execute a compiled query and return rows, row count, and column names."""

    try:
        if backend in {"sqlite", "postgresql"}:
            return _execute_sql_query(query, connection)

        if backend == "mongodb":
            return _execute_mongodb_query(query, connection, collection)

        raise ValueError(f"Unsupported backend: {backend}")
    except Exception as exc:
        return {
            "error": str(exc),
            "rows": [],
            "row_count": 0,
            "columns": [],
        }


def _execute_sql_query(query: str, connection: Any) -> dict[str, Any]:
    """Run a SQL query with SQLAlchemy and return capped row data."""

    result = connection.execute(text(query))
    all_rows = result.fetchall()
    columns = list(result.keys())
    capped_rows = all_rows[:MAX_RETURNED_ROWS]

    rows = [dict(zip(columns, row)) for row in capped_rows]
    return {
        "rows": rows,
        "row_count": len(rows),
        "columns": columns,
    }


def _execute_mongodb_query(query: str, connection: Any, collection: str | None) -> dict[str, Any]:
    """Run a MongoDB aggregation pipeline and return capped row data."""

    if not collection:
        raise ValueError("MongoDB execution requires a collection name")

    pipeline = json.loads(query)
    if not isinstance(pipeline, list):
        raise ValueError("MongoDB query must be a pipeline list")

    cursor = connection[collection].aggregate(pipeline)
    rows: list[dict[str, Any]] = []

    for document in cursor:
        row = dict(document)
        row.pop("_id", None)
        rows.append(row)

        if len(rows) >= MAX_RETURNED_ROWS:
            break

    columns = list(rows[0].keys()) if rows else []
    return {
        "rows": rows,
        "row_count": len(rows),
        "columns": columns,
    }
