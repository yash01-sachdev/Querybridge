"""Helpers for rejecting unsafe generated queries before execution."""

from __future__ import annotations

import json
import re
from typing import Any

import sqlparse

FORBIDDEN_SQL_KEYWORDS = [
    "drop",
    "delete",
    "update",
    "insert",
    "alter",
    "truncate",
    "create",
    "replace",
    "grant",
    "revoke",
]

FORBIDDEN_MONGODB_STAGES = {"$out", "$merge", "$delete"}
FORBIDDEN_MONGODB_OPERATORS = {"$where"}


def check_safety(query: str, backend: str, schema: dict[str, Any]) -> tuple[bool, str]:
    """Return whether a compiled query is safe to run."""

    if backend in {"sqlite", "postgresql"}:
        return _check_sql_safety(query)

    if backend == "mongodb":
        return _check_mongodb_safety(query)

    return False, f"Unsupported backend: {backend}"


def _check_sql_safety(query: str) -> tuple[bool, str]:
    """Apply read-only safety rules to a SQL query."""

    statements = [statement for statement in sqlparse.parse(query) if str(statement).strip()]
    if not statements:
        return False, "Query is empty"

    if len(statements) != 1:
        return False, "Only one SQL statement is allowed"

    statement_type = statements[0].get_type()
    if statement_type != "SELECT":
        return False, "Only SELECT queries are allowed"

    lowered_query = query.lower()

    for keyword in FORBIDDEN_SQL_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", lowered_query):
            return False, f"Query contains forbidden keyword: {keyword}"

    if not re.search(r"\blimit\b", lowered_query) and "count(" not in lowered_query:
        return False, "Query must include a LIMIT clause"

    return True, ""


def _check_mongodb_safety(query: str) -> tuple[bool, str]:
    """Apply read-only safety rules to a MongoDB pipeline."""

    try:
        parsed_query = json.loads(query)
    except json.JSONDecodeError as exc:
        return False, f"Invalid MongoDB query JSON: {exc}"

    stages = parsed_query if isinstance(parsed_query, list) else [parsed_query]

    if not stages:
        return False, "MongoDB pipeline is empty"

    for stage in stages:
        if not isinstance(stage, dict):
            return False, "MongoDB pipeline must contain JSON objects"

        for stage_name in stage:
            if stage_name in FORBIDDEN_MONGODB_STAGES:
                return False, "Destructive pipeline stage not allowed"

        if _contains_forbidden_mongodb_operator(stage):
            return False, "MongoDB query contains a forbidden operator"

    if not any("$limit" in stage or "$count" in stage for stage in stages):
        return False, "MongoDB pipeline must include $limit or $count"

    return True, ""


def _contains_forbidden_mongodb_operator(value: Any) -> bool:
    """Return whether a MongoDB stage contains a forbidden operator anywhere inside it."""

    if isinstance(value, dict):
        for key, nested_value in value.items():
            if key in FORBIDDEN_MONGODB_OPERATORS:
                return True

            if _contains_forbidden_mongodb_operator(nested_value):
                return True

    if isinstance(value, list):
        return any(_contains_forbidden_mongodb_operator(item) for item in value)

    return False
