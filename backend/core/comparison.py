"""Helpers for building side-by-side query previews across all backends."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from core.query_graph import QueryGraphError, plan_query_preview

SUPPORTED_BACKENDS = ("sqlite", "postgresql", "mongodb")
LIVE_SCHEMA_SOURCE = "linked live database"
DEMO_SCHEMA_SOURCE = "built-in learning schema"

DEMO_COMPARE_SCHEMA: dict[str, Any] = {
    "tables": {
        "users": {
            "columns": [
                {"name": "id", "type": "INTEGER", "primary_key": True},
                {"name": "name", "type": "TEXT", "primary_key": False},
                {"name": "email", "type": "TEXT", "primary_key": False},
                {"name": "status", "type": "TEXT", "primary_key": False},
                {"name": "created_at", "type": "TEXT", "primary_key": False},
            ],
            "foreign_keys": [],
        },
        "orders": {
            "columns": [
                {"name": "id", "type": "INTEGER", "primary_key": True},
                {"name": "user_id", "type": "INTEGER", "primary_key": False},
                {"name": "product_id", "type": "INTEGER", "primary_key": False},
                {"name": "amount", "type": "REAL", "primary_key": False},
                {"name": "status", "type": "TEXT", "primary_key": False},
                {"name": "created_at", "type": "TEXT", "primary_key": False},
            ],
            "foreign_keys": [
                {"column": "user_id", "references": "users.id"},
                {"column": "product_id", "references": "products.id"},
            ],
        },
        "products": {
            "columns": [
                {"name": "id", "type": "INTEGER", "primary_key": True},
                {"name": "title", "type": "TEXT", "primary_key": False},
                {"name": "category", "type": "TEXT", "primary_key": False},
                {"name": "price", "type": "REAL", "primary_key": False},
            ],
            "foreign_keys": [],
        },
        "subscriptions": {
            "columns": [
                {"name": "id", "type": "INTEGER", "primary_key": True},
                {"name": "user_id", "type": "INTEGER", "primary_key": False},
                {"name": "plan", "type": "TEXT", "primary_key": False},
                {"name": "status", "type": "TEXT", "primary_key": False},
                {"name": "started_at", "type": "TEXT", "primary_key": False},
            ],
            "foreign_keys": [{"column": "user_id", "references": "users.id"}],
        },
    }
}


def get_demo_compare_schema() -> dict[str, Any]:
    """Return the raw built-in relational demo schema used in compare mode."""

    return deepcopy(DEMO_COMPARE_SCHEMA)


def build_comparison_queries(
    question: str,
    backend_inputs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build model-backed preview queries for all supported backends."""

    comparisons: list[dict[str, Any]] = []

    for backend in SUPPORTED_BACKENDS:
        backend_input = backend_inputs.get(backend, {})
        schema_source = str(backend_input.get("schema_source", DEMO_SCHEMA_SOURCE))
        backend_error = backend_input.get("error")

        if isinstance(backend_error, str) and backend_error:
            comparisons.append(
                {
                    "backend": backend,
                    "schema_source": schema_source,
                    "workflow": "langgraph",
                    "model": "",
                    "query_plan": {},
                    "compiled_query": "",
                    "trace": {"steps": [], "node_count": 0},
                    "message": backend_error,
                    "success": False,
                }
            )
            continue

        target_schema = backend_input.get("schema")
        if not isinstance(target_schema, dict):
            target_schema = get_demo_compare_schema_for_backend(backend)
            schema_source = DEMO_SCHEMA_SOURCE

        try:
            preview = plan_query_preview(question, backend, target_schema)
        except QueryGraphError as exc:
            comparisons.append(
                {
                    "backend": backend,
                    "schema_source": schema_source,
                    "workflow": "langgraph",
                    "model": "",
                    "query_plan": {},
                    "compiled_query": "",
                    "trace": exc.trace,
                    "message": str(exc),
                    "success": False,
                }
            )
            continue

        comparisons.append(
            {
                "backend": backend,
                "schema_source": schema_source,
                "workflow": str(preview["workflow"]),
                "model": str(preview["model"]),
                "query_plan": preview["query_plan"],
                "compiled_query": str(preview["compiled_query"]),
                "trace": preview["trace"],
                "message": f"Model-backed preview generated for {backend}.",
                "success": True,
            }
        )

    return comparisons


def get_demo_compare_schema_for_backend(backend: str) -> dict[str, Any]:
    """Return the built-in demo schema already shaped for one backend."""

    base_schema = get_demo_compare_schema()
    return _adapt_schema_for_backend(base_schema, backend)


def _adapt_schema_for_backend(schema: dict[str, Any], backend: str) -> dict[str, Any]:
    """Convert the schema shape when a target backend uses a different structure."""

    if backend == "mongodb":
        return _convert_schema_to_mongodb(schema)

    return _convert_schema_to_sql(schema)


def _convert_schema_to_sql(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a SQL-style schema with tables, columns, and foreign keys."""

    if "tables" in schema:
        return schema

    collections = schema.get("collections", {})
    tables: dict[str, Any] = {}

    for collection_name, collection_info in collections.items():
        columns = []

        for field in collection_info.get("fields", []):
            columns.append(
                {
                    "name": field.get("name", ""),
                    "type": field.get("type", "TEXT"),
                    "primary_key": False,
                }
            )

        tables[collection_name] = {
            "columns": columns,
            "foreign_keys": [],
        }

    return {"tables": tables}


def _convert_schema_to_mongodb(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a MongoDB-style schema with collections and fields."""

    if "collections" in schema:
        return schema

    tables = schema.get("tables", {})
    collections: dict[str, Any] = {}

    for table_name, table_info in tables.items():
        fields = []

        for column in table_info.get("columns", []):
            fields.append(
                {
                    "name": column.get("name", ""),
                    "type": column.get("type", "string"),
                }
            )

        collections[table_name] = {"fields": fields}

    return {"collections": collections}
