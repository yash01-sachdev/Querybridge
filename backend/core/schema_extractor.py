"""Helpers for extracting database schema metadata across supported backends."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, inspect


def extract_schema(backend: str, connection: Any) -> dict[str, Any]:
    """Inspect the selected database connection and return a normalized schema."""

    if backend not in {"sqlite", "postgresql", "mongodb"}:
        raise ValueError(f"Unsupported backend: {backend}")

    try:
        if backend in {"sqlite", "postgresql"}:
            return _extract_sql_schema(connection)

        return _extract_mongodb_schema(connection)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Failed to extract schema for {backend}: {exc}") from exc


def _extract_sql_schema(connection: Any) -> dict[str, Any]:
    """Return table, column, and foreign key metadata for SQL backends."""

    inspector = inspect(connection)
    tables: dict[str, Any] = {}

    for table_name in sorted(inspector.get_table_names()):
        primary_key_names: set[str] = set(
            inspector.get_pk_constraint(table_name).get("constrained_columns", []) or []
        )
        columns: list[dict[str, Any]] = []

        for column in inspector.get_columns(table_name):
            column_name: str = str(column["name"])
            columns.append(
                {
                    "name": column_name,
                    "type": str(column["type"]),
                    "primary_key": column_name in primary_key_names,
                }
            )

        tables[table_name] = {
            "columns": columns,
            "foreign_keys": _extract_foreign_keys(inspector, table_name),
        }

    return {"tables": tables}


def _extract_foreign_keys(inspector: Any, table_name: str) -> list[dict[str, str]]:
    """Flatten SQLAlchemy foreign key metadata into a beginner-friendly format."""

    foreign_keys: list[dict[str, str]] = []

    for foreign_key in inspector.get_foreign_keys(table_name):
        constrained_columns: list[str] = list(foreign_key.get("constrained_columns", []) or [])
        referred_columns: list[str] = list(foreign_key.get("referred_columns", []) or [])
        referred_table: str = str(foreign_key.get("referred_table", ""))

        for index, column_name in enumerate(constrained_columns):
            referred_column: str = referred_columns[index] if index < len(referred_columns) else ""
            reference: str = f"{referred_table}.{referred_column}".rstrip(".")

            foreign_keys.append(
                {
                    "column": column_name,
                    "references": reference,
                }
            )

    return foreign_keys


def _extract_mongodb_schema(connection: Any) -> dict[str, Any]:
    """Return collection and sampled field metadata for MongoDB backends."""

    if not hasattr(connection, "list_collection_names"):
        raise ValueError(
            "MongoDB connection must provide list_collection_names(); pass a pymongo database object."
        )

    collections: dict[str, Any] = {}

    for collection_name in sorted(connection.list_collection_names()):
        collection = connection[collection_name]
        field_types: dict[str, set[str]] = {}

        for document in collection.find().limit(5):
            _collect_document_field_types(document, field_types)

        fields: list[dict[str, str]] = [
            {"name": field_name, "type": _merge_type_names(type_names)}
            for field_name, type_names in sorted(field_types.items())
        ]

        collections[collection_name] = {"fields": fields}

    return {"collections": collections}


def _collect_document_field_types(document: dict[str, Any], field_types: dict[str, set[str]]) -> None:
    """Track sampled MongoDB field types by field name."""

    for field_name, value in document.items():
        field_types.setdefault(field_name, set()).add(type(value).__name__)


def _merge_type_names(type_names: set[str]) -> str:
    """Return a stable display name for one or more sampled Python types."""

    if not type_names:
        return "Unknown"

    if len(type_names) == 1:
        return next(iter(type_names))

    return " | ".join(sorted(type_names))


if __name__ == "__main__":
    database_path = Path(__file__).resolve().parent.parent / "test.db"
    sqlite_engine = create_engine(f"sqlite:///{database_path.as_posix()}")
    schema = extract_schema("sqlite", sqlite_engine)
    print(json.dumps(schema, indent=2))
