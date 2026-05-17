"""Helpers for checking query syntax and schema awareness."""

from __future__ import annotations

import json
from typing import Any

from sqlglot import exp, parse
from sqlglot.errors import ParseError


def validate_query(query: str, backend: str, schema: dict[str, Any]) -> tuple[bool, str]:
    """Return whether a compiled query is valid for the selected backend and schema."""

    if backend in {"sqlite", "postgresql"}:
        return _validate_sql_query(query, backend, schema)

    if backend == "mongodb":
        return _validate_mongodb_query(query, schema)

    return False, f"Unsupported backend: {backend}"


def _validate_sql_query(query: str, backend: str, schema: dict[str, Any]) -> tuple[bool, str]:
    """Validate SQL syntax plus table and column references."""

    dialect = "sqlite" if backend == "sqlite" else "postgres"

    try:
        statements = parse(query, read=dialect)
    except ParseError as exc:
        return False, f"Syntax error: {exc}"

    known_tables = set(schema.get("tables", {}).keys())
    known_columns = {
        table_name: {
            str(column.get("name", "")).strip()
            for column in table_info.get("columns", [])
            if column.get("name")
        }
        for table_name, table_info in schema.get("tables", {}).items()
    }

    for statement in statements:
        alias_names = _extract_sql_alias_names(statement)
        scoped_tables, table_aliases = _extract_sql_table_scope(statement)

        for table in statement.find_all(exp.Table):
            table_name = table.name
            if table_name not in known_tables:
                return False, f"Table not found in schema: {table_name}"

        for column in statement.find_all(exp.Column):
            column_name = column.name
            if not column_name or column_name == "*":
                continue

            if column_name in alias_names:
                continue

            table_name = column.table

            if table_name:
                resolved_table_name = table_aliases.get(table_name, table_name)

                if resolved_table_name not in known_columns:
                    return False, f"Table not found in schema: {resolved_table_name}"

                if resolved_table_name not in scoped_tables:
                    return (
                        False,
                        f"Column references a table that is not joined in this query: "
                        f"{resolved_table_name}.{column_name}",
                    )

                if column_name not in known_columns[resolved_table_name]:
                    return False, f"Column not found in schema: {resolved_table_name}.{column_name}"

                continue

            is_valid, reason = _validate_unqualified_sql_column(
                column_name=column_name,
                scoped_tables=scoped_tables,
                known_columns=known_columns,
            )
            if not is_valid:
                return False, reason

        is_semantically_valid, semantic_reason = _validate_sql_select_semantics(statement)
        if not is_semantically_valid:
            return False, semantic_reason

    return True, ""


def _validate_mongodb_query(query: str, schema: dict[str, Any]) -> tuple[bool, str]:
    """Validate a MongoDB aggregation pipeline against the provided schema."""

    try:
        pipeline = json.loads(query)
    except json.JSONDecodeError as exc:
        return False, f"Invalid MongoDB query JSON: {exc}"

    if not isinstance(pipeline, list):
        return False, "MongoDB pipeline must be a list"

    collections = schema.get("collections", {})
    if not collections:
        return False, "No collections found in schema"

    available_fields = {
        str(field.get("name", "")).strip()
        for collection in collections.values()
        for field in collection.get("fields", [])
        if field.get("name")
    }

    for stage in pipeline:
        if not isinstance(stage, dict):
            return False, "MongoDB pipeline stages must be JSON objects"

        is_valid, reason, available_fields = _validate_mongodb_stage(stage, available_fields)
        if not is_valid:
            return False, reason

    return True, ""


def _extract_sql_alias_names(statement: exp.Expression) -> set[str]:
    """Collect SELECT alias names so ORDER BY aliases are treated as valid."""

    alias_names: set[str] = set()

    for alias in statement.find_all(exp.Alias):
        alias_name = alias.alias
        if alias_name:
            alias_names.add(alias_name)

    return alias_names


def _extract_sql_table_scope(statement: exp.Expression) -> tuple[set[str], dict[str, str]]:
    """Collect the real tables in scope plus any table-alias-to-table mapping."""

    scoped_tables: set[str] = set()
    table_aliases: dict[str, str] = {}

    for table in statement.find_all(exp.Table):
        table_name = table.name
        if not table_name:
            continue

        scoped_tables.add(table_name)
        alias_name = table.alias
        if alias_name:
            table_aliases[alias_name] = table_name

    return scoped_tables, table_aliases


def _validate_unqualified_sql_column(
    column_name: str,
    scoped_tables: set[str],
    known_columns: dict[str, set[str]],
) -> tuple[bool, str]:
    """Validate an unqualified column against the tables actually used in this statement."""

    candidate_tables = scoped_tables or set(known_columns.keys())
    matching_tables = {
        table_name
        for table_name in candidate_tables
        if column_name in known_columns.get(table_name, set())
    }

    if not matching_tables:
        return False, f"Column not found in schema for the referenced tables: {column_name}"

    if len(matching_tables) > 1:
        sorted_matches = ", ".join(sorted(matching_tables))
        return False, f"Ambiguous column reference: {column_name} (matches: {sorted_matches})"

    return True, ""


def _validate_sql_select_semantics(statement: exp.Expression) -> tuple[bool, str]:
    """Reject aggregate SQL select lists that would mix summary and row-level output."""

    for select_statement in statement.find_all(exp.Select):
        select_expressions = list(select_statement.expressions)
        if not select_expressions:
            continue

        has_aggregate_expression = any(
            expression.find(exp.AggFunc) is not None for expression in select_expressions
        )
        if not has_aggregate_expression:
            continue

        group_clause = select_statement.args.get("group")
        if group_clause is not None:
            continue

        for expression in select_expressions:
            if expression.find(exp.AggFunc) is not None:
                continue

            if isinstance(expression, (exp.Column, exp.Star)):
                return (
                    False,
                    "SQL query cannot mix ordinary fields with aggregations unless those fields are grouped.",
                )

            if isinstance(expression, exp.Alias) and expression.this.find(exp.AggFunc) is None:
                if expression.this.find(exp.Column) is not None or expression.this.find(exp.Star) is not None:
                    return (
                        False,
                        "SQL query cannot mix ordinary fields with aggregations unless those fields are grouped.",
                    )

    return True, ""


def _validate_mongodb_stage(
    stage: dict[str, Any],
    available_fields: set[str],
) -> tuple[bool, str, set[str]]:
    """Validate one MongoDB stage and return the next available output fields."""

    if "$match" in stage:
        match_clause = stage["$match"]
        if not isinstance(match_clause, dict):
            return False, "MongoDB $match stage must be an object", available_fields

        for field_name in _extract_mongodb_match_fields(match_clause):
            if field_name not in available_fields:
                return False, f"Field not found in schema: {field_name}", available_fields

        return True, "", available_fields

    if "$group" in stage:
        return _validate_mongodb_group_stage(stage["$group"], available_fields)

    if "$project" in stage:
        return _validate_mongodb_project_stage(stage["$project"], available_fields)

    if "$sort" in stage:
        sort_clause = stage["$sort"]
        if not isinstance(sort_clause, dict):
            return False, "MongoDB $sort stage must be an object", available_fields

        for field_name in sort_clause:
            if field_name not in available_fields:
                return False, f"Field not found in schema: {field_name}", available_fields

        return True, "", available_fields

    if "$limit" in stage or "$count" in stage:
        if "$count" in stage:
            count_name = str(stage["$count"]).strip()
            if count_name:
                return True, "", {count_name}
        return True, "", available_fields

    return True, "", available_fields


def _validate_mongodb_group_stage(
    group_clause: object,
    available_fields: set[str],
) -> tuple[bool, str, set[str]]:
    """Validate one MongoDB $group stage and expose its output aliases."""

    if not isinstance(group_clause, dict):
        return False, "MongoDB $group stage must be an object", available_fields

    group_identifier = group_clause.get("_id")
    is_valid, reason = _validate_mongodb_reference(group_identifier, available_fields)
    if not is_valid:
        return False, reason, available_fields

    next_fields = {"_id"}

    for alias_name, expression in group_clause.items():
        if alias_name == "_id":
            continue

        if not alias_name:
            return False, "MongoDB aggregation alias cannot be empty", available_fields

        if not isinstance(expression, dict) or len(expression) != 1:
            return False, f"Invalid MongoDB aggregation expression for {alias_name}", available_fields

        operator, referenced_value = next(iter(expression.items()))
        if operator not in {"$sum", "$avg", "$min", "$max"}:
            return False, f"Unsupported MongoDB aggregation operator: {operator}", available_fields

        if operator == "$sum" and referenced_value == 1:
            next_fields.add(alias_name)
            continue

        is_valid, reason = _validate_mongodb_reference(referenced_value, available_fields)
        if not is_valid:
            return False, reason, available_fields

        next_fields.add(alias_name)

    return True, "", next_fields


def _validate_mongodb_project_stage(
    project_clause: object,
    available_fields: set[str],
) -> tuple[bool, str, set[str]]:
    """Validate one MongoDB $project stage and return the projected output fields."""

    if not isinstance(project_clause, dict):
        return False, "MongoDB $project stage must be an object", available_fields

    next_fields: set[str] = set()

    for output_name, expression in project_clause.items():
        if expression in {0, False}:
            continue

        if expression in {1, True}:
            if output_name not in available_fields:
                return False, f"Field not found in schema: {output_name}", available_fields
            next_fields.add(output_name)
            continue

        is_valid, reason = _validate_mongodb_reference(expression, available_fields)
        if not is_valid:
            return False, reason, available_fields

        next_fields.add(output_name)

    return True, "", next_fields or available_fields


def _validate_mongodb_reference(value: object, available_fields: set[str]) -> tuple[bool, str]:
    """Validate one MongoDB field reference like $amount or $_id.status."""

    if value in {None, 1}:
        return True, ""

    if not isinstance(value, str):
        return True, ""

    if not value.startswith("$"):
        return True, ""

    reference = value[1:]
    if not reference:
        return False, "Empty MongoDB field reference"

    root_field = reference.split(".", 1)[0]
    if root_field not in available_fields:
        return False, f"Field not found in schema: {reference}"

    return True, ""


def _extract_mongodb_match_fields(match_clause: dict[str, Any]) -> set[str]:
    """Collect field names from nested MongoDB match conditions."""

    field_names: set[str] = set()

    for key, value in match_clause.items():
        if key in {"$and", "$or", "$nor"} and isinstance(value, list):
            for nested_clause in value:
                if isinstance(nested_clause, dict):
                    field_names.update(_extract_mongodb_match_fields(nested_clause))
            continue

        if not key.startswith("$"):
            field_names.add(key)

        if isinstance(value, dict):
            field_names.update(_extract_mongodb_match_fields(value))

    return field_names
