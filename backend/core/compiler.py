"""Helpers for compiling a shared query plan into backend-specific syntax."""

from __future__ import annotations

import json
from typing import Any

from sqlglot import exp, transpile

DEFAULT_LIMIT = 100
MAX_LIMIT = 500


def compile_query(plan: dict[str, Any], backend: str) -> str:
    """Compile a query plan into SQL or a MongoDB aggregation pipeline."""

    operation = str(plan.get("operation", "")).lower()

    if operation not in {"select", "find", "aggregate"}:
        raise ValueError("Unsupported operation: only read operations allowed")

    if backend in {"sqlite", "postgresql"}:
        if operation not in {"select", "aggregate"}:
            raise ValueError("Unsupported operation: only read operations allowed")
        return _compile_sql_query(plan, backend)

    if backend == "mongodb":
        if operation not in {"find", "aggregate"}:
            raise ValueError("Unsupported operation: only read operations allowed")
        return _compile_mongodb_pipeline(plan)

    raise ValueError(f"Unsupported backend: {backend}")


def _compile_sql_query(plan: dict[str, Any], backend: str) -> str:
    """Build a SQL query with sqlglot expressions and transpile it to the target dialect."""

    tables = plan.get("tables", [])
    if not tables:
        raise ValueError("SQL plan must include at least one table")

    select_expressions = _build_select_expressions(plan)
    query = exp.Select(expressions=select_expressions)
    query = query.from_(_build_table_expression(str(tables[0])))
    query = _add_joins(query, plan)
    query = _add_where_clause(query, plan)
    query = _add_group_by_clause(query, plan)
    query = _add_order_by_clause(query, plan)

    if _should_apply_limit(plan):
        query = query.limit(exp.convert(_get_capped_limit(plan)))

    dialect = "sqlite" if backend == "sqlite" else "postgres"
    return transpile(query.sql(), write=dialect)[0]


def _build_select_expressions(plan: dict[str, Any]) -> list[exp.Expression]:
    """Build the SELECT expression list from fields and aggregations."""

    expressions: list[exp.Expression] = []
    aggregations = [aggregation for aggregation in plan.get("aggregations", []) if isinstance(aggregation, dict)]
    group_by_fields = plan.get("group_by", [])
    field_names = plan.get("fields", []) or plan.get("select", []) or []

    if aggregations and not group_by_fields:
        field_names = []

    for field_name in field_names:
        expressions.append(_build_field_expression(str(field_name)))

    for aggregation in aggregations:
        expressions.append(_build_aggregation_expression(aggregation))

    if not expressions:
        return [exp.Star()]

    return expressions


def _build_field_expression(field_name: str) -> exp.Expression:
    """Turn a field name like users.name into a sqlglot expression."""

    if field_name == "*":
        return exp.Star()

    if field_name.endswith(".*"):
        table_name = field_name[:-2]
        return exp.Column(this=exp.Star(), table=exp.to_identifier(table_name))

    return _build_column_expression(field_name)


def _build_aggregation_expression(aggregation: dict[str, Any]) -> exp.Expression:
    """Turn an aggregation plan item into a sqlglot function expression."""

    function_name = str(aggregation.get("function", "")).upper()
    field_name = str(aggregation.get("field", ""))
    alias_name = str(aggregation.get("alias", "")).strip()

    if not function_name:
        raise ValueError("Aggregation function is required")

    if field_name == "*":
        field_expression: exp.Expression = exp.Star()
    else:
        field_expression = _build_column_expression(field_name)

    aggregation_expression = exp.func(function_name, field_expression)

    if alias_name:
        return exp.alias_(aggregation_expression, alias_name)

    return aggregation_expression


def _add_joins(query: exp.Select, plan: dict[str, Any]) -> exp.Select:
    """Attach JOIN clauses based on the plan's join definitions."""

    joined_tables = {str(plan["tables"][0])}

    for join_info in plan.get("joins", []):
        left_reference = str(join_info.get("left", ""))
        right_reference = str(join_info.get("right", ""))

        if not left_reference or not right_reference:
            continue

        join_table = _pick_join_table(left_reference, right_reference, joined_tables)
        join_condition = exp.EQ(
            this=_build_column_expression(left_reference),
            expression=_build_column_expression(right_reference),
        )

        query = query.join(_build_table_expression(join_table), on=join_condition)
        joined_tables.add(join_table)

    return query


def _pick_join_table(left_reference: str, right_reference: str, joined_tables: set[str]) -> str:
    """Choose which table should be added by the next JOIN."""

    left_table = _extract_table_name(left_reference)
    right_table = _extract_table_name(right_reference)

    if right_table and right_table not in joined_tables:
        return right_table

    if left_table and left_table not in joined_tables:
        return left_table

    if right_table:
        return right_table

    if left_table:
        return left_table

    raise ValueError("Join references must include table names")


def _add_where_clause(query: exp.Select, plan: dict[str, Any]) -> exp.Select:
    """Attach a WHERE clause if the plan contains filters."""

    filters = plan.get("filters", [])
    if not filters:
        return query

    combined_expression = _build_filter_expression(filters[0])

    for filter_info in filters[1:]:
        next_expression = _build_filter_expression(filter_info)
        connector = str(filter_info.get("combine_with_previous", "AND")).upper()

        if connector == "OR":
            combined_expression = exp.or_(combined_expression, next_expression)
            continue

        combined_expression = exp.and_(combined_expression, next_expression)

    return query.where(combined_expression)


def _build_filter_expression(filter_info: dict[str, Any]) -> exp.Expression:
    """Turn a filter rule into a sqlglot comparison expression."""

    field_name = str(filter_info.get("field", ""))
    operator = str(filter_info.get("operator", "=")).upper()
    value = filter_info.get("value")
    field_expression = _build_column_expression(field_name)

    if operator == "=":
        return exp.EQ(this=field_expression, expression=exp.convert(value))
    if operator in {"!=", "<>"}:
        return exp.NEQ(this=field_expression, expression=exp.convert(value))
    if operator == ">":
        return exp.GT(this=field_expression, expression=exp.convert(value))
    if operator == ">=":
        return exp.GTE(this=field_expression, expression=exp.convert(value))
    if operator == "<":
        return exp.LT(this=field_expression, expression=exp.convert(value))
    if operator == "<=":
        return exp.LTE(this=field_expression, expression=exp.convert(value))
    if operator == "LIKE":
        return exp.Like(this=field_expression, expression=exp.convert(value))
    if operator == "STARTS_WITH":
        return _build_case_insensitive_like(field_expression, f"{value}%")
    if operator == "NOT_STARTS_WITH":
        return _build_negated_like_expression(field_expression, f"{value}%")
    if operator == "ENDS_WITH":
        return _build_case_insensitive_like(field_expression, f"%{value}")
    if operator == "NOT_ENDS_WITH":
        return _build_negated_like_expression(field_expression, f"%{value}")
    if operator == "CONTAINS":
        if isinstance(value, list):
            return _build_any_contains_expression(field_expression, value)
        return _build_case_insensitive_like(field_expression, f"%{value}%")
    if operator == "NOT_CONTAINS":
        values = value if isinstance(value, list) else [value]
        return _build_all_negated_contains_expression(field_expression, values)
    if operator == "CONTAINS_ANY":
        values = value if isinstance(value, list) else [value]
        return _build_any_contains_expression(field_expression, values)
    if operator == "IN":
        values = value if isinstance(value, list) else [value]
        return exp.In(this=field_expression, expressions=[exp.convert(item) for item in values])

    raise ValueError(f"Unsupported filter operator: {operator}")


def _build_case_insensitive_like(
    field_expression: exp.Expression,
    value: str,
) -> exp.Expression:
    """Build a LOWER(column) LIKE lower(value) expression for text contains checks."""

    lowered_value = str(value).lower()
    lowered_field = exp.func("LOWER", field_expression.copy())

    return exp.Like(this=lowered_field, expression=exp.convert(lowered_value))


def _build_negated_like_expression(
    field_expression: exp.Expression,
    value: str,
) -> exp.Expression:
    """Build a NOT LOWER(column) LIKE pattern expression."""

    return exp.Not(this=_build_case_insensitive_like(field_expression, value))


def _build_any_contains_expression(
    field_expression: exp.Expression,
    values: list[Any],
) -> exp.Expression:
    """Build a chain of OR-ed case-insensitive LIKE expressions."""

    like_expressions = [
        _build_case_insensitive_like(field_expression, f"%{item}%")
        for item in values
    ]

    combined_expression = like_expressions[0]

    for expression in like_expressions[1:]:
        combined_expression = exp.or_(combined_expression, expression)

    return combined_expression


def _build_all_negated_contains_expression(
    field_expression: exp.Expression,
    values: list[Any],
) -> exp.Expression:
    """Build a chain of AND-ed negated LIKE expressions."""

    negated_expressions = [
        _build_negated_like_expression(field_expression, f"%{item}%")
        for item in values
    ]

    combined_expression = negated_expressions[0]

    for expression in negated_expressions[1:]:
        combined_expression = exp.and_(combined_expression, expression)

    return combined_expression


def _add_group_by_clause(query: exp.Select, plan: dict[str, Any]) -> exp.Select:
    """Attach a GROUP BY clause if present in the plan."""

    group_by_fields = plan.get("group_by", [])
    if not group_by_fields:
        return query

    group_expressions = [_build_column_expression(str(field_name)) for field_name in group_by_fields]
    return query.group_by(*group_expressions)


def _add_order_by_clause(query: exp.Select, plan: dict[str, Any]) -> exp.Select:
    """Attach an ORDER BY clause if present in the plan."""

    order_by_items = plan.get("order_by", [])
    if not order_by_items:
        return query

    ordered_expressions: list[exp.Ordered] = []

    for order_by in order_by_items:
        field_name = str(order_by.get("field", ""))
        direction = str(order_by.get("direction", "ASC")).upper()
        ordered_expressions.append(
            exp.Ordered(
                this=_build_column_expression(field_name),
                desc=direction == "DESC",
            )
        )

    return query.order_by(*ordered_expressions)


def _build_column_expression(reference: str) -> exp.Expression:
    """Turn a dotted column reference into a sqlglot column expression."""

    reference = reference.strip()
    if not reference:
        raise ValueError("Column reference is required")

    parts = reference.split(".")

    if len(parts) == 1:
        return exp.column(parts[0])

    if len(parts) == 2:
        return exp.column(parts[1], table=parts[0])

    if len(parts) == 3:
        return exp.column(parts[2], table=parts[1], db=parts[0])

    raise ValueError(f"Unsupported column reference: {reference}")


def _build_table_expression(reference: str) -> exp.Table:
    """Turn a dotted table reference into a sqlglot table expression."""

    parts = reference.split(".")

    if len(parts) == 1:
        return exp.Table(this=exp.to_identifier(parts[0]))

    if len(parts) == 2:
        return exp.Table(this=exp.to_identifier(parts[1]), db=exp.to_identifier(parts[0]))

    if len(parts) == 3:
        return exp.Table(
            this=exp.to_identifier(parts[2]),
            db=exp.to_identifier(parts[1]),
            catalog=exp.to_identifier(parts[0]),
        )

    raise ValueError(f"Unsupported table reference: {reference}")


def _extract_table_name(reference: str) -> str:
    """Return the table portion of a dotted column reference."""

    parts = reference.split(".")

    if len(parts) < 2:
        return ""

    return parts[-2]


def _compile_mongodb_pipeline(plan: dict[str, Any]) -> str:
    """Build a MongoDB aggregation pipeline and return it as JSON."""

    pipeline: list[dict[str, Any]] = []

    if plan.get("match"):
        pipeline.append({"$match": plan["match"]})

    group_by_fields = [str(field_name) for field_name in plan.get("group_by", [])]
    aggregations = [aggregation for aggregation in plan.get("aggregations", []) if isinstance(aggregation, dict)]

    if plan.get("intent") == "count" and not group_by_fields and not aggregations:
        pipeline.append({"$count": "count"})
        return json.dumps(pipeline)

    if group_by_fields or aggregations:
        pipeline.append({"$group": _build_mongodb_group_stage(group_by_fields, aggregations)})
        pipeline.append({"$project": _build_mongodb_group_project_stage(group_by_fields, aggregations)})
    elif plan.get("project"):
        pipeline.append({"$project": plan["project"]})

    if plan.get("sort"):
        pipeline.append({"$sort": plan["sort"]})

    if _should_apply_limit(plan):
        pipeline.append({"$limit": _get_capped_limit(plan)})

    return json.dumps(pipeline)


def _build_mongodb_group_stage(
    group_by_fields: list[str],
    aggregations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build one MongoDB $group stage from grouped fields and aggregations."""

    group_stage: dict[str, Any] = {
        "_id": _build_mongodb_group_id(group_by_fields),
    }

    for aggregation in aggregations:
        alias_name = str(aggregation.get("alias", "")).strip()
        function_name = str(aggregation.get("function", "")).upper()
        field_name = str(aggregation.get("field", "")).strip()

        if not alias_name:
            raise ValueError("MongoDB aggregation alias is required")

        if function_name == "COUNT":
            group_stage[alias_name] = {"$sum": 1}
            continue

        if not field_name:
            raise ValueError("MongoDB aggregation field is required")

        field_reference = f"${field_name}"

        if function_name == "AVG":
            group_stage[alias_name] = {"$avg": field_reference}
            continue
        if function_name == "SUM":
            group_stage[alias_name] = {"$sum": field_reference}
            continue
        if function_name == "MIN":
            group_stage[alias_name] = {"$min": field_reference}
            continue
        if function_name == "MAX":
            group_stage[alias_name] = {"$max": field_reference}
            continue

        raise ValueError(f"Unsupported MongoDB aggregation function: {function_name}")

    return group_stage


def _build_mongodb_group_id(group_by_fields: list[str]) -> Any:
    """Build the MongoDB _id expression for grouped aggregation pipelines."""

    if not group_by_fields:
        return None

    if len(group_by_fields) == 1:
        return f"${group_by_fields[0]}"

    return {
        _mongodb_output_name(field_name): f"${field_name}"
        for field_name in group_by_fields
    }


def _build_mongodb_group_project_stage(
    group_by_fields: list[str],
    aggregations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Project grouped fields and aggregation aliases into flat result rows."""

    project_stage: dict[str, Any] = {"_id": 0}

    if len(group_by_fields) == 1:
        output_field = _mongodb_output_name(group_by_fields[0])
        project_stage[output_field] = "$_id"
    elif len(group_by_fields) > 1:
        for field_name in group_by_fields:
            output_field = _mongodb_output_name(field_name)
            project_stage[output_field] = f"$_id.{output_field}"

    for aggregation in aggregations:
        alias_name = str(aggregation.get("alias", "")).strip()
        if alias_name:
            project_stage[alias_name] = 1

    return project_stage


def _mongodb_output_name(field_name: str) -> str:
    """Flatten one field reference into the output field name shown to the user."""

    return str(field_name).split(".")[-1]


def _should_apply_limit(plan: dict[str, Any]) -> bool:
    """Skip LIMIT for plain count queries because they already return one row."""

    fields = plan.get("fields", [])
    aggregations = plan.get("aggregations", [])
    group_by = plan.get("group_by", [])

    if fields:
        return True

    if group_by:
        return True

    if len(aggregations) != 1:
        return True

    aggregation = aggregations[0]
    function_name = str(aggregation.get("function", "")).upper()
    field_name = str(aggregation.get("field", ""))

    return not (function_name == "COUNT" and field_name == "*")


def _get_capped_limit(plan: dict[str, Any]) -> int:
    """Return a safe LIMIT value with defaults and a hard cap."""

    raw_limit = plan.get("limit", DEFAULT_LIMIT)

    try:
        limit_value = int(raw_limit)
    except (TypeError, ValueError):
        limit_value = DEFAULT_LIMIT

    if limit_value < 1:
        return DEFAULT_LIMIT

    return min(limit_value, MAX_LIMIT)


if __name__ == "__main__":
    sql_plan = {
        "operation": "select",
        "tables": ["orders", "users"],
        "fields": ["users.name"],
        "joins": [{"left": "orders.user_id", "right": "users.id"}],
        "filters": [{"field": "orders.status", "operator": "=", "value": "completed"}],
        "aggregations": [{"function": "COUNT", "field": "orders.id", "alias": "total"}],
        "group_by": ["users.name"],
        "order_by": [{"field": "total", "direction": "DESC"}],
        "limit": 10,
    }

    mongo_plan = {
        "operation": "find",
        "collection": "orders",
        "match": {"status": "completed"},
        "project": {"user_id": 1, "amount": 1},
        "sort": {"amount": -1},
        "limit": 25,
    }

    print(compile_query(sql_plan, "sqlite"))
    print(compile_query(mongo_plan, "mongodb"))
