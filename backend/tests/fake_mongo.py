"""Small in-memory MongoDB helpers for backend tests."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any


class FakeMongoDatabase:
    """Very small in-memory MongoDB stand-in for route and unit tests."""

    def __init__(self, collections: dict[str, list[dict[str, object]]]) -> None:
        self._collections = {
            name: FakeMongoCollection(documents)
            for name, documents in collections.items()
        }

    def __getitem__(self, collection_name: str) -> "FakeMongoCollection":
        return self._collections[collection_name]

    def list_collection_names(self) -> list[str]:
        """Return collection names so schema extraction can inspect them."""

        return sorted(self._collections.keys())


class FakeMongoCollection:
    """Enough aggregate and sample support to mimic the current backend flow."""

    def __init__(self, documents: list[dict[str, object]]) -> None:
        self._documents = [dict(document) for document in documents]

    def find(self) -> "FakeMongoCursor":
        """Return a cursor-like object that supports limit()."""

        return FakeMongoCursor(self._documents)

    def aggregate(self, pipeline: list[dict[str, object]]) -> list[dict[str, object]]:
        """Apply a tiny subset of MongoDB aggregation stages in memory."""

        documents = [dict(document) for document in self._documents]

        for stage in pipeline:
            if "$match" in stage:
                documents = _apply_match(documents, stage["$match"])
                continue

            if "$project" in stage:
                documents = _apply_project(documents, stage["$project"])
                continue

            if "$group" in stage:
                documents = _apply_group(documents, stage["$group"])
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


class FakeMongoCursor:
    """Very small cursor object that supports limit() and iteration."""

    def __init__(self, documents: list[dict[str, object]]) -> None:
        self._documents = [dict(document) for document in documents]

    def limit(self, count: int) -> list[dict[str, object]]:
        """Return the first N sampled documents."""

        return self._documents[:count]


def _apply_match(documents: list[dict[str, object]], match: dict[str, object]) -> list[dict[str, object]]:
    """Apply a small subset of MongoDB matching for local tests."""

    return [
        document
        for document in documents
        if _document_matches(document, match)
    ]


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
        if not _value_matches(actual_value, expected_value):
            return False

    return True


def _value_matches(actual_value: object, expected_value: object) -> bool:
    """Apply a tiny subset of MongoDB comparison operators against one value."""

    if isinstance(expected_value, dict):
        if "$not" in expected_value:
            return not _value_matches(actual_value, expected_value["$not"])

        if "$regex" in expected_value:
            regex_pattern = str(expected_value["$regex"])
            regex_flags = re.IGNORECASE if expected_value.get("$options") == "i" else 0
            return actual_value is not None and bool(re.search(regex_pattern, str(actual_value), regex_flags))

        if "$gte" in expected_value and (actual_value is None or actual_value < expected_value["$gte"]):
            return False
        if "$gt" in expected_value and (actual_value is None or actual_value <= expected_value["$gt"]):
            return False
        if "$lte" in expected_value and (actual_value is None or actual_value > expected_value["$lte"]):
            return False
        if "$lt" in expected_value and (actual_value is None or actual_value >= expected_value["$lt"]):
            return False
        if "$ne" in expected_value and actual_value == expected_value["$ne"]:
            return False
        if "$in" in expected_value and actual_value not in expected_value["$in"]:
            return False

        return True

    return actual_value == expected_value


def _apply_project(documents: list[dict[str, object]], project: dict[str, object]) -> list[dict[str, object]]:
    """Keep only projected fields for each in-memory document."""

    projected_documents: list[dict[str, object]] = []

    for document in documents:
        projected_document: dict[str, object] = {}

        for field_name, include in project.items():
            if include in {0, False}:
                continue

            if include in {1, True}:
                if field_name in document:
                    projected_document[field_name] = document.get(field_name)
                continue

            if isinstance(include, str) and include.startswith("$"):
                projected_document[field_name] = _resolve_reference(document, include)

        projected_documents.append(projected_document)

    return projected_documents


def _apply_sort(documents: list[dict[str, object]], sort: dict[str, object]) -> list[dict[str, object]]:
    """Sort in-memory documents by the first requested field."""

    if not sort:
        return documents

    field_name, direction = next(iter(sort.items()))
    reverse = int(direction) < 0
    return sorted(documents, key=lambda document: document.get(field_name), reverse=reverse)


def _apply_group(documents: list[dict[str, object]], group: dict[str, object]) -> list[dict[str, object]]:
    """Apply a tiny subset of MongoDB $group in memory."""

    grouped_rows: dict[object, dict[str, object]] = {}
    average_state: dict[object, dict[str, dict[str, float]]] = defaultdict(dict)

    for document in documents:
        group_value = _resolve_group_value(document, group.get("_id"))
        group_key = _freeze_value(group_value)

        if group_key not in grouped_rows:
            grouped_rows[group_key] = {"_id": group_value}

        grouped_row = grouped_rows[group_key]

        for alias_name, expression in group.items():
            if alias_name == "_id":
                continue

            if not isinstance(expression, dict) or len(expression) != 1:
                continue

            operator, reference = next(iter(expression.items()))

            if operator == "$sum":
                increment = 1 if reference == 1 else _coerce_number(_resolve_reference(document, reference))
                grouped_row[alias_name] = _coerce_number(grouped_row.get(alias_name, 0)) + increment
                continue

            value = _coerce_number(_resolve_reference(document, reference))

            if operator == "$avg":
                alias_state = average_state[group_key].setdefault(alias_name, {"total": 0.0, "count": 0.0})
                alias_state["total"] += value
                alias_state["count"] += 1
                grouped_row[alias_name] = alias_state["total"] / alias_state["count"]
                continue

            if operator == "$min":
                existing_value = grouped_row.get(alias_name)
                grouped_row[alias_name] = value if existing_value is None else min(_coerce_number(existing_value), value)
                continue

            if operator == "$max":
                existing_value = grouped_row.get(alias_name)
                grouped_row[alias_name] = value if existing_value is None else max(_coerce_number(existing_value), value)

    return list(grouped_rows.values())


def _resolve_group_value(document: dict[str, object], group_identifier: object) -> object:
    """Resolve the _id value used by one MongoDB group stage."""

    if group_identifier is None:
        return None

    if isinstance(group_identifier, str):
        return _resolve_reference(document, group_identifier)

    if isinstance(group_identifier, dict):
        return {
            key: _resolve_reference(document, reference)
            for key, reference in group_identifier.items()
        }

    return group_identifier


def _resolve_reference(document: dict[str, object], reference: object) -> object:
    """Resolve a reference like $amount or $_id.status against one row."""

    if not isinstance(reference, str) or not reference.startswith("$"):
        return reference

    current_value: object = document
    path_parts = reference[1:].split(".")

    for path_part in path_parts:
        if not isinstance(current_value, dict):
            return None
        current_value = current_value.get(path_part)

    return current_value


def _freeze_value(value: object) -> object:
    """Turn grouped values into hashable keys for the fake in-memory map."""

    if isinstance(value, dict):
        return tuple(sorted((key, _freeze_value(nested_value)) for key, nested_value in value.items()))

    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)

    return value


def _coerce_number(value: object) -> float:
    """Best-effort numeric coercion for fake aggregate math."""

    if isinstance(value, (int, float)):
        return float(value)

    return float(value or 0)
