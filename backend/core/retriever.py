"""Helpers for narrowing a full schema down to the most relevant parts."""

from __future__ import annotations

import json
from typing import Any

STOPWORDS: set[str] = {
    "a",
    "the",
    "in",
    "for",
    "of",
    "from",
    "where",
    "what",
    "show",
    "get",
    "find",
    "list",
    "how",
    "many",
    "is",
    "are",
    "by",
    "to",
    "total",
    "number",
    "all",
    "data",
    "records",
    "record",
    "give",
    "me",
}


def retrieve_relevant_schema(
    question: str,
    full_schema: dict[str, Any],
    backend: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """Return the tables or collections that best match the user's question."""

    keywords = _extract_keywords(question)

    if backend in {"sqlite", "postgresql"}:
        tables = full_schema.get("tables", {})
        table_scores: dict[str, int] = {}

        for table_name, table_info in tables.items():
            column_names = [str(column.get("name", "")) for column in table_info.get("columns", [])]
            table_scores[table_name] = _score_item(table_name, column_names, keywords)

        selected_tables = _select_top_matches(tables, table_scores, top_k)
        if selected_tables:
            return {"tables": selected_tables}

        raise ValueError(_build_sql_no_match_message(keywords, tables))

    if backend == "mongodb":
        collections = full_schema.get("collections", {})
        collection_scores: dict[str, int] = {}

        for collection_name, collection_info in collections.items():
            field_names = [str(field.get("name", "")) for field in collection_info.get("fields", [])]
            collection_scores[collection_name] = _score_item(collection_name, field_names, keywords)

        selected_collections = _select_top_matches(collections, collection_scores, top_k)
        if selected_collections:
            return {"collections": selected_collections}

        raise ValueError(_build_mongodb_no_match_message(keywords, collections))

    raise ValueError(f"Unsupported backend: {backend}")


def _extract_keywords(question: str) -> list[str]:
    """Lowercase the question, split it into words, and remove simple stopwords."""

    keywords: list[str] = []
    seen_keywords: set[str] = set()

    for word in question.lower().split():
        cleaned_word = word.strip(".,!?;:()[]{}\"'")

        if not cleaned_word or cleaned_word in STOPWORDS:
            continue

        if cleaned_word not in seen_keywords:
            keywords.append(cleaned_word)
            seen_keywords.add(cleaned_word)

    return keywords


def _score_item(item_name: str, field_names: list[str], keywords: list[str]) -> int:
    """Score exact schema matches higher than loose substring matches."""

    score = _score_name(item_name, keywords, is_table_name=True)

    for field_name in field_names:
        score += _score_name(field_name, keywords, is_table_name=False)

    return score


def _score_name(name: str, keywords: list[str], is_table_name: bool) -> int:
    """Score one table or field name against the user keywords."""

    score = 0
    lowered_name = name.lower()
    name_tokens = _split_name_tokens(lowered_name)

    for keyword in keywords:
        if _has_exact_match(keyword, name_tokens):
            score += 7 if is_table_name else 3
            continue

        if keyword in lowered_name:
            score += 1

    return score


def _split_name_tokens(name: str) -> list[str]:
    """Break names like user_id into simpler lowercase tokens."""

    return [token for token in name.replace(".", "_").split("_") if token]


def _has_exact_match(keyword: str, tokens: list[str]) -> bool:
    """Treat singular and plural forms as the same basic match."""

    for token in tokens:
        if keyword == token:
            return True

        if keyword.endswith("s") and keyword[:-1] == token:
            return True

        if token.endswith("s") and token[:-1] == keyword:
            return True

    return False


def _select_top_matches(
    items: dict[str, Any],
    scores: dict[str, int],
    top_k: int,
) -> dict[str, Any]:
    """Return the highest-scoring items, or an empty dict if nothing matches."""

    positive_matches = [
        name
        for name, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        if score > 0
    ]

    if not positive_matches:
        return {}

    selected_names = positive_matches[:top_k]
    return {name: items[name] for name in selected_names}


def _build_sql_no_match_message(keywords: list[str], tables: dict[str, Any]) -> str:
    """Build a clear SQL error message when no table matches the question."""

    available_tables = ", ".join(sorted(tables.keys())) or "none"
    requested_text = ", ".join(keywords) or "your question"

    return (
        f"Could not match {requested_text} to any table or column in the current database. "
        f"Available tables: {available_tables}."
    )


def _build_mongodb_no_match_message(keywords: list[str], collections: dict[str, Any]) -> str:
    """Build a clear MongoDB error message when no collection matches the question."""

    available_collections = ", ".join(sorted(collections.keys())) or "none"
    requested_text = ", ".join(keywords) or "your question"

    return (
        f"Could not match {requested_text} to any collection or field in the current database. "
        f"Available collections: {available_collections}."
    )


if __name__ == "__main__":
    sample_schema = {
        "tables": {
            "users": {
                "columns": [
                    {"name": "id", "type": "INTEGER", "primary_key": True},
                    {"name": "name", "type": "TEXT", "primary_key": False},
                ],
                "foreign_keys": [],
            },
            "orders": {
                "columns": [
                    {"name": "id", "type": "INTEGER", "primary_key": True},
                    {"name": "user_id", "type": "INTEGER", "primary_key": False},
                    {"name": "total", "type": "REAL", "primary_key": False},
                ],
                "foreign_keys": [{"column": "user_id", "references": "users.id"}],
            },
            "products": {
                "columns": [
                    {"name": "id", "type": "INTEGER", "primary_key": True},
                    {"name": "title", "type": "TEXT", "primary_key": False},
                ],
                "foreign_keys": [],
            },
            "payments": {
                "columns": [
                    {"name": "id", "type": "INTEGER", "primary_key": True},
                    {"name": "order_id", "type": "INTEGER", "primary_key": False},
                ],
                "foreign_keys": [{"column": "order_id", "references": "orders.id"}],
            },
            "categories": {
                "columns": [
                    {"name": "id", "type": "INTEGER", "primary_key": True},
                    {"name": "label", "type": "TEXT", "primary_key": False},
                ],
                "foreign_keys": [],
            },
        }
    }

    result = retrieve_relevant_schema("show orders by user", sample_schema, "sqlite")
    assert set(result["tables"].keys()) == {"orders", "users"}
    print(json.dumps(result, indent=2))
