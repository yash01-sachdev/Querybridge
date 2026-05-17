"""Shared pytest fixtures and local temp setup for backend tests."""

from __future__ import annotations

import atexit
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.runtime_paths import iter_temp_root_candidates, pick_first_writable_directory

TEST_RUNTIME_ROOT = None
TEST_TEMP_DIR = None


@pytest.fixture(scope="session")
def sqlite_backend() -> dict[str, object]:
    """Create a local SQLite database with a small relational schema."""

    engine, database_path = create_demo_sqlite_database(BACKEND_ROOT)

    yield {
        "engine": engine,
        "path": database_path,
    }

    cleanup_demo_sqlite_database(engine, database_path)


@pytest.fixture()
def sqlite_schema(sqlite_backend: dict[str, object]) -> dict[str, object]:
    """Return the extracted SQLite schema for tests that need it."""

    with sqlite_backend["engine"].connect() as connection:
        return extract_schema("sqlite", connection)


@pytest.fixture(autouse=True)
def mock_ollama_for_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace live Ollama calls with deterministic test responses."""

    sql_question_to_plan = {
        case["question"]: case["plan"]
        for case in SQL_PROMPT_CASES
    }
    sql_question_to_selection = {
        case["question"]: case.get("selection") or _derive_selection_from_case(case)
        for case in SQL_PROMPT_CASES
    }
    mongo_question_to_plan = {
        case["question"]: case["plan"]
        for case in MONGODB_PROMPT_CASES
    }
    mongo_question_to_selection = {
        case["question"]: case.get("selection") or _derive_selection_from_case(case)
        for case in MONGODB_PROMPT_CASES
    }
    repair_plan_by_question = {
        "show user emails": {
            "tables": ["users"],
            "fields": ["users.email"],
            "filters": [],
            "joins": [],
            "aggregations": [],
            "group_by": [],
            "order_by": [],
            "limit": 10,
        }
    }

    def fake_generate_json(prompt: str) -> dict[str, Any]:
        is_schema_selection_prompt = "Return this schema selection shape" in prompt
        is_sql_prompt = "SQL query" in prompt or "SQL schema context" in prompt
        is_mongo_prompt = "MongoDB query" in prompt or "MongoDB schema context" in prompt

        if "Semantic checklist:" in prompt:
            return {
                "passed": True,
                "reason": "The plan matches the question and schema.",
                "repaired_plan": None,
            }

        if "Return this schema selection shape" in prompt:
            question = _extract_prompt_value(prompt, "Question")
            if question is None:
                raise AssertionError("Could not find Question in schema-selection prompt during test.")

            if is_sql_prompt:
                selected_schema = sql_question_to_selection.get(question)
            elif is_mongo_prompt:
                selected_schema = mongo_question_to_selection.get(question)
            else:
                selected_schema = None

            if selected_schema is None:
                selected_schema = _derive_selection_from_question(question, is_mongo_prompt)

            if selected_schema is None:
                raise AssertionError(f"Unexpected schema-selection question in test: {question}")

            return selected_schema

        original_question = _extract_prompt_value(prompt, "Original question")
        if original_question is not None:
            repaired_plan = repair_plan_by_question.get(original_question)
            if repaired_plan is None:
                raise AssertionError(f"Unexpected repair question in test: {original_question}")
            return repaired_plan

        question = _extract_prompt_value(prompt, "Question")
        if question is None:
            raise AssertionError("Could not find Question in Ollama prompt during test.")

        if is_schema_selection_prompt:
            raise AssertionError("Schema selection prompt should have returned earlier.")

        if is_sql_prompt:
            planned_response = sql_question_to_plan.get(question)
        elif is_mongo_prompt:
            planned_response = mongo_question_to_plan.get(question)
        else:
            planned_response = None

        if planned_response is None:
            if _is_destructive_question(question):
                return {"error": "Only read-only queries are allowed."}

            raise AssertionError(f"Unexpected planner question in test: {question}")

        return planned_response

    def fake_generate_text(prompt: str) -> str:
        question = _extract_prompt_value(prompt, "Question")
        return f"Mock explanation for: {question or 'query result'}"

    monkeypatch.setattr("core.query_graph.generate_json", fake_generate_json)
    monkeypatch.setattr("core.query_graph.generate_text", fake_generate_text)
    monkeypatch.setattr("core.query_graph.get_active_model_name", lambda: "mock-ollama")
    monkeypatch.setattr("core.eval_runner.get_active_model_name", lambda: "mock-ollama")


def _extract_prompt_value(prompt: str, label: str) -> str | None:
    """Pull one labeled block value out of a multiline prompt."""

    pattern = rf"{re.escape(label)}:\s*\n(.+?)(?:\n[A-Z][^:\n]*:\n|\Z)"
    match = re.search(pattern, prompt, re.DOTALL)
    if not match:
        return None

    return match.group(1).strip()


def _derive_selection_from_case(case: dict[str, Any]) -> dict[str, Any]:
    """Create a small schema-selection response when the case did not spell one out."""

    plan = case["plan"]

    if "tables" in plan:
        return {"tables": list(plan["tables"])}

    if "collection" in plan:
        return {"collections": [str(plan["collection"])]}

    question = str(case["question"]).lower()
    if "order" in question:
        return {"tables": ["orders"]}

    return {"tables": ["users"]}


def _derive_selection_from_question(question: str, is_mongo_prompt: bool) -> dict[str, Any] | None:
    """Pick a nearest schema entity for ad-hoc mocked questions."""

    lowered_question = question.lower()

    if is_mongo_prompt:
        if "order" in lowered_question:
            return {"collections": ["orders"]}
        return {"collections": ["users"]}

    if "order" in lowered_question:
        return {"tables": ["orders"]}

    if "user" in lowered_question:
        return {"tables": ["users"]}

    return None


def _is_destructive_question(question: str) -> bool:
    """Return True when the prompt clearly asks for a write or schema-changing action."""

    lowered_question = question.lower()
    destructive_keywords = (
        "delete",
        "drop",
        "truncate",
        "update",
        "insert",
        "create",
        "alter",
        "remove",
        "destroy",
    )
    return any(keyword in lowered_question for keyword in destructive_keywords)


def _resolve_test_runtime_root() -> Path:
    """Return one writable temp root, falling back to a gitignored repo folder."""

    candidate_roots = [
        candidate / "nl-query-copilot-tests"
        for candidate in iter_temp_root_candidates()
    ]
    candidate_roots.append(BACKEND_ROOT / "test_runtime")

    candidate_root = pick_first_writable_directory(candidate_roots)
    if candidate_root is not None:
        return candidate_root

    raise RuntimeError("Could not create a writable pytest runtime directory.")


TEST_RUNTIME_ROOT = _resolve_test_runtime_root()
TEST_TEMP_DIR = Path(tempfile.mkdtemp(prefix="pytest-", dir=TEST_RUNTIME_ROOT))

os.environ["TEMP"] = str(TEST_TEMP_DIR)
os.environ["TMP"] = str(TEST_TEMP_DIR)
os.environ["TMPDIR"] = str(TEST_TEMP_DIR)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
tempfile.tempdir = str(TEST_TEMP_DIR)
sys.dont_write_bytecode = True


def _cleanup_test_temp_dir() -> None:
    """Best-effort cleanup for the per-run pytest temp directory."""

    shutil.rmtree(TEST_TEMP_DIR, ignore_errors=True)


atexit.register(_cleanup_test_temp_dir)

from core.eval_fixtures import cleanup_demo_sqlite_database, create_demo_sqlite_database
from core.schema_extractor import extract_schema
from tests.mongodb_prompt_matrix import MONGODB_PROMPT_CASES
from tests.sql_prompt_matrix import SQL_PROMPT_CASES
