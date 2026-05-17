"""LangGraph workflow that uses Ollama for planning, repair, and explanation."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any, Generator, TypedDict

from langgraph.graph import END, START, StateGraph

from config import settings
from core.compiler import compile_query
from core.executor import execute_query
from core.ollama_client import generate_json, generate_text, get_active_model_name
from core.retriever import retrieve_relevant_schema
from core.safety import check_safety
from core.validator import validate_query

DEFAULT_LIMIT = 10
MAX_REPAIR_ATTEMPTS = 2
STRUCTURED_REPAIR_MODEL_ORDER = (
    "qwen2.5:3b",
)
PLACEHOLDER_MODEL_ERRORS = {
    "short clear reason",
    "clear reason",
    "reason",
    "actual specific reason",
    "specific reason",
    "actual reason",
}
PLACEHOLDER_TEMPLATE_WORDS = {
    "short",
    "clear",
    "reason",
    "actual",
    "specific",
    "concrete",
    "brief",
    "detailed",
    "real",
    "helpful",
    "useful",
    "meaningful",
    "error",
}
CONCRETE_ERROR_OBJECT_INSTRUCTIONS = """
If the request is destructive, write-focused, ambiguous, or unsupported, return one JSON error object with a real sentence, for example:
{"error": "Only read-only questions are supported."}
{"error": "The question is ambiguous because it does not specify which field to summarize."}
Use a real sentence in the error field, not a placeholder or template fragment.
""".strip()


class QueryGraphError(ValueError):
    """Error raised when the graph fails after collecting a useful trace."""

    def __init__(self, message: str, trace: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.trace = {
            "steps": trace,
            "node_count": len(trace),
        }


class QueryGraphState(TypedDict, total=False):
    """State carried across the LangGraph workflow."""

    question: str
    backend: str
    database_schema: dict[str, Any]
    connection: Any
    trace_steps: list[dict[str, Any]]
    relevant_schema: dict[str, Any]
    query_plan: dict[str, Any]
    compiled_query: str
    selection_mode: str
    planning_mode: str
    semantic_passed: bool
    semantic_reason: str
    safety_passed: bool
    safety_reason: str
    validation_passed: bool
    validation_reason: str
    result: dict[str, Any]
    repaired: bool
    repair_attempts: int
    explanation: str


def run_query_graph(
    question: str,
    backend: str,
    database_schema: dict[str, Any],
    connection: Any,
) -> dict[str, Any]:
    """Run the model-backed query workflow and return the finished state."""

    graph = _get_query_graph()
    final_state = graph.invoke(
        {
            "question": question,
            "backend": backend,
            "database_schema": database_schema,
            "connection": connection,
            "repair_attempts": 0,
            "repaired": False,
        }
    )

    return _build_graph_result(final_state)


def stream_query_graph(
    question: str,
    backend: str,
    database_schema: dict[str, Any],
    connection: Any,
) -> Generator[dict[str, Any], None, dict[str, Any]]:
    """Yield workflow progress events while running the graph and return the final result."""

    graph = _get_query_graph()
    initial_state: QueryGraphState = {
        "question": question,
        "backend": backend,
        "database_schema": database_schema,
        "connection": connection,
        "repair_attempts": 0,
        "repaired": False,
    }
    current_state: QueryGraphState = dict(initial_state)
    workflow_name = "langgraph"
    active_model_name = get_active_model_name()

    yield {
        "type": "workflow_started",
        "workflow": workflow_name,
        "model": active_model_name,
        "backend": backend,
    }

    for update in graph.stream(initial_state, stream_mode="updates"):
        for node_name, node_update in update.items():
            if not isinstance(node_update, dict):
                continue

            current_state.update(node_update)
            trace_steps = list(current_state.get("trace_steps", []))
            latest_step = trace_steps[-1] if trace_steps else {
                "name": node_name,
                "status": "success",
                "summary": "Step finished.",
                "details": {},
            }
            yield {
                "type": "step",
                "workflow": workflow_name,
                "model": active_model_name,
                "step": latest_step,
                "trace": {
                    "steps": trace_steps,
                    "node_count": len(trace_steps),
                },
            }

    final_result = _build_graph_result(current_state)
    yield {
        "type": "completed",
        "workflow": workflow_name,
        "model": active_model_name,
        "response": final_result,
    }

    return final_result


def _build_graph_result(final_state: QueryGraphState) -> dict[str, Any]:
    """Turn the final graph state into the API response payload shape."""

    return {
        "relevant_schema": final_state["relevant_schema"],
        "query_plan": final_state["query_plan"],
        "compiled_query": final_state["compiled_query"],
        "trace": {
            "steps": list(final_state.get("trace_steps", [])),
            "node_count": len(final_state.get("trace_steps", [])),
        },
        "safety_check": (
            bool(final_state.get("safety_passed", False)),
            str(final_state.get("safety_reason", "")),
        ),
        "validation_check": (
            bool(final_state.get("validation_passed", False)),
            str(final_state.get("validation_reason", "")),
        ),
        "result": final_state["result"],
        "repaired": bool(final_state.get("repaired", False)),
        "repair_attempts": int(final_state.get("repair_attempts", 0)),
        "explanation": str(final_state.get("explanation", "")),
        "workflow": "langgraph",
        "model": get_active_model_name(),
    }


def plan_query_preview(
    question: str,
    backend: str,
    database_schema: dict[str, Any],
) -> dict[str, Any]:
    """Run the model-backed planning path without execution for compare mode."""

    state: QueryGraphState = {
        "question": question,
        "backend": backend,
        "database_schema": database_schema,
        "repair_attempts": 0,
        "repaired": False,
        "trace_steps": [],
    }

    state.update(_select_relevant_schema_node(state))
    state.update(_plan_query_node(state))
    state.update(_semantic_check_node(state))
    state.update(_compile_query_node(state))
    state.update(_check_safety_node(state))
    state.update(_check_validation_node(state))

    if not state.get("validation_passed"):
        raise QueryGraphError(
            f"Validation failed: {state.get('validation_reason', 'Unknown validation error.')}",
            list(state.get("trace_steps", [])),
        )

    return {
        "relevant_schema": state["relevant_schema"],
        "query_plan": state["query_plan"],
        "compiled_query": state["compiled_query"],
        "trace": {
            "steps": list(state.get("trace_steps", [])),
            "node_count": len(state.get("trace_steps", [])),
        },
        "safety_check": (
            bool(state.get("safety_passed", False)),
            str(state.get("safety_reason", "")),
        ),
        "validation_check": (
            bool(state.get("validation_passed", False)),
            str(state.get("validation_reason", "")),
        ),
        "workflow": "langgraph",
        "model": get_active_model_name(),
    }


@lru_cache(maxsize=1)
def _get_query_graph():
    """Build and cache the LangGraph workflow."""

    graph = StateGraph(QueryGraphState)
    graph.add_node("select_schema", _select_relevant_schema_node)
    graph.add_node("plan_query", _plan_query_node)
    graph.add_node("semantic_check", _semantic_check_node)
    graph.add_node("compile_query", _compile_query_node)
    graph.add_node("check_safety", _check_safety_node)
    graph.add_node("check_validation", _check_validation_node)
    graph.add_node("repair_query", _repair_query_node)
    graph.add_node("execute_query", _execute_query_node)
    graph.add_node("explain_result", _explain_result_node)

    graph.add_edge(START, "select_schema")
    graph.add_edge("select_schema", "plan_query")
    graph.add_edge("plan_query", "semantic_check")
    graph.add_edge("semantic_check", "compile_query")
    graph.add_edge("compile_query", "check_safety")
    graph.add_edge("check_safety", "check_validation")
    graph.add_conditional_edges(
        "check_validation",
        _route_after_validation,
        {
            "execute_query": "execute_query",
            "repair_query": "repair_query",
            "fail": END,
        },
    )
    graph.add_conditional_edges(
        "execute_query",
        _route_after_execution,
        {
            "explain_result": "explain_result",
            "repair_query": "repair_query",
            "fail": END,
        },
    )
    graph.add_edge("repair_query", "compile_query")
    graph.add_edge("explain_result", END)

    return graph.compile()


def _select_relevant_schema_node(state: QueryGraphState) -> QueryGraphState:
    """Ask Ollama to choose the schema slice needed for this question."""

    if _should_use_python_schema_selection():
        try:
            relevant_schema = retrieve_relevant_schema(
                question=state["question"],
                full_schema=state["database_schema"],
                backend=state["backend"],
            )
            selected_names = _get_schema_entity_names(relevant_schema)
            summary = (
                f"Selected {len(selected_names)} schema item{'s' if len(selected_names) != 1 else ''} "
                "for the question."
            )
            return _with_trace(
                state,
                "select_schema",
                "success",
                summary,
                {
                    "selection_mode": "python",
                    "selected_names": selected_names,
                },
                relevant_schema=relevant_schema,
                selection_mode="python",
            )
        except ValueError:
            pass

    try:
        raw_selection = generate_json(
            _build_schema_selection_prompt(
                question=state["question"],
                backend=state["backend"],
                database_schema=state["database_schema"],
            )
        )
        relevant_schema = _normalize_schema_selection(
            raw_selection=raw_selection,
            backend=state["backend"],
            database_schema=state["database_schema"],
            placeholder_fallback=(
                "Ollama did not choose a usable schema slice. "
                "It returned a placeholder error instead of a real reason."
            ),
        )
    except ValueError as exc:
        _raise_graph_error(
            message=str(exc),
            state=state,
            name="select_schema",
            status="failed",
            summary="Ollama could not choose a usable schema slice for the question.",
            details={"reason": str(exc)},
        )

    selected_names = _get_schema_entity_names(relevant_schema)
    summary = f"Selected {len(selected_names)} schema item{'s' if len(selected_names) != 1 else ''} for the question."

    return _with_trace(
        state,
        "select_schema",
        "success",
        summary,
        {
            "selection_mode": "model",
            "selected_names": selected_names,
        },
        relevant_schema=relevant_schema,
        selection_mode="model",
    )


def _plan_query_node(state: QueryGraphState) -> QueryGraphState:
    """Use Ollama to build a structured query plan."""

    planning_attempts = 1
    retry_placeholder_fallback = (
        "Ollama did not return a usable query plan after retrying. "
        "It kept returning placeholder text instead of a real reason."
    )

    if _should_use_python_fast_path():
        query_plan = _build_python_fast_path_plan(
            question=state["question"],
            backend=state["backend"],
            database_schema=state["database_schema"],
        )
        if query_plan is not None:
            query_plan, guardrail_reason = _apply_question_guardrails(
                question=state["question"],
                backend=state["backend"],
                query_plan=query_plan,
            )
            return _with_trace(
                state,
                "plan_query",
                "success",
                "Python matched a fast-path query plan and skipped Ollama planning.",
                {
                    **_summarize_plan(query_plan),
                    "planning_mode": "python-fast-path",
                    "planning_attempts": 0,
                    "guardrail_reason": guardrail_reason,
                },
                query_plan=query_plan,
                planning_mode="python-fast-path",
            )

    try:
        prompt = _build_planning_prompt(
            question=state["question"],
            backend=state["backend"],
            relevant_schema=state["relevant_schema"],
        )
        raw_plan = generate_json(prompt)
        raw_plan_error = _extract_model_error(raw_plan)

        if raw_plan_error and _should_retry_planning_error(state["question"], raw_plan_error):
            planning_attempts = 2
            retry_prompt = _build_planning_retry_prompt(
                question=state["question"],
                backend=state["backend"],
                relevant_schema=state["relevant_schema"],
                previous_response=raw_plan,
                previous_issue=raw_plan_error,
            )
            raw_plan = generate_json(retry_prompt)
            raw_plan_error = _extract_model_error(raw_plan)

        try:
            query_plan = _normalize_query_plan(
                raw_plan,
                state["backend"],
                placeholder_fallback=retry_placeholder_fallback
                if planning_attempts > 1
                else (
                    "Ollama did not return a usable query plan. "
                    "It returned a placeholder error instead of a real reason."
                ),
            )
        except ValueError as exc:
            if planning_attempts > 1 or not _is_placeholder_model_error(raw_plan_error or str(exc)):
                raise

            planning_attempts = 2
            retry_prompt = _build_planning_retry_prompt(
                question=state["question"],
                backend=state["backend"],
                relevant_schema=state["relevant_schema"],
                previous_response=raw_plan,
                previous_issue=raw_plan_error or str(exc),
            )
            query_plan = _normalize_query_plan(
                generate_json(retry_prompt),
                state["backend"],
                placeholder_fallback=retry_placeholder_fallback,
            )

        query_plan, canonicalization_reason = _canonicalize_model_query_plan(
            question=state["question"],
            backend=state["backend"],
            query_plan=query_plan,
            database_schema=state["database_schema"],
        )
        query_plan, guardrail_reason = _apply_question_guardrails(
            question=state["question"],
            backend=state["backend"],
            query_plan=query_plan,
        )
    except ValueError as exc:
        _raise_graph_error(
            message=str(exc),
            state=state,
            name="plan_query",
            status="failed",
            summary="Ollama could not build a usable query plan.",
            details={
                "reason": str(exc),
                "planning_attempts": planning_attempts,
            },
        )

    return _with_trace(
        state,
        "plan_query",
        "success",
        "Ollama built a structured query plan from the selected schema.",
        {
            **_summarize_plan(query_plan),
            "planning_mode": "model",
            "planning_attempts": planning_attempts,
            "guardrail_reason": guardrail_reason,
            "canonicalization_reason": canonicalization_reason,
        },
        query_plan=query_plan,
        planning_mode="model",
    )


def _semantic_check_node(state: QueryGraphState) -> QueryGraphState:
    """Ask Ollama to verify that the plan matches the question before compilation."""

    if state.get("planning_mode") == "python-fast-path":
        checked_plan, guardrail_reason = _apply_question_guardrails(
            question=state["question"],
            backend=state["backend"],
            query_plan=state["query_plan"],
        )
        try:
            _assert_question_plan_alignment(
                question=state["question"],
                backend=state["backend"],
                query_plan=checked_plan,
            )
        except ValueError as exc:
            _raise_graph_error(
                message=str(exc),
                state=state,
                name="semantic_check",
                status="failed",
                summary="Python fast-path planning produced a semantically invalid query.",
                details={"reason": str(exc)},
            )

        reason = "Skipped model semantic review because Python built a deterministic fast-path plan."
        if guardrail_reason:
            reason = f"{reason} {guardrail_reason}"

        return _with_trace(
            state,
            "semantic_check",
            "success",
            "Python fast path already matched the question and schema.",
            {
                "semantic_mode": "python-fast-path",
                "passed": True,
                "reason": reason,
                "repaired_plan": {},
            },
            query_plan=checked_plan,
            semantic_passed=True,
            semantic_reason=reason,
        )

    checked_plan: dict[str, Any]
    reason = ""
    was_repaired = False
    repair_attempts = int(state.get("repair_attempts", 0))
    semantic_mode = "model"

    if _should_skip_model_semantic_review(
        question=state["question"],
        backend=state["backend"],
        query_plan=state["query_plan"],
        planning_attempts=int(state.get("trace_steps", [{}])[-1].get("details", {}).get("planning_attempts", 1))
        if state.get("trace_steps")
        else 1,
    ):
        checked_plan = state["query_plan"]
        checked_plan, guardrail_reason = _apply_question_guardrails(
            question=state["question"],
            backend=state["backend"],
            query_plan=checked_plan,
        )
        _assert_question_plan_alignment(
            question=state["question"],
            backend=state["backend"],
            query_plan=checked_plan,
        )
        reason = "Skipped model semantic review because deterministic guardrails already confirmed the model plan."
        if guardrail_reason:
            reason = f"{reason} {guardrail_reason}"
        semantic_mode = "python-guardrails"
        return _with_trace(
            state,
            "semantic_check",
            "success",
            "Deterministic guardrails confirmed the model plan before compilation.",
            {
                "semantic_mode": semantic_mode,
                "passed": True,
                "reason": reason,
                "repaired_plan": {},
            },
            query_plan=checked_plan,
            semantic_passed=True,
            semantic_reason=reason,
        )

    try:
        raw_check = generate_json(
            _build_semantic_check_prompt(
                question=state["question"],
                backend=state["backend"],
                relevant_schema=state["relevant_schema"],
                query_plan=state["query_plan"],
            )
        )
        try:
            checked_plan, reason, was_repaired = _normalize_semantic_check(
                raw_check=raw_check,
                backend=state["backend"],
                current_plan=state["query_plan"],
                placeholder_fallback="Ollama rejected the query plan without giving a concrete reason.",
            )
        except ValueError as exc:
            repair_attempts += 1
            normalized_reason = _normalize_model_message(
                str(raw_check.get("reason", "")).strip(),
                empty_fallback="The plan does not answer the question safely.",
                placeholder_fallback="Ollama rejected the query plan without giving a concrete reason.",
            )
            if _extract_semantic_repaired_plan(
                raw_check,
                current_plan=state["query_plan"],
                passed=bool(raw_check.get("passed", False)),
                normalized_reason=normalized_reason,
            ) is None:
                reason = f"Ollama rejected the plan during semantic review, then repaired it again: {exc}"
            else:
                reason = f"Ollama returned an invalid semantic repair, then repaired it again: {exc}"
            checked_plan = _repair_plan_with_model(
                state=state,
                error_message=reason,
                repair_attempts=repair_attempts,
            )
            was_repaired = True

        checked_plan, canonicalization_reason = _canonicalize_model_query_plan(
            question=state["question"],
            backend=state["backend"],
            query_plan=checked_plan,
            database_schema=state["database_schema"],
        )
        if canonicalization_reason:
            was_repaired = True
            reason = "; ".join(part for part in (reason, canonicalization_reason) if part)

        try:
            _assert_question_plan_alignment(
                question=state["question"],
                backend=state["backend"],
                query_plan=checked_plan,
            )
        except ValueError as exc:
            repair_attempts += 1
            reason = f"Ollama returned a semantically inconsistent plan, then repaired it again: {exc}"
            checked_plan = _repair_plan_with_model(
                state=state,
                error_message=reason,
                repair_attempts=repair_attempts,
            )
            was_repaired = True

        checked_plan, guardrail_reason = _apply_question_guardrails(
            question=state["question"],
            backend=state["backend"],
            query_plan=checked_plan,
        )
        if guardrail_reason:
            was_repaired = True
            reason = "; ".join(part for part in (reason, guardrail_reason) if part)
    except ValueError as exc:
        _raise_graph_error(
            message=str(exc),
            state=state,
            name="semantic_check",
            status="failed",
            summary="Ollama rejected the query plan during semantic review.",
            details={"reason": str(exc)},
        )

    if was_repaired:
        repair_attempts = max(repair_attempts, int(state.get("repair_attempts", 0)) + 1)

    return _with_trace(
        state,
        "semantic_check",
        "repaired" if was_repaired else "success",
        "Ollama revised the plan so it better matches the question."
        if was_repaired
        else "Ollama confirmed the plan matches the question and schema.",
        {
            "semantic_mode": semantic_mode,
            "passed": True,
            "reason": reason or "Passed",
            "repaired_plan": _summarize_plan(checked_plan) if was_repaired else {},
        },
        query_plan=checked_plan,
        semantic_passed=True,
        semantic_reason=reason,
        repaired=bool(state.get("repaired", False) or was_repaired),
        repair_attempts=repair_attempts,
    )


def _compile_query_node(state: QueryGraphState) -> QueryGraphState:
    """Compile the structured plan into executable backend syntax."""

    try:
        compiled_query = compile_query(state["query_plan"], state["backend"])
    except ValueError as exc:
        repair_attempts = int(state.get("repair_attempts", 0))
        if repair_attempts >= MAX_REPAIR_ATTEMPTS:
            _raise_graph_error(
                message=str(exc),
                state=state,
                name="compile_query",
                status="failed",
                summary="The structured plan could not be compiled for the selected backend.",
                details={"reason": str(exc)},
            )

        repair_attempts += 1
        repaired_plan = _repair_plan_with_model(
            state=state,
            error_message=f"Compilation failed: {exc}",
            repair_attempts=repair_attempts,
        )

        try:
            compiled_query = compile_query(repaired_plan, state["backend"])
        except ValueError as repair_exc:
            _raise_graph_error(
                message=str(repair_exc),
                state=state,
                name="compile_query",
                status="failed",
                summary="Ollama repaired the plan, but it still could not be compiled.",
                details={
                    "original_reason": str(exc),
                    "repair_reason": str(repair_exc),
                    "attempt": repair_attempts,
                    "repaired_plan": _summarize_plan(repaired_plan),
                },
            )

        return _with_trace(
            state,
            "compile_query",
            "repaired",
            "Ollama repaired the plan after compilation failed, then Python compiled it.",
            {
                "reason": str(exc),
                "attempt": repair_attempts,
                "repaired_plan": _summarize_plan(repaired_plan),
                "compiled_query": compiled_query,
            },
            query_plan=repaired_plan,
            compiled_query=compiled_query,
            repaired=True,
            repair_attempts=repair_attempts,
        )

    return _with_trace(
        state,
        "compile_query",
        "success",
        "Compiled the plan into backend-ready query text.",
        {"compiled_query": compiled_query},
        compiled_query=compiled_query,
    )


def _check_safety_node(state: QueryGraphState) -> QueryGraphState:
    """Run the read-only safety check."""

    passed, reason = check_safety(
        state["compiled_query"],
        state["backend"],
        state["database_schema"],
    )

    if not passed:
        _raise_graph_error(
            message=f"Safety check failed: {reason}",
            state=state,
            name="check_safety",
            status="failed",
            summary="The query was blocked by the read-only safety rules.",
            details={"passed": passed, "reason": reason},
        )

    return _with_trace(
        state,
        "check_safety",
        "success",
        "Passed the read-only safety check.",
        {"passed": passed, "reason": reason or "Passed"},
        safety_passed=passed,
        safety_reason=reason,
    )


def _check_validation_node(state: QueryGraphState) -> QueryGraphState:
    """Validate syntax and schema references before execution."""

    passed, reason = validate_query(
        state["compiled_query"],
        state["backend"],
        state["database_schema"],
    )
    return _with_trace(
        state,
        "check_validation",
        "success" if passed else "failed",
        "Validated the compiled query against syntax and schema rules."
        if passed
        else "Validation found a syntax or schema issue that may need repair.",
        {"passed": passed, "reason": reason or "Passed"},
        validation_passed=passed,
        validation_reason=reason,
    )


def _repair_query_node(state: QueryGraphState) -> QueryGraphState:
    """Ask Ollama for a corrected plan after validation or execution failure."""

    repair_attempts = int(state.get("repair_attempts", 0)) + 1
    error_message = str(state.get("validation_reason") or _extract_execution_error(state))
    repair_mode = "model"
    repaired_plan = _repair_plan_with_python(
        state=state,
        error_message=error_message,
    )

    if repaired_plan is None:
        repaired_plan = _repair_plan_with_model(
            state=state,
            error_message=error_message,
            repair_attempts=repair_attempts,
        )
    else:
        repair_mode = "python"

    if repaired_plan == state["query_plan"]:
        _raise_graph_error(
            message=f"Ollama returned the same plan during repair: {error_message}",
            state=state,
            name="repair_query",
            status="failed",
            summary="Ollama repair did not change the failed plan.",
            details={
                "attempt": repair_attempts,
                "trigger_reason": error_message,
            },
        )

    return _with_trace(
        state,
        "repair_query",
        "repaired",
        f"Ollama repaired the plan on attempt {repair_attempts} after validation or execution failed.",
        {
            "attempt": repair_attempts,
            "trigger_reason": error_message,
            "repair_mode": repair_mode,
            "repaired_plan": _summarize_plan(repaired_plan),
        },
        query_plan=repaired_plan,
        repair_attempts=repair_attempts,
        repaired=True,
        validation_passed=False,
        validation_reason="",
    )


def _repair_plan_with_model(
    state: QueryGraphState,
    error_message: str,
    repair_attempts: int,
) -> dict[str, Any]:
    """Ask Ollama for a corrected plan and normalize the returned shape."""

    prompt = (
        _build_count_repair_prompt(
            question=state["question"],
            backend=state["backend"],
            relevant_schema=state["relevant_schema"],
            failed_plan=state["query_plan"],
            failed_query=str(state.get("compiled_query", "")),
            error_message=error_message,
        )
        if _is_count_question(state["question"])
        else _build_repair_prompt(
            question=state["question"],
            backend=state["backend"],
            relevant_schema=state["relevant_schema"],
            failed_plan=state["query_plan"],
            failed_query=str(state.get("compiled_query", "")),
            error_message=error_message,
        )
    )
    model_errors: dict[str, str] = {}

    for model_name in STRUCTURED_REPAIR_MODEL_ORDER:
        try:
            raw_plan = generate_json(prompt, candidate_models=[model_name])
            repaired_plan = _normalize_query_plan(
                raw_plan,
                state["backend"],
                placeholder_fallback=(
                    "Ollama could not repair the query plan because it returned a "
                    "placeholder error instead of a real reason."
                ),
            )
            repaired_plan, _ = _canonicalize_model_query_plan(
                question=state["question"],
                backend=state["backend"],
                query_plan=repaired_plan,
                database_schema=state["database_schema"],
            )
            repaired_plan, _ = _apply_question_guardrails(
                question=state["question"],
                backend=state["backend"],
                query_plan=repaired_plan,
            )
            _assert_question_plan_alignment(
                question=state["question"],
                backend=state["backend"],
                query_plan=repaired_plan,
            )
            return repaired_plan
        except ValueError as exc:
            model_errors[model_name] = str(exc)

    last_error = next(reversed(model_errors.values()), "Unknown repair failure.")
    _raise_graph_error(
        message=last_error,
        state=state,
        name="repair_query",
        status="failed",
        summary="Ollama could not repair the query plan.",
        details={
            "reason": last_error,
            "attempt": repair_attempts,
            "trigger_reason": error_message,
            "model_errors": model_errors,
        },
    )


def _repair_plan_with_python(
    state: QueryGraphState,
    error_message: str,
) -> dict[str, Any] | None:
    """Try one deterministic SQL repair before asking the model again."""

    if state["backend"] not in {"sqlite", "postgresql"}:
        return None

    repaired_plan = _repair_sql_plan_from_validation_error(
        question=state["question"],
        query_plan=state["query_plan"],
        database_schema=state["database_schema"],
        error_message=error_message,
    )
    if repaired_plan is None:
        return None

    try:
        compiled_query = compile_query(repaired_plan, state["backend"])
    except ValueError:
        return None

    passed, _ = validate_query(
        compiled_query,
        state["backend"],
        state["database_schema"],
    )
    if not passed:
        return None

    return repaired_plan


def _execute_query_node(state: QueryGraphState) -> QueryGraphState:
    """Execute the compiled read-only query."""

    collection_name = _get_collection_name(state["query_plan"], state["backend"])
    result = execute_query(
        state["compiled_query"],
        state["backend"],
        state["connection"],
        collection=collection_name,
    )
    if "error" in result:
        return _with_trace(
            state,
            "execute_query",
            "failed",
            "Execution hit a backend error and may need repair.",
            {"error": result["error"]},
            result=result,
        )

    return _with_trace(
        state,
        "execute_query",
        "success",
        "Executed the query and collected result rows.",
        {
            "row_count": result.get("row_count", 0),
            "columns": result.get("columns", []),
        },
        result=result,
    )


def _explain_result_node(state: QueryGraphState) -> QueryGraphState:
    """Ask Ollama to summarize the query result for the user."""

    if _should_use_python_explanation():
        explanation = _build_python_result_explanation(
            question=state["question"],
            result=state["result"],
        )
        explanation_mode = "python"
        summary = "Python summarized the result for the user."
    else:
        explanation = generate_text(
            _build_explanation_prompt(
                question=state["question"],
                result=state["result"],
                compiled_query=state["compiled_query"],
                backend=state["backend"],
            )
        )
        explanation_mode = "model"
        summary = "Ollama summarized the result for the user."

    return _with_trace(
        state,
        "explain_result",
        "success",
        summary,
        {
            "explanation_mode": explanation_mode,
            "preview": explanation[:160],
        },
        explanation=explanation,
    )


def _route_after_validation(state: QueryGraphState) -> str:
    """Choose whether to execute, repair, or stop after validation."""

    if state.get("validation_passed"):
        return "execute_query"

    if int(state.get("repair_attempts", 0)) < MAX_REPAIR_ATTEMPTS:
        return "repair_query"

    raise QueryGraphError(
        f"Validation failed: {state.get('validation_reason', 'Unknown validation error.')}",
        list(state.get("trace_steps", [])),
    )


def _route_after_execution(state: QueryGraphState) -> str:
    """Choose whether to explain, repair, or stop after execution."""

    result = state.get("result", {})
    if "error" not in result:
        return "explain_result"

    if int(state.get("repair_attempts", 0)) < MAX_REPAIR_ATTEMPTS:
        return "repair_query"

    raise QueryGraphError(
        f"Query execution failed: {result['error']}",
        list(state.get("trace_steps", [])),
    )


def _normalize_query_plan(
    raw_plan: dict[str, Any],
    backend: str,
    placeholder_fallback: str = "Ollama did not return a usable query plan.",
) -> dict[str, Any]:
    """Validate the shape returned by Ollama and keep only supported fields."""

    if not isinstance(raw_plan, dict):
        raise ValueError("The model did not return a plan object.")

    error_message = str(raw_plan.get("error", "")).strip()
    if error_message:
        raise ValueError(
            _normalize_model_message(
                error_message,
                empty_fallback=placeholder_fallback,
                placeholder_fallback=placeholder_fallback,
            )
        )

    if backend in {"sqlite", "postgresql"}:
        return _normalize_sql_plan(raw_plan)

    if backend == "mongodb":
        return _normalize_mongodb_plan(raw_plan)

    raise ValueError(f"Unsupported backend: {backend}")


def _is_placeholder_model_error(message: str) -> bool:
    """Return whether Ollama copied a prompt placeholder instead of reasoning."""

    normalized = _normalize_model_text(message)
    if normalized in PLACEHOLDER_MODEL_ERRORS:
        return True

    tokens = normalized.split()
    return (
        0 < len(tokens) <= 5
        and ("reason" in tokens or "error" in tokens)
        and set(tokens).issubset(PLACEHOLDER_TEMPLATE_WORDS)
    )


def _normalize_model_text(message: str) -> str:
    """Collapse punctuation and repeated whitespace before placeholder checks."""

    return " ".join(part for part in re.split(r"[^a-z0-9]+", message.lower()) if part)


def _normalize_model_message(
    message: str,
    *,
    empty_fallback: str,
    placeholder_fallback: str,
) -> str:
    """Return a concrete backend-safe message even when the model emits template text."""

    cleaned = message.strip()
    if not cleaned:
        return empty_fallback

    if _is_placeholder_model_error(cleaned):
        return placeholder_fallback

    return cleaned


def _extract_model_error(payload: Any) -> str:
    """Return the model error string from one structured payload when present."""

    if not isinstance(payload, dict):
        return ""

    return str(payload.get("error", "")).strip()


def _should_retry_planning_error(question: str, error_message: str) -> bool:
    """Decide whether one stricter planning retry is worth attempting."""

    if not error_message:
        return False

    if _is_placeholder_model_error(error_message):
        return True

    return not _looks_destructive_question(question)


def _looks_destructive_question(question: str) -> bool:
    """Return whether the natural-language request sounds write-focused."""

    normalized = _normalize_model_text(question)
    destructive_terms = {
        "delete",
        "drop",
        "remove",
        "truncate",
        "update",
        "insert",
        "create",
        "alter",
        "modify",
        "change",
        "rename",
        "replace",
        "overwrite",
        "set",
        "write",
    }
    return any(term in normalized.split() for term in destructive_terms)


def _normalize_schema_selection(
    raw_selection: dict[str, Any],
    backend: str,
    database_schema: dict[str, Any],
    placeholder_fallback: str = "Ollama did not choose a usable schema slice.",
) -> dict[str, Any]:
    """Validate and materialize the model's chosen schema subset."""

    if not isinstance(raw_selection, dict):
        raise ValueError("The model did not return a schema selection object.")

    error_message = str(raw_selection.get("error", "")).strip()
    if error_message:
        raise ValueError(
            _normalize_model_message(
                error_message,
                empty_fallback=placeholder_fallback,
                placeholder_fallback=placeholder_fallback,
            )
        )

    if backend in {"sqlite", "postgresql"}:
        available_tables = database_schema.get("tables", {})
        selected_names = _string_list(raw_selection.get("tables"))
        selected_tables = {
            table_name: available_tables[table_name]
            for table_name in selected_names
            if table_name in available_tables
        }

        if not selected_tables:
            raise ValueError("The model did not choose any valid SQL tables.")

        return {"tables": selected_tables}

    if backend == "mongodb":
        available_collections = database_schema.get("collections", {})
        selected_names = _string_list(raw_selection.get("collections"))
        selected_collections = {
            collection_name: available_collections[collection_name]
            for collection_name in selected_names
            if collection_name in available_collections
        }

        if not selected_collections:
            raise ValueError("The model did not choose any valid MongoDB collections.")

        return {"collections": selected_collections}

    raise ValueError(f"Unsupported backend: {backend}")


def _normalize_semantic_check(
    raw_check: dict[str, Any],
    backend: str,
    current_plan: dict[str, Any],
    placeholder_fallback: str = "Ollama rejected the query plan without giving a concrete reason.",
) -> tuple[dict[str, Any], str, bool]:
    """Normalize Ollama's semantic review and optional repaired plan."""

    if not isinstance(raw_check, dict):
        raise ValueError("The model did not return a semantic check object.")

    error_message = str(raw_check.get("error", "")).strip()
    if error_message:
        raise ValueError(
            _normalize_model_message(
                error_message,
                empty_fallback=placeholder_fallback,
                placeholder_fallback=placeholder_fallback,
            )
        )

    reason = str(raw_check.get("reason", "")).strip()
    normalized_reason = _normalize_model_message(
        reason,
        empty_fallback="The plan matches the question and schema."
        if bool(raw_check.get("passed", False))
        else "The plan does not answer the question safely.",
        placeholder_fallback=placeholder_fallback,
    )
    repaired_plan = _extract_semantic_repaired_plan(
        raw_check,
        current_plan=current_plan,
        passed=bool(raw_check.get("passed", False)),
        normalized_reason=normalized_reason,
    )

    if repaired_plan is not None:
        checked_plan = _normalize_query_plan(
            repaired_plan,
            backend,
            placeholder_fallback=(
                "Ollama returned an invalid semantic repair. "
                "It used a placeholder error instead of a repaired plan."
            ),
        )
        if checked_plan == current_plan and (
            not bool(raw_check.get("passed", False)) or _semantic_reason_requires_repair(normalized_reason)
        ):
            raise ValueError(normalized_reason)
        return checked_plan, normalized_reason, checked_plan != current_plan

    if bool(raw_check.get("passed", False)):
        if _semantic_reason_requires_repair(normalized_reason):
            raise ValueError(normalized_reason)
        return (
            current_plan,
            normalized_reason,
            False,
        )

    raise ValueError(
        normalized_reason
    )


def _semantic_reason_requires_repair(reason: str) -> bool:
    """Treat contradictory 'passed' messages as failed semantic review."""

    lowered_reason = reason.lower()
    contradiction_fragments = (
        "should include",
        "should use",
        "should filter",
        "should group",
        "should join",
        "should count",
        "should exclude",
        "should instead",
        "the plan does not",
        "the plan doesn't",
        "missing",
        "is not in the selected schema",
        "does not filter",
        "does not match",
        "does not answer",
        "incorrect",
        "wrong",
    )
    return any(fragment in lowered_reason for fragment in contradiction_fragments)


def _extract_semantic_repaired_plan(
    raw_check: dict[str, Any],
    *,
    current_plan: dict[str, Any],
    passed: bool,
    normalized_reason: str,
) -> dict[str, Any] | None:
    """Return the repaired plan from a semantic check response when one is present."""

    for key in ("repaired_plan", "corrected_plan"):
        value = raw_check.get(key)
        if isinstance(value, dict) and value:
            return value

    if passed and not _semantic_reason_requires_repair(normalized_reason):
        return None

    for key in ("query_plan", "plan"):
        value = raw_check.get(key)
        if isinstance(value, dict) and value and value != current_plan:
            return value

    return None


def _normalize_sql_plan(raw_plan: dict[str, Any]) -> dict[str, Any]:
    """Normalize a SQL plan into the shared compiler format."""

    tables = _string_list(raw_plan.get("tables"))
    if not tables:
        raise ValueError("The model did not choose any SQL tables.")

    aggregations = _normalize_aggregations(raw_plan.get("aggregations"))

    return {
        "operation": "select",
        "tables": tables,
        "fields": _string_list(raw_plan.get("fields")),
        "filters": _normalize_sql_filters(raw_plan.get("filters")),
        "joins": _normalize_sql_joins(raw_plan.get("joins")),
        "aggregations": aggregations,
        "group_by": _string_list(raw_plan.get("group_by")),
        "order_by": _normalize_sql_order_by(raw_plan.get("order_by")),
        "limit": _normalize_limit(raw_plan.get("limit")),
    }


def _normalize_mongodb_plan(raw_plan: dict[str, Any]) -> dict[str, Any]:
    """Normalize a MongoDB plan into the shared compiler format."""

    collection_name = str(raw_plan.get("collection", "")).strip()
    if not collection_name:
        raise ValueError("The model did not choose a MongoDB collection.")

    raw_aggregations = _normalize_aggregations(raw_plan.get("aggregations"))
    raw_group_by = _string_list(raw_plan.get("group_by"))
    operation = str(raw_plan.get("operation", "find")).lower()
    if raw_aggregations or raw_group_by:
        operation = "aggregate"

    if operation not in {"find", "aggregate"}:
        raise ValueError("MongoDB plan must use 'find' or 'aggregate'.")

    return {
        "operation": operation,
        "collection": collection_name,
        "intent": str(raw_plan.get("intent", "")).lower(),
        "match": raw_plan.get("match", {}) if isinstance(raw_plan.get("match", {}), dict) else {},
        "project": raw_plan.get("project", {}) if isinstance(raw_plan.get("project", {}), dict) else {},
        "sort": raw_plan.get("sort", {}) if isinstance(raw_plan.get("sort", {}), dict) else {},
        "group_by": raw_group_by,
        "aggregations": raw_aggregations,
        "limit": _normalize_limit(raw_plan.get("limit")),
    }


def _normalize_aggregations(value: Any) -> list[dict[str, Any]]:
    """Validate aggregation items enough to avoid compiler-level placeholders."""

    aggregations = _dict_list(value)

    for aggregation in aggregations:
        function_name = str(aggregation.get("function", "")).strip().upper()
        if not function_name:
            raise ValueError("Aggregation function is required")

        if function_name not in {"COUNT", "AVG", "SUM", "MIN", "MAX"}:
            raise ValueError(f"Unsupported aggregation function: {function_name}")

        field_name = str(aggregation.get("field", "")).strip()
        if not field_name:
            raise ValueError("Aggregation field is required")

    return aggregations


def _assert_question_plan_alignment(
    question: str,
    backend: str,
    query_plan: dict[str, Any],
) -> None:
    """Reject plans that contradict the clearest high-level user intent."""

    lowered_question = question.lower()

    if not _is_count_question(question):
        _assert_projection_alignment(lowered_question, backend, query_plan)
        _assert_filter_alignment(lowered_question, backend, query_plan)
        _assert_join_alignment(backend, query_plan)
        _assert_aggregation_alignment(lowered_question, backend, query_plan)
        _assert_sort_alignment(lowered_question, backend, query_plan)
        return

    if backend in {"sqlite", "postgresql"}:
        aggregations = query_plan.get("aggregations", [])
        has_count = any(
            str(aggregation.get("function", "")).strip().upper() == "COUNT"
            for aggregation in aggregations
            if isinstance(aggregation, dict)
        )
        if not has_count:
            raise ValueError("Count questions must use a COUNT aggregation.")

        group_by = _string_list(query_plan.get("group_by"))
        if not group_by and _string_list(query_plan.get("fields")):
            raise ValueError("Plain count questions must not select ordinary row fields.")

        _assert_filter_alignment(lowered_question, backend, query_plan)
        _assert_join_alignment(backend, query_plan)
        return

    if backend == "mongodb":
        intent = str(query_plan.get("intent", "")).strip().lower()
        aggregations = query_plan.get("aggregations", [])
        if intent == "count":
            return

        if any(
            str(aggregation.get("function", "")).strip().upper() == "COUNT"
            for aggregation in aggregations
            if isinstance(aggregation, dict)
        ):
            _assert_filter_alignment(lowered_question, backend, query_plan)
            return

        raise ValueError("Count questions must use MongoDB count intent or COUNT aggregation.")


def _assert_projection_alignment(
    lowered_question: str,
    backend: str,
    query_plan: dict[str, Any],
) -> None:
    """Catch the clearest field-selection mismatches before execution."""

    if backend not in {"sqlite", "postgresql"}:
        return

    if _is_count_question(lowered_question):
        return

    fields = _string_list(query_plan.get("fields"))
    if "email" in lowered_question and not any(field.endswith(".email") for field in fields):
        raise ValueError("Questions asking for email data must project an email field.")

    if "amount" in lowered_question and "average" not in lowered_question and "sum" not in lowered_question:
        projections = set(fields)
        projections.update(
            str(order_info.get("field", "")).strip()
            for order_info in query_plan.get("order_by", [])
            if isinstance(order_info, dict)
        )
        if not any(field.endswith(".amount") for field in projections):
            raise ValueError("Questions asking for amounts must project or order by an amount field.")


def _assert_filter_alignment(
    lowered_question: str,
    backend: str,
    query_plan: dict[str, Any],
) -> None:
    """Catch missing or mismatched filters for the most common user phrasing."""

    if backend not in {"sqlite", "postgresql"}:
        return

    filters = [filter_info for filter_info in query_plan.get("filters", []) if isinstance(filter_info, dict)]
    operators = {str(filter_info.get("operator", "")).strip().upper() for filter_info in filters}

    needs_filter = any(
        phrase in lowered_question
        for phrase in (
            " where ",
            " name is ",
            " email is ",
            " status is ",
            " contains ",
            " starts with ",
            " ends with ",
            " at least ",
            " more than ",
            " less than ",
            " greater than ",
            " fewer than ",
        )
    )
    needs_filter = needs_filter or lowered_question.startswith(
        (
            "name is ",
            "email is ",
            "status is ",
            "contains ",
            "starts with ",
            "ends with ",
        )
    )
    if needs_filter and not filters:
        raise ValueError("The question includes a filter, but the plan has no filters.")

    if "does not contain" in lowered_question and "NOT_CONTAINS" not in operators:
        raise ValueError("Negated text questions must use a NOT_CONTAINS filter.")
    if " contains " in lowered_question and "does not contain" not in lowered_question:
        if operators.isdisjoint({"CONTAINS", "CONTAINS_ANY"}):
            raise ValueError("Contains questions must use a CONTAINS-style filter.")
    if " starts with " in lowered_question and "STARTS_WITH" not in operators:
        raise ValueError("Starts-with questions must use a STARTS_WITH filter.")
    if " ends with " in lowered_question and "ENDS_WITH" not in operators:
        raise ValueError("Ends-with questions must use an ENDS_WITH filter.")


def _assert_sort_alignment(
    lowered_question: str,
    backend: str,
    query_plan: dict[str, Any],
) -> None:
    """Reject plans that ignore obvious latest/descending sort intent."""

    if backend not in {"sqlite", "postgresql"} or "latest" not in lowered_question:
        return

    order_by = [order_info for order_info in query_plan.get("order_by", []) if isinstance(order_info, dict)]
    if not order_by:
        raise ValueError("Latest questions must include descending order.")

    first_direction = str(order_by[0].get("direction", "")).strip().upper()
    if first_direction != "DESC":
        raise ValueError("Latest questions must sort in descending order.")


def _assert_join_alignment(
    backend: str,
    query_plan: dict[str, Any],
) -> None:
    """Reject multi-table SQL plans that forgot to connect those tables."""

    if backend not in {"sqlite", "postgresql"}:
        return

    if len(_string_list(query_plan.get("tables"))) <= 1:
        return

    joins = [join_info for join_info in query_plan.get("joins", []) if isinstance(join_info, dict)]
    if not joins:
        raise ValueError("Multi-table SQL plans must include a join.")


def _assert_aggregation_alignment(
    lowered_question: str,
    backend: str,
    query_plan: dict[str, Any],
) -> None:
    """Catch the clearest aggregate mismatches without paying for another model round-trip."""

    if backend not in {"sqlite", "postgresql"}:
        return

    aggregation_functions = {
        str(aggregation.get("function", "")).strip().upper()
        for aggregation in query_plan.get("aggregations", [])
        if isinstance(aggregation, dict)
    }
    group_by = _string_list(query_plan.get("group_by"))

    if any(token in lowered_question for token in ("average", " avg ")):
        if "AVG" not in aggregation_functions:
            raise ValueError("Average questions must use an AVG aggregation.")

    if any(token in lowered_question for token in (" by user", " per user")) and not group_by:
        raise ValueError("Questions grouped by user must include a GROUP BY field.")


def _is_count_question(question: str) -> bool:
    """Return whether the prompt is clearly asking for a count."""

    lowered_question = question.lower()
    count_phrases = (
        "how many",
        "count",
        "number of",
        "total number",
    )
    return any(phrase in lowered_question for phrase in count_phrases)


def _apply_question_guardrails(
    question: str,
    backend: str,
    query_plan: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Apply deterministic plan cleanup for the clearest question intents."""

    if not _is_count_question(question):
        return query_plan, ""

    if backend in {"sqlite", "postgresql"}:
        return _apply_sql_count_guardrails(query_plan)

    if backend == "mongodb":
        return _apply_mongodb_count_guardrails(query_plan)

    return query_plan, ""


def _apply_sql_count_guardrails(query_plan: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Canonicalize plain SQL count plans so they cannot leak row fields into the result."""

    aggregations = [
        dict(aggregation)
        for aggregation in query_plan.get("aggregations", [])
        if isinstance(aggregation, dict)
    ]
    count_aggregation = next(
        (
            aggregation
            for aggregation in aggregations
            if str(aggregation.get("function", "")).strip().upper() == "COUNT"
        ),
        None,
    )
    if count_aggregation is None:
        raise ValueError("Count questions must use a COUNT aggregation.")

    group_by = _string_list(query_plan.get("group_by"))
    if group_by:
        cleaned_fields = [
            field_name
            for field_name in _string_list(query_plan.get("fields"))
            if field_name in group_by
        ]
        if cleaned_fields == _string_list(query_plan.get("fields")):
            return query_plan, ""

        next_plan = dict(query_plan)
        next_plan["fields"] = cleaned_fields or group_by
        return next_plan, "Python removed non-grouped row fields from a grouped count plan."

    alias_name = str(count_aggregation.get("alias", "")).strip() or "count"
    normalized_count = {
        "function": "COUNT",
        "field": "*",
        "alias": alias_name,
    }
    next_plan = dict(query_plan)
    next_plan["fields"] = []
    next_plan["aggregations"] = [normalized_count]
    next_plan["order_by"] = []

    if (
        next_plan["fields"] == query_plan.get("fields", [])
        and next_plan["aggregations"] == query_plan.get("aggregations", [])
        and next_plan["order_by"] == query_plan.get("order_by", [])
    ):
        return query_plan, ""

    return next_plan, "Python canonicalized a plain count plan to return only the count result."


def _apply_mongodb_count_guardrails(query_plan: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Canonicalize plain MongoDB count plans so they use count intent cleanly."""

    group_by = _string_list(query_plan.get("group_by"))
    if group_by:
        return query_plan, ""

    next_plan = dict(query_plan)
    next_plan["operation"] = "aggregate"
    next_plan["intent"] = "count"
    next_plan["project"] = {}
    next_plan["sort"] = {}
    next_plan["aggregations"] = []

    changed = any(
        next_plan.get(key) != query_plan.get(key)
        for key in ("operation", "intent", "project", "sort", "aggregations")
    )
    if not changed:
        return query_plan, ""

    return next_plan, "Python canonicalized a plain MongoDB count plan to use count intent."


def _repair_sql_plan_from_validation_error(
    question: str,
    query_plan: dict[str, Any],
    database_schema: dict[str, Any],
    error_message: str,
) -> dict[str, Any] | None:
    """Repair common SQL schema-scope mistakes using the extracted schema itself."""

    if not _is_sql_schema_scope_error(error_message):
        return None

    known_columns = _build_sql_known_columns_map(database_schema)
    if not known_columns:
        return None

    repaired_plan = _clone_sql_plan(query_plan)
    did_change = False

    for index, field_name in enumerate(list(repaired_plan.get("fields", []))):
        qualified_field = _qualify_sql_plan_reference(field_name, repaired_plan, known_columns)
        if qualified_field != field_name:
            did_change = True
        repaired_plan["fields"][index] = qualified_field

    for index, aggregation in enumerate(repaired_plan.get("aggregations", [])):
        field_name = str(aggregation.get("field", "")).strip()
        if not field_name or field_name == "*":
            continue
        qualified_field = _qualify_sql_plan_reference(field_name, repaired_plan, known_columns)
        if qualified_field != field_name:
            did_change = True
        next_aggregation = dict(aggregation)
        next_aggregation["field"] = qualified_field
        repaired_plan["aggregations"][index] = next_aggregation

    for index, field_name in enumerate(repaired_plan.get("group_by", [])):
        qualified_field = _qualify_sql_plan_reference(field_name, repaired_plan, known_columns)
        if qualified_field != field_name:
            did_change = True
        repaired_plan["group_by"][index] = qualified_field

    for index, order_by in enumerate(repaired_plan.get("order_by", [])):
        field_name = str(order_by.get("field", "")).strip()
        if not field_name:
            continue
        qualified_field = _qualify_sql_plan_reference(field_name, repaired_plan, known_columns)
        if qualified_field != field_name:
            did_change = True
        next_order_by = dict(order_by)
        next_order_by["field"] = qualified_field
        repaired_plan["order_by"][index] = next_order_by

    for index, filter_info in enumerate(repaired_plan.get("filters", [])):
        field_name = str(filter_info.get("field", "")).strip()
        if not field_name:
            continue
        qualified_field = _qualify_sql_plan_reference(field_name, repaired_plan, known_columns)
        if qualified_field != field_name:
            did_change = True
        next_filter = dict(filter_info)
        next_filter["field"] = qualified_field
        repaired_plan["filters"][index] = next_filter

    referenced_tables = _collect_sql_plan_tables(repaired_plan)
    for table_name in referenced_tables:
        if table_name not in repaired_plan["tables"]:
            repaired_plan["tables"].append(table_name)
            did_change = True

    repaired_plan, grouping_changed = _promote_sql_group_labels(
        question=question,
        query_plan=repaired_plan,
        database_schema=database_schema,
    )
    did_change = did_change or grouping_changed

    repaired_plan, join_changed = _ensure_sql_join_connectivity(
        query_plan=repaired_plan,
        database_schema=database_schema,
    )
    did_change = did_change or join_changed

    if repaired_plan.get("group_by") and repaired_plan.get("aggregations") and not repaired_plan.get("fields"):
        repaired_plan["fields"] = list(repaired_plan["group_by"])
        did_change = True

    return repaired_plan if did_change else None


def _canonicalize_model_query_plan(
    question: str,
    backend: str,
    query_plan: dict[str, Any],
    database_schema: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Normalize model-shaped plans so aliases and missing joins resolve to real schema names."""

    if backend not in {"sqlite", "postgresql"}:
        return query_plan, ""

    known_columns = _build_sql_known_columns_map(database_schema)
    if not known_columns:
        return query_plan, ""

    next_plan = _clone_sql_plan(query_plan)
    changed = False

    normalized_tables: list[str] = []
    for table_name in next_plan.get("tables", []):
        normalized_table = _resolve_sql_table_name_alias(str(table_name).strip(), known_columns)
        if normalized_table != table_name:
            changed = True
        if normalized_table and normalized_table not in normalized_tables:
            normalized_tables.append(normalized_table)
    next_plan["tables"] = normalized_tables or list(next_plan.get("tables", []))

    for index, field_name in enumerate(list(next_plan.get("fields", []))):
        normalized_field = _qualify_sql_plan_reference(field_name, next_plan, known_columns)
        if normalized_field != field_name:
            changed = True
        next_plan["fields"][index] = normalized_field

    for index, aggregation in enumerate(next_plan.get("aggregations", [])):
        field_name = str(aggregation.get("field", "")).strip()
        if not field_name or field_name == "*":
            continue
        normalized_field = _qualify_sql_plan_reference(field_name, next_plan, known_columns)
        if normalized_field != field_name:
            changed = True
        next_aggregation = dict(aggregation)
        next_aggregation["field"] = normalized_field
        next_plan["aggregations"][index] = next_aggregation

    for index, field_name in enumerate(list(next_plan.get("group_by", []))):
        normalized_field = _qualify_sql_plan_reference(field_name, next_plan, known_columns)
        if normalized_field != field_name:
            changed = True
        next_plan["group_by"][index] = normalized_field

    for index, order_by in enumerate(next_plan.get("order_by", [])):
        field_name = str(order_by.get("field", "")).strip()
        if not field_name:
            continue
        normalized_field = _qualify_sql_plan_reference(field_name, next_plan, known_columns)
        if normalized_field != field_name:
            changed = True
        next_order_by = dict(order_by)
        next_order_by["field"] = normalized_field
        next_plan["order_by"][index] = next_order_by

    for index, filter_info in enumerate(next_plan.get("filters", [])):
        field_name = str(filter_info.get("field", "")).strip()
        if not field_name:
            continue
        normalized_field = _qualify_sql_plan_reference(field_name, next_plan, known_columns)
        if normalized_field != field_name:
            changed = True
        next_filter = dict(filter_info)
        next_filter["field"] = normalized_field
        next_plan["filters"][index] = next_filter

    for index, join_info in enumerate(next_plan.get("joins", [])):
        left_reference = str(join_info.get("left", "")).strip()
        right_reference = str(join_info.get("right", "")).strip()
        normalized_left = _qualify_sql_plan_reference(left_reference, next_plan, known_columns)
        normalized_right = _qualify_sql_plan_reference(right_reference, next_plan, known_columns)
        if normalized_left != left_reference or normalized_right != right_reference:
            changed = True
        next_plan["joins"][index] = {
            "left": normalized_left,
            "right": normalized_right,
        }

    next_plan, filter_cleanup_changed = _drop_sql_join_literal_filters(next_plan, known_columns)
    changed = changed or filter_cleanup_changed

    referenced_tables = _collect_sql_plan_tables(next_plan)
    for table_name in referenced_tables:
        if table_name not in next_plan["tables"]:
            next_plan["tables"].append(table_name)
            changed = True

    next_plan, grouping_changed = _promote_sql_group_labels(
        question=question,
        query_plan=next_plan,
        database_schema=database_schema,
    )
    changed = changed or grouping_changed

    next_plan, join_changed = _ensure_sql_join_connectivity(
        query_plan=next_plan,
        database_schema=database_schema,
    )
    changed = changed or join_changed

    if next_plan.get("group_by") and next_plan.get("aggregations") and not next_plan.get("fields"):
        next_plan["fields"] = list(next_plan["group_by"])
        changed = True

    if not changed:
        return query_plan, ""

    return next_plan, "Python canonicalized model output to match the real SQL schema."


def _is_sql_schema_scope_error(error_message: str) -> bool:
    """Return whether the validator is complaining about SQL schema scope."""

    normalized = error_message.lower()
    return any(
        fragment in normalized
        for fragment in (
            "column not found",
            "ambiguous column reference",
            "table not found",
            "no such column",
            "no such table",
            "not joined in this query",
            "references a table that is not joined",
        )
    )


def _build_sql_known_columns_map(database_schema: dict[str, Any]) -> dict[str, set[str]]:
    """Return a simple table-to-column lookup from the extracted SQL schema."""

    return {
        table_name: {
            str(column.get("name", "")).strip()
            for column in table_info.get("columns", [])
            if column.get("name")
        }
        for table_name, table_info in database_schema.get("tables", {}).items()
    }


def _clone_sql_plan(query_plan: dict[str, Any]) -> dict[str, Any]:
    """Clone the mutable parts of one SQL plan so repair can edit it safely."""

    return {
        "operation": str(query_plan.get("operation", "select")),
        "tables": list(query_plan.get("tables", [])),
        "fields": list(query_plan.get("fields", [])),
        "filters": [dict(filter_info) for filter_info in query_plan.get("filters", []) if isinstance(filter_info, dict)],
        "joins": [dict(join_info) for join_info in query_plan.get("joins", []) if isinstance(join_info, dict)],
        "aggregations": [
            dict(aggregation)
            for aggregation in query_plan.get("aggregations", [])
            if isinstance(aggregation, dict)
        ],
        "group_by": list(query_plan.get("group_by", [])),
        "order_by": [dict(order_info) for order_info in query_plan.get("order_by", []) if isinstance(order_info, dict)],
        "limit": query_plan.get("limit", DEFAULT_LIMIT),
    }


def _qualify_sql_plan_reference(
    reference: str,
    query_plan: dict[str, Any],
    known_columns: dict[str, set[str]],
) -> str:
    """Qualify a one-part SQL reference when the schema makes the target table clear."""

    trimmed_reference = str(reference).strip()
    if not trimmed_reference or trimmed_reference == "*":
        return trimmed_reference

    if "." in trimmed_reference:
        table_name, column_name = trimmed_reference.split(".", 1)
        if table_name in known_columns and column_name in known_columns[table_name]:
            return trimmed_reference

        resolved_from_alias = _resolve_sql_alias_reference(
            table_name,
            column_name,
            query_plan,
            known_columns,
        )
        if resolved_from_alias is not None:
            return resolved_from_alias

        resolved_reference = _resolve_sql_column_reference(column_name, query_plan, known_columns)
        return resolved_reference or trimmed_reference

    return _resolve_sql_column_reference(trimmed_reference, query_plan, known_columns) or trimmed_reference


def _resolve_sql_alias_reference(
    alias_name: str,
    column_name: str,
    query_plan: dict[str, Any],
    known_columns: dict[str, set[str]],
) -> str | None:
    """Resolve one alias-like SQL reference such as o.amount to a real table.column."""

    candidate_tables: list[str] = []

    for table_name in query_plan.get("tables", []):
        if table_name not in known_columns:
            continue

        lowered_alias = alias_name.lower()
        lowered_table = table_name.lower()
        if lowered_table.startswith(lowered_alias) or _table_initials(lowered_table).startswith(lowered_alias):
            candidate_tables.append(table_name)

    candidate_tables = [table_name for table_name in candidate_tables if column_name in known_columns.get(table_name, set())]
    if len(candidate_tables) == 1:
        return f"{candidate_tables[0]}.{column_name}"

    return None


def _resolve_sql_table_name_alias(alias_name: str, known_columns: dict[str, set[str]]) -> str:
    """Resolve one alias-like table token back to a real schema table when that is obvious."""

    if alias_name in known_columns:
        return alias_name

    lowered_alias = alias_name.lower()
    candidate_tables = [
        table_name
        for table_name in known_columns
        if table_name.lower().startswith(lowered_alias) or _table_initials(table_name).startswith(lowered_alias)
    ]
    if len(candidate_tables) == 1:
        return candidate_tables[0]

    return alias_name


def _drop_sql_join_literal_filters(
    query_plan: dict[str, Any],
    known_columns: dict[str, set[str]],
) -> tuple[dict[str, Any], bool]:
    """Remove filters that accidentally restate a join as a string literal comparison."""

    join_pairs = {
        frozenset(
            {
                str(join_info.get("left", "")).strip(),
                str(join_info.get("right", "")).strip(),
            }
        )
        for join_info in query_plan.get("joins", [])
        if isinstance(join_info, dict)
    }
    if not join_pairs:
        return query_plan, False

    kept_filters: list[dict[str, Any]] = []
    changed = False

    for filter_info in query_plan.get("filters", []):
        if not isinstance(filter_info, dict):
            continue

        field_name = str(filter_info.get("field", "")).strip()
        operator = str(filter_info.get("operator", "")).strip().upper()
        value = filter_info.get("value")
        normalized_value = _normalize_sql_filter_reference_value(value, query_plan, known_columns)
        if (
            operator == "="
            and isinstance(normalized_value, str)
            and frozenset({field_name, normalized_value}) in join_pairs
        ):
            changed = True
            continue

        next_filter = dict(filter_info)
        if normalized_value != value:
            next_filter["value"] = normalized_value
            changed = True
        kept_filters.append(next_filter)

    if not changed:
        return query_plan, False

    next_plan = _clone_sql_plan(query_plan)
    next_plan["filters"] = kept_filters
    return next_plan, True


def _normalize_sql_filter_reference_value(
    value: Any,
    query_plan: dict[str, Any],
    known_columns: dict[str, set[str]],
) -> Any:
    """Canonicalize string values that are really schema references, otherwise leave them literal."""

    if not isinstance(value, str):
        return value

    trimmed_value = value.strip()
    if "." not in trimmed_value:
        return value

    return _qualify_sql_plan_reference(trimmed_value, query_plan, known_columns)


def _resolve_sql_column_reference(
    column_name: str,
    query_plan: dict[str, Any],
    known_columns: dict[str, set[str]],
) -> str | None:
    """Return one fully qualified SQL column when the target table is unambiguous."""

    current_tables = [
        table_name
        for table_name in query_plan.get("tables", [])
        if column_name in known_columns.get(table_name, set())
    ]
    if len(current_tables) == 1:
        return f"{current_tables[0]}.{column_name}"

    candidate_tables = [
        table_name
        for table_name, columns in known_columns.items()
        if column_name in columns
    ]
    if len(candidate_tables) == 1:
        return f"{candidate_tables[0]}.{column_name}"

    return None


def _table_initials(table_name: str) -> str:
    """Return a short alias-like signature for one table name."""

    parts = [part for part in re.split(r"[^a-z0-9]+", table_name.lower()) if part]
    if len(parts) > 1:
        return "".join(part[0] for part in parts if part)

    return table_name[:1]


def _collect_sql_plan_tables(query_plan: dict[str, Any]) -> set[str]:
    """Collect every table referenced directly by the repaired SQL plan."""

    referenced_tables: set[str] = set(query_plan.get("tables", []))

    for field_name in query_plan.get("fields", []):
        _add_sql_reference_table(referenced_tables, field_name)

    for aggregation in query_plan.get("aggregations", []):
        _add_sql_reference_table(referenced_tables, aggregation.get("field"))

    for field_name in query_plan.get("group_by", []):
        _add_sql_reference_table(referenced_tables, field_name)

    for order_by in query_plan.get("order_by", []):
        _add_sql_reference_table(referenced_tables, order_by.get("field"))

    for filter_info in query_plan.get("filters", []):
        _add_sql_reference_table(referenced_tables, filter_info.get("field"))

    for join_info in query_plan.get("joins", []):
        _add_sql_reference_table(referenced_tables, join_info.get("left"))
        _add_sql_reference_table(referenced_tables, join_info.get("right"))

    return referenced_tables


def _add_sql_reference_table(referenced_tables: set[str], reference: object) -> None:
    """Add the table portion of one dotted SQL reference when present."""

    if not isinstance(reference, str):
        return

    parts = reference.split(".")
    if len(parts) >= 2 and parts[0]:
        referenced_tables.add(parts[0])


def _promote_sql_group_labels(
    question: str,
    query_plan: dict[str, Any],
    database_schema: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Prefer user-facing grouped labels like users.name over raw foreign-key ids."""

    if not query_plan.get("aggregations") or not query_plan.get("group_by"):
        return query_plan, False

    lowered_question = question.lower()
    if "user" not in lowered_question or "user id" in lowered_question:
        return query_plan, False

    known_columns = _build_sql_known_columns_map(database_schema)
    if "users" not in known_columns or "name" not in known_columns["users"]:
        return query_plan, False

    group_by_fields = list(query_plan.get("group_by", []))
    if not any(
        field_name.endswith(".user_id") or field_name in {"user_id", "users.id"}
        for field_name in group_by_fields
    ):
        return query_plan, False

    next_plan = _clone_sql_plan(query_plan)
    next_plan["group_by"] = ["users.name"]
    next_plan["fields"] = ["users.name"]

    if not next_plan.get("order_by"):
        next_plan["order_by"] = [{"field": "users.name", "direction": "ASC"}]

    if "users" not in next_plan["tables"]:
        next_plan["tables"].append("users")

    return next_plan, True


def _ensure_sql_join_connectivity(
    query_plan: dict[str, Any],
    database_schema: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Add straightforward foreign-key joins for any selected SQL tables that are not yet connected."""

    if len(query_plan.get("tables", [])) <= 1:
        return query_plan, False

    next_plan = _clone_sql_plan(query_plan)
    changed = False
    connected_tables = {str(next_plan["tables"][0])}

    for join_info in next_plan.get("joins", []):
        connected_tables.update(_extract_join_tables(join_info))

    for table_name in next_plan["tables"][1:]:
        if table_name in connected_tables:
            continue

        join_info = _find_sql_join_between_sets(
            source_table=table_name,
            candidate_tables=connected_tables,
            database_schema=database_schema,
        )
        if join_info is None:
            continue

        if join_info not in next_plan["joins"]:
            next_plan["joins"].append(join_info)
            changed = True
        connected_tables.update(_extract_join_tables(join_info))

    return next_plan, changed


def _find_sql_join_between_sets(
    source_table: str,
    candidate_tables: set[str],
    database_schema: dict[str, Any],
) -> dict[str, str] | None:
    """Find a direct foreign-key join between one table and any currently connected table."""

    for candidate_table in candidate_tables:
        join_info = _find_sql_join_between_tables(
            left_table=source_table,
            right_table=candidate_table,
            database_schema=database_schema,
        )
        if join_info is not None:
            return join_info

    return None


def _find_sql_join_between_tables(
    left_table: str,
    right_table: str,
    database_schema: dict[str, Any],
) -> dict[str, str] | None:
    """Return one join that follows a direct foreign-key edge between two tables."""

    tables = database_schema.get("tables", {})
    left_info = tables.get(left_table, {})
    right_info = tables.get(right_table, {})

    for foreign_key in left_info.get("foreign_keys", []):
        references = str(foreign_key.get("references", "")).strip()
        referenced_table = references.split(".", 1)[0]
        if referenced_table == right_table:
            return {
                "left": f"{left_table}.{foreign_key['column']}",
                "right": references,
            }

    for foreign_key in right_info.get("foreign_keys", []):
        references = str(foreign_key.get("references", "")).strip()
        referenced_table = references.split(".", 1)[0]
        if referenced_table == left_table:
            return {
                "left": f"{right_table}.{foreign_key['column']}",
                "right": references,
            }

    return None


def _extract_join_tables(join_info: dict[str, Any]) -> set[str]:
    """Return the table names referenced by one join payload."""

    join_tables: set[str] = set()
    _add_sql_reference_table(join_tables, join_info.get("left"))
    _add_sql_reference_table(join_tables, join_info.get("right"))
    return join_tables


def _build_planning_prompt(
    question: str,
    backend: str,
    relevant_schema: dict[str, Any],
) -> str:
    """Build the main planning prompt for Ollama."""

    if backend in {"sqlite", "postgresql"}:
        return f"""
You are planning a read-only SQL query for a natural-language database copilot.

Return exactly one JSON object and nothing else.
{CONCRETE_ERROR_OBJECT_INSTRUCTIONS}

Use only exact table and column names from this schema:
{json.dumps(relevant_schema, indent=2)}

Question:
{question}

Return this plan shape:
{{
  "tables": ["users"],
  "fields": ["users.name"],
  "filters": [],
  "joins": [],
  "aggregations": [],
  "group_by": [],
  "order_by": [],
  "limit": 10
}}

Examples:
- show user emails
  -> {{"tables":["users"],"fields":["users.email"],"filters":[],"joins":[],"aggregations":[],"group_by":[],"order_by":[],"limit":10}}
- how many orders where name is Alice
  -> {{"tables":["orders","users"],"fields":[],"filters":[{{"field":"users.name","operator":"=","value":"Alice"}}],"joins":[{{"left":"orders.user_id","right":"users.id"}}],"aggregations":[{{"function":"COUNT","field":"*","alias":"count"}}],"group_by":[],"order_by":[],"limit":10}}
- show users where name does not contain ali
  -> {{"tables":["users"],"fields":["users.name"],"filters":[{{"field":"users.name","operator":"NOT_CONTAINS","value":"ali"}}],"joins":[],"aggregations":[],"group_by":[],"order_by":[{{"field":"users.name","direction":"ASC"}}],"limit":10}}
- show order amounts with user names
  -> {{"tables":["orders","users"],"fields":["orders.amount","users.name"],"filters":[],"joins":[{{"left":"orders.user_id","right":"users.id"}}],"aggregations":[],"group_by":[],"order_by":[{{"field":"orders.id","direction":"ASC"}}],"limit":10}}
- average order amount by user
  -> {{"tables":["orders","users"],"fields":["users.name"],"filters":[],"joins":[{{"left":"orders.user_id","right":"users.id"}}],"aggregations":[{{"function":"AVG","field":"orders.amount","alias":"average_amount"}}],"group_by":["users.name"],"order_by":[{{"field":"users.name","direction":"ASC"}}],"limit":10}}

Rules:
- only plan read-only SELECT queries
- list every used table in "tables"
- fully qualify SQL references like "users.email"
- joins must use exact foreign keys from the schema
- allowed filter operators: =, !=, >, >=, <, <=, STARTS_WITH, ENDS_WITH, CONTAINS, CONTAINS_ANY, NOT_CONTAINS, IN
- "does not contain" => NOT_CONTAINS
- "at least" => >=
- count questions use COUNT(*)
- grouped metrics use group_by plus AVG/SUM/MIN/MAX/COUNT
- sort questions use order_by
- include a limit for non-count results
- never invent tables or columns
""".strip()

    return f"""
You are planning a read-only MongoDB query for a natural-language database copilot.

Return exactly one JSON object and nothing else.
{CONCRETE_ERROR_OBJECT_INSTRUCTIONS}

Use only the exact collection and field names from this schema.
Schema:
{json.dumps(relevant_schema, indent=2)}

Question:
{question}

Return this MongoDB plan shape:
{{
  "operation": "find",
  "collection": "users",
  "intent": "",
  "match": {{}},
  "project": {{"name": 1, "email": 1}},
  "sort": {{"name": 1}},
  "group_by": [],
  "aggregations": [],
  "limit": 10
}}

Examples:
1. Question:
show user emails
Return:
{{
  "operation": "find",
  "collection": "users",
  "intent": "",
  "match": {{}},
  "project": {{"email": 1}},
  "sort": {{}},
  "group_by": [],
  "aggregations": [],
  "limit": 10
}}

2. Question:
how many users
Return:
{{
  "operation": "aggregate",
  "collection": "users",
  "intent": "count",
  "match": {{}},
  "project": {{}},
  "sort": {{}},
  "group_by": [],
  "aggregations": [],
  "limit": 10
}}

3. Question:
show users where name contains alice or email contains bob
Return:
{{
  "operation": "find",
  "collection": "users",
  "intent": "",
  "match": {{
    "$or": [
      {{"name": {{"$regex": "alice", "$options": "i"}}}},
      {{"email": {{"$regex": "bob", "$options": "i"}}}}
    ]
  }},
  "project": {{"name": 1, "email": 1}},
  "sort": {{}},
  "group_by": [],
  "aggregations": [],
  "limit": 10
}}

4. Question:
count users by status
Return:
{{
  "operation": "aggregate",
  "collection": "users",
  "intent": "group",
  "match": {{}},
  "project": {{}},
  "sort": {{"status": 1}},
  "group_by": ["status"],
  "aggregations": [
    {{"function": "COUNT", "field": "*", "alias": "count"}}
  ],
  "limit": 10
}}

5. Question:
average order amount by user name
Return:
{{
  "operation": "aggregate",
  "collection": "orders",
  "intent": "group",
  "match": {{}},
  "project": {{}},
  "sort": {{"user_name": 1}},
  "group_by": ["user_name"],
  "aggregations": [
    {{"function": "AVG", "field": "amount", "alias": "avg_amount"}}
  ],
  "limit": 10
}}

6. Question:
show orders where amount is at least 100
Return:
{{
  "operation": "find",
  "collection": "orders",
  "intent": "",
  "match": {{
    "amount": {{"$gte": 100}}
  }},
  "project": {{"user_name": 1, "amount": 1}},
  "sort": {{}},
  "group_by": [],
  "aggregations": [],
  "limit": 10
}}

Rules:
- only plan read-only queries
- use operation "aggregate" with intent "count" for count questions
- use operation "aggregate" when the question asks for grouped summaries like "by status", "per user", or averages grouped by one field
- use Mongo-style $match-compatible objects inside "match"
- use regex with "$regex" and "$options": "i" for case-insensitive text matching
- use $gt, $gte, $lt, $lte, $ne, $in, and $not when the question clearly asks for those comparisons
- use group_by plus aggregations for grouped metrics
- longer polite wording like "could you please list" still means the same query intent as a direct prompt
- do not invent fields or collections
""".strip()


def _build_planning_retry_prompt(
    question: str,
    backend: str,
    relevant_schema: dict[str, Any],
    previous_response: dict[str, Any],
    previous_issue: str,
) -> str:
    """Build a stricter second planning prompt when the first answer was unusable."""

    if backend in {"sqlite", "postgresql"}:
        return f"""
You are replanning a read-only SQL query after an earlier failed answer.

Return exactly one JSON object and nothing else.
Re-evaluate the question from scratch. Build the plan if it is safe and answerable.

Schema:
{json.dumps(relevant_schema, indent=2)}

Question:
{question}

Previous issue:
{previous_issue}

Previous response:
{json.dumps(previous_response, indent=2)}

Return this exact shape:
{{
  "tables": ["users"],
  "fields": [],
  "filters": [],
  "joins": [],
  "aggregations": [
    {{"function": "COUNT", "field": "*", "alias": "count"}}
  ],
  "group_by": [],
  "order_by": [],
  "limit": 10
}}

Rules:
- Use only exact schema table and column names.
- For "how many ..." questions, use COUNT with field "*".
- For filters, use fully qualified fields like "users.name".
- For joins, use exact foreign keys from the schema.
- Keep negation explicit. For prompts like "does not contain", use NOT_CONTAINS instead of dropping the filter.
""".strip()

    return f"""
You are replanning a read-only MongoDB query after an earlier failed answer.

Return exactly one JSON object and nothing else.
The previous answer was not usable. Re-evaluate the question from scratch.
If the question is already read-only and answerable from the schema, build the plan instead of rejecting it.
Do not return an "error" object unless the question is truly unsafe or impossible.
If you must return an "error" object, use a real sentence like {{"error": "Only read-only questions are supported."}}.

Schema:
{json.dumps(relevant_schema, indent=2)}

Question:
{question}

Previous issue:
{previous_issue}

Previous response:
{json.dumps(previous_response, indent=2)}

Return this exact shape:
{{
  "operation": "aggregate",
  "collection": "users",
  "intent": "count",
  "match": {{}},
  "project": {{}},
  "sort": {{}},
  "group_by": [],
  "aggregations": [],
  "limit": 10
}}

Rules:
- Use only exact schema collection and field names.
- For "how many ..." questions, use operation "aggregate" with intent "count".
- For list/show questions, use operation "find" and put requested fields in "project".
- For text filters, use case-insensitive $regex when appropriate.
""".strip()


def _build_schema_selection_prompt(
    question: str,
    backend: str,
    database_schema: dict[str, Any],
) -> str:
    """Build the prompt that asks Ollama for the smallest relevant schema slice."""

    if backend in {"sqlite", "postgresql"}:
        return f"""
You are choosing the minimum SQL schema context needed to answer a database question.

Return exactly one JSON object and nothing else.
If the question is unclear, still choose the closest useful tables.

Full schema:
{json.dumps(database_schema, indent=2)}

Question:
{question}

Return this schema selection shape:
{{
  "tables": ["users", "orders"]
}}

Examples:
1. Question:
show user emails
Return:
{{"tables": ["users"]}}

2. Question:
how many orders where name is Alice
Return:
{{"tables": ["orders", "users"]}}

3. Question:
for each user, tell me how many orders they have placed
Return:
{{"tables": ["orders", "users"]}}

Rules:
- choose only exact table names from the schema
- include any related table needed for joins or filters
- do not include every table unless the question truly needs them
""".strip()

    return f"""
You are choosing the minimum MongoDB schema context needed to answer a database question.

Return exactly one JSON object and nothing else.
If the question is unclear, still choose the closest useful collections.

Full schema:
{json.dumps(database_schema, indent=2)}

Question:
{question}

Return this schema selection shape:
{{
  "collections": ["users"]
}}

Examples:
1. Question:
show user emails
Return:
{{"collections": ["users"]}}

2. Question:
how many users
Return:
{{"collections": ["users"]}}

3. Question:
average order amount by user name
Return:
{{"collections": ["orders"]}}

Rules:
- choose only exact collection names from the schema
- include only the collections needed for this question
""".strip()


def _build_semantic_check_prompt(
    question: str,
    backend: str,
    relevant_schema: dict[str, Any],
    query_plan: dict[str, Any],
) -> str:
    """Build the prompt that asks Ollama to validate plan semantics."""

    return f"""
You are the semantic reviewer for a read-only {backend} natural-language database copilot.

Return exactly one JSON object and nothing else.

Decide whether the current plan faithfully answers the user's question using only this schema.
If the plan is correct, return:
{{
  "passed": true,
  "reason": "short reason",
  "repaired_plan": null
}}

If the plan is close but wrong, return:
{{
  "passed": false,
  "reason": "what was wrong",
  "repaired_plan": {{ ...the corrected plan in the same shape... }}
}}

If the request cannot be answered safely or read-only:
{CONCRETE_ERROR_OBJECT_INSTRUCTIONS}

Schema:
{json.dumps(relevant_schema, indent=2)}

Question:
{question}

Current plan:
{json.dumps(query_plan, indent=2)}

Semantic checklist:
- The selected tables or collection must match the nouns in the question.
- Selected fields and projections must match what the user asked to see.
- Filters must preserve AND/OR intent, negation, ranges, text matching, and values.
- Aggregations, grouping, ordering, and limits must match the wording.
- Join plans must use only foreign-key relationships present in the schema.
- Do not invent schema names.
- Do not approve destructive or write-focused requests.
- Never return "passed": true if your reason says the plan is missing a filter, join, field, grouping, or uses the wrong operator.
- If the plan is wrong but repairable, return "passed": false and include a corrected "repaired_plan".
""".strip()


def _build_repair_prompt(
    question: str,
    backend: str,
    relevant_schema: dict[str, Any],
    failed_plan: dict[str, Any],
    failed_query: str,
    error_message: str,
) -> str:
    """Build a repair prompt that asks Ollama for one corrected plan."""

    return f"""
You are repairing a read-only {backend} query plan for a natural-language database copilot.

Return exactly one JSON object and nothing else.
If the question cannot be repaired safely:
{CONCRETE_ERROR_OBJECT_INSTRUCTIONS}

Schema:
{json.dumps(relevant_schema, indent=2)}

Original question:
{question}

Failed plan:
{json.dumps(failed_plan, indent=2)}

Failed compiled query:
{failed_query}

Error:
{error_message}

Repair the plan so it uses only valid schema names and still answers the original question.
Keep the same plan shape as before.
If a field or table name was invented, replace it with the closest valid schema name instead of keeping it.
If the question asks "how many" or otherwise asks for a count, the repaired plan must use COUNT instead of listing rows.
If the question says "does not contain", preserve that negation with NOT_CONTAINS instead of removing the filter.
If the question says "by user" for relational data, prefer grouping by users.name and join orders.user_id to users.id when both tables exist.
For SQL count questions without grouping, return:
{{
  "tables": ["users"],
  "fields": [],
  "filters": [],
  "joins": [],
  "aggregations": [
    {{"function": "COUNT", "field": "*", "alias": "count"}}
  ],
  "group_by": [],
  "order_by": [],
  "limit": 10
}}
""".strip()


def _build_count_repair_prompt(
    question: str,
    backend: str,
    relevant_schema: dict[str, Any],
    failed_plan: dict[str, Any],
    failed_query: str,
    error_message: str,
) -> str:
    """Build a narrower repair prompt for count questions."""

    if backend in {"sqlite", "postgresql"}:
        return f"""
You are fixing a read-only SQL query plan for a count question.

Return exactly one JSON object and nothing else.
The user is asking for a count, so the repaired plan must use COUNT and must not list ordinary rows.

Schema:
{json.dumps(relevant_schema, indent=2)}

Question:
{question}

Failed plan:
{json.dumps(failed_plan, indent=2)}

Failed compiled query:
{failed_query}

Error:
{error_message}

Return this exact shape:
{{
  "tables": ["users"],
  "fields": [],
  "filters": [],
  "joins": [],
  "aggregations": [
    {{"function": "COUNT", "field": "*", "alias": "count"}}
  ],
  "group_by": [],
  "order_by": [],
  "limit": 10
}}

Rules:
- Use only exact table and column names from the schema.
- If the question asks "how many" and does not ask for grouping, "fields" should be empty.
- For a plain count question, "aggregations" must contain COUNT on "*".
- Do not return an "error" object unless the request is unsafe or impossible.
""".strip()

    return f"""
You are fixing a read-only MongoDB query plan for a count question.

Return exactly one JSON object and nothing else.
The user is asking for a count, so the repaired plan must use MongoDB count intent or COUNT aggregation and must not list ordinary rows.

Schema:
{json.dumps(relevant_schema, indent=2)}

Question:
{question}

Failed plan:
{json.dumps(failed_plan, indent=2)}

Failed compiled query:
{failed_query}

Error:
{error_message}

Return this exact shape:
{{
  "operation": "aggregate",
  "collection": "users",
  "intent": "count",
  "match": {{}},
  "project": {{}},
  "sort": {{}},
  "group_by": [],
  "aggregations": [],
  "limit": 10
}}

Rules:
- Use only exact collection and field names from the schema.
- If the question asks "how many" and does not ask for grouping, use operation "aggregate" with intent "count".
- Do not return an "error" object unless the request is unsafe or impossible.
""".strip()


def _build_explanation_prompt(
    question: str,
    backend: str,
    compiled_query: str,
    result: dict[str, Any],
) -> str:
    """Build the final plain-English explanation prompt."""

    preview_rows = list(result.get("rows", []))[:3]

    return f"""
Explain this database query result in 2 or 3 short sentences for a beginner.
Be accurate and avoid hype.
Say what the query answered in plain words, then summarize the most important returned value or row pattern.

Backend: {backend}
Question: {question}
Compiled query:
{compiled_query}

Result summary:
row_count={result.get("row_count", 0)}
columns={result.get("columns", [])}
preview_rows={json.dumps(preview_rows, indent=2)}
""".strip()


def _string_list(value: Any) -> list[str]:
    """Turn a raw model value into a clean string list."""

    if not isinstance(value, list):
        return []

    cleaned_items: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            cleaned_items.append(text)

    return cleaned_items


def _dict_list(value: Any) -> list[dict[str, Any]]:
    """Keep only list items that are dictionaries."""

    if not isinstance(value, list):
        return []

    return [item for item in value if isinstance(item, dict)]


def _normalize_limit(value: Any) -> int:
    """Return a safe integer limit from model output."""

    try:
        limit_value = int(value)
    except (TypeError, ValueError):
        return DEFAULT_LIMIT

    if limit_value < 1:
        return DEFAULT_LIMIT

    return limit_value


def _normalize_sql_filters(value: Any) -> list[dict[str, Any]]:
    """Drop malformed SQL filter items before they become invalid compiled SQL."""

    normalized_filters: list[dict[str, Any]] = []

    for filter_info in _dict_list(value):
        field_name = str(filter_info.get("field", "")).strip()
        operator = str(filter_info.get("operator", "")).strip()
        if not field_name or not operator or "value" not in filter_info:
            continue

        next_filter = dict(filter_info)
        next_filter["field"] = field_name
        next_filter["operator"] = operator
        combine_with_previous = str(filter_info.get("combine_with_previous", "")).strip().upper()
        if combine_with_previous in {"AND", "OR"}:
            next_filter["combine_with_previous"] = combine_with_previous
        else:
            next_filter.pop("combine_with_previous", None)
        normalized_filters.append(next_filter)

    return normalized_filters


def _normalize_sql_joins(value: Any) -> list[dict[str, Any]]:
    """Keep only join definitions that name both sides of the relationship."""

    normalized_joins: list[dict[str, Any]] = []

    for join_info in _dict_list(value):
        left_reference = str(join_info.get("left", "")).strip()
        right_reference = str(join_info.get("right", "")).strip()
        if not left_reference or not right_reference:
            continue

        normalized_joins.append(
            {
                "left": left_reference,
                "right": right_reference,
            }
        )

    return normalized_joins


def _normalize_sql_order_by(value: Any) -> list[dict[str, Any]]:
    """Keep only usable ORDER BY items with a non-empty field name."""

    normalized_order_by: list[dict[str, Any]] = []

    for order_info in _dict_list(value):
        field_name = str(order_info.get("field", "")).strip()
        if not field_name:
            continue

        direction = str(order_info.get("direction", "ASC")).strip().upper()
        if direction not in {"ASC", "DESC"}:
            direction = "ASC"

        normalized_order_by.append(
            {
                "field": field_name,
                "direction": direction,
            }
        )

    return normalized_order_by


def _should_skip_model_semantic_review(
    question: str,
    backend: str,
    query_plan: dict[str, Any],
    planning_attempts: int,
) -> bool:
    """Skip the extra Ollama semantic hop for straightforward plans the local guardrails already cover."""

    if (
        backend not in {"sqlite", "postgresql"}
        or planning_attempts != 1
        or get_active_model_name() == "mock-ollama"
    ):
        return False

    joins = [join_info for join_info in query_plan.get("joins", []) if isinstance(join_info, dict)]
    aggregations = [
        aggregation
        for aggregation in query_plan.get("aggregations", [])
        if isinstance(aggregation, dict)
    ]
    if len(joins) > 1 or len(aggregations) > 1:
        return False

    filters = [filter_info for filter_info in query_plan.get("filters", []) if isinstance(filter_info, dict)]
    if len(filters) > 1:
        return False

    if any(
        str(filter_info.get("combine_with_previous", "")).strip().upper() == "OR"
        for filter_info in filters
    ):
        return False

    lowered_question = question.lower()
    if any(phrase in lowered_question for phrase in (" latest ", " newest ")):
        return False

    return True


def _should_use_python_fast_path() -> bool:
    """Use deterministic planning for real local runs while preserving mocked test coverage."""

    return settings.query_graph_fast_path_enabled and get_active_model_name() != "mock-ollama"


def _should_use_python_schema_selection() -> bool:
    """Use deterministic schema narrowing to keep model prompts small and fast."""

    return get_active_model_name() != "mock-ollama"


def _should_use_python_explanation() -> bool:
    """Use a deterministic result summary so Ollama spends time on reasoning, not narration."""

    return get_active_model_name() != "mock-ollama"


def _build_python_fast_path_plan(
    question: str,
    backend: str,
    database_schema: dict[str, Any],
) -> dict[str, Any] | None:
    """Return one deterministic plan for the common demo prompts when the schema supports it."""

    lowered_question = question.lower().strip()
    normalized_question = _normalize_question_text(question)
    requested_limit = _extract_requested_limit(normalized_question)

    if backend == "mongodb":
        if normalized_question == "show user emails" and _mongodb_schema_supports(
            database_schema,
            {"users": {"email"}},
        ):
            return {
                "operation": "find",
                "collection": "users",
                "intent": "",
                "match": {},
                "project": {"email": 1},
                "sort": {},
                "limit": requested_limit,
                "group_by": [],
                "aggregations": [],
            }

        if normalized_question == "how many users" and _mongodb_schema_supports(
            database_schema,
            {"users": set()},
        ):
            return {
                "operation": "aggregate",
                "collection": "users",
                "intent": "count",
                "match": {},
                "project": {},
                "sort": {},
                "limit": requested_limit,
                "group_by": [],
                "aggregations": [],
            }

        if normalized_question == "count users by status" and _mongodb_schema_supports(
            database_schema,
            {"users": {"status"}},
        ):
            return {
                "operation": "aggregate",
                "collection": "users",
                "intent": "group",
                "match": {},
                "project": {},
                "sort": {"status": 1},
                "limit": requested_limit,
                "group_by": ["status"],
                "aggregations": [{"function": "COUNT", "field": "*", "alias": "count"}],
            }

        if normalized_question == "average order amount by user name" and _mongodb_schema_supports(
            database_schema,
            {"orders": {"user_name", "amount"}},
        ):
            return {
                "operation": "aggregate",
                "collection": "orders",
                "intent": "group",
                "match": {},
                "project": {},
                "sort": {"user_name": 1},
                "limit": requested_limit,
                "group_by": ["user_name"],
                "aggregations": [{"function": "AVG", "field": "amount", "alias": "avg_amount"}],
            }

        return None

    if backend not in {"sqlite", "postgresql"}:
        return None

    if normalized_question == "show user emails" and _sql_schema_supports(
        database_schema,
        {"users": {"email"}},
    ):
        return {
            "operation": "select",
            "tables": ["users"],
            "fields": ["users.email"],
            "filters": [],
            "joins": [],
            "aggregations": [],
            "group_by": [],
            "order_by": [],
            "limit": requested_limit,
        }

    if normalized_question == "how many users" and _sql_schema_supports(
        database_schema,
        {"users": {"id"}},
    ):
        return {
            "operation": "select",
            "tables": ["users"],
            "fields": [],
            "filters": [],
            "joins": [],
            "aggregations": [{"function": "COUNT", "field": "*", "alias": "count"}],
            "group_by": [],
            "order_by": [],
            "limit": requested_limit,
        }

    if normalized_question in {
        "show latest orders first",
        "please show the latest two orders first",
    } and _sql_schema_supports(database_schema, {"orders": {"id", "amount", "status"}}):
        return {
            "operation": "select",
            "tables": ["orders"],
            "fields": ["orders.id", "orders.amount", "orders.status"],
            "filters": [],
            "joins": [],
            "aggregations": [],
            "group_by": [],
            "order_by": [{"field": "orders.id", "direction": "DESC"}],
            "limit": requested_limit,
        }

    if normalized_question == "show order amounts with user names" and _sql_schema_supports(
        database_schema,
        {
            "orders": {"id", "amount", "user_id"},
            "users": {"id", "name"},
        },
    ):
        return {
            "operation": "select",
            "tables": ["orders", "users"],
            "fields": ["orders.amount", "users.name"],
            "filters": [],
            "joins": [{"left": "orders.user_id", "right": "users.id"}],
            "aggregations": [],
            "group_by": [],
            "order_by": [{"field": "orders.id", "direction": "ASC"}],
            "limit": requested_limit,
        }

    if normalized_question == "how many orders where name is alice" and _sql_schema_supports(
        database_schema,
        {
            "orders": {"user_id"},
            "users": {"id", "name"},
        },
    ):
        return {
            "operation": "select",
            "tables": ["orders", "users"],
            "fields": [],
            "filters": [{"field": "users.name", "operator": "=", "value": "Alice"}],
            "joins": [{"left": "orders.user_id", "right": "users.id"}],
            "aggregations": [{"function": "COUNT", "field": "*", "alias": "count"}],
            "group_by": [],
            "order_by": [],
            "limit": requested_limit,
        }

    if normalized_question == "average order amount by user" and _sql_schema_supports(
        database_schema,
        {
            "orders": {"amount", "user_id"},
            "users": {"id", "name"},
        },
    ):
        return {
            "operation": "select",
            "tables": ["orders", "users"],
            "fields": ["users.name"],
            "filters": [],
            "joins": [{"left": "orders.user_id", "right": "users.id"}],
            "aggregations": [{"function": "AVG", "field": "orders.amount", "alias": "avg_amount"}],
            "group_by": ["users.name"],
            "order_by": [{"field": "users.name", "direction": "ASC"}],
            "limit": requested_limit,
        }

    if normalized_question in {
        "count orders per user",
        "for each user tell me how many orders they have placed",
    } and _sql_schema_supports(
        database_schema,
        {
            "orders": {"user_id"},
            "users": {"id", "name"},
        },
        ):
        return {
            "operation": "select",
            "tables": ["orders", "users"],
            "fields": ["users.name"],
            "filters": [],
            "joins": [{"left": "orders.user_id", "right": "users.id"}],
            "aggregations": [{"function": "COUNT", "field": "*", "alias": "count"}],
            "group_by": ["users.name"],
            "order_by": [{"field": "users.name", "direction": "ASC"}],
            "limit": requested_limit,
        }

    if (
        lowered_question.startswith("show users where name does not contain ")
        and _sql_schema_supports(database_schema, {"users": {"name"}})
    ):
        blocked_text = question.rsplit("contain", maxsplit=1)[-1].strip(" .!?\"'")
        if blocked_text:
            return {
                "operation": "select",
                "tables": ["users"],
                "fields": ["users.name"],
                "filters": [{"field": "users.name", "operator": "NOT_CONTAINS", "value": blocked_text}],
                "joins": [],
                "aggregations": [],
                "group_by": [],
                "order_by": [{"field": "users.name", "direction": "ASC"}],
                "limit": requested_limit,
            }

    if (
        lowered_question.startswith("user with name ending with ")
        and _sql_schema_supports(database_schema, {"users": {"name"}})
    ):
        suffix = question.rsplit("ending with", maxsplit=1)[-1].strip(" .!?\"'")
        if suffix:
            return {
                "operation": "select",
                "tables": ["users"],
                "fields": ["users.name"],
                "filters": [{"field": "users.name", "operator": "ENDS_WITH", "value": suffix}],
                "joins": [],
                "aggregations": [],
                "group_by": [],
                "order_by": [],
                "limit": requested_limit,
            }

    contains_or_match = re.fullmatch(
        r"show users where name contains ([a-z0-9@._-]+) or email contains ([a-z0-9@._-]+)",
        lowered_question,
    )
    if contains_or_match and _sql_schema_supports(database_schema, {"users": {"name", "email"}}):
        left_value, right_value = contains_or_match.groups()
        return {
            "operation": "select",
            "tables": ["users"],
            "fields": ["users.name", "users.email"],
            "filters": [
                {"field": "users.name", "operator": "CONTAINS", "value": left_value},
                {
                    "field": "users.email",
                    "operator": "CONTAINS",
                    "value": right_value,
                    "combine_with_previous": "OR",
                },
            ],
            "joins": [],
            "aggregations": [],
            "group_by": [],
            "order_by": [],
            "limit": requested_limit,
        }

    contains_and_match = re.fullmatch(
        r"show users where name contains ([a-z0-9@._-]+) and email contains ([a-z0-9@._-]+)",
        lowered_question,
    )
    if contains_and_match and _sql_schema_supports(database_schema, {"users": {"name", "email"}}):
        left_value, right_value = contains_and_match.groups()
        return {
            "operation": "select",
            "tables": ["users"],
            "fields": ["users.name", "users.email"],
            "filters": [
                {"field": "users.name", "operator": "CONTAINS", "value": left_value},
                {
                    "field": "users.email",
                    "operator": "CONTAINS",
                    "value": right_value,
                    "combine_with_previous": "AND",
                },
            ],
            "joins": [],
            "aggregations": [],
            "group_by": [],
            "order_by": [],
            "limit": requested_limit,
        }

    exact_name_match = re.fullmatch(r"name is ([a-z0-9@._-]+)", lowered_question)
    if exact_name_match and _sql_schema_supports(database_schema, {"users": {"name", "email"}}):
        return {
            "operation": "select",
            "tables": ["users"],
            "fields": ["users.name", "users.email"],
            "filters": [{"field": "users.name", "operator": "=", "value": exact_name_match.group(1).title()}],
            "joins": [],
            "aggregations": [],
            "group_by": [],
            "order_by": [],
            "limit": requested_limit,
        }

    negated_name_match = re.fullmatch(r"name is not ([a-z0-9@._-]+)", lowered_question)
    if negated_name_match and _sql_schema_supports(database_schema, {"users": {"name"}}):
        return {
            "operation": "select",
            "tables": ["users"],
            "fields": ["users.name"],
            "filters": [{"field": "users.name", "operator": "!=", "value": negated_name_match.group(1).title()}],
            "joins": [],
            "aggregations": [],
            "group_by": [],
            "order_by": [{"field": "users.name", "direction": "ASC"}],
            "limit": requested_limit,
        }

    contains_any_match = re.fullmatch(
        r"user with name containing any one letter ([a-z0-9@._-]+),([a-z0-9@._-]+)",
        lowered_question,
    )
    if contains_any_match and _sql_schema_supports(database_schema, {"users": {"name"}}):
        return {
            "operation": "select",
            "tables": ["users"],
            "fields": ["users.name"],
            "filters": [
                {
                    "field": "users.name",
                    "operator": "CONTAINS_ANY",
                    "value": list(contains_any_match.groups()),
                }
            ],
            "joins": [],
            "aggregations": [],
            "group_by": [],
            "order_by": [],
            "limit": requested_limit,
        }

    if normalized_question == "show users ordered by name" and _sql_schema_supports(
        database_schema,
        {"users": {"name"}},
    ):
        return {
            "operation": "select",
            "tables": ["users"],
            "fields": ["users.name"],
            "filters": [],
            "joins": [],
            "aggregations": [],
            "group_by": [],
            "order_by": [{"field": "users.name", "direction": "ASC"}],
            "limit": requested_limit,
        }

    amount_at_least_match = re.fullmatch(
        r"(?:show orders where amount >=|show me any orders where the amount is at least) ([0-9]+(?:\.[0-9]+)?)",
        lowered_question,
    )
    if amount_at_least_match and _sql_schema_supports(database_schema, {"orders": {"amount"}}):
        amount_value = float(amount_at_least_match.group(1))
        if amount_value.is_integer():
            amount_value = int(amount_value)
        return {
            "operation": "select",
            "tables": ["orders"],
            "fields": ["orders.amount"],
            "filters": [{"field": "orders.amount", "operator": ">=", "value": amount_value}],
            "joins": [],
            "aggregations": [],
            "group_by": [],
            "order_by": [{"field": "orders.amount", "direction": "ASC"}],
            "limit": requested_limit,
        }

    return None


def _sql_schema_supports(
    database_schema: dict[str, Any],
    required_columns: dict[str, set[str]],
) -> bool:
    """Return whether the active SQL schema contains each required table and column."""

    tables = database_schema.get("tables", {})
    if not isinstance(tables, dict):
        return False

    for table_name, expected_columns in required_columns.items():
        table_info = tables.get(table_name)
        if not isinstance(table_info, dict):
            return False

        actual_columns = {
            str(column.get("name", "")).strip()
            for column in table_info.get("columns", [])
            if isinstance(column, dict) and column.get("name")
        }
        if not expected_columns.issubset(actual_columns):
            return False

    return True


def _mongodb_schema_supports(
    database_schema: dict[str, Any],
    required_fields: dict[str, set[str]],
) -> bool:
    """Return whether the active MongoDB schema contains each required collection and field."""

    collections = database_schema.get("collections", {})
    if not isinstance(collections, dict):
        return False

    for collection_name, expected_fields in required_fields.items():
        collection_info = collections.get(collection_name)
        if not isinstance(collection_info, dict):
            return False

        actual_fields = {
            str(field.get("name", "")).strip()
            for field in collection_info.get("fields", [])
            if isinstance(field, dict) and field.get("name")
        }
        if not expected_fields.issubset(actual_fields):
            return False

    return True


def _normalize_question_text(question: str) -> str:
    """Lowercase one prompt and collapse punctuation so exact fast-path checks stay stable."""

    return re.sub(r"[^a-z0-9]+", " ", question.lower()).strip()


def _extract_requested_limit(normalized_question: str) -> int:
    """Read small explicit limits like 'top 2' or 'latest two' from one normalized prompt."""

    for pattern in (r"\btop (\d+)\b", r"\blatest (\d+)\b", r"\bfirst (\d+)\b"):
        match = re.search(pattern, normalized_question)
        if match is not None:
            return _normalize_limit(match.group(1))

    word_limits = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    for word, value in word_limits.items():
        if any(
            phrase in normalized_question
            for phrase in (f"top {word}", f"latest {word}", f"first {word}")
        ):
            return value

    return DEFAULT_LIMIT


def _build_python_result_explanation(question: str, result: dict[str, Any]) -> str:
    """Summarize the returned rows without an extra model round-trip."""

    row_count = int(result.get("row_count", 0))
    columns = [str(column) for column in result.get("columns", []) if str(column).strip()]
    rows = [row for row in result.get("rows", []) if isinstance(row, dict)]

    if row_count == 0:
        return f'The query for "{question}" ran successfully but returned no rows.'

    if row_count == 1 and rows:
        return (
            f'The query for "{question}" returned 1 row. '
            f"The result was { _format_result_row(rows[0], columns) }."
        )

    preview = _format_result_row(rows[0], columns) if rows else f"{row_count} rows"
    return (
        f'The query for "{question}" returned {row_count} rows. '
        f"The first result was {preview}."
    )


def _format_result_row(row: dict[str, Any], columns: list[str]) -> str:
    """Render one result row into a compact human-readable fragment."""

    ordered_columns = columns or list(row.keys())
    parts = [
        f"{column}={row[column]!r}"
        for column in ordered_columns
        if column in row
    ]
    return ", ".join(parts) if parts else "an empty row"


def _get_collection_name(query_plan: dict[str, Any], backend: str) -> str | None:
    """Return the MongoDB collection name when the executor needs one."""

    if backend != "mongodb":
        return None

    collection_name = str(query_plan.get("collection", "")).strip()
    if not collection_name:
        raise ValueError("MongoDB query plan must include a collection name.")

    return collection_name


def _extract_execution_error(state: QueryGraphState) -> str:
    """Read the last execution error from the graph state."""

    result = state.get("result", {})
    if isinstance(result, dict):
        return str(result.get("error", "Unknown execution error."))

    return "Unknown execution error."


def _with_trace(
    state: QueryGraphState,
    name: str,
    status: str,
    summary: str,
    details: dict[str, Any] | None = None,
    **updates: Any,
) -> QueryGraphState:
    """Return one state update payload with an appended workflow trace step."""

    next_trace = list(state.get("trace_steps", []))
    next_trace.append(
        {
            "name": name,
            "status": status,
            "summary": summary,
            "details": details or {},
        }
    )
    updates["trace_steps"] = next_trace
    return updates


def _raise_graph_error(
    message: str,
    state: QueryGraphState,
    name: str,
    status: str,
    summary: str,
    details: dict[str, Any] | None = None,
) -> None:
    """Raise a graph error that keeps the trace gathered so far."""

    safe_message = _normalize_model_message(
        message,
        empty_fallback=summary,
        placeholder_fallback=(
            f"{summary} The model returned a placeholder error instead of a concrete reason."
        ),
    )
    trace = list(state.get("trace_steps", []))
    trace.append(
        {
            "name": name,
            "status": status,
            "summary": summary,
            "details": details or {},
        }
    )
    raise QueryGraphError(safe_message, trace)


def _get_schema_entity_names(schema: dict[str, Any]) -> list[str]:
    """Return selected table or collection names from one schema slice."""

    tables = schema.get("tables", {})
    if isinstance(tables, dict) and tables:
        return list(tables.keys())

    collections = schema.get("collections", {})
    if isinstance(collections, dict) and collections:
        return list(collections.keys())

    return []


def _summarize_plan(query_plan: dict[str, Any]) -> dict[str, Any]:
    """Return a short plan summary for the debug trace."""

    if "tables" in query_plan:
        return {
            "tables": list(query_plan.get("tables", [])),
            "fields": list(query_plan.get("fields", [])),
            "filter_count": len(query_plan.get("filters", [])),
            "join_count": len(query_plan.get("joins", [])),
            "aggregation_count": len(query_plan.get("aggregations", [])),
            "group_by": list(query_plan.get("group_by", [])),
            "order_by": list(query_plan.get("order_by", [])),
            "limit": query_plan.get("limit"),
        }

    return {
        "collection": query_plan.get("collection", ""),
        "operation": query_plan.get("operation", ""),
        "intent": query_plan.get("intent", ""),
        "match_keys": sorted(query_plan.get("match", {}).keys())
        if isinstance(query_plan.get("match", {}), dict)
        else [],
        "project_keys": sorted(query_plan.get("project", {}).keys())
        if isinstance(query_plan.get("project", {}), dict)
        else [],
        "group_by": list(query_plan.get("group_by", [])),
        "aggregation_count": len(query_plan.get("aggregations", [])),
        "limit": query_plan.get("limit"),
    }
