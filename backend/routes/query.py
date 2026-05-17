"""Query route definitions for the nl-query-copilot API."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from core.comparison import (
    DEMO_SCHEMA_SOURCE,
    LIVE_SCHEMA_SOURCE,
    SUPPORTED_BACKENDS,
    build_comparison_queries,
    get_demo_compare_schema_for_backend,
)
from core.connection import open_backend_connection
from core.connection_registry import get_connection, register_connection, remove_connection
from core.eval_runner import run_eval_suite
from core.ollama_client import OllamaResponseError, OllamaUnavailableError
from core.query_graph import QueryGraphError, run_query_graph, stream_query_graph
from core.retriever import retrieve_relevant_schema
from core.schema_extractor import extract_schema
from models.request import (
    BackendComparison,
    ConnectionDisconnectRequest,
    ConnectionDisconnectResponse,
    ConnectionLinkRequest,
    ConnectionLinkResponse,
    ConnectionTestRequest,
    ConnectionTestResponse,
    EvalCaseResult,
    EvalRunRequest,
    EvalRunResponse,
    QueryTrace,
    QueryTraceStep,
    QueryCompareRequest,
    QueryCheck,
    QueryCompareResponse,
    QueryRequest,
    QueryResponse,
    QueryResult,
)

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def create_query(request: QueryRequest) -> QueryResponse:
    """Run the LangGraph query workflow and return the finished query result."""

    started_at = perf_counter()

    try:
        connection_overrides = _resolve_connection_overrides(
            backend=request.backend,
            connection_id=request.connection_id,
            connection=request.connection,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        connection, close_connection = open_backend_connection(
            request.backend,
            connection_overrides,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        database_schema = extract_schema(request.backend, connection)
        graph_result = run_query_graph(
            question=request.question,
            backend=request.backend,
            database_schema=database_schema,
            connection=connection,
        )
    except QueryGraphError as exc:
        raise HTTPException(
            status_code=400,
            detail=_build_traceable_error_detail(str(exc), exc.trace),
        ) from exc
    except OllamaUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except OllamaResponseError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        close_connection()

    execution_result = graph_result["result"]
    if "error" in execution_result:
        raise HTTPException(status_code=500, detail=f"Query execution failed: {execution_result['error']}")

    execution_time_ms = (perf_counter() - started_at) * 1000

    return _build_query_response(
        request=request,
        database_schema=database_schema,
        graph_result=graph_result,
        execution_time_ms=execution_time_ms,
    )


@router.post("/query/stream")
def stream_query(request: QueryRequest) -> StreamingResponse:
    """Run the query workflow and stream step-by-step progress events as SSE."""

    try:
        connection_overrides = _resolve_connection_overrides(
            backend=request.backend,
            connection_id=request.connection_id,
            connection=request.connection,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        connection, close_connection = open_backend_connection(
            request.backend,
            connection_overrides,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    def event_stream():
        started_at = perf_counter()

        try:
            database_schema = extract_schema(request.backend, connection)
            yield _format_sse_event(
                "schema_ready",
                {
                    "backend": request.backend,
                    "schema_summary": _schema_summary_payload(database_schema),
                },
            )

            event_generator = stream_query_graph(
                question=request.question,
                backend=request.backend,
                database_schema=database_schema,
                connection=connection,
            )

            for event_payload in event_generator:
                event_type = str(event_payload.get("type", "message"))

                if event_type == "completed":
                    graph_result = event_payload.get("response", {})
                    if not isinstance(graph_result, dict):
                        raise ValueError("The streamed workflow did not return a valid response payload.")

                    execution_result = graph_result.get("result", {})
                    if isinstance(execution_result, dict) and "error" in execution_result:
                        raise ValueError(f"Query execution failed: {execution_result['error']}")

                    query_response = _build_query_response(
                        request=request,
                        database_schema=database_schema,
                        graph_result=graph_result,
                        execution_time_ms=(perf_counter() - started_at) * 1000,
                    )
                    yield _format_sse_event("completed", query_response.model_dump())
                    continue

                yield _format_sse_event(event_type, event_payload)

        except QueryGraphError as exc:
            yield _format_sse_event(
                "error",
                _build_traceable_error_detail(str(exc), exc.trace),
            )
        except OllamaUnavailableError as exc:
            yield _format_sse_event("error", {"message": str(exc), "workflow": "langgraph"})
        except OllamaResponseError as exc:
            yield _format_sse_event("error", {"message": str(exc), "workflow": "langgraph"})
        except ValueError as exc:
            yield _format_sse_event("error", {"message": str(exc), "workflow": "langgraph"})
        finally:
            close_connection()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/evals/run", response_model=EvalRunResponse)
def run_built_in_evals(request: EvalRunRequest) -> EvalRunResponse:
    """Run the built-in prompt suite against one backend's demo fixture."""

    try:
        eval_result = run_eval_suite(request.backend, router_root_path())
    except QueryGraphError as exc:
        raise HTTPException(
            status_code=400,
            detail=_build_traceable_error_detail(str(exc), exc.trace),
        ) from exc
    except OllamaUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except OllamaResponseError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return EvalRunResponse(
        backend=request.backend,
        workflow=str(eval_result["workflow"]),
        model=str(eval_result["model"]),
        dataset_source=str(eval_result["dataset_source"]),
        total_cases=int(eval_result["total_cases"]),
        passed_cases=int(eval_result["passed_cases"]),
        failed_cases=int(eval_result["failed_cases"]),
        pass_rate=float(eval_result["pass_rate"]),
        avg_latency_ms=float(eval_result["avg_latency_ms"]),
        cases=[
            EvalCaseResult(
                name=str(case["name"]),
                question=str(case["question"]),
                expected=str(case["expected"]),
                status=str(case["status"]),
                passed=bool(case["passed"]),
                message=str(case["message"]),
                compiled_query=str(case["compiled_query"]),
                row_count=int(case["row_count"]),
                latency_ms=float(case["latency_ms"]),
                trace=_to_query_trace(case["trace"]),
            )
            for case in eval_result["cases"]
        ],
    )


@router.post("/connection/test", response_model=ConnectionTestResponse)
def test_connection(request: ConnectionTestRequest) -> ConnectionTestResponse:
    """Validate a live database link and return a quick schema summary."""

    connection_overrides = _connection_overrides(request.connection)

    try:
        connection, close_connection = open_backend_connection(
            request.backend,
            connection_overrides,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        schema = extract_schema(request.backend, connection)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        close_connection()

    entity_type, entity_names = _schema_summary(schema)
    entity_count = len(entity_names)
    entity_label = "table" if entity_type == "tables" else "collection"
    plural_suffix = "" if entity_count == 1 else "s"

    return ConnectionTestResponse(
        backend=request.backend,
        success=True,
        message=f"Connected successfully. Found {entity_count} {entity_label}{plural_suffix}.",
        entity_type=entity_type,
        entity_count=entity_count,
        entity_names=entity_names[:12],
    )


@router.post("/connection/link", response_model=ConnectionLinkResponse)
def link_connection(request: ConnectionLinkRequest) -> ConnectionLinkResponse:
    """Store a tested connection on the backend and return a temporary opaque id."""

    connection_overrides = _connection_overrides(request.connection)
    if connection_overrides is None:
        raise HTTPException(status_code=400, detail="Connection details are required.")

    try:
        connection, close_connection = open_backend_connection(
            request.backend,
            connection_overrides,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        schema = extract_schema(request.backend, connection)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        close_connection()

    connection_id = register_connection(request.backend, connection_overrides)
    entity_type, entity_names = _schema_summary(schema)
    entity_count = len(entity_names)

    return ConnectionLinkResponse(
        backend=request.backend,
        success=True,
        connection_id=connection_id,
        message="Database linked securely for this backend session.",
        entity_type=entity_type,
        entity_count=entity_count,
        entity_names=entity_names[:12],
    )


@router.post("/connection/disconnect", response_model=ConnectionDisconnectResponse)
def disconnect_connection(request: ConnectionDisconnectRequest) -> ConnectionDisconnectResponse:
    """Remove a linked connection from backend memory."""

    was_removed = remove_connection(request.connection_id)

    if not was_removed:
        raise HTTPException(status_code=404, detail="That database link no longer exists.")

    return ConnectionDisconnectResponse(
        success=True,
        message="Database link removed from backend memory.",
    )


@router.post("/compare", response_model=QueryCompareResponse)
def compare_backends(request: QueryCompareRequest) -> QueryCompareResponse:
    """Build preview queries for all backends using live links when they exist."""

    try:
        backend_inputs = _build_compare_backend_inputs(request.connection_ids)
        selected_schema, schema_source = _select_compare_schema_for_backend(
            request.backend,
            backend_inputs,
        )
        relevant_schema = retrieve_relevant_schema(
            request.question,
            selected_schema,
            request.backend,
        )
        comparisons = build_comparison_queries(request.question, backend_inputs)
    except QueryGraphError as exc:
        raise HTTPException(
            status_code=400,
            detail=_build_traceable_error_detail(str(exc), exc.trace),
        ) from exc
    except OllamaUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except OllamaResponseError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return QueryCompareResponse(
        question=request.question,
        schema_source=schema_source,
        relevant_schema=relevant_schema,
        comparisons=[BackendComparison(**comparison) for comparison in comparisons],
    )


def _connection_overrides(connection: object) -> dict[str, str | None] | None:
    """Turn an optional connection model into a plain dict for the core layer."""

    if connection is None:
        return None

    model_dump = getattr(connection, "model_dump", None)
    if callable(model_dump):
        return model_dump()

    return None


def _build_success_message(repaired: bool, repair_attempts: int) -> str:
    """Return a short status message for the finished pipeline."""

    if not repaired:
        return "Schema extracted, model-planned, semantically checked, compiled, validated, and executed successfully."

    return (
        "Schema extracted, model-planned, semantically checked, repaired, validated, and executed successfully "
        f"after {repair_attempts} repair attempt{'s' if repair_attempts != 1 else ''}."
    )


def _resolve_connection_overrides(
    backend: str,
    connection_id: str | None,
    connection: object,
) -> dict[str, str | None] | None:
    """Choose whether the query should use a linked connection id or direct details."""

    if connection_id:
        return get_connection(connection_id, backend)

    return _connection_overrides(connection)


def _build_query_response(
    request: QueryRequest,
    database_schema: dict[str, object],
    graph_result: dict[str, object],
    execution_time_ms: float,
) -> QueryResponse:
    """Convert the raw graph output into the shared query response model."""

    execution_result = graph_result["result"]
    return QueryResponse(
        question=request.question,
        backend=request.backend,
        workflow=str(graph_result["workflow"]),
        model=str(graph_result["model"]),
        database_schema=database_schema,
        relevant_schema=graph_result["relevant_schema"],
        query_plan=graph_result["query_plan"],
        compiled_query=graph_result["compiled_query"],
        safety_check=QueryCheck(
            passed=graph_result["safety_check"][0],
            reason=graph_result["safety_check"][1],
        ),
        validation_check=QueryCheck(
            passed=graph_result["validation_check"][0],
            reason=graph_result["validation_check"][1],
        ),
        result=QueryResult(
            rows=execution_result["rows"],
            row_count=execution_result["row_count"],
            columns=execution_result["columns"],
        ),
        trace=_to_query_trace(graph_result["trace"]),
        explanation=str(graph_result["explanation"]),
        repaired=bool(graph_result["repaired"]),
        repair_attempts=int(graph_result["repair_attempts"]),
        execution_time_ms=round(execution_time_ms, 2),
        message=_build_success_message(
            repaired=bool(graph_result["repaired"]),
            repair_attempts=int(graph_result["repair_attempts"]),
        ),
    )


def _schema_summary(schema: dict[str, object]) -> tuple[str, list[str]]:
    """Return the top-level entity type and names from an extracted schema."""

    tables = schema.get("tables")
    if isinstance(tables, dict):
        return "tables", list(tables.keys())

    collections = schema.get("collections")
    if isinstance(collections, dict):
        return "collections", list(collections.keys())

    return "tables", []


def _schema_summary_payload(schema: dict[str, object]) -> dict[str, object]:
    """Return a small schema summary that is safe to send early in stream mode."""

    entity_type, entity_names = _schema_summary(schema)
    return {
        "entity_type": entity_type,
        "entity_count": len(entity_names),
        "entity_names": entity_names[:12],
    }


def _build_compare_backend_inputs(connection_ids: object) -> dict[str, dict[str, object]]:
    """Choose the schema source for each compare backend."""

    connection_id_map = _compare_connection_id_map(connection_ids)
    backend_inputs: dict[str, dict[str, object]] = {}

    for backend in SUPPORTED_BACKENDS:
        connection_id = connection_id_map.get(backend)
        if not connection_id:
            backend_inputs[backend] = {
                "schema": get_demo_compare_schema_for_backend(backend),
                "schema_source": DEMO_SCHEMA_SOURCE,
            }
            continue

        try:
            connection_overrides = get_connection(connection_id, backend)
            connection, close_connection = open_backend_connection(
                backend,
                connection_overrides,
            )
        except ValueError as exc:
            backend_inputs[backend] = {
                "error": str(exc),
                "schema_source": LIVE_SCHEMA_SOURCE,
            }
            continue

        try:
            schema = extract_schema(backend, connection)
        except ValueError as exc:
            backend_inputs[backend] = {
                "error": str(exc),
                "schema_source": LIVE_SCHEMA_SOURCE,
            }
        else:
            backend_inputs[backend] = {
                "schema": schema,
                "schema_source": LIVE_SCHEMA_SOURCE,
            }
        finally:
            close_connection()

    return backend_inputs


def _compare_connection_id_map(connection_ids: object) -> dict[str, str]:
    """Turn compare-mode connection ids into a plain backend-to-id mapping."""

    if connection_ids is None:
        return {}

    model_dump = getattr(connection_ids, "model_dump", None)
    if callable(model_dump):
        raw_mapping = model_dump()
    elif isinstance(connection_ids, dict):
        raw_mapping = connection_ids
    else:
        return {}

    normalized_mapping: dict[str, str] = {}
    for backend in SUPPORTED_BACKENDS:
        raw_value = raw_mapping.get(backend)
        if isinstance(raw_value, str):
            trimmed_value = raw_value.strip()
            if trimmed_value:
                normalized_mapping[backend] = trimmed_value

    return normalized_mapping


def _select_compare_schema_for_backend(
    backend: str,
    backend_inputs: dict[str, dict[str, object]],
) -> tuple[dict[str, object], str]:
    """Pick one schema for the top-level compare summary in the selected backend."""

    selected_input = backend_inputs.get(backend, {})
    selected_schema = selected_input.get("schema")

    if isinstance(selected_schema, dict):
        schema_source = str(selected_input.get("schema_source", DEMO_SCHEMA_SOURCE))
        return selected_schema, schema_source

    return get_demo_compare_schema_for_backend(backend), DEMO_SCHEMA_SOURCE


def _to_query_trace(trace_payload: dict[str, object]) -> QueryTrace:
    """Convert a plain dict trace payload into the response model."""

    raw_steps = trace_payload.get("steps", [])
    steps = [
        QueryTraceStep(
            name=str(step.get("name", "")),
            status=str(step.get("status", "")),
            summary=str(step.get("summary", "")),
            details=dict(step.get("details", {}))
            if isinstance(step.get("details", {}), dict)
            else {},
        )
        for step in raw_steps
        if isinstance(step, dict)
    ]

    return QueryTrace(
        steps=steps,
        node_count=int(trace_payload.get("node_count", len(steps))),
    )


def _build_traceable_error_detail(message: str, trace_payload: dict[str, object]) -> dict[str, object]:
    """Return one structured error payload that keeps the workflow trace."""

    trace = _to_query_trace(trace_payload)
    return {
        "message": message,
        "workflow": "langgraph",
        "trace": trace.model_dump(),
    }


def _format_sse_event(event_name: str, payload: dict[str, object]) -> str:
    """Serialize one server-sent event."""

    return f"event: {event_name}\ndata: {json.dumps(payload)}\n\n"


def router_root_path() -> Path:
    """Return the backend root path for eval fixtures."""

    return Path(__file__).resolve().parent.parent
