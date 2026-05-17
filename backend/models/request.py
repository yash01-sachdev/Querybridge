"""Pydantic models used by the query API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ConnectionSettings(BaseModel):
    """Optional live database connection details sent from the frontend."""

    sqlite_path: str | None = None
    postgres_url: str | None = None
    mongo_url: str | None = None


class CompareConnectionIds(BaseModel):
    """Optional linked connection ids used by compare mode for each backend."""

    sqlite: str | None = None
    postgresql: str | None = None
    mongodb: str | None = None


class QueryRequest(BaseModel):
    """Incoming request payload for the natural-language query endpoint."""

    question: str = Field(..., min_length=1)
    backend: Literal["sqlite", "postgresql", "mongodb"]
    connection_id: str | None = None
    connection: ConnectionSettings | None = None


class QueryCompareRequest(BaseModel):
    """Incoming request payload for the compare-all-backends endpoint."""

    question: str = Field(..., min_length=1)
    backend: Literal["sqlite", "postgresql", "mongodb"]
    connection_ids: "CompareConnectionIds | None" = None


class QueryResult(BaseModel):
    """Executed query output returned by the backend."""

    rows: list[dict[str, Any]]
    row_count: int
    columns: list[str]


class QueryCheck(BaseModel):
    """Outcome of one pipeline guard check."""

    passed: bool
    reason: str


class QueryTraceStep(BaseModel):
    """One recorded step from the LangGraph workflow."""

    name: str
    status: str
    summary: str
    details: dict[str, Any] = Field(default_factory=dict)


class QueryTrace(BaseModel):
    """Structured workflow trace returned for debugging and evaluation."""

    steps: list[QueryTraceStep] = Field(default_factory=list)
    node_count: int = 0


class ConnectionTestRequest(BaseModel):
    """Incoming request payload for testing a live database connection."""

    backend: Literal["sqlite", "postgresql", "mongodb"]
    connection: ConnectionSettings | None = None


class ConnectionLinkRequest(BaseModel):
    """Incoming request payload for securely linking a live database."""

    backend: Literal["sqlite", "postgresql", "mongodb"]
    connection: ConnectionSettings


class ConnectionLinkResponse(BaseModel):
    """Response returned after securely linking a live database."""

    backend: Literal["sqlite", "postgresql", "mongodb"]
    success: bool
    connection_id: str
    message: str
    entity_type: Literal["tables", "collections"]
    entity_count: int
    entity_names: list[str]


class ConnectionDisconnectRequest(BaseModel):
    """Incoming request payload for removing a linked database session."""

    connection_id: str = Field(..., min_length=1)


class ConnectionDisconnectResponse(BaseModel):
    """Response returned after removing a linked database session."""

    success: bool
    message: str


class ConnectionTestResponse(BaseModel):
    """Response returned after testing a live database connection."""

    backend: Literal["sqlite", "postgresql", "mongodb"]
    success: bool
    message: str
    entity_type: Literal["tables", "collections"]
    entity_count: int
    entity_names: list[str]


class QueryResponse(BaseModel):
    """Temporary response payload returned while the query pipeline is being built."""

    question: str
    backend: Literal["sqlite", "postgresql", "mongodb"]
    workflow: str
    model: str
    database_schema: dict[str, Any]
    relevant_schema: dict[str, Any]
    query_plan: dict[str, Any]
    compiled_query: str
    safety_check: QueryCheck
    validation_check: QueryCheck
    result: QueryResult
    trace: QueryTrace
    explanation: str
    repaired: bool
    repair_attempts: int
    execution_time_ms: float
    message: str


class EvalRunRequest(BaseModel):
    """Request payload for running the built-in GenAI evaluation suite."""

    backend: Literal["sqlite", "postgresql", "mongodb"]


class EvalCaseResult(BaseModel):
    """Outcome of one built-in evaluation case."""

    name: str
    question: str
    expected: Literal["success", "error"]
    status: Literal["Pass", "Fail"]
    passed: bool
    message: str
    compiled_query: str = ""
    row_count: int = 0
    latency_ms: float = 0
    trace: QueryTrace


class EvalRunResponse(BaseModel):
    """Summary plus case-by-case results for one backend eval run."""

    backend: Literal["sqlite", "postgresql", "mongodb"]
    workflow: str
    model: str
    dataset_source: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    pass_rate: float
    avg_latency_ms: float
    cases: list[EvalCaseResult]


class BackendComparison(BaseModel):
    """One backend's query preview in compare mode."""

    backend: Literal["sqlite", "postgresql", "mongodb"]
    schema_source: str = ""
    workflow: str = ""
    model: str = ""
    query_plan: dict[str, Any] = Field(default_factory=dict)
    compiled_query: str = ""
    trace: QueryTrace = Field(default_factory=QueryTrace)
    message: str
    success: bool


class QueryCompareResponse(BaseModel):
    """Side-by-side preview response for all supported backends."""

    question: str
    schema_source: str
    relevant_schema: dict[str, Any]
    comparisons: list[BackendComparison]
