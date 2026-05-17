import { formatBackendLabel } from "./backendOptions";

const RAW_API_BASE_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
const API_BASE_URL = RAW_API_BASE_URL.replace(/\/+$/, "");
const HEALTH_TIMEOUT_MS = 5000;
const DEFAULT_TIMEOUT_MS = 30000;
const COMPARE_TIMEOUT_MS = 180000;
const EVAL_TIMEOUT_MS = 300000;
const STREAM_CONNECT_TIMEOUT_MS = 15000;
const STREAM_IDLE_TIMEOUT_MS = 120000;
const STREAM_TIMEOUT_MS = 300000;
const PLACEHOLDER_ERROR_WORDS = new Set([
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
]);

export async function requestHealth() {
  const payload = await requestJson(`${API_BASE_URL}/health`, {
    fallbackMessage: "The backend returned an unexpected health error.",
    timeoutMs: HEALTH_TIMEOUT_MS,
  });

  return payload;
}

export async function requestQueryStream(
  question,
  backend,
  connectionId = null,
  {
    onWorkflowStarted = () => {},
    onSchemaReady = () => {},
    onStep = () => {},
    timeoutMs = STREAM_TIMEOUT_MS,
    idleTimeoutMs = STREAM_IDLE_TIMEOUT_MS,
    connectTimeoutMs = STREAM_CONNECT_TIMEOUT_MS,
  } = {},
) {
  const controller = new AbortController();
  let abortReason = "";
  let connectTimeoutId = 0;
  let idleTimeoutId = 0;
  const overallTimeoutId = window.setTimeout(() => abortStreamRequest("overall"), timeoutMs);
  let response;

  function abortStreamRequest(reason) {
    abortReason = reason;
    if (!controller.signal.aborted) {
      controller.abort();
    }
  }

  function resetIdleTimeout() {
    if (idleTimeoutId) {
      window.clearTimeout(idleTimeoutId);
    }
    idleTimeoutId = window.setTimeout(() => abortStreamRequest("idle"), idleTimeoutMs);
  }

  function clearStreamTimeouts() {
    window.clearTimeout(overallTimeoutId);
    if (connectTimeoutId) {
      window.clearTimeout(connectTimeoutId);
    }
    if (idleTimeoutId) {
      window.clearTimeout(idleTimeoutId);
    }
  }

  connectTimeoutId = window.setTimeout(() => abortStreamRequest("connect"), connectTimeoutMs);

  try {
    response = await fetch(`${API_BASE_URL}/query/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question,
        backend,
        connection_id: connectionId,
      }),
      signal: controller.signal,
    });
  } catch (error) {
    if (error?.name === "AbortError") {
      clearStreamTimeouts();
      throw new Error(buildStreamTimeoutMessage(abortReason));
    }
    clearStreamTimeouts();
    throw new Error("Could not reach the backend. Make sure the API server is running.");
  }

  if (!response.ok) {
    clearStreamTimeouts();
    let payload = null;
    try {
      payload = await response.json();
    } catch {
      payload = null;
    }
    throw buildApiError(payload, "The backend returned an unexpected streamed-query error.");
  }

  if (!response.body) {
    clearStreamTimeouts();
    throw new Error("The backend did not open a streamed query response.");
  }

  window.clearTimeout(connectTimeoutId);
  resetIdleTimeout();

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }

      resetIdleTimeout();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      buffer = buffer.replace(/\r\n/g, "\n");

      let separatorIndex = buffer.indexOf("\n\n");
      while (separatorIndex !== -1) {
        const chunk = buffer.slice(0, separatorIndex).trim();
        buffer = buffer.slice(separatorIndex + 2);

        if (chunk) {
          const eventPayload = parseSseEvent(chunk);

          if (eventPayload.event === "workflow_started") {
            onWorkflowStarted(eventPayload.data);
          } else if (eventPayload.event === "schema_ready") {
            onSchemaReady(eventPayload.data);
          } else if (eventPayload.event === "step") {
            onStep(eventPayload.data);
          } else if (eventPayload.event === "completed") {
            return eventPayload.data;
          } else if (eventPayload.event === "error") {
            throw buildApiError(
              { detail: eventPayload.data },
              "The streamed query failed unexpectedly.",
            );
          }
        }

        separatorIndex = buffer.indexOf("\n\n");
      }

    }
  } catch (error) {
    if (error?.name === "AbortError") {
      throw new Error(buildStreamTimeoutMessage(abortReason));
    }
    throw error;
  } finally {
    clearStreamTimeouts();
    reader.releaseLock();
  }

  throw new Error("The streamed query ended before the backend returned a final result.");
}

export async function requestEvalRun(backend) {
  const payload = await requestJson(`${API_BASE_URL}/evals/run`, {
    method: "POST",
    fallbackMessage: "The backend returned an unexpected eval error.",
    timeoutMs: EVAL_TIMEOUT_MS,
    body: {
      backend,
    },
  });

  return {
    backend: payload?.backend || backend,
    workflow: payload?.workflow || "langgraph",
    model: payload?.model || "",
    datasetSource: payload?.dataset_source || "built-in demo fixture",
    totalCases: payload?.total_cases || 0,
    passedCases: payload?.passed_cases || 0,
    failedCases: payload?.failed_cases || 0,
    passRate: payload?.pass_rate || 0,
    avgLatencyMs: payload?.avg_latency_ms || 0,
    cases: payload?.cases || [],
  };
}

export async function testDatabaseConnection(backend, connection) {
  return requestJson(`${API_BASE_URL}/connection/test`, {
    method: "POST",
    fallbackMessage: "The backend returned an unexpected error.",
    timeoutMs: DEFAULT_TIMEOUT_MS,
    body: {
      backend,
      connection,
    },
  });
}

export async function linkDatabaseConnection(backend, connection) {
  return requestJson(`${API_BASE_URL}/connection/link`, {
    method: "POST",
    fallbackMessage: "The backend returned an unexpected error.",
    timeoutMs: DEFAULT_TIMEOUT_MS,
    body: {
      backend,
      connection,
    },
  });
}

export async function disconnectDatabaseConnection(connectionId) {
  return requestJson(`${API_BASE_URL}/connection/disconnect`, {
    method: "POST",
    fallbackMessage: "The backend returned an unexpected error.",
    timeoutMs: DEFAULT_TIMEOUT_MS,
    body: {
      connection_id: connectionId,
    },
  });
}

export async function requestQueryComparison(question, backend, linkedConnections = {}) {
  const connectionIds = buildCompareConnectionIds(linkedConnections);
  const payload = await requestJson(`${API_BASE_URL}/compare`, {
    method: "POST",
    fallbackMessage: "The backend returned an unexpected error.",
    timeoutMs: COMPARE_TIMEOUT_MS,
    body: {
      question,
      backend,
      connection_ids: connectionIds,
    },
  });

  const comparisons = payload?.comparisons || [];

  return {
    question: payload?.question || question,
    schemaSource: payload?.schema_source || "built-in learning schema",
    relevantSchema: payload?.relevant_schema || {},
    comparisons: comparisons.map((item) => ({
      backend: item.backend,
      label: formatBackendLabel(item.backend),
      ok: item.success,
      schemaSource: item.schema_source || "built-in learning schema",
      workflow: item.workflow || "",
      model: item.model || "",
      queryPlan: item.query_plan,
      compiledQuery: item.compiled_query,
      trace: item.trace || null,
      message: item.message,
      errorMessage: item.message,
    })),
  };
}

function buildCompareConnectionIds(linkedConnections) {
  return {
    sqlite: linkedConnections?.sqlite?.connectionId || null,
    postgresql: linkedConnections?.postgresql?.connectionId || null,
    mongodb: linkedConnections?.mongodb?.connectionId || null,
  };
}

async function requestJson(
  url,
  {
    method = "GET",
    body = null,
    fallbackMessage = "The backend returned an unexpected error.",
    timeoutMs = DEFAULT_TIMEOUT_MS,
  } = {},
) {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);
  let response;

  try {
    response = await fetch(url, {
      method,
      headers: {
        "Content-Type": "application/json",
      },
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });
  } catch (error) {
    window.clearTimeout(timeoutId);
    if (error?.name === "AbortError") {
      throw new Error("The request took too long and was cancelled. Please try again.");
    }
    throw new Error("Could not reach the backend. Make sure the API server is running.");
  }

  let payload = null;

  try {
    payload = await response.json();
  } catch {
    payload = null;
  }

  if (!response.ok) {
    window.clearTimeout(timeoutId);
    throw buildApiError(payload, fallbackMessage);
  }

  window.clearTimeout(timeoutId);
  return payload;
}

function buildApiError(payload, fallbackMessage) {
  const detail = payload?.detail;

  if (typeof detail === "string" && detail) {
    return new Error(normalizeApiErrorMessage(detail, fallbackMessage));
  }

  if (detail && typeof detail === "object") {
    const error = new Error(normalizeApiErrorMessage(detail.message, fallbackMessage));
    if (detail.trace) {
      error.trace = detail.trace;
    }
    if (detail.workflow) {
      error.workflow = detail.workflow;
    }
    return error;
  }

  return new Error(fallbackMessage);
}

function buildStreamTimeoutMessage(abortReason) {
  if (abortReason === "connect") {
    return "The backend did not start the streamed query in time. Make sure the API server is running and responsive.";
  }

  if (abortReason === "idle") {
    return "The query is still taking too long for Ollama. The model may still be warming up or running slowly.";
  }

  if (abortReason === "overall") {
    return "The query ran too long overall for the browser session. Try again after Ollama warms up, or switch to a smaller model.";
  }

  return "The query timed out while waiting for Ollama. Try again, or switch to a smaller model.";
}

function normalizeApiErrorMessage(message, fallbackMessage) {
  if (typeof message !== "string") {
    return fallbackMessage;
  }

  const trimmed = message.trim();
  if (!trimmed) {
    return fallbackMessage;
  }

  if (isPlaceholderErrorMessage(trimmed)) {
    return "The model rejected the request without giving a concrete reason. Please try the query again.";
  }

  return trimmed;
}

function isPlaceholderErrorMessage(message) {
  const normalized = message
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
  const tokens = normalized.split(/\s+/).filter(Boolean);

  return (
    tokens.length > 0 &&
    tokens.length <= 5 &&
    (tokens.includes("reason") || tokens.includes("error")) &&
    tokens.every((token) => PLACEHOLDER_ERROR_WORDS.has(token))
  );
}

function parseSseEvent(chunk) {
  const eventPayload = {
    event: "message",
    data: {},
  };
  const dataLines = [];

  for (const line of chunk.split("\n")) {
    if (line.startsWith("event:")) {
      eventPayload.event = line.slice("event:".length).trim();
      continue;
    }

    if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).trimStart());
    }
  }

  const rawData = dataLines.join("\n");
  if (!rawData) {
    return eventPayload;
  }

  try {
    eventPayload.data = JSON.parse(rawData);
  } catch {
    eventPayload.data = { message: rawData };
  }

  return eventPayload;
}
