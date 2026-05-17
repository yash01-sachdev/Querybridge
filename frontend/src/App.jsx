import { useEffect, useState } from "react";
import { Database, Menu } from "lucide-react";

import EvalPanel from "./components/EvalPanel";
import ComparePanel from "./components/ComparePanel";
import QueryBox from "./components/QueryBox";
import QueryHistorySidebar from "./components/QueryHistorySidebar";
import ResultPanel from "./components/ResultPanel";
import { historyItems as initialHistoryItems } from "./mockData";
import {
  disconnectDatabaseConnection,
  linkDatabaseConnection,
  requestEvalRun,
  requestHealth,
  requestQueryStream,
  requestQueryComparison,
  testDatabaseConnection,
} from "./api";
import {
  appendRunRecord,
  buildEvalRows,
  buildEvalSuiteRows,
  buildEvalSuiteSummary,
  buildEvalSummary,
  createFailedRun,
  createSuccessfulRun,
} from "./evalMetrics";
import { formatBackendLabel } from "./backendOptions";
import {
  buildConnectionPayload,
  clearConnectionDraft,
  createEmptyConnectionDrafts,
} from "./connectionSettings";

const QUERY_LOADING_STAGES = [
  {
    headline: "Checking runtime",
    detail: "Making sure the backend and local model are ready for a query run.",
  },
  {
    headline: "Selecting schema",
    detail: "Narrowing the database down to the most relevant tables or collections.",
  },
  {
    headline: "Planning query",
    detail: "Turning your question into a structured backend query plan.",
  },
  {
    headline: "Validating query",
    detail: "Compiling the query, checking safety rules, and confirming schema references.",
  },
  {
    headline: "Fetching results",
    detail: "Executing the read-only query and preparing the result summary.",
  },
];

const COMPARE_LOADING_STAGES = [
  {
    headline: "Preparing compare mode",
    detail: "Collecting schema context for each backend you want to compare.",
  },
  {
    headline: "Planning each backend",
    detail: "Generating side-by-side query previews for SQLite, PostgreSQL, and MongoDB.",
  },
  {
    headline: "Wrapping up comparison",
    detail: "Formatting the previews so you can compare query plans and compiled output.",
  },
];

const LIVE_TRACE_PRELUDE = [
  {
    name: "workflow_started",
    summary: "Connecting to the FastAPI runtime and confirming the active Ollama model.",
  },
  {
    name: "schema_ready",
    summary: "Inspecting the linked database so the workflow knows which entities it can use.",
  },
];

const LIVE_TRACE_STEP_LIBRARY = {
  select_schema: "Choosing the most relevant tables or collections for this question.",
  plan_query: "Turning the prompt into a structured database query plan.",
  semantic_check: "Checking whether the plan really matches the question and selected schema.",
  compile_query: "Compiling the structured plan into a runnable SQL or MongoDB query.",
  check_safety: "Applying read-only safety rules before any query can run.",
  check_validation: "Validating syntax and schema references against the selected backend.",
  repair_query: "Repairing the failed plan before retrying the compile and validation path.",
  execute_query: "Running the query against the live database and collecting rows.",
  explain_result: "Summarizing the result in plain language for the user.",
};

const DEFAULT_LIVE_FLOW = [
  "select_schema",
  "plan_query",
  "semantic_check",
  "compile_query",
  "check_safety",
  "check_validation",
  "execute_query",
  "explain_result",
];
const EMPTY_CONNECTIONS = {
  sqlite: null,
  postgresql: null,
  mongodb: null,
};

export default function App() {
  const [activeTab, setActiveTab] = useState("query");
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);
  const [historyItems, setHistoryItems] = useState(initialHistoryItems);
  const [question, setQuestion] = useState("");
  const [backend, setBackend] = useState("sqlite");
  const [connectionDrafts, setConnectionDrafts] = useState(() => createEmptyConnectionDrafts());
  const [linkedConnections, setLinkedConnections] = useState(EMPTY_CONNECTIONS);
  const [defaultConnections, setDefaultConnections] = useState(EMPTY_CONNECTIONS);
  const [connectionFeedback, setConnectionFeedback] = useState({});
  const [selectedHistory, setSelectedHistory] = useState(null);
  const [queryResponse, setQueryResponse] = useState(null);
  const [queryRuns, setQueryRuns] = useState([]);
  const [comparisonResponses, setComparisonResponses] = useState([]);
  const [comparisonSchemaSource, setComparisonSchemaSource] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [errorTrace, setErrorTrace] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isTestingConnection, setIsTestingConnection] = useState(false);
  const [isLinkingConnection, setIsLinkingConnection] = useState(false);
  const [isRunningEvalSuite, setIsRunningEvalSuite] = useState(false);
  const [resultMode, setResultMode] = useState("single");
  const [loadingMode, setLoadingMode] = useState("single");
  const [healthStatus, setHealthStatus] = useState(null);
  const [evalRunResponse, setEvalRunResponse] = useState(null);
  const [evalSuiteError, setEvalSuiteError] = useState("");
  const [loadingTrace, setLoadingTrace] = useState(() =>
    buildLiveTrace({
      workflowStarted: false,
      schemaReady: false,
      actualSteps: [],
      schemaSummary: null,
    }),
  );
  const [loadingRuntime, setLoadingRuntime] = useState({
    workflow: "langgraph",
    model: "",
    schemaSummary: null,
  });
  const [loadingHintIndex, setLoadingHintIndex] = useState(0);
  const [loadingElapsedMs, setLoadingElapsedMs] = useState(0);
  const evalSummary = buildEvalSummary(queryRuns);
  const evalRows = buildEvalRows(queryRuns);
  const evalSuiteSummary = buildEvalSuiteSummary(evalRunResponse);
  const evalSuiteRows = buildEvalSuiteRows(evalRunResponse);
  const loadingState = buildLoadingState({
    isLoading,
    loadingMode,
    loadingTrace,
    loadingRuntime,
    loadingHintIndex,
    loadingElapsedMs,
  });

  useEffect(() => {
    let isMounted = true;

    async function loadHealth() {
      try {
        const nextHealthStatus = await requestHealth();
        if (isMounted) {
          setHealthStatus(nextHealthStatus);
          setDefaultConnections(buildDefaultConnections(nextHealthStatus));
        }
      } catch {
        if (isMounted) {
          setHealthStatus({
            status: "offline",
            workflow: "langgraph",
            model: "",
            ollama: {
              available: false,
              installed: false,
            },
          });
          setDefaultConnections(EMPTY_CONNECTIONS);
        }
      }
    }

    loadHealth();
    const intervalId = window.setInterval(loadHealth, 20000);

    return () => {
      isMounted = false;
      window.clearInterval(intervalId);
    };
  }, []);

  useEffect(() => {
    if (!isLoading) {
      setLoadingHintIndex(0);
      setLoadingElapsedMs(0);
      return undefined;
    }

    const startedAt = Date.now();
    const tickId = window.setInterval(() => {
      setLoadingElapsedMs(Date.now() - startedAt);
    }, 500);
    const hintId = window.setInterval(() => {
      setLoadingHintIndex((currentIndex) => currentIndex + 1);
    }, 2800);

    return () => {
      window.clearInterval(tickId);
      window.clearInterval(hintId);
    };
  }, [isLoading]);

  function handleHistorySelect(item) {
    setActiveTab("query");
    setSelectedHistory(item);
    setQuestion(item.title);
    setBackend(item.backend.toLowerCase());
    setQueryResponse(null);
    setComparisonResponses([]);
    setComparisonSchemaSource("");
    setErrorMessage("");
    setErrorTrace(null);
    setLoadingTrace({ steps: [], node_count: 0, total_count: 0, completed_count: 0 });
    setResultMode("single");
  }

  function handleConnectionValueChange(fieldKey, value) {
    setConnectionDrafts((currentDrafts) => ({
      ...currentDrafts,
      [backend]: {
        ...currentDrafts[backend],
        [fieldKey]: value,
      },
    }));
    setConnectionFeedback((currentFeedback) => ({
      ...currentFeedback,
      [backend]: null,
    }));
  }

  async function handleTestConnection() {
    const connection = buildConnectionPayload(backend, connectionDrafts);

    if (!connection) {
      setConnectionFeedback((currentFeedback) => ({
        ...currentFeedback,
        [backend]: {
          kind: "error",
          message: getMissingConnectionMessage(backend),
        },
      }));
      return;
    }

    setIsTestingConnection(true);
    setConnectionFeedback((currentFeedback) => ({
      ...currentFeedback,
      [backend]: {
        kind: "info",
        message: "Testing the live database link...",
      },
    }));

    try {
      const response = await testDatabaseConnection(backend, connection);
      const entityPreview =
        response.entity_names.length > 0 ? ` ${response.entity_names.join(", ")}` : "";

      setConnectionFeedback((currentFeedback) => ({
        ...currentFeedback,
        [backend]: {
          kind: "success",
          message: `${response.message}${entityPreview}`,
        },
      }));
    } catch (error) {
      setConnectionFeedback((currentFeedback) => ({
        ...currentFeedback,
        [backend]: {
          kind: "error",
          message: error.message || "Could not connect to that database.",
        },
      }));
    } finally {
      setIsTestingConnection(false);
    }
  }

  async function handleLinkConnection() {
    const connection = buildConnectionPayload(backend, connectionDrafts);

    if (!connection) {
      setConnectionFeedback((currentFeedback) => ({
        ...currentFeedback,
        [backend]: {
          kind: "error",
          message: getMissingConnectionMessage(backend),
        },
      }));
      return;
    }

    setIsLinkingConnection(true);
    setConnectionFeedback((currentFeedback) => ({
      ...currentFeedback,
      [backend]: {
        kind: "info",
        message: "Linking the database securely through the backend...",
      },
    }));

    try {
      const response = await linkDatabaseConnection(backend, connection);

      setLinkedConnections((currentConnections) => ({
        ...currentConnections,
        [backend]: {
          connectionId: response.connection_id,
          entityCount: response.entity_count,
          entityLabel: response.entity_type,
          previewNames: response.entity_names || [],
        },
      }));

      setConnectionDrafts((currentDrafts) => clearConnectionDraft(backend, currentDrafts));
      setConnectionFeedback((currentFeedback) => ({
        ...currentFeedback,
        [backend]: {
          kind: "success",
          message:
            "Linked securely. The browser cleared the raw connection value after the backend stored a temporary session id.",
        },
      }));
    } catch (error) {
      setConnectionFeedback((currentFeedback) => ({
        ...currentFeedback,
        [backend]: {
          kind: "error",
          message: error.message || "Could not link that database.",
        },
      }));
    } finally {
      setIsLinkingConnection(false);
    }
  }

  async function handleDisconnectConnection() {
    const currentLink = linkedConnections[backend];

    if (!currentLink?.connectionId) {
      setConnectionFeedback((currentFeedback) => ({
        ...currentFeedback,
        [backend]: {
          kind: "error",
          message: "There is no linked database session to disconnect.",
        },
      }));
      return;
    }

    try {
      await disconnectDatabaseConnection(currentLink.connectionId);
    } catch (error) {
      setConnectionFeedback((currentFeedback) => ({
        ...currentFeedback,
        [backend]: {
          kind: "error",
          message: error.message || "Could not disconnect that database link.",
        },
      }));
      return;
    }

    setLinkedConnections((currentConnections) => ({
      ...currentConnections,
      [backend]: null,
    }));
    setConnectionFeedback((currentFeedback) => ({
      ...currentFeedback,
      [backend]: {
        kind: "success",
        message: "Database link removed from backend memory.",
      },
    }));
  }

  async function handleGenerate() {
    const trimmedQuestion = question.trim();

    if (!trimmedQuestion) {
      setQueryResponse(null);
      setErrorMessage("Enter a question before generating a query.");
      setErrorTrace(null);
      setLoadingTrace({ steps: [], node_count: 0, total_count: 0, completed_count: 0 });
      return;
    }

    setIsLoading(true);
    setLoadingMode("single");
    setErrorMessage("");
    setErrorTrace(null);
    setResultMode("single");
    setQueryResponse(null);
    setComparisonResponses([]);
    setComparisonSchemaSource("");
    setLoadingTrace(
      buildLiveTrace({
        workflowStarted: false,
        schemaReady: false,
        actualSteps: [],
        schemaSummary: null,
      }),
    );
    setLoadingRuntime({
      workflow: "langgraph",
      model: "",
      schemaSummary: null,
    });
    setSelectedHistory((currentItem) => {
      if (!currentItem) {
        return null;
      }

      const isSameQuestion = currentItem.title === trimmedQuestion;
      const isSameBackend = currentItem.backend.toLowerCase() === backend;

      return isSameQuestion && isSameBackend ? currentItem : null;
    });

    try {
      const connectionId = linkedConnections[backend]?.connectionId || null;
      const nextResponse = await requestQueryStream(trimmedQuestion, backend, connectionId, {
        onWorkflowStarted: (eventPayload) => {
          setLoadingRuntime((currentState) => ({
            ...currentState,
            workflow: eventPayload.workflow || currentState.workflow,
            model: eventPayload.model || currentState.model,
          }));
          setLoadingTrace((currentTrace) =>
            evolveLiveTrace(currentTrace, {
              workflowStarted: true,
            }),
          );
        },
        onSchemaReady: (eventPayload) => {
          setLoadingRuntime((currentState) => ({
            ...currentState,
            schemaSummary: eventPayload.schema_summary || null,
          }));
          setLoadingTrace((currentTrace) =>
            evolveLiveTrace(currentTrace, {
              workflowStarted: true,
              schemaReady: true,
              schemaSummary: eventPayload.schema_summary || null,
            }),
          );
        },
        onStep: (eventPayload) => {
          setLoadingTrace((currentTrace) =>
            evolveLiveTrace(currentTrace, {
              workflowStarted: true,
              schemaReady: true,
              actualSteps: eventPayload.trace?.steps || [],
            }),
          );
          setLoadingRuntime((currentState) => ({
            ...currentState,
            workflow: eventPayload.workflow || currentState.workflow,
            model: eventPayload.model || currentState.model,
          }));
        },
      });
      setQueryResponse(nextResponse);
      setHistoryItems((currentItems) => addHistoryItem(currentItems, trimmedQuestion, backend));
      setQueryRuns((currentRuns) =>
        appendRunRecord(currentRuns, createSuccessfulRun(trimmedQuestion, backend, nextResponse)),
      );
    } catch (error) {
      setQueryResponse(null);
      const nextErrorMessage = error.message || "Something went wrong while generating the query.";
      setErrorMessage(nextErrorMessage);
      setErrorTrace(error.trace || null);
      setQueryRuns((currentRuns) =>
        appendRunRecord(
          currentRuns,
          createFailedRun(trimmedQuestion, backend, nextErrorMessage, error.trace || null),
        ),
      );
    } finally {
      setIsLoading(false);
    }
  }

  async function handleCompare() {
    const trimmedQuestion = question.trim();

    if (!trimmedQuestion) {
      setComparisonResponses([]);
      setComparisonSchemaSource("");
      setErrorMessage("Enter a question before comparing all backends.");
      setErrorTrace(null);
      setResultMode("compare");
      return;
    }

    setIsLoading(true);
    setLoadingMode("compare");
    setErrorMessage("");
    setErrorTrace(null);
    setResultMode("compare");
    setQueryResponse(null);
    setComparisonResponses([]);
    setComparisonSchemaSource("");
    setLoadingTrace(
      buildLiveTrace({
        workflowStarted: false,
        schemaReady: false,
        actualSteps: [],
        schemaSummary: null,
      }),
    );
    setLoadingRuntime({
      workflow: "langgraph",
      model: "",
      schemaSummary: null,
    });
    setSelectedHistory((currentItem) => {
      if (!currentItem) {
        return null;
      }

      return currentItem.title === trimmedQuestion ? currentItem : null;
    });

    try {
      const comparisonPayload = await requestQueryComparison(
        trimmedQuestion,
        backend,
        linkedConnections,
      );
      setComparisonResponses(comparisonPayload.comparisons);
      setComparisonSchemaSource(comparisonPayload.schemaSource);
    } catch (error) {
      setComparisonResponses([]);
      setComparisonSchemaSource("");
      setErrorMessage(error.message || "Something went wrong while comparing the backends.");
      setErrorTrace(null);
    } finally {
      setIsLoading(false);
    }
  }

  async function handleRunEvalSuite() {
    setIsRunningEvalSuite(true);
    setEvalSuiteError("");

    try {
      const nextEvalRun = await requestEvalRun(backend);
      setEvalRunResponse(nextEvalRun);
    } catch (error) {
      setEvalRunResponse(null);
      setEvalSuiteError(error.message || "Could not run the built-in GenAI suite.");
    } finally {
      setIsRunningEvalSuite(false);
    }
  }

  return (
    <div className="page-shell">
      <div className="app-frame">
        <QueryHistorySidebar
          isOpen={isHistoryOpen}
          items={historyItems}
          onClose={() => setIsHistoryOpen(false)}
          onSelect={handleHistorySelect}
        />

        <main className="content-panel">
          <header className="topbar">
            <button
              type="button"
              className="icon-button"
              aria-label="Open query history"
              onClick={() => setIsHistoryOpen(true)}
            >
              <Menu className="tiny-icon" />
            </button>

            <div className="brand-lockup">
              <div className="brand-icon">
                <Database className="brand-icon-svg" />
              </div>
              <div className="brand-name">Query Bridge</div>
            </div>

            <nav className="tab-row" aria-label="Primary tabs">
              <button
                type="button"
                className={activeTab === "query" ? "tab-button active" : "tab-button"}
                onClick={() => setActiveTab("query")}
              >
                Query
              </button>
              <button
                type="button"
                className={activeTab === "evals" ? "tab-button active" : "tab-button"}
                onClick={() => setActiveTab("evals")}
              >
                Evals
              </button>
            </nav>

            <div className={buildRuntimeClassName(healthStatus)}>
              <span className="runtime-status-label">{buildRuntimeHeadline(healthStatus)}</span>
              <strong className="runtime-status-value">{buildRuntimeDetail(healthStatus)}</strong>
            </div>
          </header>

          {activeTab === "query" ? (
            <section className="panel-stack">
              <QueryBox
                question={question}
                backend={backend}
                isLoading={isLoading}
                loadingMode={loadingMode}
                connectionDrafts={connectionDrafts}
                linkedConnection={linkedConnections[backend]}
                defaultConnection={defaultConnections[backend]}
                connectionStatus={connectionFeedback[backend] || null}
                isTestingConnection={isTestingConnection}
                isLinkingConnection={isLinkingConnection}
                onGenerate={handleGenerate}
                onCompare={handleCompare}
                onQuestionChange={setQuestion}
                onBackendChange={setBackend}
                loadingState={loadingState}
                loadingTrace={loadingTrace}
                onConnectionValueChange={handleConnectionValueChange}
                onLinkConnection={handleLinkConnection}
                onDisconnectConnection={handleDisconnectConnection}
                onTestConnection={handleTestConnection}
              />

              {resultMode === "compare" ? (
                <ComparePanel
                  question={question}
                  responses={comparisonResponses}
                  schemaSource={comparisonSchemaSource}
                  errorMessage={errorMessage}
                  isLoading={isLoading}
                  loadingState={loadingState}
                  selectedHistory={selectedHistory}
                />
              ) : (
                <ResultPanel
                  response={queryResponse}
                  errorMessage={errorMessage}
                  errorTrace={errorTrace}
                  isLoading={isLoading}
                  loadingTrace={loadingTrace}
                  loadingState={loadingState}
                  selectedHistory={selectedHistory}
                />
              )}
            </section>
          ) : (
            <EvalPanel
              activeBackend={backend}
              summaryCards={evalSummary}
              rows={evalRows}
              suiteSummaryCards={evalSuiteSummary}
              suiteRows={evalSuiteRows}
              suiteResponse={evalRunResponse}
              suiteErrorMessage={evalSuiteError}
              isRunningSuite={isRunningEvalSuite}
              onRunSuite={handleRunEvalSuite}
            />
          )}
        </main>
      </div>
    </div>
  );
}

function buildRuntimeHeadline(healthStatus) {
  if (!healthStatus) {
    return "Checking runtime";
  }

  if (healthStatus?.status !== "ok") {
    return "Backend offline";
  }

  if (!healthStatus?.ollama?.available) {
    return "Ollama offline";
  }

  if (!healthStatus?.ollama?.installed) {
    return "Model missing";
  }

  return "GenAI ready";
}

function buildRuntimeDetail(healthStatus) {
  if (!healthStatus) {
    return "Connecting...";
  }

  const workflow = healthStatus?.workflow || "langgraph";
  const model = healthStatus?.model || "No model";

  if (healthStatus?.status !== "ok") {
    return "Start the FastAPI server";
  }

  if (!healthStatus?.ollama?.available) {
    return `${workflow} needs Ollama`;
  }

  if (!healthStatus?.ollama?.installed) {
    return `Pull ${model}`;
  }

  if (healthStatus?.ollama?.using_fallback) {
    const configuredModel = healthStatus?.ollama?.configured_model || "configured model";
    return `${workflow} · ${model} (fallback for ${configuredModel})`;
  }

  return `${workflow} · ${model}`;
}

function buildLoadingState({
  isLoading,
  loadingMode,
  loadingTrace,
  loadingRuntime,
  loadingHintIndex,
  loadingElapsedMs,
}) {
  if (!isLoading) {
    return null;
  }

  const stagedMessages =
    loadingMode === "compare" ? COMPARE_LOADING_STAGES : QUERY_LOADING_STAGES;
  const stagedMessage =
    stagedMessages[Math.min(loadingHintIndex, stagedMessages.length - 1)] || stagedMessages[0];
  const workflow = loadingRuntime?.workflow || "langgraph";
  const model = loadingRuntime?.model || "";
  const runtimeLabel = model ? `${workflow} · ${model}` : workflow;
  const elapsedLabel = formatDurationLabel(loadingElapsedMs);

  if (loadingMode === "compare") {
    const progressTotal = stagedMessages.length;
    const progressCompleted = Math.min(progressTotal, loadingHintIndex + 1);

    return {
      headline: stagedMessage.headline,
      detail: stagedMessage.detail,
      runtimeLabel,
      elapsedLabel,
      schemaSummary: loadingRuntime?.schemaSummary || null,
      note: buildSlowRunNote(loadingElapsedMs),
      latestStep: null,
      completedSteps: progressCompleted,
      totalSteps: progressTotal,
      progressRatio: progressCompleted / progressTotal,
      progressLabel: `${progressCompleted}/${progressTotal} stages`,
    };
  }

  const latestStep = getHighlightedLiveTraceStep(loadingTrace);
  const progress = summarizeLiveTraceProgress(loadingTrace);

  if (latestStep) {
    return {
      headline: formatWorkflowStepName(latestStep.name),
      detail: latestStep.summary || stagedMessage.detail,
      runtimeLabel,
      elapsedLabel,
      schemaSummary: loadingRuntime?.schemaSummary || null,
      note: buildSlowRunNote(loadingElapsedMs),
      latestStep,
      completedSteps: progress.completed,
      totalSteps: progress.total,
      progressRatio: progress.ratio,
      progressLabel: `${progress.completed}/${progress.total} steps`,
    };
  }

  return {
    headline: stagedMessage.headline,
    detail: stagedMessage.detail,
    runtimeLabel,
    elapsedLabel,
    schemaSummary: loadingRuntime?.schemaSummary || null,
    note: buildSlowRunNote(loadingElapsedMs),
    latestStep: null,
    completedSteps: progress.completed,
    totalSteps: progress.total,
    progressRatio: progress.ratio,
    progressLabel: `${progress.completed}/${progress.total} steps`,
  };
}

function buildSlowRunNote(loadingElapsedMs) {
  if (loadingElapsedMs >= 60000) {
    return "Still working. Ollama runs can take a minute or two, especially right after the model loads.";
  }

  if (loadingElapsedMs >= 18000) {
    return "This is taking longer than usual. The Ollama model may still be warming up.";
  }

  return "";
}

function buildRuntimeClassName(healthStatus) {
  if (!healthStatus) {
    return "runtime-status-card neutral";
  }

  if (
    healthStatus?.status === "ok" &&
    healthStatus?.ollama?.available &&
    healthStatus?.ollama?.installed
  ) {
    return "runtime-status-card ready";
  }

  return "runtime-status-card warning";
}

function formatWorkflowStepName(stepName) {
  return String(stepName || "")
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function formatDurationLabel(durationMs) {
  const totalSeconds = Math.max(0, Math.round(durationMs / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;

  if (minutes === 0) {
    return `${seconds}s`;
  }

  return `${minutes}m ${String(seconds).padStart(2, "0")}s`;
}

function getMissingConnectionMessage(backend) {
  if (backend === "sqlite") {
    return "Enter a SQLite file path before testing the connection.";
  }

  if (backend === "postgresql") {
    return "Enter a PostgreSQL connection URL before testing the connection.";
  }

  return "Enter a MongoDB connection URL before testing the connection.";
}

function addHistoryItem(items, question, backend) {
  const nextItem = {
    id: Date.now(),
    title: question,
    timestamp: formatTimestamp(new Date()),
    backend: formatBackendLabel(backend),
  };

  const remainingItems = items.filter(
    (item) => !(item.title === question && item.backend.toLowerCase() === backend),
  );

  return [nextItem, ...remainingItems].slice(0, 8);
}

function buildDefaultConnections(healthStatus) {
  const demoDatabase = healthStatus?.demo_database;

  if (!demoDatabase?.available || demoDatabase?.backend !== "sqlite") {
    return EMPTY_CONNECTIONS;
  }

  return {
    ...EMPTY_CONNECTIONS,
    sqlite: {
      connectionId: null,
      entityCount: demoDatabase.entity_count || 0,
      entityLabel: demoDatabase.entity_type || "tables",
      previewNames: demoDatabase.entity_names || [],
      autoConnected: Boolean(demoDatabase.auto_connected),
      label: demoDatabase.label || "Bundled demo database",
      message:
        demoDatabase.message ||
        "Bundled demo database ready. SQLite queries work immediately without linking.",
    },
  };
}

function buildLiveTrace({
  workflowStarted,
  schemaReady,
  actualSteps,
  schemaSummary,
}) {
  const steps = [];

  steps.push({
    name: "workflow_started",
    status: workflowStarted ? "success" : "active",
    summary: LIVE_TRACE_PRELUDE[0].summary,
    details: {},
    synthetic: true,
  });

  steps.push({
    name: "schema_ready",
    status: schemaReady ? "success" : workflowStarted ? "active" : "pending",
    summary: LIVE_TRACE_PRELUDE[1].summary,
    details: schemaSummary ? { schema_summary: schemaSummary } : {},
    synthetic: true,
  });

  const realSteps = Array.isArray(actualSteps) ? actualSteps : [];
  steps.push(...realSteps);

  const pendingSteps = buildPendingLiveSteps(realSteps, schemaReady);
  steps.push(...pendingSteps);

  const completed = steps.filter((step) =>
    ["success", "repaired", "failed"].includes(step.status),
  ).length;

  return {
    steps,
    node_count: realSteps.length,
    total_count: steps.length,
    completed_count: completed,
  };
}

function evolveLiveTrace(currentTrace, patch = {}) {
  return buildLiveTrace({
    workflowStarted:
      patch.workflowStarted ?? hasTraceStepWithState(currentTrace, "workflow_started", ["success", "active"]),
    schemaReady:
      patch.schemaReady ?? hasTraceStepWithState(currentTrace, "schema_ready", ["success"]),
    actualSteps: patch.actualSteps ?? extractActualTraceSteps(currentTrace),
    schemaSummary: patch.schemaSummary ?? getTraceSchemaSummary(currentTrace),
  });
}

function buildPendingLiveSteps(actualSteps, schemaReady) {
  if (!schemaReady) {
    return [];
  }

  const remainingNames = getRemainingLiveStepNames(actualSteps);
  return remainingNames.map((name, index) => ({
    name,
    status: index === 0 ? "active" : "pending",
    summary: LIVE_TRACE_STEP_LIBRARY[name] || "Waiting for this workflow stage.",
    details: { preview: "Waiting for this step to start." },
    synthetic: true,
  }));
}

function getRemainingLiveStepNames(actualSteps) {
  if (!Array.isArray(actualSteps) || actualSteps.length === 0) {
    return [...DEFAULT_LIVE_FLOW];
  }

  const latestStep = actualSteps[actualSteps.length - 1];
  const latestName = latestStep?.name;

  if (latestName === "repair_query") {
    return ["compile_query", "check_safety", "check_validation", "execute_query", "explain_result"];
  }

  if (latestName === "check_validation") {
    if (latestStep?.status === "failed" || latestStep?.details?.passed === false) {
      return ["repair_query", "compile_query", "check_safety", "check_validation", "execute_query", "explain_result"];
    }

    return ["execute_query", "explain_result"];
  }

  if (latestName === "execute_query") {
    if (latestStep?.status === "failed") {
      return ["repair_query", "compile_query", "check_safety", "check_validation", "execute_query", "explain_result"];
    }

    return ["explain_result"];
  }

  if (latestName === "explain_result") {
    return [];
  }

  const currentIndex = DEFAULT_LIVE_FLOW.indexOf(latestName);
  if (currentIndex === -1) {
    return [];
  }

  return DEFAULT_LIVE_FLOW.slice(currentIndex + 1);
}

function extractActualTraceSteps(trace) {
  return (trace?.steps || []).filter((step) => !step.synthetic);
}

function getTraceSchemaSummary(trace) {
  const schemaStep = (trace?.steps || []).find((step) => step.name === "schema_ready");
  return schemaStep?.details?.schema_summary || null;
}

function hasTraceStepWithState(trace, stepName, statuses) {
  return (trace?.steps || []).some(
    (step) => step.name === stepName && statuses.includes(step.status),
  );
}

function getHighlightedLiveTraceStep(trace) {
  const steps = trace?.steps || [];
  const activeStep = steps.find((step) => step.status === "active");
  if (activeStep) {
    return activeStep;
  }

  const completedSteps = steps.filter((step) =>
    ["failed", "repaired", "success"].includes(step.status),
  );
  return completedSteps.length > 0 ? completedSteps[completedSteps.length - 1] : null;
}

function summarizeLiveTraceProgress(trace) {
  const steps = trace?.steps || [];
  const total = steps.length || 1;
  const completed = steps.filter((step) =>
    ["success", "repaired", "failed"].includes(step.status),
  ).length;

  return {
    completed,
    total,
    ratio: Math.min(1, completed / total),
  };
}

function formatTimestamp(date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  const hours = String(date.getHours()).padStart(2, "0");
  const minutes = String(date.getMinutes()).padStart(2, "0");

  return `${year}-${month}-${day} ${hours}:${minutes}`;
}
