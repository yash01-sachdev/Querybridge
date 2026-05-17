import { useState } from "react";
import {
  Check,
  CheckCircle2,
  Copy,
  Database,
  Shield,
  Sparkles,
  Wrench,
  Zap,
} from "lucide-react";

import { DatabaseBadge, StatusBadge } from "./Badges";
import TracePanel from "./TracePanel";

export default function ResultPanel({
  response,
  errorMessage,
  errorTrace,
  isLoading,
  loadingTrace,
  loadingState,
  selectedHistory,
}) {
  const [copied, setCopied] = useState(false);
  const compiledQuery = response?.compiled_query || "";
  const rows = response?.result?.rows || [];
  const rowCount = response?.result?.row_count || 0;
  const columns = response?.result?.columns || [];
  const queryPlan = response?.query_plan || null;
  const relevantNames = getRelevantNames(response?.relevant_schema);
  const explanation = response?.explanation || response?.message || "Query finished successfully.";
  const workflow = response?.workflow || "";
  const model = response?.model || "";
  const trace = response?.trace || null;
  const liveDetailRows = summarizeStepDetails(loadingState?.latestStep?.details || {});

  function copyQuery() {
    if (!compiledQuery) {
      return;
    }

    navigator.clipboard.writeText(compiledQuery);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1800);
  }

  if (errorMessage) {
    return (
      <div className="panel-stack">
        <section className="panel-card error-card">
          <h2 className="section-title no-margin">Request Failed</h2>
          <p className="error-copy">{errorMessage}</p>
        </section>
        <TracePanel
          trace={errorTrace}
          title="Failure Trace"
          subtitle="These are the LangGraph steps that completed before the request was rejected."
        />
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="panel-stack">
        <section className="panel-card live-workflow-card">
          <div className="live-workflow-hero">
            <div className="live-workflow-copy">
              <span className="live-workflow-label">Live workflow</span>
              <h2 className="section-title no-margin">
                <Sparkles className="section-icon accent" />
                {loadingState?.headline || "Running query"}
              </h2>
              <p className="section-subtitle">
                {loadingState?.detail ||
                  "The LangGraph workflow is reading the schema, planning the query, and fetching rows."}
              </p>
            </div>
            <div className="live-workflow-progress">
              <span>{loadingState?.progressLabel || "0/0 steps"}</span>
              <strong>{loadingState?.elapsedLabel || "0s"}</strong>
            </div>
          </div>

          <div className="live-progress-track" aria-hidden="true">
            <div
              className="live-progress-fill"
              style={{ width: `${Math.max(8, Math.round((loadingState?.progressRatio || 0) * 100))}%` }}
            />
          </div>

          <div className="summary-grid">
            <div className="mini-note compact">
              <span>Runtime</span>
              <strong>{loadingState?.runtimeLabel || "langgraph"}</strong>
            </div>
            <div className="mini-note compact">
              <span>Elapsed</span>
              <strong>{loadingState?.elapsedLabel || "0s"}</strong>
            </div>
            <div className="mini-note compact">
              <span>Trace Steps</span>
              <strong>{loadingTrace?.node_count || loadingTrace?.steps?.length || 0}</strong>
            </div>
            {loadingState?.schemaSummary?.entity_count ? (
              <div className="mini-note compact">
                <span>Schema Scope</span>
                <strong>
                  {loadingState.schemaSummary.entity_count} {loadingState.schemaSummary.entity_type}
                </strong>
              </div>
            ) : null}
          </div>

          {liveDetailRows.length > 0 ? (
            <div className="live-detail-grid">
              {liveDetailRows.map((detail) => (
                <div key={detail.label} className="mini-note compact live-detail-note">
                  <span>{detail.label}</span>
                  <strong>{detail.value}</strong>
                </div>
              ))}
            </div>
          ) : null}

          {loadingState?.note ? <p className="section-subtitle centered">{loadingState.note}</p> : null}
        </section>

        <TracePanel
          trace={loadingTrace}
          title="Live Workflow"
          subtitle="These steps appear immediately, then fill in as each backend stage completes."
        />
      </div>
    );
  }

  if (!compiledQuery) {
    return (
      <section className="panel-card empty-card">
        <div className="empty-state">
          <Sparkles className="empty-state-icon" />
          <h2 className="section-title no-margin">No query run yet</h2>
          <p className="section-subtitle centered">
            Enter a prompt above and click Run Query to see the compiled query and live results.
          </p>
        </div>
      </section>
    );
  }

  return (
    <div className="panel-stack">
      <section className="panel-card">
        {workflow || model ? (
          <div className="genai-runtime-banner">
            <span className="runtime-pill">{workflow || "workflow"}</span>
            <span className="runtime-copy">
              {model ? `Model: ${model}` : "Local model-backed run"}
            </span>
          </div>
        ) : null}
        <div className="panel-header-row">
          <h2 className="section-title no-margin">
            <CheckCircle2 className="section-icon success" />
            Generated Query
          </h2>

          <button type="button" className="ghost-button" onClick={copyQuery}>
            {copied ? <Check className="tiny-icon" /> : <Copy className="tiny-icon" />}
            {copied ? "Copied!" : "Copy"}
          </button>
        </div>
        <pre className="code-block">{compiledQuery}</pre>
      </section>

      <section className="panel-card">
        <div className="panel-header-row">
          <h2 className="section-title no-margin">
            <Sparkles className="section-icon accent" />
            Query Plan
          </h2>
          <span className="row-count-badge">{queryPlan?.operation || "select"}</span>
        </div>
        <pre className="code-block">{JSON.stringify(queryPlan, null, 2)}</pre>
      </section>

      <TracePanel
        trace={trace}
        subtitle="Open any step to see what the workflow selected, validated, repaired, or executed."
      />

      <section className="panel-card">
        <div className="panel-header-row">
          <h2 className="section-title no-margin">
            <Database className="section-icon" />
            Results
          </h2>
          <span className="row-count-badge">{rowCount} rows</span>
        </div>
        {rows.length > 0 ? (
          <div className="result-table-wrapper">
            <table className="result-table">
              <thead>
                <tr>
                  {columns.map((column) => (
                    <th key={column}>{column}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row, rowIndex) => (
                  <tr key={`row-${rowIndex}`}>
                    {columns.map((column) => (
                      <td key={`${rowIndex}-${column}`}>{formatCellValue(row[column])}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="section-subtitle tighter">The query ran successfully, but no rows matched.</p>
        )}
      </section>

      <section className="panel-card">
        <h2 className="section-title">
          <Sparkles className="section-icon accent" />
          Run Summary
        </h2>
        <p className="body-copy">{explanation}</p>
        <div className="summary-grid">
          <div className="mini-note compact">
            <span>Backend</span>
            <strong>
              <DatabaseBadge backend={response?.backend || "sqlite"} />
            </strong>
          </div>
          <div className="mini-note compact">
            <span>Workflow</span>
            <strong>{workflow || "—"}</strong>
          </div>
          <div className="mini-note compact">
            <span>Model</span>
            <strong>{model || "—"}</strong>
          </div>
          <div className="mini-note compact">
            <span>Rows Returned</span>
            <strong>{rowCount}</strong>
          </div>
          <div className="mini-note compact">
            <span>Safety Check</span>
            <strong>
              <StatusBadge status={formatCheckStatus(response?.safety_check?.passed)} />
            </strong>
          </div>
          <div className="mini-note compact">
            <span>Validation</span>
            <strong>
              <StatusBadge status={formatCheckStatus(response?.validation_check?.passed)} />
            </strong>
          </div>
          <div className="mini-note compact">
            <span>Repair Flow</span>
            <strong>{response?.repaired ? `${response?.repair_attempts || 0} attempt(s)` : "Not needed"}</strong>
          </div>
          <div className="mini-note compact">
            <span>Latency</span>
            <strong>
              {typeof response?.execution_time_ms === "number"
                ? `${response.execution_time_ms}ms`
                : "—"}
            </strong>
          </div>
          <div className="mini-note compact">
            <span>Relevant Scope</span>
            <strong>{relevantNames || "Not available"}</strong>
          </div>
          <div className="mini-note compact">
            <span>Columns</span>
            <strong>{columns.length}</strong>
          </div>
        </div>
        <div className="meta-note-row">
          <div className="mini-note inline-note">
            <span>
              <Shield className="tiny-icon" />
              Safety
            </span>
            <strong>{response?.safety_check?.reason || "Passed read-only safety rules."}</strong>
          </div>
          <div className="mini-note inline-note">
            <span>
              <CheckCircle2 className="tiny-icon" />
              Validation
            </span>
            <strong>{response?.validation_check?.reason || "Matched the selected schema."}</strong>
          </div>
          <div className="mini-note inline-note">
            <span>
              <Zap className="tiny-icon" />
              Status
            </span>
            <strong>{response?.message || "Query finished successfully."}</strong>
          </div>
          {response?.repaired ? (
            <div className="mini-note inline-note">
              <span>
                <Wrench className="tiny-icon" />
                Repair
              </span>
              <strong>{`Recovered after ${response.repair_attempts} attempt(s).`}</strong>
            </div>
          ) : null}
        </div>
        {selectedHistory ? (
          <div className="mini-note">
            <span>Loaded example</span>
            <strong>{selectedHistory.title}</strong>
          </div>
        ) : null}
      </section>
    </div>
  );
}

function formatCellValue(value) {
  if (value === null || value === undefined) {
    return "—";
  }

  if (typeof value === "object") {
    return JSON.stringify(value);
  }

  return String(value);
}

function getRelevantNames(schema) {
  if (!schema) {
    return "";
  }

  const tableNames = Object.keys(schema.tables || {});
  if (tableNames.length > 0) {
    return tableNames.join(", ");
  }

  const collectionNames = Object.keys(schema.collections || {});
  return collectionNames.join(", ");
}

function formatCheckStatus(value) {
  if (value === true) {
    return "Pass";
  }

  if (value === false) {
    return "Fail";
  }

  return "Pending";
}

function summarizeStepDetails(details) {
  if (!details || typeof details !== "object") {
    return [];
  }

  return Object.entries(details)
    .filter(([, value]) => value !== undefined && value !== null && value !== "")
    .slice(0, 4)
    .map(([label, value]) => ({
      label: formatDetailLabel(label),
      value: formatDetailValue(value),
    }));
}

function formatDetailLabel(label) {
  return String(label)
    .replace(/_/g, " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function formatDetailValue(value) {
  if (Array.isArray(value)) {
    return value.length > 0 ? value.join(", ") : "—";
  }

  if (typeof value === "object") {
    return JSON.stringify(value);
  }

  if (typeof value === "boolean") {
    return value ? "Yes" : "No";
  }

  return String(value);
}
