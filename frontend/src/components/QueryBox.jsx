import { Database, GitCompareArrows, Sparkles } from "lucide-react";

import { BACKEND_OPTIONS } from "../backendOptions";
import ConnectionPanel from "./ConnectionPanel";

export default function QueryBox({
  question,
  backend,
  isLoading,
  loadingMode,
  loadingState,
  loadingTrace,
  connectionDrafts,
  linkedConnection,
  defaultConnection,
  connectionStatus,
  isTestingConnection,
  isLinkingConnection,
  onGenerate,
  onCompare,
  onQuestionChange,
  onBackendChange,
  onConnectionValueChange,
  onLinkConnection,
  onDisconnectConnection,
  onTestConnection,
}) {
  return (
    <section className="panel-card">
      <h2 className="section-title">
        <Sparkles className="section-icon accent" />
        Describe your query
      </h2>
      <textarea
        className="query-textarea"
        value={question}
        onChange={(event) => onQuestionChange(event.target.value)}
        placeholder="e.g., show user emails or how many users..."
      />

      <div className="query-controls">
        <div className="control-group">
          <label className="control-label" htmlFor="backend-select">
            <Database className="tiny-icon" />
            Database Type
          </label>
          <select
            id="backend-select"
            className="query-select"
            value={backend}
            onChange={(event) => onBackendChange(event.target.value)}
          >
            {BACKEND_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>

          <ConnectionPanel
            backend={backend}
            drafts={connectionDrafts}
            linkedConnection={linkedConnection}
            defaultConnection={defaultConnection}
            status={connectionStatus}
            isTesting={isTestingConnection}
            isLinking={isLinkingConnection}
            onValueChange={onConnectionValueChange}
            onLink={onLinkConnection}
            onDisconnect={onDisconnectConnection}
            onTest={onTestConnection}
          />
        </div>

        <div className="query-action-row">
          <button
            type="button"
            className="compare-button"
            onClick={onCompare}
            disabled={isLoading}
          >
            <GitCompareArrows className="tiny-icon" />
            {isLoading && loadingMode === "compare" ? "Comparing..." : "Compare All Backends"}
          </button>

          <button
            type="button"
            className="primary-button"
            onClick={onGenerate}
            disabled={isLoading}
          >
            <Sparkles className="tiny-icon" />
            {isLoading && loadingMode === "single" ? "Running..." : "Run Query"}
          </button>
        </div>
      </div>

      {isLoading && loadingMode === "single" ? (
        <div className="query-live-strip">
          <div className="query-live-copy">
            <span className="query-live-label">Live workflow</span>
            <strong>{loadingState?.headline || "Starting query run"}</strong>
            <p>{loadingState?.detail || "Preparing the runtime and reading the selected schema."}</p>
          </div>
          <div className="query-live-metrics">
            <span>{loadingState?.runtimeLabel || "langgraph"}</span>
            <span>{loadingState?.elapsedLabel || "0s"}</span>
            <span>
              {loadingState?.progressLabel ||
                `${loadingTrace?.completed_count || 0}/${loadingTrace?.total_count || 0} steps`}
            </span>
          </div>
        </div>
      ) : null}
    </section>
  );
}
