import { useState } from "react";
import { AlertTriangle, Check, Copy, GitCompareArrows, Layers3 } from "lucide-react";

import { DatabaseBadge } from "./Badges";

export default function ComparePanel({
  question,
  responses,
  schemaSource,
  errorMessage,
  isLoading,
  loadingState,
  selectedHistory,
}) {
  const [copiedBackend, setCopiedBackend] = useState("");
  const successCount = responses.filter((item) => item.ok).length;

  function copyQuery(backend, query) {
    if (!query) {
      return;
    }

    navigator.clipboard.writeText(query);
    setCopiedBackend(backend);
    window.setTimeout(() => setCopiedBackend(""), 1800);
  }

  if (errorMessage) {
    return (
      <section className="panel-card error-card">
        <h2 className="section-title no-margin">Comparison Failed</h2>
        <p className="error-copy">{errorMessage}</p>
      </section>
    );
  }

  if (isLoading) {
    return (
      <section className="panel-card empty-card">
        <div className="empty-state">
          <GitCompareArrows className="empty-state-icon" />
          <h2 className="section-title no-margin">
            {loadingState?.headline || "Comparing all backends"}
          </h2>
          <p className="section-subtitle centered">
            {loadingState?.detail ||
              "Running the same question on SQLite, PostgreSQL, and MongoDB at the same time."}
          </p>
          <p className="section-subtitle centered">
            {loadingState?.runtimeLabel
              ? `${loadingState.runtimeLabel} · ${loadingState.elapsedLabel || "0s"} elapsed`
              : loadingState?.elapsedLabel
                ? `${loadingState.elapsedLabel} elapsed`
                : ""}
          </p>
          {loadingState?.note ? <p className="section-subtitle centered">{loadingState.note}</p> : null}
        </div>
      </section>
    );
  }

  if (!responses.length) {
    return (
      <section className="panel-card empty-card">
        <div className="empty-state">
          <GitCompareArrows className="empty-state-icon" />
          <h2 className="section-title no-margin">No comparison run yet</h2>
          <p className="section-subtitle centered">
            Click Compare All Backends to see three compiled queries side by side.
          </p>
        </div>
      </section>
    );
  }

  return (
    <div className="panel-stack">
      <section className="panel-card comparison-hero-card">
        <div className="comparison-hero-copy">
          <h2 className="section-title no-margin">
            <GitCompareArrows className="section-icon accent" />
            Cross-backend comparison
          </h2>
          <p className="section-subtitle">
            Same English question, three backend compilers, side-by-side output for demos and
            interviews.
          </p>
        </div>

        <div className="comparison-pill-row">
          <div className="comparison-pill">
            <span>Question</span>
            <strong>{question.trim()}</strong>
          </div>
          <div className="comparison-pill">
            <span>Successful runs</span>
            <strong>
              {successCount} / {responses.length}
            </strong>
          </div>
          {selectedHistory ? (
            <div className="comparison-pill">
              <span>Loaded example</span>
              <strong>{selectedHistory.title}</strong>
            </div>
          ) : null}
        </div>
      </section>

      <section className="comparison-grid">
        {responses.map((item) => {
          const compiledQuery = item.compiledQuery || "";
          const queryPlan = item.queryPlan || null;
          const comparedEntity = getComparedEntity(queryPlan);
          const traceSteps = item.trace?.node_count || item.trace?.steps?.length || 0;
          const schemaSourceLabel = item.schemaSource || "built-in learning schema";

          return (
            <article
              key={item.backend}
              className={
                item.ok
                  ? "compare-card panel-card"
                  : "compare-card panel-card compare-card-error"
              }
            >
              <div className="compare-card-top">
                <DatabaseBadge backend={item.label} />
                <span className={item.ok ? "compare-status success" : "compare-status danger"}>
                  {item.ok ? "Preview Ready" : "Unavailable"}
                </span>
              </div>

              {item.ok ? (
                <>
                  {item.workflow || item.model ? (
                    <div className="genai-runtime-banner">
                      <span className="runtime-pill">{item.workflow || "workflow"}</span>
                      <span className="runtime-copy">
                        {item.model ? `Model: ${item.model}` : "Local model-backed preview"}
                      </span>
                    </div>
                  ) : null}

                  <p className="section-subtitle tighter">Schema source: {schemaSourceLabel}</p>

                  <div className="panel-header-row compare-card-header">
                    <h3 className="compare-card-title">Compiled Query</h3>
                    <button
                      type="button"
                      className="ghost-button compact-button"
                      onClick={() => copyQuery(item.backend, compiledQuery)}
                    >
                      {copiedBackend === item.backend ? (
                        <Check className="tiny-icon" />
                      ) : (
                        <Copy className="tiny-icon" />
                      )}
                      {copiedBackend === item.backend ? "Copied" : "Copy"}
                    </button>
                  </div>

                  <pre className="code-block compare-code-block">{compiledQuery}</pre>

                  <div className="compare-stat-row">
                    <div className="compare-stat">
                      <span>Operation</span>
                      <strong>{queryPlan?.operation || "select"}</strong>
                    </div>
                    <div className="compare-stat">
                      <span>Entity</span>
                      <strong>{comparedEntity}</strong>
                    </div>
                    <div className="compare-stat">
                      <span>Trace Steps</span>
                      <strong>{traceSteps}</strong>
                    </div>
                  </div>

                  <div className="compare-preview-block">
                    <h3 className="compare-card-title">
                      <Layers3 className="tiny-icon" />
                      Query Plan
                    </h3>
                    <pre className="compare-preview-code">{JSON.stringify(queryPlan, null, 2)}</pre>
                  </div>
                </>
              ) : (
                <div className="compare-error-block">
                  <div className="compare-error-icon">
                    <AlertTriangle className="section-icon" />
                  </div>
                  <h3 className="compare-card-title">Backend unavailable for this question</h3>
                  <p className="error-copy compare-error-copy">{item.errorMessage}</p>
                  <p className="section-subtitle tighter">Schema source: {schemaSourceLabel}</p>
                  {traceSteps ? (
                    <p className="section-subtitle tighter">{traceSteps} trace steps completed before failure.</p>
                  ) : null}
                </div>
              )}

            </article>
          );
        })}
      </section>

      <p className="comparison-note">
        Compare mode is using {schemaSource || "the built-in learning schema"} for the selected
        backend summary. Each card switches to its own linked live database as soon as you link
        that backend.
      </p>
    </div>
  );
}

function getComparedEntity(queryPlan) {
  if (!queryPlan) {
    return "Unknown";
  }

  if (Array.isArray(queryPlan.tables) && queryPlan.tables.length > 0) {
    return queryPlan.tables.join(", ");
  }

  if (typeof queryPlan.collection === "string" && queryPlan.collection) {
    return queryPlan.collection;
  }

  return "Unknown";
}
