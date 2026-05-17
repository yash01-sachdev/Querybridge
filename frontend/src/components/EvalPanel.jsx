import { Activity, CheckCircle2, Clock3, Shield, Sparkles, Wrench, Zap } from "lucide-react";

import { DatabaseBadge, StatusBadge } from "./Badges";

const metricIcons = {
  runs: <Activity className="metric-icon info" />,
  success: <CheckCircle2 className="metric-icon success" />,
  safety: <Shield className="metric-icon success" />,
  validation: <Sparkles className="metric-icon info" />,
  repair: <Wrench className="metric-icon warning" />,
  latency: <Zap className="metric-icon info" />,
};

export default function EvalPanel({
  activeBackend,
  summaryCards,
  rows,
  suiteSummaryCards,
  suiteRows,
  suiteResponse,
  suiteErrorMessage,
  isRunningSuite,
  onRunSuite,
}) {
  return (
    <section className="panel-stack">
      <div className="eval-header">
        <div>
          <h2 className="section-title no-margin">Evaluation Dashboard</h2>
          <p className="section-subtitle">
            Live session metrics from the queries you run in this app
          </p>
        </div>
        <button type="button" className="secondary-button" disabled>
          <Clock3 className="tiny-icon" />
          Live Session
        </button>
      </div>

      <div className="metric-grid">
        {summaryCards.map((card) => (
          <article key={card.label} className="metric-card">
            <div className="metric-header">
              <div className="metric-label">{card.label}</div>
              {metricIcons[card.iconKey]}
            </div>
            <div className="metric-value">{card.value}</div>
            {card.progress ? (
              <div className="metric-progress-block">
                <div className="metric-progress-track">
                  <div
                    className={`metric-progress-fill ${card.progress.tone}`}
                    style={{ width: card.progress.width }}
                  />
                </div>
                <span className={`metric-progress-label ${card.progress.tone}`}>
                  {card.progress.label}
                </span>
              </div>
            ) : (
              <div className="metric-sublabel">{card.sublabel}</div>
            )}
          </article>
        ))}
      </div>

      <section className="panel-card">
        <h2 className="section-title no-margin">Recent Query Runs</h2>
        <p className="section-subtitle tighter">
          Safety, validation, repair, and latency details from your latest live requests
        </p>
        {rows.length === 0 ? (
          <div className="empty-state eval-empty-state">
            <Sparkles className="empty-state-icon" />
            <p className="section-subtitle centered">
              Run a few queries in the Query tab and the live evaluation table will fill in here.
            </p>
          </div>
        ) : (
          <div className="result-table-wrapper">
            <table className="result-table eval-table">
              <thead>
                <tr>
                  <th>Question</th>
                  <th>Backend</th>
                  <th>Workflow</th>
                  <th>Model</th>
                  <th>Compiled Query</th>
                  <th>Rows</th>
                  <th>Safety</th>
                  <th>Validation</th>
                  <th>Repair</th>
                  <th>Trace</th>
                  <th>Latency</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.id}>
                    <td title={row.message}>{row.question}</td>
                    <td>
                      <DatabaseBadge backend={row.backend} />
                    </td>
                    <td>{row.workflow}</td>
                    <td title={row.model}>{row.model}</td>
                    <td className="mono-cell" title={row.compiledQuery}>
                      {row.compiledQuery}
                    </td>
                    <td>{row.rowCount}</td>
                    <td>
                      <StatusBadge status={row.safetyStatus} />
                    </td>
                    <td>
                      <StatusBadge status={row.validationStatus} />
                    </td>
                    <td>{row.repairStatus}</td>
                    <td>{row.traceSteps}</td>
                    <td>{row.latencyLabel}</td>
                    <td>
                      <StatusBadge status={row.status} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="panel-card">
        <div className="panel-header-row">
          <div>
            <h2 className="section-title no-margin">Built-in GenAI Suite</h2>
            <p className="section-subtitle tighter">
              Runs the real LangGraph workflow against a fixed demo dataset for{" "}
              <strong>{formatBackendName(activeBackend)}</strong>.
            </p>
          </div>
          <button
            type="button"
            className="secondary-button"
            onClick={onRunSuite}
            disabled={isRunningSuite}
          >
            <Sparkles className="tiny-icon" />
            {isRunningSuite ? "Running Suite..." : `Run ${formatBackendName(activeBackend)} Suite`}
          </button>
        </div>

        {suiteErrorMessage ? <p className="error-copy">{suiteErrorMessage}</p> : null}

        {suiteSummaryCards.length > 0 ? (
          <div className="metric-grid suite-metric-grid">
            {suiteSummaryCards.map((card) => (
              <article key={card.label} className="metric-card">
                <div className="metric-header">
                  <div className="metric-label">{card.label}</div>
                  {metricIcons[card.iconKey]}
                </div>
                <div className="metric-value">{card.value}</div>
                {card.progress ? (
                  <div className="metric-progress-block">
                    <div className="metric-progress-track">
                      <div
                        className={`metric-progress-fill ${card.progress.tone}`}
                        style={{ width: card.progress.width }}
                      />
                    </div>
                    <span className={`metric-progress-label ${card.progress.tone}`}>
                      {card.progress.label}
                    </span>
                  </div>
                ) : (
                  <div className="metric-sublabel">{card.sublabel}</div>
                )}
              </article>
            ))}
          </div>
        ) : (
          <div className="empty-state eval-empty-state">
            <Sparkles className="empty-state-icon" />
            <p className="section-subtitle centered">
              Run the built-in suite to measure how the current local model handles the demo prompts.
            </p>
          </div>
        )}

        {suiteResponse ? (
          <div className="result-table-wrapper">
            <table className="result-table eval-table">
              <thead>
                <tr>
                  <th>Question</th>
                  <th>Expected</th>
                  <th>Compiled Query / Message</th>
                  <th>Rows</th>
                  <th>Trace</th>
                  <th>Latency</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {suiteRows.map((row) => (
                  <tr key={row.id}>
                    <td title={row.message}>{row.question}</td>
                    <td>{row.expected}</td>
                    <td className="mono-cell" title={row.compiledQuery}>
                      {row.compiledQuery}
                    </td>
                    <td>{row.rowCount}</td>
                    <td>{row.traceSteps}</td>
                    <td>{row.latencyLabel}</td>
                    <td>
                      <StatusBadge status={row.status} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
      </section>
    </section>
  );
}

function formatBackendName(backend) {
  if (backend === "sqlite") {
    return "SQLite";
  }

  if (backend === "postgresql") {
    return "PostgreSQL";
  }

  if (backend === "mongodb") {
    return "MongoDB";
  }

  return "Backend";
}
