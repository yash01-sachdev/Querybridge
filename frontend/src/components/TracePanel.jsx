import { Clock3, GitBranch, LoaderCircle, Wrench } from "lucide-react";

export default function TracePanel({ trace, title = "Workflow Trace", subtitle = "" }) {
  const steps = trace?.steps || [];
  const completedCount =
    trace?.completed_count ||
    steps.filter((step) => ["success", "repaired", "failed"].includes(step.status)).length;
  const totalCount = trace?.total_count || steps.length;

  if (steps.length === 0) {
    return null;
  }

  return (
    <section className="panel-card">
      <div className="panel-header-row">
        <div>
          <h2 className="section-title no-margin">
            <GitBranch className="section-icon accent" />
            {title}
          </h2>
          {subtitle ? <p className="section-subtitle tighter">{subtitle}</p> : null}
        </div>
        <div className="trace-panel-meta">
          <span className="row-count-badge">{trace?.node_count || steps.length} backend steps</span>
          <span className="row-count-badge">
            {completedCount}/{totalCount} visible stages
          </span>
        </div>
      </div>

      <div className="trace-step-list">
        {steps.map((step, index) => (
          <details
            key={`${step.name}-${index}`}
            className={`trace-step-card ${step.status || "neutral"} ${step.synthetic ? "synthetic" : ""}`}
            open={step.status === "active" || step.status === "failed" || step.status === "repaired"}
          >
            <summary className="trace-step-summary">
              <div className="trace-step-main">
                <span className="trace-step-index">{index + 1}</span>
                <div>
                  <div className="trace-step-heading-row">
                    <strong className="trace-step-name">{formatStepName(step.name)}</strong>
                    {step.synthetic ? (
                      <span className="trace-step-kind">
                        <Clock3 className="tiny-icon" />
                        Live
                      </span>
                    ) : null}
                  </div>
                  <p className="trace-step-copy">{step.summary}</p>
                </div>
              </div>
              <span className={`trace-step-status ${step.status || "neutral"}`}>
                {step.status === "repaired" ? <Wrench className="tiny-icon" /> : null}
                {step.status === "active" ? <LoaderCircle className="tiny-icon spin-icon" /> : null}
                {step.status || "success"}
              </span>
            </summary>
            {hasRenderableDetails(step.details) ? (
              <pre className="code-block trace-code-block">
                {JSON.stringify(step.details || {}, null, 2)}
              </pre>
            ) : (
              <div className="trace-step-empty">Waiting for this step to produce runtime details.</div>
            )}
          </details>
        ))}
      </div>
    </section>
  );
}

function formatStepName(stepName) {
  return String(stepName || "")
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function hasRenderableDetails(details) {
  return details && typeof details === "object" && Object.keys(details).length > 0;
}
