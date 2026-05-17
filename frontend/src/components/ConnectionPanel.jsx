import { CheckCircle2, DatabaseZap, LockKeyhole, ShieldX, Unplug } from "lucide-react";

import { getConnectionFieldConfig } from "../connectionSettings";

export default function ConnectionPanel({
  backend,
  drafts,
  linkedConnection,
  defaultConnection,
  status,
  isTesting,
  isLinking,
  onValueChange,
  onLink,
  onDisconnect,
  onTest,
}) {
  const field = getConnectionFieldConfig(backend, drafts);
  const activeConnection = linkedConnection || defaultConnection;
  const isBundledDemo =
    backend === "sqlite" && !linkedConnection && Boolean(defaultConnection?.autoConnected);
  const panelCopy = isBundledDemo
    ? "SQLite is already pointed at the bundled demo database, so you can run queries immediately. Use this form only if you want to replace it with another live database."
    : "Link a real database without storing its password in the browser. The backend keeps a temporary secure session id instead. Compare All Backends uses your linked live databases when available and falls back to the built-in learning schema for the rest.";
  const cardCopy =
    activeConnection?.previewNames?.length > 0
      ? `Visible now: ${activeConnection.previewNames.join(", ")}`
      : isBundledDemo
        ? "The backend bundled the SQLite demo file, so the browser does not need a connection secret."
        : "The backend has the link, and the browser no longer needs the secret.";

  return (
    <div className="connection-panel">
      <div className="connection-panel-header">
        <div>
          <h3 className="connection-panel-title">Live Database Link</h3>
          <p className="connection-panel-copy">{panelCopy}</p>
        </div>
      </div>

      <div className="connection-form">
        <label className="control-label" htmlFor={`connection-${backend}`}>
          <DatabaseZap className="tiny-icon" />
          {field.label}
        </label>

        <input
          id={`connection-${backend}`}
          className="query-input"
          type={field.inputType}
          value={field.value}
          placeholder={field.placeholder}
          onChange={(event) => onValueChange(field.fieldKey, event.target.value)}
          autoComplete="off"
        />

        <p className="connection-help">{field.helpText}</p>

        {activeConnection ? (
          <div className={isBundledDemo ? "connection-linked-card demo" : "connection-linked-card"}>
            <div className="connection-linked-top">
              <span className="connection-linked-badge">
                {isBundledDemo ? activeConnection.label || "Bundled demo database" : "Linked for this session"}
              </span>
              <span className="connection-linked-summary">
                {activeConnection.entityCount} {activeConnection.entityLabel}
              </span>
            </div>
            <p className="connection-linked-copy">{cardCopy}</p>
            {isBundledDemo && activeConnection.message ? (
              <p className="connection-linked-note">{activeConnection.message}</p>
            ) : null}
          </div>
        ) : (
          <div className="connection-status neutral">
            <LockKeyhole className="tiny-icon" />
            <span>No live database linked for this backend yet.</span>
          </div>
        )}

        {status?.message ? (
          <div className={status.kind === "error" ? "connection-status error" : "connection-status"}>
            {status.kind === "success" ? <CheckCircle2 className="tiny-icon" /> : null}
            {status.kind === "error" ? <ShieldX className="tiny-icon" /> : null}
            <span>{status.message}</span>
          </div>
        ) : null}
      </div>

      <div className="connection-action-row">
        <button type="button" className="ghost-button" onClick={onTest} disabled={isTesting || isLinking}>
          <DatabaseZap className="tiny-icon" />
          {isTesting ? "Testing..." : "Test Connection"}
        </button>
        <button type="button" className="secondary-button" onClick={onLink} disabled={isLinking}>
          <LockKeyhole className="tiny-icon" />
          {isLinking ? "Linking..." : "Link Database"}
        </button>
        {linkedConnection?.connectionId ? (
          <button type="button" className="ghost-button" onClick={onDisconnect}>
            <Unplug className="tiny-icon" />
            Disconnect
          </button>
        ) : null}
      </div>
    </div>
  );
}
