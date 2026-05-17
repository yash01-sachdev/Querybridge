import { formatBackendLabel } from "../backendOptions";

export function DatabaseBadge({ backend }) {
  const normalizedBackend = String(backend).toLowerCase();
  const toneClass = `database-badge ${normalizedBackend}`;
  return <span className={toneClass}>{formatBackendLabel(normalizedBackend)}</span>;
}

export function StatusBadge({ status }) {
  let toneClass = "status-badge neutral";

  if (status === "Pass") {
    toneClass = "status-badge pass";
  } else if (status === "Fail") {
    toneClass = "status-badge fail";
  }

  return <span className={toneClass}>{status}</span>;
}
