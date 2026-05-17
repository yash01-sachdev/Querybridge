export const BACKEND_OPTIONS = [
  { value: "sqlite", label: "SQLite" },
  { value: "postgresql", label: "PostgreSQL" },
  { value: "mongodb", label: "MongoDB" },
];

export const COMPARE_BACKENDS = BACKEND_OPTIONS.map((backend) => backend.value);

export function formatBackendLabel(backend) {
  const matchedBackend = BACKEND_OPTIONS.find((option) => option.value === backend);
  return matchedBackend ? matchedBackend.label : backend;
}
