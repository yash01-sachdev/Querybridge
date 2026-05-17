export function createEmptyConnectionDrafts() {
  return {
    sqlite: {
      sqlitePath: "",
    },
    postgresql: {
      postgresUrl: "",
    },
    mongodb: {
      mongoUrl: "",
    },
  };
}

export function buildConnectionPayload(backend, drafts) {
  if (backend === "sqlite") {
    const sqlitePath = drafts?.sqlite?.sqlitePath?.trim();
    return sqlitePath ? { sqlite_path: sqlitePath } : null;
  }

  if (backend === "postgresql") {
    const postgresUrl = drafts?.postgresql?.postgresUrl?.trim();
    return postgresUrl ? { postgres_url: postgresUrl } : null;
  }

  if (backend === "mongodb") {
    const mongoUrl = drafts?.mongodb?.mongoUrl?.trim();
    return mongoUrl ? { mongo_url: mongoUrl } : null;
  }

  return null;
}

export function clearConnectionDraft(backend, drafts) {
  const nextDrafts = createEmptyConnectionDrafts();
  nextDrafts.sqlite = { ...drafts.sqlite };
  nextDrafts.postgresql = { ...drafts.postgresql };
  nextDrafts.mongodb = { ...drafts.mongodb };

  if (backend === "sqlite") {
    nextDrafts.sqlite.sqlitePath = "";
  } else if (backend === "postgresql") {
    nextDrafts.postgresql.postgresUrl = "";
  } else if (backend === "mongodb") {
    nextDrafts.mongodb.mongoUrl = "";
  }

  return nextDrafts;
}

export function getConnectionFieldConfig(backend, drafts) {
  if (backend === "sqlite") {
    return {
      fieldKey: "sqlitePath",
      label: "SQLite Database File",
      placeholder: "E:/data/shop.db",
      helpText: "Only the backend keeps this link after you connect it.",
      value: drafts?.sqlite?.sqlitePath || "",
      inputType: "text",
    };
  }

  if (backend === "postgresql") {
    return {
      fieldKey: "postgresUrl",
      label: "PostgreSQL Connection URL",
      placeholder: "postgresql+psycopg2://postgres:password@localhost:5432/shop",
      helpText: "The browser uses this only for linking, then clears it from the form.",
      value: drafts?.postgresql?.postgresUrl || "",
      inputType: "password",
    };
  }

  return {
    fieldKey: "mongoUrl",
    label: "MongoDB Connection URL",
    placeholder: "mongodb://localhost:27017/shop",
    helpText: "Include the database name at the end, then link it securely for this session.",
    value: drafts?.mongodb?.mongoUrl || "",
    inputType: "password",
  };
}
