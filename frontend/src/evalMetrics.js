const MAX_RECORDED_RUNS = 20;

export function createSuccessfulRun(question, backend, response) {
  return {
    id: Date.now(),
    question,
    backend,
    workflow: response?.workflow || "",
    model: response?.model || "",
    compiledQuery: response?.compiled_query || "",
    rowCount: response?.result?.row_count || 0,
    safetyPassed: response?.safety_check?.passed ?? null,
    validationPassed: response?.validation_check?.passed ?? null,
    repaired: response?.repaired || false,
    repairAttempts: response?.repair_attempts || 0,
    latencyMs: response?.execution_time_ms || 0,
    traceStepCount: response?.trace?.node_count || response?.trace?.steps?.length || 0,
    status: "Pass",
    message: response?.message || "Query completed successfully.",
  };
}

export function createFailedRun(question, backend, errorMessage, errorTrace = null) {
  return {
    id: Date.now(),
    question,
    backend,
    workflow: "",
    model: "",
    compiledQuery: "",
    rowCount: 0,
    safetyPassed: null,
    validationPassed: null,
    repaired: false,
    repairAttempts: 0,
    latencyMs: null,
    traceStepCount: errorTrace?.node_count || errorTrace?.steps?.length || 0,
    status: "Fail",
    message: errorMessage || "Query failed.",
  };
}

export function appendRunRecord(runs, nextRun) {
  return [nextRun, ...runs].slice(0, MAX_RECORDED_RUNS);
}

export function buildEvalSummary(runs) {
  const totalRuns = runs.length;
  const successfulRuns = runs.filter((run) => run.status === "Pass").length;
  const repairedRuns = runs.filter((run) => run.repaired).length;
  const safetyChecks = runs.filter((run) => typeof run.safetyPassed === "boolean");
  const validationChecks = runs.filter((run) => typeof run.validationPassed === "boolean");
  const latencyRuns = runs.filter((run) => typeof run.latencyMs === "number");

  return [
    {
      label: "Total Runs",
      value: String(totalRuns),
      sublabel: totalRuns === 1 ? "1 live query run" : `${totalRuns} live query runs`,
      iconKey: "runs",
    },
    buildRateCard("Successful Runs", successfulRuns, totalRuns, "success"),
    buildRateCard(
      "Safety Pass Rate",
      safetyChecks.filter((run) => run.safetyPassed).length,
      safetyChecks.length,
      "success",
    ),
    buildRateCard(
      "Validation Pass Rate",
      validationChecks.filter((run) => run.validationPassed).length,
      validationChecks.length,
      "info",
    ),
    buildRateCard("Repair Usage", repairedRuns, successfulRuns, "warning"),
    {
      label: "Avg Latency",
      value:
        latencyRuns.length > 0
          ? `${average(latencyRuns.map((run) => run.latencyMs)).toFixed(0)}ms`
          : "—",
      sublabel: latencyRuns.length > 0 ? "from successful runs" : "run queries to measure latency",
      iconKey: "latency",
    },
  ];
}

export function buildEvalRows(runs) {
  return runs.map((run) => ({
    id: run.id,
    question: run.question,
    backend: run.backend,
    workflow: run.workflow || "—",
    model: run.model || "—",
    compiledQuery: run.compiledQuery || run.message,
    rowCount: run.rowCount,
    safetyStatus: toStatus(run.safetyPassed),
    validationStatus: toStatus(run.validationPassed),
    repairStatus: run.repaired ? `${run.repairAttempts} attempt${run.repairAttempts === 1 ? "" : "s"}` : "No",
    traceSteps: run.traceStepCount || 0,
    latencyLabel: typeof run.latencyMs === "number" ? `${run.latencyMs.toFixed(0)}ms` : "—",
    status: run.status,
    message: run.message,
  }));
}

export function buildEvalSuiteSummary(evalRunResponse) {
  if (!evalRunResponse) {
    return [];
  }

  return [
    {
      label: "Suite Cases",
      value: String(evalRunResponse.totalCases),
      sublabel: evalRunResponse.datasetSource,
      iconKey: "runs",
    },
    {
      label: "Pass Rate",
      value: `${evalRunResponse.passedCases}/${evalRunResponse.totalCases}`,
      sublabel: `${evalRunResponse.passRate.toFixed(0)}%`,
      iconKey: "success",
      progress: {
        width: `${evalRunResponse.passRate}%`,
        label: `${evalRunResponse.passRate.toFixed(0)}%`,
        tone: "success",
      },
    },
    {
      label: "Failures",
      value: String(evalRunResponse.failedCases),
      sublabel:
        evalRunResponse.failedCases === 1
          ? "1 prompt needs attention"
          : `${evalRunResponse.failedCases} prompts need attention`,
      iconKey: "repair",
    },
    {
      label: "Avg Latency",
      value: `${evalRunResponse.avgLatencyMs.toFixed(0)}ms`,
      sublabel: `${evalRunResponse.workflow} · ${evalRunResponse.model || "local model"}`,
      iconKey: "latency",
    },
  ];
}

export function buildEvalSuiteRows(evalRunResponse) {
  if (!evalRunResponse) {
    return [];
  }

  return (evalRunResponse.cases || []).map((caseResult) => ({
    id: `${evalRunResponse.backend}-${caseResult.name}`,
    question: caseResult.question,
    expected: caseResult.expected,
    status: caseResult.status,
    compiledQuery: caseResult.compiled_query || caseResult.message,
    rowCount: caseResult.row_count || 0,
    latencyLabel:
      typeof caseResult.latency_ms === "number" ? `${caseResult.latency_ms.toFixed(0)}ms` : "—",
    traceSteps: caseResult.trace?.node_count || caseResult.trace?.steps?.length || 0,
    message: caseResult.message,
  }));
}

function buildRateCard(label, passedCount, totalCount, tone) {
  if (totalCount === 0) {
    return {
      label,
      value: "—",
      sublabel: "no live runs yet",
      iconKey: tone === "warning" ? "repair" : tone === "info" ? "validation" : "safety",
    };
  }

  const percentage = (passedCount / totalCount) * 100;

  return {
    label,
    value: `${passedCount}/${totalCount}`,
    sublabel: `${percentage.toFixed(0)}%`,
    iconKey: tone === "warning" ? "repair" : tone === "info" ? "validation" : "safety",
    progress: {
      width: `${percentage}%`,
      label: `${percentage.toFixed(0)}%`,
      tone,
    },
  };
}

function average(values) {
  if (values.length === 0) {
    return 0;
  }

  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function toStatus(value) {
  if (value === true) {
    return "Pass";
  }

  if (value === false) {
    return "Fail";
  }

  return "Pending";
}
