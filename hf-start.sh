#!/usr/bin/env bash
set -euo pipefail

export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://${OLLAMA_HOST}}"
export SQLITE_PATH="${SQLITE_PATH:-/app/backend/test.db}"

ollama serve >/tmp/ollama.log 2>&1 &
OLLAMA_PID=$!

cleanup() {
  if kill -0 "$OLLAMA_PID" 2>/dev/null; then
    kill "$OLLAMA_PID"
    wait "$OLLAMA_PID" || true
  fi
}

trap cleanup EXIT

echo "Waiting for Ollama on ${OLLAMA_BASE_URL}..."
ready=0
for attempt in $(seq 1 60); do
  if curl -fsS "${OLLAMA_BASE_URL}/api/tags" >/dev/null; then
    ready=1
    break
  fi
  sleep 2
done

if [ "$ready" -ne 1 ]; then
  cat /tmp/ollama.log || true
  exit 1
fi

echo "Ensuring model ${OLLAMA_MODEL} is available..."
ollama pull "${OLLAMA_MODEL}"

cd /app/backend
exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-7860}"
