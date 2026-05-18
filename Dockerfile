FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_BASE_URL=http://127.0.0.1:11434 \
    OLLAMA_MODEL=qwen2.5:1.5b \
    OLLAMA_TIMEOUT_SECONDS=30 \
    OLLAMA_KEEP_ALIVE=15m \
    OLLAMA_MAX_TOKENS=384 \
    OLLAMA_MULTI_MODEL_FALLBACK_ENABLED=false \
    QUERY_GRAPH_FAST_PATH_ENABLED=false \
    SQLITE_PATH=/app/backend/test.db \
    ALLOWED_ORIGINS=*

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates curl zstd \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

COPY backend/requirements.txt /app/backend/requirements.txt

RUN python -m pip install --upgrade pip \
    && python -m pip install -r /app/backend/requirements.txt

COPY backend /app/backend
COPY hf-start.sh /app/hf-start.sh

RUN chmod +x /app/hf-start.sh \
    && bash -lc "ollama serve >/tmp/ollama-build.log 2>&1 & \
        ollama_pid=\$!; \
        ready=0; \
        for attempt in \$(seq 1 60); do \
            if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null; then \
                ready=1; \
                break; \
            fi; \
            sleep 2; \
        done; \
        if [ \"\$ready\" -ne 1 ]; then \
            cat /tmp/ollama-build.log; \
            exit 1; \
        fi; \
        ollama pull \"\$OLLAMA_MODEL\"; \
        kill \"\$ollama_pid\"; \
        wait \"\$ollama_pid\" || true"

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD curl -fsS http://127.0.0.1:7860/health || exit 1

CMD ["/app/hf-start.sh"]
