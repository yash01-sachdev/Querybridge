# Vercel + Hugging Face Deploy Guide

This project is wired for a free demo deployment with:

- `frontend/` on Vercel
- repo root on a Hugging Face Docker Space
- Ollama running inside the Docker Space
- bundled SQLite demo data already available at startup

## 1. What is already wired

- The backend Docker image starts `ollama serve`, ensures `qwen2.5:3b` is available, then launches FastAPI on port `7860`.
- The backend defaults `SQLITE_PATH` to the bundled demo file in `backend/test.db`.
- The frontend reads `VITE_API_URL` and trims trailing slashes automatically.
- The health endpoint now reports whether the bundled demo database is ready, and the UI shows SQLite as ready without a manual link.

## 2. Deploy the backend to Hugging Face Spaces

Create a new Hugging Face Space and choose `Docker` as the SDK. This repo already includes:

- root `Dockerfile`
- root `README.md` with the Space YAML block
- `hf-start.sh` startup script

### Recommended Space variables

Set these in the Space settings:

- `ALLOWED_ORIGINS=https://YOUR-VERCEL-PROJECT.vercel.app`
- `OLLAMA_MODEL=qwen2.5:3b`
- `OLLAMA_TIMEOUT_SECONDS=60`
- `OLLAMA_KEEP_ALIVE=15m`
- `OLLAMA_MAX_TOKENS=384`
- `OLLAMA_MULTI_MODEL_FALLBACK_ENABLED=false`
- `QUERY_GRAPH_FAST_PATH_ENABLED=false`
- `SQLITE_PATH=/app/backend/test.db`

If you also use a custom frontend domain, add it to `ALLOWED_ORIGINS` as a comma-separated second value.

Example:

```text
ALLOWED_ORIGINS=https://query-bridge.vercel.app,https://your-custom-domain.com
```

### Backend smoke check

After the Space finishes building, open:

- `https://YOUR-SPACE-NAME.hf.space/health`

You should see:

- `"status": "ok"`
- `"ollama": { "available": true, ... }`
- `"demo_database": { "available": true, ... }`

## 3. Deploy the frontend to Vercel

Import the same GitHub repo into Vercel, then set:

- Root Directory: `frontend`
- Build Command: `npm run build`
- Output Directory: `dist`

Add this environment variable in Vercel:

```text
VITE_API_URL=https://YOUR-SPACE-NAME.hf.space
```

The frontend already uses `import.meta.env.VITE_API_URL`, so no extra code changes are needed.

## 4. Demo flow after deploy

Once both sides are live:

1. Open the Vercel URL.
2. Leave SQLite selected.
3. Confirm the connection panel shows the bundled demo database as ready.
4. Run the prompts below without linking anything manually.

## 5. Suggested demo prompts

- `show user emails`
- `show users where name does not contain ali`
- `show order amounts with user names`
- `average order amount by user`

## 6. Important Hugging Face note

Free Spaces can sleep when idle. After a cold wake-up, the first request may take longer while the Space and Ollama warm up. After that, the normal demo flow should be much quicker.
