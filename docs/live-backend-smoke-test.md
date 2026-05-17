# Live Backend Smoke Test

This project can now run the same GenAI workflow against:

- SQLite
- PostgreSQL
- MongoDB

## 1. Start PostgreSQL and MongoDB

From the repo root:

```powershell
docker compose -f docker-compose.dev.yml up -d
```

This starts:

- PostgreSQL on `localhost:5432`
- MongoDB on `localhost:27017`

Both are seeded with a small demo dataset.

## 2. Start Ollama

Make sure Ollama is running and the model exists:

```powershell
ollama pull qwen2.5:3b
ollama serve
```

## 3. Start the backend

In `E:\nl-query-copilot\backend`:

```powershell
copy .env.example .env
python -m pip install -r requirements.txt
uvicorn main:app --reload
```

## 4. Start the frontend

In `E:\nl-query-copilot\frontend`:

```powershell
npm install
npm run dev
```

## 5. Use the bundled SQLite demo or link the others

Open [http://localhost:5173](http://localhost:5173).

SQLite now auto-wires to the bundled demo database, so you can run SQLite prompts immediately without linking it manually.

If you want to test live links, use these connection values:

- SQLite override: `E:\nl-query-copilot\backend\test.db`
- PostgreSQL: `postgresql://postgres:postgres@localhost:5432/app`
- MongoDB: `mongodb://localhost:27017/app`

Use `Test Connection`, then `Link Database` for any backend you want to override.

## 6. Good smoke-test prompts

Try these in the Query tab:

- `show user emails`
- `how many users`
- `user with name ending with b`
- `show users where name contains alice or email contains bob`
- `count users by status`
- `average order amount by user name`

Then try `Compare All Backends` after linking all three.

Each compare card should say `Schema source: linked live database`.

## 7. What success looks like

- `/health` shows GenAI ready
- Query tab returns compiled query plus rows
- Workflow Trace shows model schema selection, model planning, semantic check, validation, execution, and explanation steps
- Compare mode uses live schema for linked backends
- Evals tab still runs the built-in suites

## 8. Stop the demo databases

From the repo root:

```powershell
docker compose -f docker-compose.dev.yml down
```
