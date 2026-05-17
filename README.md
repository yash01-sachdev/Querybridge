---
title: Query Bridge API
emoji: 🧠
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
short_description: FastAPI + Ollama backend for the Query Bridge resume demo.
---

# Query Bridge

This repository is set up for the same free deployment split you used before:

- `frontend/` deploys to Vercel
- the repo root deploys to a Hugging Face Docker Space for the FastAPI + Ollama backend

The backend image bundles a SQLite demo database at `backend/test.db`, and the frontend now shows it as ready automatically. That means SQLite prompts work immediately without manual linking.

## Good demo prompts

- `show user emails`
- `show users where name does not contain ali`
- `show order amounts with user names`
- `average order amount by user`

## Deployment guide

Use [docs/vercel-huggingface-deploy.md](/docs/vercel-huggingface-deploy.md) for the Vercel + Hugging Face setup steps.
