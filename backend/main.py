"""FastAPI entry point for the nl-query-copilot backend."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from core.connection import open_backend_connection
from core.ollama_client import get_active_model_name, get_ollama_status
from core.schema_extractor import extract_schema
from routes.query import router as query_router

app = FastAPI(title="nl-query-copilot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allowed_origins) or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router)


@app.get("/health")
def health() -> dict[str, object]:
    """Return API and GenAI runtime status for the frontend."""

    ollama_status = get_ollama_status()

    return {
        "status": "ok",
        "workflow": "langgraph",
        "model": get_active_model_name(),
        "ollama": ollama_status,
        "demo_database": _get_demo_database_status(),
    }


def _get_demo_database_status() -> dict[str, object]:
    """Describe whether the bundled SQLite demo database is ready to use."""

    try:
        connection, close_connection = open_backend_connection("sqlite")
    except ValueError as exc:
        return {
            "backend": "sqlite",
            "available": False,
            "auto_connected": False,
            "label": settings.demo_database_label,
            "message": str(exc),
        }

    try:
        schema = extract_schema("sqlite", connection)
    except ValueError as exc:
        return {
            "backend": "sqlite",
            "available": False,
            "auto_connected": False,
            "label": settings.demo_database_label,
            "message": str(exc),
        }
    finally:
        close_connection()

    raw_tables = schema.get("tables", {})
    table_names = list(raw_tables.keys()) if isinstance(raw_tables, dict) else []

    return {
        "backend": "sqlite",
        "available": True,
        "auto_connected": True,
        "label": settings.demo_database_label,
        "entity_type": "tables",
        "entity_count": len(table_names),
        "entity_names": table_names[:12],
        "message": "Bundled demo database ready. SQLite queries work immediately without linking.",
    }
