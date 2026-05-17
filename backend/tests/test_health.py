"""Health endpoint tests for backend and Ollama runtime status."""

from __future__ import annotations

from fastapi.testclient import TestClient

from main import app


def test_health_reports_genai_runtime(monkeypatch) -> None:
    """Health should expose workflow, model, and Ollama runtime status."""

    monkeypatch.setattr("main.get_active_model_name", lambda: "llama3:latest")
    monkeypatch.setattr(
        "main.get_ollama_status",
        lambda: {
            "available": True,
            "base_url": "http://127.0.0.1:11434",
            "model": "llama3:latest",
            "configured_model": "llama3.1:8b",
            "configured_model_installed": False,
            "installed": True,
            "using_fallback": True,
        },
    )
    monkeypatch.setattr(
        "main._get_demo_database_status",
        lambda: {
            "backend": "sqlite",
            "available": True,
            "auto_connected": True,
            "label": "Bundled demo database",
            "entity_type": "tables",
            "entity_count": 2,
            "entity_names": ["orders", "users"],
            "message": "Bundled demo database ready. SQLite queries work immediately without linking.",
        },
    )

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["workflow"] == "langgraph"
    assert payload["model"] == "llama3:latest"
    assert payload["ollama"]["available"] is True
    assert payload["demo_database"]["available"] is True
    assert payload["demo_database"]["entity_names"] == ["orders", "users"]
