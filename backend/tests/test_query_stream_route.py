"""Streaming route tests for live workflow progress."""

from __future__ import annotations

from fastapi.testclient import TestClient

from main import app


def test_query_stream_emits_progress_events_and_final_payload(
    sqlite_backend: dict[str, object],
) -> None:
    """The streamed query endpoint should emit progress events before the final response."""

    client = TestClient(app)
    payload = {
        "question": "show user emails",
        "backend": "sqlite",
        "connection": {"sqlite_path": str(sqlite_backend["path"])},
    }

    with client.stream("POST", "/query/stream", json=payload) as response:
        assert response.status_code == 200
        stream_text = "".join(response.iter_text())

    assert "event: workflow_started" in stream_text
    assert "event: schema_ready" in stream_text
    assert "event: step" in stream_text
    assert "event: completed" in stream_text
    assert '"compiled_query": "SELECT users.email FROM users LIMIT 10"' in stream_text
