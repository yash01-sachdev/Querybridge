"""Tests for local Ollama model resolution and fallback behavior."""

from __future__ import annotations

from types import SimpleNamespace

from core import ollama_client


class _FakeResponse:
    """Small requests-like response object used in Ollama client tests."""

    def __init__(self, status_code: int, payload: dict[str, object]) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self) -> dict[str, object]:
        return self._payload


def test_ollama_status_falls_back_to_installed_generation_model(monkeypatch) -> None:
    """Status should report a usable installed model when the configured tag is missing."""

    monkeypatch.setattr(
        ollama_client,
        "settings",
        SimpleNamespace(
            ollama_base_url="http://127.0.0.1:11434",
            ollama_model="llama3.1:8b",
            ollama_timeout_seconds=90,
            ollama_keep_alive="10m",
            ollama_max_tokens=384,
            ollama_multi_model_fallback_enabled=False,
        ),
    )
    monkeypatch.setattr(
        ollama_client.requests,
        "get",
        lambda *args, **kwargs: _FakeResponse(
            200,
            {
                "models": [
                    {"name": "nomic-embed-text:latest"},
                    {"name": "llama3:latest"},
                    {"name": "sqlcoder:latest"},
                ]
            },
        ),
    )

    status = ollama_client.get_ollama_status()

    assert status["available"] is True
    assert status["installed"] is True
    assert status["model"] == "llama3:latest"
    assert status["configured_model"] == "llama3.1:8b"
    assert status["configured_model_installed"] is False
    assert status["using_fallback"] is True


def test_generate_text_uses_resolved_fallback_model(monkeypatch) -> None:
    """Generation requests should target the resolved fallback model, not the missing configured tag."""

    requested_models: list[str] = []

    monkeypatch.setattr(
        ollama_client,
        "settings",
        SimpleNamespace(
            ollama_base_url="http://127.0.0.1:11434",
            ollama_model="llama3.1:8b",
            ollama_timeout_seconds=90,
            ollama_keep_alive="10m",
            ollama_max_tokens=384,
            ollama_multi_model_fallback_enabled=False,
        ),
    )
    monkeypatch.setattr(
        ollama_client.requests,
        "get",
        lambda *args, **kwargs: _FakeResponse(
            200,
            {
                "models": [
                    {"name": "qwen3:8b"},
                    {"name": "llama3:latest"},
                ]
            },
        ),
    )

    def fake_post(*args, **kwargs):
        requested_models.append(kwargs["json"]["model"])
        return _FakeResponse(200, {"response": "hello from fallback"})

    monkeypatch.setattr(ollama_client.requests, "post", fake_post)

    response_text = ollama_client.generate_text("Say hello.")

    assert response_text == "hello from fallback"
    assert requested_models == ["llama3:latest"]


def test_generate_text_retries_smaller_model_after_runner_failure(monkeypatch) -> None:
    """Runner startup failures should automatically retry a lighter installed fallback model."""

    requested_models: list[str] = []

    monkeypatch.setattr(
        ollama_client,
        "settings",
        SimpleNamespace(
            ollama_base_url="http://127.0.0.1:11434",
            ollama_model="llama3:latest",
            ollama_timeout_seconds=30,
            ollama_keep_alive="10m",
            ollama_max_tokens=384,
            ollama_multi_model_fallback_enabled=True,
        ),
    )
    monkeypatch.setattr(
        ollama_client.requests,
        "get",
        lambda *args, **kwargs: _FakeResponse(
            200,
            {
                "models": [
                    {"name": "llama3:latest"},
                    {"name": "qwen2.5:3b"},
                ]
            },
        ),
    )

    def fake_post(*args, **kwargs):
        requested_models.append(kwargs["json"]["model"])
        if len(requested_models) == 1:
            return _FakeResponse(
                500,
                {"error": "llama runner process has terminated: %!w(<nil>)"},
            )
        return _FakeResponse(200, {"response": "fallback reply"})

    monkeypatch.setattr(ollama_client.requests, "post", fake_post)

    response_text = ollama_client.generate_text("Say hello.")

    assert response_text == "fallback reply"
    assert requested_models == ["llama3:latest", "qwen2.5:3b"]


def test_generate_text_requested_model_does_not_expand_to_default_candidates(monkeypatch) -> None:
    """Explicit candidate model requests should stay scoped to the requested list for speed."""

    requested_models: list[str] = []

    monkeypatch.setattr(
        ollama_client,
        "settings",
        SimpleNamespace(
            ollama_base_url="http://127.0.0.1:11434",
            ollama_model="llama3:latest",
            ollama_timeout_seconds=30,
            ollama_keep_alive="10m",
            ollama_max_tokens=384,
            ollama_multi_model_fallback_enabled=True,
        ),
    )
    monkeypatch.setattr(
        ollama_client.requests,
        "get",
        lambda *args, **kwargs: _FakeResponse(
            200,
            {
                "models": [
                    {"name": "llama3:latest"},
                    {"name": "qwen2.5:3b"},
                ]
            },
        ),
    )

    def fake_post(*args, **kwargs):
        requested_models.append(kwargs["json"]["model"])
        return _FakeResponse(200, {"response": "requested-only reply"})

    monkeypatch.setattr(ollama_client.requests, "post", fake_post)

    response_text = ollama_client.generate_text(
        "Say hello.",
        candidate_models=["qwen2.5:3b"],
    )

    assert response_text == "requested-only reply"
    assert requested_models == ["qwen2.5:3b"]
