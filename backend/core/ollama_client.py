"""Small Ollama client helpers used by the LangGraph query workflow."""

from __future__ import annotations

import json
from typing import Any

import requests

from config import settings

DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"
MAX_GENERATION_TIMEOUT_SECONDS = 30
DEV_FALLBACK_MODEL_ORDER = (
    "qwen2.5:3b",
    "qwen3:8b",
    "llama3:latest",
    "sqlcoder:latest",
)
PREFERRED_GENERATION_MODEL_PREFIXES = (
    "qwen",
    "llama",
    "gemma",
    "mistral",
    "deepseek",
    "phi",
    "codellama",
    "sqlcoder",
)
RETRYABLE_OLLAMA_ERROR_FRAGMENTS = (
    "runner process has terminated",
    "timed out waiting",
    "load failed",
    "context canceled",
    "context cancelled",
    "connection closed before server finished loading",
)


class OllamaUnavailableError(RuntimeError):
    """Raised when the local Ollama service cannot be reached."""


class OllamaResponseError(RuntimeError):
    """Raised when Ollama returns an unusable response."""


def get_active_model_name() -> str:
    """Return the configured or auto-resolved local Ollama model name."""

    model_name, _ = _resolve_model_name()
    return model_name


def get_ollama_status() -> dict[str, Any]:
    """Return whether the local Ollama server is reachable and model metadata."""

    installed_names = _fetch_installed_model_names()
    if installed_names is None:
        return {
            "available": False,
            "base_url": settings.ollama_base_url,
            "model": _configured_model_name(),
            "configured_model": _configured_model_name(),
            "installed": False,
        }

    model_name, using_fallback = _resolve_model_name(installed_names)
    configured_model = _configured_model_name()
    configured_model_installed = configured_model in installed_names

    return {
        "available": True,
        "base_url": settings.ollama_base_url,
        "model": model_name,
        "configured_model": configured_model,
        "configured_model_installed": configured_model_installed,
        "installed": bool(installed_names),
        "using_fallback": using_fallback,
        "available_models": sorted(installed_names),
    }


def generate_json(prompt: str, candidate_models: list[str] | None = None) -> dict[str, Any]:
    """Ask Ollama for JSON output and parse the returned object."""

    response_text = _generate(prompt, response_format="json", candidate_models=candidate_models)

    try:
        parsed_response = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise OllamaResponseError(f"Ollama did not return valid JSON: {exc}") from exc

    if not isinstance(parsed_response, dict):
        raise OllamaResponseError("Ollama must return one JSON object.")

    return parsed_response


def generate_text(prompt: str, candidate_models: list[str] | None = None) -> str:
    """Ask Ollama for plain text output."""

    return _generate(prompt, response_format=None, candidate_models=candidate_models).strip()


def _generate(
    prompt: str,
    response_format: str | None,
    candidate_models: list[str] | None = None,
) -> str:
    """Send one non-streaming generation request to the local Ollama service."""

    endpoint = f"{settings.ollama_base_url.rstrip('/')}/api/generate"
    installed_names = _fetch_installed_model_names()
    candidate_models = _requested_or_default_candidate_models(candidate_models, installed_names)
    last_error_message = ""

    for index, model_name in enumerate(candidate_models):
        payload: dict[str, Any] = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "keep_alive": settings.ollama_keep_alive,
            "options": {
                "temperature": 0,
                "num_predict": settings.ollama_max_tokens,
            },
        }

        if response_format is not None:
            payload["format"] = response_format

        try:
            response = requests.post(
                endpoint,
                json=payload,
                timeout=_generation_timeout_seconds(),
            )
        except requests.RequestException as exc:
            raise OllamaUnavailableError(
                "Could not reach Ollama. Start it with 'ollama serve' and make sure the "
                f"model '{model_name}' is available."
            ) from exc

        if response.status_code >= 400:
            last_error_message = (
                f"Ollama request failed with status {response.status_code} for model "
                f"'{model_name}': {response.text}"
            )
            if (
                index + 1 < len(candidate_models)
                and _is_retryable_runtime_error(response.status_code, response.text)
            ):
                continue

            raise OllamaResponseError(last_error_message)

        try:
            response_payload = response.json()
        except ValueError as exc:
            raise OllamaResponseError("Ollama returned a non-JSON HTTP response.") from exc

        response_text = str(response_payload.get("response", "")).strip()
        if response_text:
            return response_text

        last_error_message = f"Ollama returned an empty response for model '{model_name}'."
        if index + 1 < len(candidate_models):
            continue

        raise OllamaResponseError(last_error_message)

    raise OllamaResponseError(last_error_message or "Ollama did not return a usable response.")


def _requested_or_default_candidate_models(
    requested_models: list[str] | None,
    installed_names: set[str] | None,
) -> list[str]:
    """Return the requested models, or fall back to the default candidate list."""

    default_candidates = _generation_candidate_models(installed_names)
    if not requested_models:
        return default_candidates

    candidate_models: list[str] = []

    for model_name in requested_models:
        cleaned_name = str(model_name).strip()
        if cleaned_name and cleaned_name not in candidate_models:
            candidate_models.append(cleaned_name)

    return candidate_models or default_candidates


def _resolve_model_name(installed_names: set[str] | None = None) -> tuple[str, bool]:
    """Pick the configured model when present, otherwise fall back to a sensible installed model."""

    configured_model = _configured_model_name()
    if installed_names is None:
        installed_names = _fetch_installed_model_names()

    if not installed_names:
        return configured_model, False

    if configured_model in installed_names:
        return configured_model, False

    for prefix in _configured_model_prefixes(configured_model):
        candidate = _find_matching_model(installed_names, prefix)
        if candidate:
            return candidate, True

    for prefix in PREFERRED_GENERATION_MODEL_PREFIXES:
        candidate = _find_matching_model(installed_names, prefix)
        if candidate:
            return candidate, True

    for candidate in sorted(installed_names):
        if _is_generation_model(candidate):
            return candidate, True

    return configured_model, False


def _generation_candidate_models(installed_names: set[str] | None) -> list[str]:
    """Return the ordered generation models to try for one request."""

    primary_model, _ = _resolve_model_name(installed_names)
    candidate_models = [primary_model]

    if not installed_names:
        return candidate_models

    if not settings.ollama_multi_model_fallback_enabled:
        return candidate_models

    retry_fallback = _find_retryable_fallback_model(primary_model, installed_names)
    if retry_fallback and retry_fallback not in candidate_models:
        candidate_models.append(retry_fallback)

    return candidate_models


def _generation_timeout_seconds() -> int:
    """Clamp one generation request to a short interactive budget."""

    try:
        configured_timeout = int(settings.ollama_timeout_seconds)
    except (TypeError, ValueError):
        configured_timeout = MAX_GENERATION_TIMEOUT_SECONDS

    return max(1, min(configured_timeout, MAX_GENERATION_TIMEOUT_SECONDS))


def _find_retryable_fallback_model(primary_model: str, installed_names: set[str]) -> str | None:
    """Return one lighter backup model for runner startup failures."""

    for preferred_name in DEV_FALLBACK_MODEL_ORDER:
        if preferred_name in installed_names and preferred_name != primary_model:
            return preferred_name

    for prefix in PREFERRED_GENERATION_MODEL_PREFIXES:
        candidate = _find_matching_model(installed_names, prefix)
        if candidate and candidate != primary_model:
            return candidate

    for candidate in sorted(installed_names, key=_model_sort_key):
        if _is_generation_model(candidate) and candidate != primary_model:
            return candidate

    return None


def _configured_model_name() -> str:
    """Return one non-empty configured model name."""

    configured_model = settings.ollama_model.strip()
    return configured_model or DEFAULT_OLLAMA_MODEL


def _fetch_installed_model_names() -> set[str] | None:
    """Return the installed Ollama model names, or None when the server is unavailable."""

    endpoint = f"{settings.ollama_base_url.rstrip('/')}/api/tags"

    try:
        response = requests.get(
            endpoint,
            timeout=min(settings.ollama_timeout_seconds, 10),
        )
    except requests.RequestException:
        return None

    if response.status_code >= 400:
        return None

    try:
        payload = response.json()
    except ValueError:
        return None

    installed_models = payload.get("models", [])
    return {
        str(model_info.get("name", "")).strip()
        for model_info in installed_models
        if isinstance(model_info, dict) and str(model_info.get("name", "")).strip()
    }


def _configured_model_prefixes(configured_model: str) -> list[str]:
    """Return the strongest family prefixes for the configured model name."""

    family = configured_model.split(":", maxsplit=1)[0].strip().lower()
    if not family:
        return []

    prefixes = [family]
    if "." in family:
        prefixes.append(family.split(".", maxsplit=1)[0])

    return prefixes


def _find_matching_model(installed_names: set[str], prefix: str) -> str | None:
    """Return the first installed model whose family matches the given prefix."""

    if not prefix:
        return None

    candidates = [
        name
        for name in sorted(installed_names, key=_model_sort_key)
        if _model_family(name).startswith(prefix) and _is_generation_model(name)
    ]

    if not candidates:
        return None

    return candidates[0]


def _model_family(model_name: str) -> str:
    """Return the lowercased family portion of one model name."""

    return model_name.split(":", maxsplit=1)[0].strip().lower()


def _is_generation_model(model_name: str) -> bool:
    """Skip embedding-only models when picking a fallback generation model."""

    lowered_model_name = model_name.lower()
    return "embed" not in lowered_model_name


def _model_sort_key(model_name: str) -> tuple[int, str]:
    """Prefer the smaller dev-friendly installed models before larger fallbacks."""

    if model_name in DEV_FALLBACK_MODEL_ORDER:
        return (DEV_FALLBACK_MODEL_ORDER.index(model_name), model_name)

    return (len(DEV_FALLBACK_MODEL_ORDER), model_name)


def _is_retryable_runtime_error(status_code: int, response_text: str) -> bool:
    """Return whether this Ollama failure looks like a model runner startup problem."""

    if status_code < 500:
        return False

    lowered_text = response_text.lower()
    return any(fragment in lowered_text for fragment in RETRYABLE_OLLAMA_ERROR_FRAGMENTS)
