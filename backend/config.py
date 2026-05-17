"""Environment-backed configuration helpers for database connections."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env_flag(name: str, default: bool) -> bool:
    """Parse one boolean-like environment variable."""

    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _env_csv(name: str, default: str) -> tuple[str, ...]:
    """Parse one comma-separated environment variable into a stable tuple."""

    raw_value = os.getenv(name, default)
    return tuple(
        item.strip()
        for item in raw_value.split(",")
        if item and item.strip()
    )


@dataclass(frozen=True, slots=True)
class Settings:
    """Container for the database connection strings used by the backend."""

    sqlite_path: str = os.getenv("SQLITE_PATH", "./data/app.db")
    allowed_origins: tuple[str, ...] = field(
        default_factory=lambda: _env_csv("ALLOWED_ORIGINS", "http://localhost:5173")
    )
    demo_database_label: str = os.getenv("DEMO_DATABASE_LABEL", "Bundled demo database")
    postgres_url: str = os.getenv("POSTGRES_URL", "postgresql://user:password@localhost:5432/app")
    mongo_url: str = os.getenv("MONGO_URL", "mongodb://localhost:27017")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    ollama_timeout_seconds: int = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60"))
    ollama_keep_alive: str = os.getenv("OLLAMA_KEEP_ALIVE", "10m")
    ollama_max_tokens: int = int(os.getenv("OLLAMA_MAX_TOKENS", "384"))
    ollama_multi_model_fallback_enabled: bool = _env_flag("OLLAMA_MULTI_MODEL_FALLBACK_ENABLED", False)
    query_graph_fast_path_enabled: bool = _env_flag("QUERY_GRAPH_FAST_PATH_ENABLED", False)


settings = Settings()
