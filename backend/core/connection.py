"""Connection helpers for opening the selected backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from pymongo import MongoClient
from sqlalchemy import create_engine

from config import settings

ConnectionHandle = tuple[Any, Callable[[], None]]
ConnectionOverrides = dict[str, str | None] | None


def open_backend_connection(
    backend: str,
    connection_overrides: ConnectionOverrides = None,
) -> ConnectionHandle:
    """Open the requested backend and return the connection with a cleanup function."""

    if backend == "sqlite":
        return _open_sqlite_connection(connection_overrides)

    if backend == "postgresql":
        return _open_postgresql_connection(connection_overrides)

    if backend == "mongodb":
        return _open_mongodb_connection(connection_overrides)

    raise ValueError(f"Unsupported backend: {backend}")


def _open_sqlite_connection(connection_overrides: ConnectionOverrides) -> ConnectionHandle:
    """Open a SQLite connection using the configured path or the local test file."""

    sqlite_path = _get_connection_value(
        key="sqlite_path",
        connection_overrides=connection_overrides,
        default_value=settings.sqlite_path,
    )
    database_path = _resolve_sqlite_path(sqlite_path)

    try:
        engine = create_engine(f"sqlite:///{database_path.as_posix()}")
        connection = engine.connect()
    except Exception as exc:
        raise ValueError(f"Failed to connect to sqlite: {exc}") from exc

    def close_connection() -> None:
        connection.close()
        engine.dispose()

    return connection, close_connection


def _open_postgresql_connection(connection_overrides: ConnectionOverrides) -> ConnectionHandle:
    """Open a PostgreSQL connection using the configured URL."""

    postgres_url = _get_connection_value(
        key="postgres_url",
        connection_overrides=connection_overrides,
        default_value=settings.postgres_url,
    )

    try:
        engine = create_engine(postgres_url)
        connection = engine.connect()
    except Exception as exc:
        raise ValueError(f"Failed to connect to postgresql: {exc}") from exc

    def close_connection() -> None:
        connection.close()
        engine.dispose()

    return connection, close_connection


def _open_mongodb_connection(connection_overrides: ConnectionOverrides) -> ConnectionHandle:
    """Open a MongoDB database connection using the configured URL."""

    mongo_url = _get_connection_value(
        key="mongo_url",
        connection_overrides=connection_overrides,
        default_value=settings.mongo_url,
    )

    try:
        client = MongoClient(mongo_url, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        database = _select_mongodb_database(client, mongo_url)
    except Exception as exc:
        raise ValueError(f"Failed to connect to mongodb: {exc}") from exc

    def close_connection() -> None:
        client.close()

    return database, close_connection


def _get_connection_value(
    key: str,
    connection_overrides: ConnectionOverrides,
    default_value: str,
) -> str:
    """Use an in-request connection value when present, otherwise fall back to settings."""

    if connection_overrides:
        raw_value = connection_overrides.get(key)

        if isinstance(raw_value, str):
            trimmed_value = raw_value.strip()
            if trimmed_value:
                return trimmed_value

    return default_value


def _resolve_sqlite_path(raw_path: str) -> Path:
    """Resolve the SQLite file path relative to the backend folder."""

    backend_root = Path(__file__).resolve().parent.parent
    configured_path = Path(raw_path)

    if not configured_path.is_absolute():
        configured_path = (backend_root / configured_path).resolve()

    if configured_path.exists():
        return configured_path

    local_test_path = backend_root / "test.db"
    if local_test_path.exists():
        return local_test_path

    raise ValueError(
        "SQLite database file not found. Set SQLITE_PATH or create backend/test.db."
    )


def _select_mongodb_database(client: MongoClient, mongo_url: str) -> Any:
    """Choose the MongoDB database named in the URL, or fall back to app."""

    parsed_url = urlparse(mongo_url)
    database_name = parsed_url.path.lstrip("/") or "app"
    return client[database_name]
