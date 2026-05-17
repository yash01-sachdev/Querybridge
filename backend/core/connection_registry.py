"""Temporary server-side storage for linked database connections."""

from __future__ import annotations

from dataclasses import dataclass
from secrets import token_urlsafe
from threading import Lock
from time import time

LINK_TTL_SECONDS = 60 * 60 * 2


@dataclass(slots=True)
class StoredConnection:
    """One linked database kept only in backend memory."""

    backend: str
    connection: dict[str, str | None]
    created_at: float
    last_used_at: float


_connections: dict[str, StoredConnection] = {}
_registry_lock = Lock()


def register_connection(backend: str, connection: dict[str, str | None]) -> str:
    """Store a connection server-side and return an opaque temporary id."""

    now = time()
    connection_id = token_urlsafe(24)

    with _registry_lock:
        _cleanup_expired_connections(now)
        _connections[connection_id] = StoredConnection(
            backend=backend,
            connection=dict(connection),
            created_at=now,
            last_used_at=now,
        )

    return connection_id


def get_connection(connection_id: str, backend: str) -> dict[str, str | None]:
    """Return a linked connection if it still exists and matches the backend."""

    now = time()

    with _registry_lock:
        _cleanup_expired_connections(now)
        stored_connection = _connections.get(connection_id)

        if stored_connection is None:
            raise ValueError("That database link has expired or does not exist anymore.")

        if stored_connection.backend != backend:
            raise ValueError("That database link was created for a different backend.")

        stored_connection.last_used_at = now
        return dict(stored_connection.connection)


def remove_connection(connection_id: str) -> bool:
    """Delete a linked connection and report whether one was removed."""

    with _registry_lock:
        return _connections.pop(connection_id, None) is not None


def _cleanup_expired_connections(now: float) -> None:
    """Drop idle linked connections after the in-memory security window ends."""

    expired_ids = [
        connection_id
        for connection_id, stored_connection in _connections.items()
        if now - stored_connection.last_used_at > LINK_TTL_SECONDS
    ]

    for connection_id in expired_ids:
        _connections.pop(connection_id, None)
