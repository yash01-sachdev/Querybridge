"""Shared demo databases for built-in GenAI eval runs."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

from sqlalchemy import create_engine, text

from core.runtime_paths import iter_temp_root_candidates, pick_first_writable_directory
from tests.fake_mongo import FakeMongoDatabase

FIXTURE_RUNTIME_ROOT_NAME = "nl-query-copilot-fixtures"


def create_demo_sqlite_database(base_dir: Path) -> tuple[object, Path]:
    """Create a small relational demo database for SQL evals."""

    runtime_root = _resolve_fixture_runtime_root(base_dir)
    database_path = runtime_root / f"sqlite-fixture-{uuid4().hex}.db"
    database_path.unlink(missing_ok=True)
    engine = create_engine(f"sqlite:///{database_path.as_posix()}")

    with engine.begin() as connection:
        # This workspace blocks SQLite's default file-backed journaling on new temp files.
        connection.execute(text("PRAGMA journal_mode=MEMORY"))
        connection.execute(text("PRAGMA synchronous=OFF"))
        connection.execute(
            text(
                """
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    email TEXT
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE orders (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER,
                    amount REAL,
                    status TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO users (id, name, email) VALUES
                (1, 'Alice', 'alice@example.com'),
                (2, 'Bob', 'bob@example.com'),
                (3, 'Charlie', 'charlie@example.com')
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO orders (id, user_id, amount, status) VALUES
                (1, 1, 120.5, 'completed'),
                (2, 2, 60.0, 'pending'),
                (3, 1, 300.0, 'completed'),
                (4, 3, 90.0, 'cancelled')
                """
            )
        )

    return engine, database_path


def cleanup_demo_sqlite_database(engine: object, database_path: Path) -> None:
    """Dispose a demo SQLite engine and best-effort remove its temp files."""

    dispose = getattr(engine, "dispose", None)
    if callable(dispose):
        dispose()

    _remove_sqlite_sidecar_files(database_path)
    _remove_file_if_possible(database_path)


def create_demo_mongo_database() -> FakeMongoDatabase:
    """Create a small in-memory MongoDB-like demo database."""

    return FakeMongoDatabase(
        {
            "users": [
                {"name": "Alice", "email": "alice@example.com", "status": "active"},
                {"name": "Bob", "email": "bob@example.com", "status": "active"},
                {"name": "Charlie", "email": "charlie@example.com", "status": "inactive"},
            ],
            "orders": [
                {"user_name": "Alice", "amount": 120.5, "status": "completed"},
                {"user_name": "Bob", "amount": 60.0, "status": "pending"},
                {"user_name": "Alice", "amount": 300.0, "status": "completed"},
                {"user_name": "Charlie", "amount": 90.0, "status": "cancelled"},
            ],
        }
    )


def _remove_sqlite_sidecar_files(database_path: Path) -> None:
    """Remove SQLite sidecar files left behind after a test session."""

    for suffix in ("-journal", "-shm", "-wal"):
        _remove_file_if_possible(database_path.with_name(f"{database_path.name}{suffix}"))


def _remove_file_if_possible(file_path: Path) -> None:
    """Best-effort cleanup for Windows paths that may still be locked briefly."""

    try:
        file_path.unlink(missing_ok=True)
    except PermissionError:
        pass


def _resolve_fixture_runtime_root(base_dir: Path) -> Path:
    """Prefer the system temp directory so eval runs do not clutter the repo."""

    candidate_roots: list[Path] = []
    temp_root = _best_effort_temp_root()
    if temp_root is not None:
        candidate_roots.append(temp_root / FIXTURE_RUNTIME_ROOT_NAME)

    candidate_roots.append(base_dir / "test_runtime" / "eval-fixtures")

    for candidate_root in candidate_roots:
        try:
            candidate_root.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue

        return candidate_root

    raise ValueError("Could not create a writable runtime directory for eval fixtures.")


def _best_effort_temp_root() -> Path | None:
    """Return one usable temp root without crashing when the system temp is misconfigured."""

    return pick_first_writable_directory(iter_temp_root_candidates())
