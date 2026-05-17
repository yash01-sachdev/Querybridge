"""Seed reusable SQLite demo databases with a small relational shop schema."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TARGETS = (
    Path(r"E:\data\shop.db"),
    REPO_ROOT / "backend" / "test.db",
)

USERS = (
    (1, "Alice", "alice@example.com", "active", "2026-01-10"),
    (2, "Bob", "bob@example.com", "active", "2026-01-12"),
    (3, "Charlie", "charlie@example.com", "inactive", "2026-02-01"),
    (4, "Diana", "diana@example.com", "active", "2026-02-14"),
    (5, "Ethan", "ethan@example.com", "active", "2026-03-03"),
)

ORDERS = (
    (1, 1, 120.50, "completed", "2026-04-01"),
    (2, 2, 60.00, "pending", "2026-04-02"),
    (3, 1, 300.00, "completed", "2026-04-05"),
    (4, 3, 90.00, "cancelled", "2026-04-06"),
    (5, 4, 45.00, "completed", "2026-04-07"),
    (6, 4, 210.00, "completed", "2026-04-08"),
    (7, 2, 150.00, "completed", "2026-04-10"),
    (8, 5, 500.00, "refunded", "2026-04-11"),
)

SCHEMA_SQL = """
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS users;

CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    amount REAL NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(id)
);
"""


def seed_database(database_path: Path) -> None:
    """Create one demo SQLite database with repeatable sample data."""

    database_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(SCHEMA_SQL)
        connection.executemany(
            """
            INSERT INTO users (id, name, email, status, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            USERS,
        )
        connection.executemany(
            """
            INSERT INTO orders (id, user_id, amount, status, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            ORDERS,
        )
        connection.commit()


def main(argv: list[str]) -> int:
    """Seed one or more explicit targets, or the default live/demo targets."""

    targets = [Path(value) for value in argv] if argv else list(DEFAULT_TARGETS)

    for target in targets:
        seed_database(target)
        print(f"Seeded {target}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
