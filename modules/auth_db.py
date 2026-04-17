"""
auth_db.py

Database management for Smart AI Data Intelligence System.

Improvements:
- Thread-safe connections (check_same_thread=False)
- Context manager for auto-close / rollback
- WAL journal mode for better write concurrency
- Indexes on username & email for faster lookups
"""

import sqlite3
from contextlib import contextmanager

DB_NAME = "smart_ai.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    conn.row_factory = sqlite3.Row          # dict-like rows
    conn.execute("PRAGMA journal_mode=WAL") # better write concurrency
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def db_connection():
    """Context manager – commits on success, rolls back on error, always closes."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def create_tables() -> None:
    with db_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                user_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                username   TEXT    NOT NULL UNIQUE,
                email      TEXT    NOT NULL UNIQUE,
                password   TEXT    NOT NULL,
                google_id  TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
            CREATE INDEX IF NOT EXISTS idx_users_email    ON users(email);
        """)