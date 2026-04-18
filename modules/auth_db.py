"""
auth_db.py

Database management for Smart AI Data Intelligence System.

ADDITIONS:
- email_verified column (users cannot login until verified)
- email_verification_tokens table (token, expiry, used flag)
- password_reset_tokens table (for future forgot-password flow)
- google_id + google_email columns for OAuth users
- Thread-safe, WAL mode, indexed for fast lookup
"""

import sqlite3
from contextlib import contextmanager

DB_NAME = "smart_ai.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def db_connection():
    """Context manager — commits on success, rolls back on error, always closes."""
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
            -- ── Users ──────────────────────────────────────────────────────
            CREATE TABLE IF NOT EXISTS users (
                user_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                username       TEXT    NOT NULL UNIQUE,
                email          TEXT    NOT NULL UNIQUE,
                password       TEXT,                          -- NULL for Google-only accounts
                google_id      TEXT    UNIQUE,                -- Google OAuth sub
                google_email   TEXT,
                email_verified INTEGER NOT NULL DEFAULT 0,    -- 0 = unverified, 1 = verified
                created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_users_username  ON users(username);
            CREATE INDEX IF NOT EXISTS idx_users_email     ON users(email);
            CREATE INDEX IF NOT EXISTS idx_users_google_id ON users(google_id);

            -- ── Email Verification Tokens ───────────────────────────────────
            CREATE TABLE IF NOT EXISTS email_verification_tokens (
                token_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                token      TEXT    NOT NULL UNIQUE,
                expires_at TIMESTAMP NOT NULL,
                used       INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_evt_token   ON email_verification_tokens(token);
            CREATE INDEX IF NOT EXISTS idx_evt_user_id ON email_verification_tokens(user_id);

            -- ── Password Reset Tokens ───────────────────────────────────────
            CREATE TABLE IF NOT EXISTS password_reset_tokens (
                token_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                token      TEXT    NOT NULL UNIQUE,
                expires_at TIMESTAMP NOT NULL,
                used       INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_prt_token ON password_reset_tokens(token);
        """)


# ============================================================
# Helper queries used by auth_service
# ============================================================

def get_user_by_email(email: str):
    """Return full user row by email, or None."""
    with db_connection() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE email = ?",
            (email.strip().lower(),)
        ).fetchone()


def get_user_by_google_id(google_id: str):
    """Return full user row by google_id, or None."""
    with db_connection() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE google_id = ?",
            (google_id,)
        ).fetchone()


def get_user_by_id(user_id: int):
    """Return full user row by user_id, or None."""
    with db_connection() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE user_id = ?",
            (user_id,)
        ).fetchone()


def mark_email_verified(user_id: int) -> None:
    with db_connection() as conn:
        conn.execute(
            "UPDATE users SET email_verified = 1 WHERE user_id = ?",
            (user_id,)
        )


def save_verification_token(user_id: int, token: str, expires_at) -> None:
    with db_connection() as conn:
        # Invalidate any previous unused tokens for this user
        conn.execute(
            "UPDATE email_verification_tokens SET used = 1 WHERE user_id = ? AND used = 0",
            (user_id,)
        )
        conn.execute(
            "INSERT INTO email_verification_tokens (user_id, token, expires_at) VALUES (?, ?, ?)",
            (user_id, token, expires_at)
        )


def get_verification_token_row(token: str):
    with db_connection() as conn:
        return conn.execute(
            "SELECT * FROM email_verification_tokens WHERE token = ?",
            (token,)
        ).fetchone()


def consume_verification_token(token_id: int) -> None:
    with db_connection() as conn:
        conn.execute(
            "UPDATE email_verification_tokens SET used = 1 WHERE token_id = ?",
            (token_id,)
        )