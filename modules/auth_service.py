"""
auth_service.py

Authentication service for login & registration.

Improvements:
- Uses db_connection() context manager (no manual close)
- Returns typed Optional[int] instead of falsy None
- Clearer exception handling
"""

import bcrypt
from typing import Optional
from auth_db import db_connection


# ===============================
# Password Hashing
# ===============================

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


# ===============================
# Register User
# ===============================

def register_user(username: str, email: str, password: str) -> bool:
    """Returns True on success, False if username/email already taken."""
    if not username or not email or not password:
        return False

    hashed_password = hash_password(password)
    try:
        with db_connection() as conn:
            conn.execute(
                "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                (username.strip(), email.strip().lower(), hashed_password),
            )
        return True
    except Exception:
        return False


# ===============================
# Login User
# ===============================

def login_user(username: str, password: str) -> Optional[int]:
    """Returns user_id on success, None on failure."""
    if not username or not password:
        return None

    try:
        with db_connection() as conn:
            row = conn.execute(
                "SELECT user_id, password FROM users WHERE username = ?",
                (username.strip(),),
            ).fetchone()
    except Exception:
        return None

    if row and verify_password(password, row["password"]):
        return row["user_id"]

    return None