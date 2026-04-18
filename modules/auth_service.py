"""
auth_service.py

Authentication service for Smart AI Data Intelligence System.

ADDITIONS:
- register_user() creates unverified account + sends verification email
- verify_email_token() marks account verified + sends welcome email
- login_user() blocks unverified accounts with clear message
- upsert_google_user() creates / fetches user from Google OAuth payload
- get_username_by_id() helper for display
"""

import secrets
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Tuple

from auth_db import (
    db_connection,
    get_user_by_email,
    get_user_by_google_id,
    get_user_by_id,
    mark_email_verified,
    save_verification_token,
    get_verification_token_row,
    consume_verification_token,
)
from email_service import send_verification_email, send_welcome_email


# ── Token TTL ────────────────────────────────────────────────────────────────
TOKEN_EXPIRY_HOURS = 24


# ============================================================
# Password helpers
# ============================================================

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


# ============================================================
# Register (email + password)
# ============================================================

def register_user(username: str, email: str, password: str) -> Tuple[bool, str]:
    """
    Creates an UNVERIFIED account and sends a verification email.

    Returns:
        (True,  "")            on success
        (False, reason_string) on failure
    """
    if not username or not email or not password:
        return False, "All fields are required."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    email    = email.strip().lower()
    username = username.strip()
    hashed   = hash_password(password)

    try:
        with db_connection() as conn:
            conn.execute(
                "INSERT INTO users (username, email, password, email_verified) VALUES (?, ?, ?, 0)",
                (username, email, hashed),
            )
    except Exception:
        return False, "Username or email already in use."

    # Fetch the newly created user_id
    row = get_user_by_email(email)
    if not row:
        return False, "Account creation failed. Please try again."

    user_id = row["user_id"]

    # Generate and persist a verification token
    token      = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRY_HOURS)
    save_verification_token(user_id, token, expires_at)

    # Send verification email (non-blocking failure)
    send_verification_email(email, username, token)

    return True, ""


# ============================================================
# Email Verification
# ============================================================

def verify_email_token(token: str) -> Tuple[bool, str]:
    """
    Validates a verification token.

    Returns:
        (True,  "")             token valid → account is now active
        (False, reason_string)  invalid / expired / already used
    """
    row = get_verification_token_row(token)

    if not row:
        return False, "Invalid or already-used verification link."

    if row["used"]:
        return False, "This verification link has already been used."

    expires_at = datetime.fromisoformat(str(row["expires_at"]))
    if datetime.utcnow() > expires_at:
        return False, "Verification link has expired. Please register again."

    # Mark token consumed and user verified
    consume_verification_token(row["token_id"])
    mark_email_verified(row["user_id"])

    # Fetch user details for the welcome email
    user = get_user_by_id(row["user_id"])
    if user:
        send_welcome_email(user["email"], user["username"])

    return True, ""


# ============================================================
# Login (email + password)
# ============================================================

def login_user(username: str, password: str) -> Tuple[Optional[int], str]:
    """
    Returns:
        (user_id, "")           success
        (None,    reason_str)   failure — reason shown in UI
    """
    if not username or not password:
        return None, "Username and password are required."

    try:
        with db_connection() as conn:
            row = conn.execute(
                "SELECT user_id, password, email_verified FROM users WHERE username = ?",
                (username.strip(),),
            ).fetchone()
    except Exception:
        return None, "Database error. Please try again."

    if not row:
        return None, "Invalid username or password."

    if not row["password"]:
        # Google-only account
        return None, "This account uses Google Sign-In. Please log in with Google."

    if not verify_password(password, row["password"]):
        return None, "Invalid username or password."

    if not row["email_verified"]:
        return None, "Please verify your email address before logging in. Check your inbox."

    return row["user_id"], ""


# ============================================================
# Resend Verification Email
# ============================================================

def resend_verification(email: str) -> Tuple[bool, str]:
    """Resend a fresh verification token to an unverified account."""
    row = get_user_by_email(email)
    if not row:
        return False, "No account found with that email."
    if row["email_verified"]:
        return False, "This account is already verified."

    token      = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRY_HOURS)
    save_verification_token(row["user_id"], token, expires_at)
    send_verification_email(row["email"], row["username"], token)

    return True, ""


# ============================================================
# Google OAuth — upsert user
# ============================================================

def upsert_google_user(google_id: str, email: str, name: str) -> Optional[int]:
    """
    Creates a new user from Google OAuth data, or returns existing user_id.
    Google accounts are automatically email-verified.

    Args:
        google_id: The 'sub' field from the Google ID token
        email:     Verified email from Google
        name:      Display name from Google

    Returns:
        user_id on success, None on failure
    """
    print("🔹 START Google Login")
    print("google_id:", google_id)
    print("email:", email)
    print("name:", name)
    email = email.strip().lower()

    # 1. Known Google user → return existing id
    existing = get_user_by_google_id(google_id)
    print("STEP 1 existing:", existing)
    if existing:
        print("✅ Found existing Google user")

        return existing["user_id"]

    # 2. Email already registered (password account) → link Google
    by_email = get_user_by_email(email)
    print("STEP 2 by_email:", by_email)

    if by_email:
        print("🔹 Linking Google account...")
        try:
            with db_connection() as conn:
                conn.execute(
                    """UPDATE users
                       SET google_id = ?, google_email = ?, email_verified = 1
                       WHERE user_id = ?""",
                    (google_id, email, by_email["user_id"]),
                )
            print("✅ UPDATE SUCCESS")
            return by_email["user_id"]
        except Exception as e:
            print("❌ UPDATE ERROR:", e)
            return None

    # 3. Brand-new Google user → create account (no password, auto-verified)
    print("🔹 Creating new Google user...")
    username = _unique_username(name)
    try:
        with db_connection() as conn:
            conn.execute(
                """INSERT INTO users
                   (username, email, password, google_id, google_email, email_verified)
                   VALUES (?, ?, NULL, ?, ?, 1)""",
                (username, email, google_id, email),
            )

            print("✅ INSERT SUCCESS")
    except Exception as e:
            print("❌ INSERT ERROR:", e)
            return None

    row = get_user_by_email(email)
    print("STEP 3 row:", row)
    if not row:
        print("❌ FETCH FAILED")
        return None

    # Send welcome email for new Google users
    send_welcome_email(email, username)
    print("✅ USER CREATED")

    return row["user_id"]


def _unique_username(display_name: str) -> str:
    """Derive a safe username from a Google display name."""
    base = "".join(c for c in display_name.lower().replace(" ", "_") if c.isalnum() or c == "_")
    base = base[:20] or "user"
    # Check uniqueness
    try:
        with db_connection() as conn:
            existing = conn.execute(
                "SELECT username FROM users WHERE username = ?", (base,)
            ).fetchone()
        if not existing:
            return base
        # Append short random suffix
        suffix = secrets.token_hex(3)
        return f"{base}_{suffix}"
    except Exception:
        return f"user_{secrets.token_hex(4)}"


# ============================================================
# Helper
# ============================================================

def get_username_by_id(user_id: int) -> str:
    row = get_user_by_id(user_id)
    return row["username"] if row else str(user_id)