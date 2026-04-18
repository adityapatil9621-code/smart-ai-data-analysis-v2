"""
google_oauth.py

Google OAuth 2.0 integration for Smart AI Data Intelligence System.

HOW IT WORKS
────────────
1. User clicks "Continue with Google"
2. App redirects to Google's consent screen via the OAuth URL
3. Google redirects back to the app with ?code=...
4. App exchanges the code for an ID token
5. ID token is verified and user is upserted

SETUP (one-time)
────────────────
1. Go to https://console.cloud.google.com/apis/credentials
2. Create a project (or select existing)
3. Click "Create Credentials" → "OAuth client ID"
4. Application type: "Web application"
5. Authorised redirect URI: http://localhost:8501  (or your deployed URL)
6. Copy Client ID and Client Secret

Add to .streamlit/secrets.toml:

    GOOGLE_CLIENT_ID     = "xxx.apps.googleusercontent.com"
    GOOGLE_CLIENT_SECRET = "GOCSPX-xxxx"
    APP_BASE_URL         = "http://localhost:8501"

Or set them as environment variables with the same names.
"""

import os
import json
import urllib.parse
import urllib.request
from typing import Optional, Dict

import streamlit as st


# ============================================================
# Config helpers
# ============================================================

def _cfg(key: str, fallback: str = "") -> str:
    try:
        return st.secrets.get(key, os.getenv(key, fallback))
    except Exception:
        return os.getenv(key, fallback)


GOOGLE_AUTH_URL  = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO  = "https://www.googleapis.com/oauth2/v3/userinfo"
SCOPES           = "openid email profile"


# ============================================================
# Build the OAuth URL the user clicks
# ============================================================

def get_google_login_url() -> str:
    """
    Build the Google OAuth consent-screen URL.
    Returns an empty string if credentials are not configured.
    """
    client_id = _cfg("GOOGLE_CLIENT_ID")
    base_url  = _cfg("APP_BASE_URL", "http://localhost:8501")

    if not client_id:
        return ""

    params = {
        "client_id":     client_id,
        "redirect_uri":  base_url,
        "response_type": "code",
        "scope":         SCOPES,
        "access_type":   "online",
        "prompt":        "select_account",
    }
    return f"{GOOGLE_AUTH_URL}?{urllib.parse.urlencode(params)}"


# ============================================================
# Exchange auth code → user info dict
# ============================================================

def exchange_code_for_user(code: str) -> Optional[Dict]:
    """
    Exchange the OAuth authorization code for user information.

    Returns a dict with keys: sub, email, name
    Returns None on any failure.
    """
    client_id     = _cfg("GOOGLE_CLIENT_ID")
    client_secret = _cfg("GOOGLE_CLIENT_SECRET")
    base_url      = _cfg("APP_BASE_URL", "http://localhost:8501")

    if not client_id or not client_secret:
        return None

    # ── Step 1: Exchange code for access token ────────────────
    token_data = urllib.parse.urlencode({
        "code":          code,
        "client_id":     client_id,
        "client_secret": client_secret,
        "redirect_uri":  base_url,
        "grant_type":    "authorization_code",
    }).encode()

    try:
        req = urllib.request.Request(
            GOOGLE_TOKEN_URL,
            data=token_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            token_response = json.loads(resp.read())
    except Exception as e:
        print(f"[GoogleOAuth] Token exchange failed: {e}")
        return None

    access_token = token_response.get("access_token")
    if not access_token:
        print(f"[GoogleOAuth] No access_token in response: {token_response}")
        return None

    # ── Step 2: Fetch user info ───────────────────────────────
    try:
        req2 = urllib.request.Request(
            GOOGLE_USERINFO,
            headers={"Authorization": f"Bearer {access_token}"},
            method="GET",
        )
        with urllib.request.urlopen(req2, timeout=10) as resp2:
            user_info = json.loads(resp2.read())
    except Exception as e:
        print(f"[GoogleOAuth] Userinfo fetch failed: {e}")
        return None

    return {
        "sub":   user_info.get("sub", ""),
        "email": user_info.get("email", ""),
        "name":  user_info.get("name", user_info.get("email", "User")),
    }


# ============================================================
# Streamlit helper — call at top of app to handle callback
# ============================================================

def handle_google_callback() -> Optional[Dict]:
    """
    Checks URL query params for ?code=... set by Google after consent.
    If found, exchanges code for user info and removes the param.
    Returns user_info dict or None.
    """
    params = st.query_params

    if "code" not in params:
        return None

    code = params["code"]

    # Remove code from URL immediately (security + UX)
    st.query_params.clear()

    user_info = exchange_code_for_user(code)
    return user_info


def is_google_configured() -> bool:
    """Returns True if Google OAuth credentials are present."""
    return bool(_cfg("GOOGLE_CLIENT_ID")) and bool(_cfg("GOOGLE_CLIENT_SECRET"))