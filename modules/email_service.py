"""
email_service.py

Email Service for Smart AI Data Intelligence System.

Handles:
- Email verification  (token link sent on registration)
- Welcome email       (sent after email is confirmed)
- Password reset      (placeholder, same pattern)

Configuration via environment variables or Streamlit secrets:
    SMTP_HOST      e.g. smtp.gmail.com
    SMTP_PORT      e.g. 587
    SMTP_USER      your Gmail / SMTP address
    SMTP_PASSWORD  app-password (not your Google account password)
    APP_BASE_URL   e.g. http://localhost:8501

For Gmail you must:
  1. Enable 2-Factor Auth on your Google account
  2. Create an App Password at https://myaccount.google.com/apppasswords
  3. Use that 16-char password as SMTP_PASSWORD
"""

import os
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

import streamlit as st


# ============================================================
# Config helpers
# ============================================================

def _cfg(key: str, fallback: str = "") -> str:
    """Read from st.secrets first, then os.environ, then fallback."""
    try:
        return st.secrets.get(key, os.getenv(key, fallback))
    except Exception:
        return os.getenv(key, fallback)


def get_smtp_config() -> dict:
    return {
        "host":     _cfg("SMTP_HOST",     "smtp.gmail.com"),
        "port":     int(_cfg("SMTP_PORT", "587")),
        "user":     _cfg("SMTP_USER",     ""),
        "password": _cfg("SMTP_PASSWORD", ""),
        "base_url": _cfg("APP_BASE_URL",  "http://localhost:8501"),
    }


# ============================================================
# Core send function
# ============================================================

def _send_email(to_address: str, subject: str, html_body: str, text_body: str) -> bool:
    """
    Send an email via SMTP TLS.
    Returns True on success, False on any failure.
    """
    cfg = get_smtp_config()

    if not cfg["user"] or not cfg["password"]:
        # Dev mode — print to console instead of crashing
        print(f"\n[EmailService] ⚠️  SMTP not configured.")
        print(f"[EmailService] TO:      {to_address}")
        print(f"[EmailService] SUBJECT: {subject}")
        print(f"[EmailService] BODY:\n{text_body}\n")
        return True   # treat as success so the app keeps working

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"Smart AI Intelligence <{cfg['user']}>"
    msg["To"]      = to_address

    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(cfg["host"], cfg["port"], timeout=15) as server:
            server.ehlo()
            server.starttls(context=context)
            server.login(cfg["user"], cfg["password"])
            server.sendmail(cfg["user"], to_address, msg.as_string())
        return True
    except Exception as e:
        print(f"[EmailService] Send failed: {e}")
        return False


# ============================================================
# Email Templates
# ============================================================

_BASE_STYLE = """
<style>
  body { font-family: 'Inter', Arial, sans-serif; background: #f6fafe; margin: 0; padding: 0; }
  .wrapper { max-width: 560px; margin: 40px auto; background: #ffffff;
             border-radius: 16px; overflow: hidden;
             box-shadow: 0 8px 30px rgba(0,101,146,0.1); }
  .header { background: linear-gradient(135deg, #006592, #005980);
            padding: 36px 40px; text-align: center; }
  .header h1 { color: #f5f9ff; font-size: 24px; margin: 0;
               font-family: 'Space Grotesk', Arial, sans-serif; }
  .header p  { color: rgba(255,255,255,0.75); margin: 6px 0 0; font-size: 13px; }
  .body  { padding: 36px 40px; color: #2a3439; }
  .body h2 { font-size: 20px; margin: 0 0 12px; color: #006592; }
  .body p  { font-size: 14px; line-height: 1.7; color: #576067; margin: 0 0 16px; }
  .btn { display: inline-block; padding: 14px 32px;
         background: linear-gradient(135deg, #006592, #005980);
         color: #f5f9ff !important; text-decoration: none;
         border-radius: 30px; font-weight: 700; font-size: 14px;
         margin: 8px 0 24px; }
  .footer { padding: 20px 40px; background: #eff4f9;
            text-align: center; font-size: 11px; color: #a9b3ba; }
  .divider { height: 1px; background: #e8eff4; margin: 20px 0; }
  .chip { display: inline-block; background: rgba(0,109,74,0.1);
          color: #006d4a; border-radius: 20px; padding: 3px 10px;
          font-size: 11px; font-weight: 700; letter-spacing: 0.05em; }
</style>
"""


def _verification_html(username: str, verify_url: str) -> str:
    return f"""<!DOCTYPE html><html><head>{_BASE_STYLE}</head><body>
<div class="wrapper">
  <div class="header">
    <h1>🧠 Smart AI Intelligence</h1>
    <p>Data Intelligence Platform</p>
  </div>
  <div class="body">
    <h2>Verify your email address</h2>
    <p>Hi <strong>{username}</strong>,</p>
    <p>Thanks for signing up! Click the button below to verify your email address
       and activate your account. The link expires in <strong>24 hours</strong>.</p>
    <center>
      <a class="btn" href="{verify_url}">✅ Verify My Email</a>
    </center>
    <div class="divider"></div>
    <p style="font-size:12px">If you didn't create this account you can safely ignore this email.</p>
    <p style="font-size:12px; word-break:break-all; color:#a9b3ba">
      Or copy this link: {verify_url}
    </p>
  </div>
  <div class="footer">© Smart AI Intelligence · You received this because you signed up.</div>
</div>
</body></html>"""


def _verification_text(username: str, verify_url: str) -> str:
    return (
        f"Hi {username},\n\n"
        f"Please verify your email by visiting:\n{verify_url}\n\n"
        f"This link expires in 24 hours.\n\n"
        f"Smart AI Intelligence"
    )


def _welcome_html(username: str) -> str:
    return f"""<!DOCTYPE html><html><head>{_BASE_STYLE}</head><body>
<div class="wrapper">
  <div class="header">
    <h1>🎉 Welcome aboard!</h1>
    <p>Smart AI Data Intelligence Platform</p>
  </div>
  <div class="body">
    <h2>Your account is ready, {username}!</h2>
    <p>We're thrilled to have you. Your email has been verified and your
       <strong>Smart AI</strong> account is now fully active.</p>
    <p>Here's what you can do right away:</p>
    <table style="width:100%;border-collapse:collapse">
      <tr>
        <td style="padding:10px 0;border-bottom:1px solid #e8eff4">
          📂 <strong>Upload a CSV dataset</strong>
        </td>
        <td style="padding:10px 0;border-bottom:1px solid #e8eff4;color:#576067">
          Any structured data file
        </td>
      </tr>
      <tr>
        <td style="padding:10px 0;border-bottom:1px solid #e8eff4">
          🔍 <strong>Run the AI pipeline</strong>
        </td>
        <td style="padding:10px 0;border-bottom:1px solid #e8eff4;color:#576067">
          Auto-cleans, models & scores
        </td>
      </tr>
      <tr>
        <td style="padding:10px 0;border-bottom:1px solid #e8eff4">
          📊 <strong>Explore visual insights</strong>
        </td>
        <td style="padding:10px 0;border-bottom:1px solid #e8eff4;color:#576067">
          Charts, SHAP, forecasts
        </td>
      </tr>
      <tr>
        <td style="padding:10px 0">
          💬 <strong>Chat with your data</strong>
        </td>
        <td style="padding:10px 0;color:#576067">
          Ask questions in plain English
        </td>
      </tr>
    </table>
    <div class="divider"></div>
    <p>If you have any questions, just reply to this email — we read every message.</p>
    <p>Happy analysing! 🚀</p>
    <p style="margin:0"><strong>The Smart AI Team</strong></p>
  </div>
  <div class="footer">© Smart AI Intelligence · You received this because you just verified your account.</div>
</div>
</body></html>"""


def _welcome_text(username: str) -> str:
    return (
        f"Hi {username},\n\n"
        f"Welcome to Smart AI Intelligence! Your account is now active.\n\n"
        f"You can now:\n"
        f"  • Upload CSV datasets\n"
        f"  • Run the full AI analysis pipeline\n"
        f"  • Explore visual insights and forecasts\n"
        f"  • Chat with your data\n\n"
        f"Happy analysing!\n"
        f"The Smart AI Team"
    )


# ============================================================
# Public API
# ============================================================

def send_verification_email(to_address: str, username: str, token: str) -> bool:
    """Send the email-verification link."""
    cfg        = get_smtp_config()
    verify_url = f"{cfg['base_url']}?verify_token={token}"
    subject    = "Verify your Smart AI account"
    return _send_email(
        to_address,
        subject,
        _verification_html(username, verify_url),
        _verification_text(username, verify_url),
    )


def send_welcome_email(to_address: str, username: str) -> bool:
    """Send the welcome / onboarding email after verification."""
    subject = f"Welcome to Smart AI, {username}! 🎉"
    return _send_email(
        to_address,
        subject,
        _welcome_html(username),
        _welcome_text(username),
    )