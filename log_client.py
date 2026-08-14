"""
Best-effort activity logging to the shared portfolio dashboard.
Never raises — a failed/slow log call must never break the app.
"""
import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # ensures PORTFOLIO_LOG_SECRET is loaded even on pages that don't call this themselves

LOG_URL = "https://portfolio-raj.pages.dev/api/log"
LOG_SECRET = os.getenv("PORTFOLIO_LOG_SECRET", "")
APP_NAME = "hallucinationdetector"


def _visitor_info():
    """Best-effort real visitor IP/User-Agent from the incoming browser
    request, so repeat visits from the same person show up as the same
    hashed visitor in the shared dashboard instead of every log looking
    identical (previously all logs came from this server's own IP)."""
    try:
        headers = st.context.headers
        forwarded = headers.get("X-Forwarded-For") or headers.get("Cf-Connecting-Ip") or ""
        ip = forwarded.split(",")[0].strip() if forwarded else None
        return ip, headers.get("User-Agent"), headers.get("Referer")
    except Exception:
        return None, None, None


def log_event(event_type: str, detail: str = None, question: str = None, reply: str = None):
    if not LOG_SECRET:
        return
    ip, ua, referrer = _visitor_info()
    try:
        requests.post(
            LOG_URL,
            json={
                "app": APP_NAME,
                "type": event_type,
                "detail": detail,
                "question": question,
                "reply": reply,
                "visitorIp": ip,
                "userAgent": ua,
                "referrer": referrer,
            },
            headers={"X-Log-Secret": LOG_SECRET},
            timeout=2,
        )
    except Exception:
        pass
