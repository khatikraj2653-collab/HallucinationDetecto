"""
Best-effort activity logging to the shared portfolio dashboard.
Never raises — a failed/slow log call must never break the app.
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()  # ensures PORTFOLIO_LOG_SECRET is loaded even on pages that don't call this themselves

LOG_URL = "https://portfolio-raj.pages.dev/api/log"
LOG_SECRET = os.getenv("PORTFOLIO_LOG_SECRET", "")
APP_NAME = "hallucinationdetector"


def log_event(event_type: str, detail: str = None, question: str = None, reply: str = None):
    if not LOG_SECRET:
        return
    try:
        requests.post(
            LOG_URL,
            json={
                "app": APP_NAME,
                "type": event_type,
                "detail": detail,
                "question": question,
                "reply": reply,
            },
            headers={"X-Log-Secret": LOG_SECRET},
            timeout=2,
        )
    except Exception:
        pass
