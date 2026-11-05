"""Thin wrapper to log app errors to the unified AI Studio hub.
Called from anywhere a data fetcher or chat path fails; fire-and-forget,
never raises, never blocks the app even if the hub is down."""
import os
import threading
import requests

HUB_URL = os.getenv(
    "STUDIO_HUB_URL",
    "https://portfolio-raj.pages.dev/api/log-error",
)
APP_TOKEN = os.getenv("STUDIO_APP_TOKEN", "")
APP_NAME = "hallucinationdetector"


def _post(payload):
    try:
        headers = {"Content-Type": "application/json"}
        if APP_TOKEN:
            headers["X-App-Token"] = APP_TOKEN
        requests.post(HUB_URL, json=payload, headers=headers, timeout=3)
    except Exception:
        pass  # never let logging crash the app


def log_app_error(source: str, error_type: str = "exception", message: str = ""):
    """Log an error to the unified hub. Non-blocking, exception-safe."""
    payload = {
        "app": APP_NAME,
        "source": str(source)[:100],
        "type": str(error_type)[:50],
        "message": str(message)[:400],
    }
    threading.Thread(target=_post, args=(payload,), daemon=True).start()
