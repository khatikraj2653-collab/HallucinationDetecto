"""
Best-effort activity logging to the shared portfolio dashboard.
Never raises — a failed/slow log call must never break the app.
"""
import os
import re
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # ensures PORTFOLIO_LOG_SECRET is loaded even on pages that don't call this themselves

LOG_URL = "https://portfolio-raj.pages.dev/api/log"
LOG_SECRET = os.getenv("PORTFOLIO_LOG_SECRET", "")
APP_NAME = "hallucinationdetector"

# Mirrors portfolio-raj/_middleware.js so the shared dashboard sees the same
# human-vs-bot split regardless of which app the log came from. Catches:
# search engine / LLM crawlers (bingbot, gptbot, claude-web, perplexity,
# applebot, ...), uptime monitors (uptimerobot, pingdom), scrapers
# (semrush, ahrefs, dataforseo), headless HTTP clients (python-requests,
# curl, axios), social-preview fetchers (facebookexternalhit, slackbot,
# linkedinbot), and our own keep-alive workflow (tagged KeepAliveBot/1.0).
# Case-insensitive; no UA at all is treated as a bot since every real
# browser sends one.
_BOT_PATTERN = re.compile(
    r"bot|crawl|spider|slurp|facebookexternalhit|preview|monitor|uptimerobot|pingdom|statuscake|"
    r"gptbot|google-inspectiontool|claude-?web|anthropic|openai|duckduckgo|yandex|baiduspider|"
    r"semrush|ahrefs|mj12|dotbot|petalbot|chatgpt|perplexity|headlesschrome|lighthouse|pagespeed|"
    r"dataforseo|curl|wget|python-requests|httpclient|axios|java/",
    re.IGNORECASE,
)


def _is_bot(ua):
    if not ua:
        return True
    return bool(_BOT_PATTERN.search(ua))


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
    if _is_bot(ua):
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
                "visitorIp": ip,
                "userAgent": ua,
                "referrer": referrer,
            },
            headers={"X-Log-Secret": LOG_SECRET},
            timeout=2,
        )
    except Exception:
        pass
