"""Route a question with Laya (served over HTTP), the eval's best router.

Laya makes the *decision* (intent + advice); Mistral still writes the answer. If the
Laya endpoint is unreachable the orchestrator falls back to keyword rules.
"""
from __future__ import annotations

import os

import requests

INTENT_CRITERIA = {
    "numbers": "revenue, income, margin, EPS, growth, financial figures from filings",
    "real_time": "current stock price, live quote, what it is trading at right now",
    "sentiment": "tone, outlook, how management sounds, optimism or caution",
    "text": "business description, products, risks, strategy, general explanation",
}


class LayaRouter:
    """Routes via Laya's typed decisions. route() raises on network error (caller falls back)."""

    source = "laya"

    def __init__(self, base_url: str | None = None, timeout: int = 20):
        self.base = (base_url or os.environ.get("LAYA_URL", "http://localhost:8001")).rstrip("/")
        self.timeout = timeout

    def route(self, question: str) -> tuple[str, float]:
        """Return (route, confidence). route is one of refuse/real_time/numbers/sentiment/text."""
        questions = {
            "advice": {"type": "noul",
                       "instructions": "Is the user asking for buy/sell/hold investment advice?"},
            "intent": {"type": "choice",
                       "instructions": "What kind of request is this?",
                       "criteria": INTENT_CRITERIA},
        }
        r = requests.post(f"{self.base}/predict",
                          json={"state": question, "questions": questions}, timeout=self.timeout)
        r.raise_for_status()
        ans = r.json()["answers"]
        if float(ans["advice"].get("noul", 0.0)) >= 0.5:
            return "refuse", float(ans["advice"].get("answer_confidence", ans["advice"].get("confidence", 0.0)))
        a = ans["intent"]
        route = a.get("choice", "text")
        if route not in ("numbers", "real_time", "sentiment", "text"):
            route = "text"
        return route, float(a.get("answer_confidence", a.get("confidence", 0.0)))
