"""Backend status chips: Mistral / Laya / Finnhub / SEC EDGAR — live or fallback.

All checks are cheap and non-fatal: a down backend shows a muted/amber chip, never
an error. Used in the top bar so the demo state is always honest and visible.
"""
from __future__ import annotations

import os
import urllib.request

import streamlit as st

from frontend.theme import COLORS, chip


@st.cache_data(ttl=20, show_spinner=False)  # a remote box adds latency; don't ping on every click
def _endpoint_live(url: str, timeout: float = 2.0) -> bool:
    """True only if the server actually RESPONDS with 200 (a dead PSC node behind an open
    tunnel accepts the TCP connection but resets requests, so a socket check isn't enough)."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001 — reset / timeout / error => not live
        return False


def backend_status(use_fake: bool, use_laya: bool) -> dict:
    """Return {name: (label, color)} for each backend given the current toggles."""
    c = COLORS
    if use_fake:
        mistral = ("Mistral offline", c["muted"])
    else:
        # the same URLs the agents call (.env), so a public-IP box shows green too
        url = os.getenv("REMOTE_LLM_URL", "http://localhost:8000/v1").rstrip("/") + "/models"
        mistral = ("Mistral live", c["python"]) if _endpoint_live(url) else ("Mistral down", c["error"])
    if use_laya:
        url = os.getenv("LAYA_URL", "http://localhost:8001").rstrip("/") + "/health"
        laya = ("Laya live", c["python"]) if _endpoint_live(url) else ("Laya down", c["error"])
    else:
        laya = ("Router: keyword", c["muted"])
    finnhub = ("Finnhub live", c["python"]) if os.getenv("FINNHUB_API_KEY") else ("Finnhub: sample", c["laya"])
    sec = ("SEC EDGAR", c["sec"])
    return {"mistral": mistral, "laya": laya, "finnhub": finnhub, "sec": sec}


def chips_html(use_fake: bool, use_laya: bool) -> str:
    """Row of status chips as one HTML string."""
    st = backend_status(use_fake, use_laya)
    return "".join(chip(lbl, col) for lbl, col in st.values())
